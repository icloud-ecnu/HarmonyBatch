from abc import ABC, abstractmethod
from typing import Optional, List, Tuple
import numpy as np
import harmonybatch.algorithm.algorithm as algorithm
from harmonybatch.predictor.cost import FunctionCost
from harmonybatch.predictor.latency import CPULatency, GPULatency, CPUConstraint, CPUMinRelative
from harmonybatch.predictor.util import Apps, Cfg, Instance, App, Mem
import math


def equivalent_timeout(t, r):
    t1, t2 = t[0], t[1]
    r1, r2 = r[0], r[1]
    return t1 + r2 / (r1+r2) * (1-np.exp(-r1 * (t2-t1))) / r1

def equivalent_timeout_multi(timeouts : List[float], rps: List[float]):
    if len(timeouts) == 1:
        return timeouts[0]
    elif len(timeouts) == 2:
        return equivalent_timeout(timeouts, rps)
    else:
        t = equivalent_timeout(timeouts[0:2], rps[0:2])
        return equivalent_timeout_multi([t] + timeouts[2:], [sum(rps[0:2])] + rps[2:])


def get_min_timeout(T : float, slo : List[float], rps : List[float]) -> List[float]:
    l = 0
    h = T
    t = [0.0] * len(slo)
    while l < h:
        m = (l+h) / 2
        t = [m]
        for i in range(len(slo)-1):
            t.append(slo[i+1] - slo[i] + t[-1])
        if equivalent_timeout_multi(t, rps) > T:
            h = m
        else:
            l = m
        if abs(h-l) < 0.001:
            return t
    return t

class FunCfgBase(ABC):
    def __init__(self, config:dict) -> None:
        pass
    
    @abstractmethod
    def get_config(self, apps : Apps) -> Optional[Cfg]:
        pass


class CPUConfig(FunCfgBase):
    def __init__(self, config : dict) -> None:
        self.config = config
        self.model_name = config['model_name']
        self.model_config = config['model_config']
        self.cpu_lat_cal = CPULatency(self.model_config[self.model_name]['CPU'], self.model_name)
        self.cpu_constraint = CPUConstraint(self.model_config[self.model_name]['CPU'], self.model_name)
        self.cpu_relative = CPUMinRelative(self.model_config[self.model_name]['CPU'], self.model_name)
        self.cpu_cost_cal = FunctionCost()
        self.mem_cal = Mem(self.model_config, self.model_name)
        self.mini_cpu = self.mem_cal.get_mini_cpu_batch(self.config["B_CPU"][1])
    
    def get_config(self, apps: Apps) -> Optional[Cfg]:
        cfg = None
        B_CPU_low, B_CPU_high = self.config["B_CPU"]
        Res_CPU_low, Res_CPU_high = self.config["Res_CPU"]

        rps = [app.rps for app in apps.apps]
        R = sum(rps)
        slo = [app.slo for app in apps.apps]

        for b in range(B_CPU_low, B_CPU_high+1):
            min_res = self.mini_cpu[b-1]
            T = (b - 1) / R
            t = get_min_timeout(T, slo, rps)
            t_min = t[0]

            c0 = max(Res_CPU_low, min_res)
            c1 = Res_CPU_high
            # SLO 限制
            cpu = self.cpu_constraint.get_constraint_cpu(b, min(slo) - t_min)
            if cpu is not None:
                c0 = max(c0, cpu)
            else:
                continue
            # 极小值
            c2 = self.cpu_relative.get_min_relative(b, c0, c1)
            
            ins0 = Instance(c0, self.mem_cal.get_cpu_mem(c0, b), None)
            lat0 = self.cpu_lat_cal.lat_avg(ins0, b)
            cost0 = self.cpu_cost_cal.cost(lat0, b, ins0, True)

            ins1 = Instance(c1, self.mem_cal.get_cpu_mem(c1, b), None)
            lat1 = self.cpu_lat_cal.lat_avg(ins1, b)
            cost1 = self.cpu_cost_cal.cost(lat1, b, ins1, True)

            if cfg is None or cost0 < cfg.cost:
                cfg = Cfg(ins0, b, cost0, rps, slo, t,self.cpu_lat_cal.lat_max(ins0, b))

            if cfg is None or cost1 < cfg.cost:   
                cfg = Cfg(ins1, b, cost1, rps, slo, t, self.cpu_lat_cal.lat_max(ins1, b))

            if c2 is not None:
                ins2 = Instance(c2, self.mem_cal.get_cpu_mem(c2, b), None)
                lat2 = self.cpu_lat_cal.lat_avg(ins2, b)
                cost2 = self.cpu_cost_cal.cost(lat2, b, ins2, True)
                if cfg is None or cost2 < cfg.cost:
                    cfg = Cfg(ins2, b, cost2, rps, slo, t, self.cpu_lat_cal.lat_max(ins2, b))
                
        return cfg
    
class GPUConfig(FunCfgBase):
    def __init__(self, config : dict) -> None:
        self.config = config
        self.model_name = config['model_name']
        self.model_config = config['model_config']
        self.gpu_lat_cal = GPULatency(self.model_config[self.model_name]['GPU']['A10'], self.model_name)
        self.gpu_cost_cal = FunctionCost()
        self.mem_cal = Mem(self.model_config, self.model_name)

    
    def get_config(self, apps : Apps) -> Optional[Cfg]:
        cfg = None
        B_GPU_low, B_GPU_high = self.config["B_GPU"]
        Res_GPU_low, Res_GPU_high = self.config["Res_GPU"]
        Res_GPU = list(range(Res_GPU_low, Res_GPU_high+1, 1))

        rps = [app.rps for app in apps.apps]
        R = sum(rps)
        slo = [app.slo for app in apps.apps]

        for g in Res_GPU:
            b_low, b_high = B_GPU_low, B_GPU_high
            while b_low <= b_high:
                b = (b_low + b_high + 1) // 2
                gpu = self.mem_cal.get_gpu_gpu_mem(b)
                if gpu > g:
                    b_high = min(b, b_high-1)
                    continue
                cpu = g / 3
                mem = self.mem_cal.get_gpu_mem(g, b)
                if mem is None:
                    continue
                ins = Instance(cpu, mem, g)
                lat = self.gpu_lat_cal.lat_max(ins, b)
                if max(slo) < lat:
                    b_high = min(b, b_high-1)
                else:
                    t = [s - lat for s in slo]
                    if b == 1:
                        T = 0
                    else:
                        T = equivalent_timeout_multi(t, rps)
                    # 请求率限制
                    if math.floor(R*T + 1) == b:
                        c = self.gpu_cost_cal.cost(self.gpu_lat_cal.lat_avg(ins, b), b, ins, True)
                        if cfg is None or cfg.cost > c:
                            cfg = Cfg(ins, b, c, rps, slo, t, lat)
                        break
                    elif math.floor(R*T + 1) < b:
                        b_high = min(b, b_high-1)
                    else:
                        if b_low == b:
                            c = self.gpu_cost_cal.cost(self.gpu_lat_cal.lat_avg(ins, b), b, ins, True)
                            if cfg is None or cfg.cost > c:
                                cfg = Cfg(ins, b, c, rps, slo, t, lat)
                            break
                        else:
                            b_low = b
        return cfg


class FunCfg(FunCfgBase):
    def __init__(self, config: dict) -> None:
        self.cpu_cfg = CPUConfig(config)
        self.gpu_cfg = GPUConfig(config)

    def get_cpu_cfg(self):
        return self.cpu_cfg

    def get_gpu_cfg(self):
        return self.gpu_cfg
    
    def get_config(self, apps: Apps) -> Optional[Cfg]:
        cfg1 = self.cpu_cfg.get_config(apps)
        cfg2 = self.gpu_cfg.get_config(apps)
        cfg = cfg1
        if cfg1 is None:
            cfg = cfg2
        elif cfg2 is not None and cfg1.cost > cfg2.cost:
            cfg = cfg2
        return cfg

class HarmonyGroup():
    def __init__(self, apps : Apps, ratio : float = 1) -> None:
        self.apps = apps
        self.ratio = ratio
        self.cfg : Optional[Cfg] = None
    
    def get_cfg(self):
        assert self.cfg is not None
        return self.cfg
    
    def add_group_apps(self, group: 'HarmonyGroup'):
        self.apps.add(group.apps.get_apps())
        self.ratio += group.ratio
    
    def get_cost(self):
        assert self.cfg is not None
        return self.cfg.cost * self.ratio
    
    def get_config(self, func_cfg, update_cfg = False):
        cfg = func_cfg.get_config(self.apps)
        if update_cfg or self.cfg is None:
            self.cfg = cfg
        else:
            if self.cfg.cost > cfg.cost:
                self.cfg = cfg
        return self.cfg

    def __str__(self) -> str:
        return str(self.cfg)

    def __repr__(self) -> str:
        return str(self.cfg)
    #     return "group_ratio: {}".format(self.ratio) + str(self.cfg)


def merge(groups: List[HarmonyGroup], low, high, func_cfg: FunCfgBase, merge = False) -> Tuple[List[HarmonyGroup], bool]:
    is_merged = False
    if low >= high:
        return groups, is_merged

    group = HarmonyGroup(Apps([]))
    cost = sum([groups[i].get_cost() for i in range(low, high)])
    for i in range(low, high):
        group.add_group_apps(groups[i])
    group.get_config(func_cfg)
    if merge:
        groups = groups[:low] + [group] + groups[high:]
        is_merged = True
    else:
        if group.get_cost() <= cost:
            groups = groups[:low] + [group] + groups[high:]
            is_merged = True
    return groups, is_merged

class Harmony(algorithm.FunctionCfg):
    def __init__(self, config: dict):
        super().__init__(config)
        self.cfg = FunCfg(config)
    
    def init_knee_point(self, slos):
        self.knee = {}
        rps_low = 0.01
        rps_high = 40
        for slo in slos:
            if slo in self.knee:
                continue
            low = rps_low
            high = rps_high
            while high - low > 0.01:
                rps = (high + low) / 2
                cfg = self.cfg.get_config(Apps([App("test", slo, rps)]))
                assert cfg is not None, str(slo) + "," + str(rps)
                if cfg.instance.gpu is not None:
                    high = rps
                else:
                    low = rps
                # print(slo, rps, cfg)
            self.knee[slo] = high
        # print(self.knee)
                
    
    def init_group(self, apps: algorithm.Apps) -> List[HarmonyGroup]:
        harmony_group = []
        app_list = apps.get_apps()
        R = sum([app.rps for app in app_list])
        for app in app_list:
            harmony_group.append(HarmonyGroup(Apps([app]), app.rps / R))
        return harmony_group
    
    def get_config(self, apps: algorithm.Apps, function_type: algorithm.List[str] = ["CPU", "GPU"]):
        slos = [app.slo for app in apps.apps]
        self.init_knee_point(slos)

        harmony_group = self.init_group(apps)

        func_cfg = self.cfg
        if "CPU" in function_type and "GPU" not in function_type:
            func_cfg = self.cfg.get_cpu_cfg()
        if "GPU" in function_type and "CPU" not in function_type:
            func_cfg = self.cfg.get_gpu_cfg()
        
        for group in harmony_group:
            group.get_config(func_cfg)

        # Stage 1
        i = 0
        j = 0
        r = 0
        r_knee = self.knee[harmony_group[0].apps.apps[0].slo]
        while i < len(harmony_group):
            if harmony_group[i].get_cfg().instance.gpu is None:
                r += harmony_group[i].get_cfg().get_total_rps()
                if r > r_knee:
                    harmony_group, _ = merge(harmony_group, j, i+1, func_cfg, True)
                    i = j
                    r = 0
                    j = j+1
            else:
                j = i+1
                r = 0
            i = i+1

        # Stage 2
        i = 0
        while i < len(harmony_group) - 1:
            if harmony_group[i].get_cfg().instance.gpu is not None or harmony_group[i+1].get_cfg().instance.gpu is not None:
                slo1 = harmony_group[i].get_cfg().slo[-1]
                slo2 = harmony_group[i+1].get_cfg().slo[0]
                index1 = 2
                index2 = 2
                if self.knee[slo1] < 0.05:
                    index1 = 1
                elif self.knee[slo1] > 39.95:
                    index1 = 3
                if self.knee[slo2] < 0.05:
                    index2 = 1
                elif self.knee[slo2] > 39.95:
                    index2 = 3
                if index1 == index2:
                    harmony_group, is_merged = merge(harmony_group, i, i+2, func_cfg)
                    if is_merged:
                        i = i-1
            i = i+1
        
        i = 0
        while i < len(harmony_group) - 1:
            if harmony_group[i].get_cfg().instance.gpu is not None or harmony_group[i+1].get_cfg().instance.gpu is not None:
                harmony_group, is_merged = merge(harmony_group, i, i+2, func_cfg)
                if is_merged:
                    i = i-1
            i = i+1
        
        return harmony_group
                

                