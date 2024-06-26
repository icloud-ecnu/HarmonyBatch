from typing import  Union, List, Any, Tuple
import numpy as np

class Mem:
    def __init__(self, model_config: dict, model_name: str) -> None:
        self.model_config = model_config[model_name]
        # CPU
        self.a, self.b = self.model_config["CPU"]["mem"]
        # GPU
        self.mem = self.model_config["GPU"]["mem"]
        self.gpu_mem = self.model_config["GPU"]["gpu_mem"]

    def get_mem(self, cpu, mem):
        mem = mem / 1024
        if 4 * cpu < mem:
            return None
        elif cpu > mem:
            mem = cpu
        mem = int(mem * 1024)
        if mem % 64 != 0:
            mem = ((mem // 64) + 1) * 64
        return mem / 1024

    def get_mini_cpu_batch(self, max_batch: int):
        cpus = []
        for batch in range(1, max_batch+1):
            mem = self.a * batch + self.b
            mem = int(mem)
            if mem % 64 != 0:
                mem = ((mem // 64) + 1) * 64
            cpus.append(mem / 4 / 1024)
        return cpus

    def get_cpu_mem(self, cpu: float, batch: int):
        mem = self.a * batch + self.b
        mem = int(mem)
        if mem % 64 != 0:
            mem = ((mem // 64) + 1) * 64
        return self.get_mem(cpu, mem)

    def get_gpu_mem(self, gpu: float, batch: Union[int, None] = None):
        mem = self.mem
        mem = int(mem)
        if mem % 64 != 0:
            mem = ((mem // 64) + 1) * 64
        cpu_min = gpu / 3
        mem = mem / 1024
        if mem > cpu_min * 4:
            return None
        return mem

    def get_gpu_gpu_mem(self, batch: int):
        for i in range(len(self.gpu_mem)):
            if batch <= self.gpu_mem[i]:
                return i+1
        return len(self.gpu_mem) + 1


class Instance:
    def __init__(self, cpu: float, mem: Union[float, None], gpu: Union[int, None]) -> None:
        self.cpu = cpu
        if mem is None:
            self.mem = self.cpu * 2
        else:
            self.mem = mem
        self.gpu = gpu
    
    def __eq__(self, other : "Instance") -> bool:
        return self.cpu == other.cpu and self.mem == other.mem and self.gpu == other.gpu


class Cfg:
    def __init__(self, instance: Instance, batch_size: int, cost: float, 
                rps: List[float], slo: List[float], 
                timeout: List[float], latency: float) -> None:
        self.instance : Instance = instance
        self.batch_size = batch_size
        self.cost = cost
        self.rps = rps
        self.timeout = timeout
        self.slo = slo
        self.latency = latency

    def set_apps(self, apps):
        self.apps = apps
    
    def get_total_rps(self):
        return sum(self.rps)

    def __str__(self):
        ret = "cpu:\t\t{%0.2f}" % self.instance.cpu + "\n" + \
            "batch:\t\t{%d}" % self.batch_size + "\n" + \
            "rps:\t\t" + str(self.rps) + "\n" + \
            "timeout:\t" + str(self.timeout) + "\n" + \
            "cost:\t\t{%0.3e}" % self.cost + "\n" \
            "slo:\t\t" + str(self.slo) + "\n"

        if self.instance.gpu is not None:
            ret = "gpu:\t\t{%d}" % self.instance.gpu + "\n" + ret
        return ret + "-----"
    
    def update(self, cfg : Union["Cfg", None]):
        if cfg is not None and cfg.cost < self.cost:
            self.instance = cfg.instance
            self.batch_size = cfg.batch_size
            self.cost = cfg.cost
            self.rps = cfg.rps
            self.timeout = cfg.timeout
            self.slo = cfg.slo
            self.latency = cfg.latency
        return self

class App:
    def __init__(self, name: str, slo: float, rps: float) -> None:
        self.name = name
        self.slo = slo
        self.rps = rps
    
    def __str__(self):
        return self.name + " " + str(round(self.slo, 1)) + " " + str(round(self.rps,1))
    
    def __repr__(self) -> str:
        return self.__str__()

class Apps:
    def __init__(self, apps: List[App]) -> None:
        self.apps = apps
        self.apps.sort(key=lambda app: app.slo)
        if len(apps) > 0:
            self.apps_rps = sum(app.rps for app in self.apps)
            self.slo = min(self.apps, key=lambda x: x.slo).slo
        else:
            self.slo = np.Inf
            self.apps_rps = 0

    def add(self, app: Union[App, 'Apps', List[Any]]):
        if isinstance(app, App):
            self.apps.append(app)
            self.apps_rps += app.rps
            self.apps.sort(key=lambda app: app.slo)
            self.slo = min(self.slo, app.slo)
        else:
            if isinstance(app, Apps):
                app_list = app.apps
            else:
                app_list = app
            for app in app_list:
                self.add(app)
        
    
    def get_rps_slo(self) -> Tuple[List[float], List[float]]:
        rpses = []
        slos = []
        for app in self.apps:
            rpses.append(app.rps)
            slos.append(app.slo)
        return rpses, slos

    def remove(self, name: str):
        self.apps = [app for app in self.apps if app.name != name]
        self.apps_rps = sum(app.rps for app in self.apps)
        self.slo = min(self.apps, key=lambda x: x.slo).slo
    
    def set_cfg(self, cfg : Cfg):
        self.cfg = cfg

    def get_apps(self):
        return self.apps

    def get_rps(self):
        return self.apps_rps
    
    def __str__(self):
        self.apps.sort(key=lambda app: app.slo)
        name = [app.name for app in self.apps]
        return str(name)

    def __repr__(self) -> str:
        return self.__str__()
