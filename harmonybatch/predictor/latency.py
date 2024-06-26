from typing import Union, Optional
import math

from harmonybatch.predictor.util import Instance
import numpy as np

from abc import ABC, abstractmethod


class CPUConstraint():
    def __init__(self, params : dict, model_name : str, fitting_method : str = 'Exponential') -> None:
        self.model_name = model_name
        self.fitting_method = fitting_method
        self.params_max = params['max'][self.fitting_method]
    
    def get_constraint_cpu(self, batch_size : int, slo : float) -> Optional[float]:
        assert self.fitting_method == 'Exponential'
        g = self.params_max[batch_size-1]
        if slo <= g[2]:
            return None
        cpu = -np.log((slo - g[2]) / g[0]) * g[1]
        if cpu <= 0:
            return None
        return cpu
    
class CPUMinRelative:
    def __init__(self, params: dict, model_name: str, fitting_metod : str = 'Exponential',) -> None:
        self.model_name = model_name
        self.fitting_metod = fitting_metod
        self.params_avg = params['avg'][self.fitting_metod]
    
    def get_min_relative(self, batch_size : int, cpu_min :float, cpu_max:float) -> Optional[float]:
        assert self.fitting_metod == 'Exponential'
        g = self.params_avg[batch_size-1]
        df = lambda c: g[0] * (1-c / g[1]) * np.exp(-c / g[1]) + g[2]
        l = max(2 * g[1], cpu_min)
        r = cpu_max

        if l > r:
            return None
        
        if df(l) > 0 or df(r) < 0:
            return None
        
        pre = 0
        while l < r:
            m = (l + r) / 2
            m = round(m / 0.05) * 0.05
            if pre == m:
                break
            pre = m
            target = df(m)
            if target < 0:
                l = m + 0.05
            elif target > 0:
                r = m
            else:
                return m
            if l + 0.05 > r:
                break
        return l


class Latency(ABC):
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def lat_avg(self, instance: Instance, batch_size: int) -> float:
        pass
    

class CPULatency(Latency):
    def __init__(self, params: dict, model_name: str, fitting_metod : str = 'Exponential', predict_error = 0.0) -> None:
        super().__init__()
        self.model_name = model_name
        self.fitting_metod = fitting_metod
        self.predict_error = predict_error

        self.params_avg = params['avg'][self.fitting_metod]
        self.params_max = params['max'][self.fitting_metod]

    def lat_avg(self, instance: Instance, batch_size: int) -> float:
        cpu = instance.cpu
        if self.fitting_metod == 'Exponential':
            g = self.params_avg[batch_size-1]
            G = g[0] * np.exp(-cpu / g[1]) + g[2]
            return G * (1 + self.predict_error)
        elif self.fitting_metod == 'Polynomial':
            f = self.params_avg['f']
            g = self.params_avg['g']
            k = self.params_avg['k']
            F = f[0] * batch_size + f[1]
            G = cpu + g[0]
            return (F / G + k[0]) * (1 + self.predict_error)
        return np.Inf

    def lat_max(self, instance: Instance, batch_size: int) -> float:
        cpu = instance.cpu
        if self.fitting_metod == 'Exponential':
            g = self.params_max[batch_size-1]
            G = g[0] * np.exp(-cpu / g[1]) + g[2]
            return G * (1 + self.predict_error)
        elif self.fitting_metod == 'Polynomial':
            f = self.params_max['f']
            g = self.params_max['g']
            k = self.params_max['k']
            F = f[0] * batch_size + f[1]
            G = cpu + g[0]
            return (F / G + k[0]) * (1 + self.predict_error)
        return np.Inf


class GPULatency(Latency):
    def __init__(self, params: dict, model_name: str, predict_error = 0.0) -> None:
        super().__init__()
        self.model_name = model_name

        self.g1 = params['l1']
        self.g2 = params['l2']
        self.t = params['t']
        self.G = params['G']

        self.a = None
        self.b = None

        if 'a' in params:
            self.a = params['a']
        if 'b' in params:
            self.b = params['b']
    

    def lat_avg(self, instance: Instance, batch_size: int, a : Union[float, None] = None, b : Union[float, None] = None)->float:
        gpu = instance.gpu
        c = instance.cpu
        if c > 1:
            c = 1

        if a is None:
            a = self.a
        if b is None:
            b = self.b
        
        if a is None:
            a = 1
        if b is None:
            b = 0

        L = self.g1 * batch_size + self.g2
        L1 = L * a
        L2 = L * b
        L = L1
        return self.G / gpu * L + L2 / c

    def lat_max(self, instance: Instance, batch_size: int, scale = 1.2, a : Union[float, None] = None, b : Union[float, None] = None)->float:
        gpu = instance.gpu
        c = instance.cpu
        if c > 1:
            c = 1

        if a is None:
            a = self.a
        if b is None:
            b = self.b
        
        if a is None:
            a = 1
        if b is None:
            b = 0
        
        if gpu == 24:
            scale = 1
        L = self.g1 * batch_size + self.g2
        L1 = L * a
        L2 = L * b
        L = L1 
        n = math.ceil(L / (gpu * self.t))
        return ((self.G - gpu) * n * self.t + L) * scale + L2 / c