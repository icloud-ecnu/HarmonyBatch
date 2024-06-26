import math
from harmonybatch.predictor.util import Instance

class FunctionCost():
    def __init__(self) -> None:
        self.cpu_cost = 0.00009
        self.mem_cost = 0.000009
        self.gpu_cost = 0.00011
        self.invocation_cost = 0.009 / 10000

    def cost(self, duration: float, batch: int, instance: Instance, billed_second : bool = True) -> float:
        if instance.gpu is None or billed_second is False:
            gpu = 0
        else:
            gpu = instance.gpu
            duration = math.ceil(duration)
        return (self.invocation_cost +
                (instance.cpu * self.cpu_cost +
                 instance.mem * self.mem_cost +
                    gpu * self.gpu_cost) * duration) / batch
