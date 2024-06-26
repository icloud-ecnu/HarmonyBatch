from abc import ABC, abstractmethod
from typing import List

from harmonybatch.predictor.util import Apps

class FunctionCfg(ABC):
    def __init__(self, config: dict):
        self.config = config
    
    @abstractmethod
    def get_config(self, apps: Apps, function_type : List[str]):
        pass
    