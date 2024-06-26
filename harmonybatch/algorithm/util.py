import harmonybatch.algorithm.harmony as harmony
from harmonybatch.config import get_config

def NewCfg():
    config = get_config()
    algorithm = config["algorithm"]
    if algorithm == "Harmony":
        return harmony.Harmony(config)
    return harmony.Harmony(config)
    
