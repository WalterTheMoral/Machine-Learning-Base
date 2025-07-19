import numpy as np
from NetworkClasses.Cost import Cost
from dataclasses import dataclass

@dataclass
class NetworkConfiguration:
    name: str
    cost: Cost
    threshold: float = 0.5