from dataclasses import dataclass
from typing import Tuple
from Activation import Activation
from Initialisation import Initialisation
from LearningStrategy import LearningStrategy, Standard

@dataclass
class LayerConfiguration:
    name: str
    unit_count: int
    input_shape: Tuple[int,]
    activation: Activation
    initialisation: Initialisation
    learning_strategy: LearningStrategy = Standard()
    learning_rate: float = 0.01

    def __post_init__(self):
        pass
        #TODO: Do checks on values
