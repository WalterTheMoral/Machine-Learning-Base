from dataclasses import dataclass
from typing import Tuple
from LayerClasses.Activation import Activation
from LayerClasses.Initialisation import Initialisation
from LayerClasses.LearningStrategy import LearningStrategy, Standard

@dataclass
class LayerConfiguration:
    name: str
    unit_count: int
    input_shape: Tuple[int,]
    activation: Activation
    initialisation: Initialisation
    learning_strategy: LearningStrategy = Standard(0.01)

    def __post_init__(self):
        pass
        #TODO: Do checks on values
