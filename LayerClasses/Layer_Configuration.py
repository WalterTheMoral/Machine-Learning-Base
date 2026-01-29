from dataclasses import dataclass
from typing import Tuple
from LayerClasses.Activation import Activation
from LayerClasses.Initialisation import Initialisation, Random
from LayerClasses.LearningStrategy import LearningStrategy, Standard
from LayerClasses.Regularisation import Regularisation, L2, Dropout

@dataclass
class LayerConfiguration:
    name: str
    unit_count: int
    input_shape: Tuple[int,]
    activation: Activation
    initialisation: Initialisation = Random()
    learning_strategy: LearningStrategy = Standard(0.01)
    regularisation: Regularisation = Regularisation()

    def __post_init__(self) -> None:
        pass
        #TODO: Do checks on values
