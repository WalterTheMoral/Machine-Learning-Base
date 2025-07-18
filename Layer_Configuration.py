from dataclasses import dataclass
from typing import Tuple
from Activation import Activation
from Initialisation import Initialisation, File

@dataclass
class LayerConfiguration:
    name: str
    unit_count: int
    input_shape: Tuple[int,]
    activation: Activation
    initialisation: Initialisation
    adaptive: bool = False

    learning_rate: float = 0.01
    random_scale: float = 0.01
    leaky_relu_d: float = 0.01
    adaptive_cont: float = 1.1
    adaptive_switch: float = 0.5
    activation_trim: float = 1e-10

    file_name: str = ""

    def __post_init__(self):
        if self.initialisation == File and self.file_name == "":
            raise ValueError("Initialisation from file requires file name")

        #TODO: Find more checks
