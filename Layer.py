import numpy as np
from typing import Tuple
import os

from Activation import *
from Initialisation import *
from Layer_Configuration import LayerConfiguration

class Layer:
    def __init__(self, configuration: LayerConfiguration):
        self.name: str = configuration.name
        self.unit_count: int = configuration.unit_count
        self.input_shape: Tuple[int,] = configuration.input_shape

        self.adaptive: bool = configuration.adaptive

        self.learning_rate: float = configuration.learning_rate
        self.random_scale: float = configuration.random_scale
        self.leaky_relu_d: float = configuration.leaky_relu_d
        self.adaptive_cont: float = configuration.adaptive_cont
        self.adaptive_switch: float = configuration.adaptive_switch
        self.activation_trim: float = configuration.activation_trim

        self.file_name: str = configuration.file_name

        self.W, self.b = (
            configuration.initialisation.initialise_weights(self.unit_count, self.input_shape))
        self.adaptive_alpha_W, self.adaptive_alpha_b = (
            initialise_adaptive(self.unit_count, self.input_shape, self.learning_rate))

    def save_weights(self, path: str, file_name: str):
        if not os.path.exists(path):
            os.makedirs(path)

        with h5py.File(f"{path}/{file_name}.h5", 'w') as hf:
            hf.create_dataset("W", data=self.W)
            hf.create_dataset("b", data=self.b)


