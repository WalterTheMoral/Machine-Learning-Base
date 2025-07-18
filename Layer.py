import numpy as np
from typing import Tuple
import os

from Util import *
from LayerClasses.Activation import *
from LayerClasses.Initialisation import *
from LayerClasses.LearningStrategy import *
from LayerClasses.Layer_Configuration import *

class Layer:
    def __init__(self, configuration: LayerConfiguration):
        self.name: str = configuration.name
        self.unit_count: int = configuration.unit_count
        self.input_shape: Tuple[int,] = configuration.input_shape

        self.learning_strategy: LearningStrategy = configuration.learning_strategy
        self.activation = configuration.activation
        self.initialisation = configuration.initialisation

        self.m = np.sum(self.input_shape)
        self.W, self.b = self.initialisation.initialise_weights(self.unit_count, self.input_shape)
        self.learning_strategy.initialise_adaptive(self.unit_count, self.input_shape)

    def save_weights(self, path: str, file_name: str) -> None:
        """
        Saves the weights W and b of current layer in a h5 file
        :param path: Path of directory in which weights are saved
        :param file_name: Name of file in which weights are saved
        """

        if not os.path.exists(path):
            os.makedirs(path)

        with h5py.File(f"{path}/{file_name}.h5", 'w') as hf:
            hf.create_dataset("W", data=self.W)
            hf.create_dataset("b", data=self.b)

    def feedforward(self, input_activation: np.ndarray) -> np.ndarray:
        """
        Propagates the layer forward, updating internal _previous_later and _Z variables
        :param input_activation: Layer prior to current layer
        :return: Vector of output of all perceptrons in layer
        """

        self._previous_layer = np.array(input_activation, copy=True)
        self._Z = np.dot(self.W, input_activation) + self.b
        return self.activation.calculate(self._Z)

    def backward_propagation(self, dA) -> np.ndarray:
        """
        Propagates the layer backward, updating internal variables dW and db
        :param dA: Derivative of the loss function with respect to the activation output of current layer
        :return: Derivative of the loss function with respect to the activation output of previous layer
        """

        dZ = self.activation.gradient(self._Z) * dA
        self.dW = (1.0 / self.m) * np.dot(dZ, self._previous_layer.T)
        self.db = np.mean(dZ, axis=1, keepdims=True)

        return np.dot(self.W.T, dZ)

    def update_parameters(self) -> None:
        """
        Update W and b weights in accordance to learning strategy and calculated gradients
        """

        self.learning_strategy.update_parameters(self.W, self.b, self.dW, self.db)

    def __str__(self):
        lines = [
            f"Name: {self.name}",
            f"Number of Units: {self.unit_count}",
            f"Shape of Input: {self.input_shape}",
            str(self.initialisation),
            str(self.activation),
            str(self.learning_strategy),
            f"Weights:",
            f"\tb: {self.b.T}",
            f"\tW Shape: {self.W.shape}"
        ]

        return "\n".join([lines[0]] + [indent_string(line) for line in lines[1:]])
