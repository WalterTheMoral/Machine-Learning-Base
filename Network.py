import numpy as np
from typing import List
import os
import h5py

from NetworkClasses.Network_Configuration import *
from Layer import Layer
from Util.Util import indent_string


class Network:
    def __init__(self, configuration: NetworkConfiguration) -> None:
        self.name: str = configuration.name
        self.cost: Cost = configuration.cost
        self.threshold: float = configuration.threshold

        self.layers: List[Layer] = []
        self._is_compiled = False

    def add(self, *layers: Layer) -> None:
        """
        Adds layers to the network.

        :param layers: One or more Layer instances to be added to the network.
        """
        for layer in layers: self.layers.append(layer)

    def __add__(self, other: Layer) -> None:
        self.layers.append(other)

    def save_weights(self, path: str) -> None:
        """
        Saves the weights of all layers in the network to different files in the specified path.

        :param path: The directory or where the weights will be saved.
        """
        for i, layer in enumerate(self.layers):
            layer.save_weights(path, f"Layer{i}")

    def network_forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass through the neural network.

        :param inputs: Input data to be passed through the network.

        :return: The output of the network after processing the input through all layers.
        """
        layer_output = inputs
        for layer in self.layers:
            layer_output = layer.feedforward(layer_output)

        return layer_output

    def network_backward(self, d_output: np.ndarray) -> None:
        """
        Performs the backward pass through the network, updating internal weights.

        :param d_output: The gradient of the loss with respect to the network's output.
        """
        for layer in reversed(self.layers):
            d_output = layer.backward_propagation(d_output)
            layer.update_parameters()

    def train(self, inputs: np.ndarray, expected_output: np.ndarray, iterations: int = 10000) -> List[float]:
        """
        Trains the neural network using the provided input data and expected output.

        :param inputs: Input data for training, typically of shape (num_samples, num_features).
        :param expected_output: Expected output (labels or targets) corresponding to the input data.
        :param iterations: Number of training iterations. Defaults to 10000.

        :return: List of cost values recorded at regular intervals during training.
        """
        costs = []

        for i in range(iterations):
            network_output = inputs

            network_output = self.network_forward(network_output)
            self.network_backward( self.cost.gradient(network_output, expected_output) )

            if i % max(iterations // 100, 1) == 0:
                cost = self.cost.compute_cost(network_output, expected_output)
                costs.append(cost)
                print(f"Cost after {i // max(iterations // 100, 1)}%: {cost}")

        return costs

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Predicts the output class for the given input data using the trained network.

        :param inputs: Input data to be evaluated by the network.

        :return: Boolean array indicating whether each network output exceeds the threshold.
        """

        network_output = self.network_forward(inputs)
        # return network_output > self.threshold

        if network_output.shape[0] > 1: # softmax
            predictions = np.where(network_output==network_output.max(axis=0), 1, 0)
            return predictions
        else:
            return network_output > self.threshold

    def __str__(self) -> str:
        lines = [
            f"{self.name}:",
            str(self.cost),
            f"Threshold: {self.threshold}",
            *(indent_string(str(layer)) for layer in self.layers)
        ]

        return "\n".join([lines[0]] + [indent_string(line) for line in lines[1:]])
