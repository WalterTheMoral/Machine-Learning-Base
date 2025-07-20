import numpy as np
from typing import List
import os
import h5py

from NetworkClasses.Network_Configuration import *
from Layer import Layer
from Util.Util import *


class Network:
    def __init__(self, configuration: NetworkConfiguration):
        self.name: str = configuration.name
        self.cost: Cost = configuration.cost
        self.threshold: float = configuration.threshold

        self.layers: List[Layer] = []
        self._is_compiled = False

    def add(self, *layers: Layer):
        for layer in layers: self.layers.append(layer)

    def __add__(self, other: Layer):
        self.layers.append(other)

    def save_weights(self, path):
        for i, layer in enumerate(self.layers):
            layer.save_weights(path, f"Layer{i}")

    def network_forward(self, inputs: np.ndarray) -> np.ndarray:
        layer_output = inputs
        for layer in self.layers:
            layer_output = layer.feedforward(layer_output)

        return layer_output

    def network_backward(self, d_output: np.ndarray):        
        for layer in reversed(self.layers):
            d_output = layer.backward_propagation(d_output)
            layer.update_parameters()

    def train(self, inputs: np.ndarray, expected_output: np.ndarray, iterations: int = 10000) -> List[float]:
        costs = []

        for i in range(iterations):
            network_output = inputs

            network_output = self.network_forward(network_output)
            self.network_backward( self.cost.gradient(network_output, expected_output) )

            if i % max(iterations // 100, 1) == 0: #TODO: Do correct logging
                cost = self.cost.compute_cost(network_output, expected_output)
                costs.append(cost)
                print(f"Cost after {i // max(iterations // 100, 1)}%: {cost}")

        return costs

    def predict(self, inputs: np.ndarray):
        network_output = self.network_forward(inputs)
        return network_output > self.threshold

    def verify_derivatives(self, inputs: np.ndarray, targets: np.ndarray, epsilon: float = 1e-7):
        relative_difference = lambda approximations, derivatives: np.linalg.norm(derivatives - approximations) / ((np.linalg.norm(derivatives)) + np.linalg.norm(approximations))

        output = self.network_forward(inputs)
        self.network_backward(self.cost.gradient(output, targets))

        for layer_number, layer in reversed(self.layers):
            parameters = parameters_to_vector(layer.W, layer.b)
            derivatives = derivatives_to_vector(layer.dW, layer.db)

            approximations = np.zeros( derivatives.shape[0] )

            for i in range( len(approximations) ):
                addition_parameters = np.array(parameters, copy=True)
                addition_parameters[i] += epsilon
                layer.parameters_from_vector(addition_parameters)
                addition_output = self.network_forward(inputs)
                addition_cost = self.cost.compute_cost(addition_output, targets)

                subtraction_parameters = np.array(parameters, copy=True)
                subtraction_parameters[i] -= epsilon
                layer.parameters_from_vector(subtraction_parameters)
                subtraction_output = self.network_forward(inputs)
                subtraction_cost = self.cost.compute_cost(subtraction_output, targets)

                approximations[i] = abs(addition_cost - subtraction_cost) / (2 * epsilon)

            layer.parameters_from_vector(parameters)



    def __str__(self):
        lines = [
            f"{self.name}:",
            str(self.cost),
            f"Threshold: {self.threshold}",
            *(indent_string(str(layer)) for layer in self.layers)
        ]

        return "\n".join([lines[0]] + [indent_string(line) for line in lines[1:]])
