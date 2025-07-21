import numpy as np
from typing import Tuple
from Util.Util import *

class LearningStrategy:
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate

    def initialise_adaptive(self, unit_count: int, input_shape: Tuple[int,]) -> None:
        """
        Initialise adaptive learning rates for layer. Does nothing if LearningStrategy is Standard
        :param unit_count:
        :param input_shape:
        :return:
        """
        raise NotImplementedError

    def update_parameters(self, W: np.ndarray, b: np.ndarray, dW: np.ndarray, db: np.ndarray) \
            -> Tuple[np.ndarray, np.ndarray]:
        """
        Update layer parameters in accordance with gradients and LearningStrategy
        :param W: Current weights of layer
        :param b: Current biases of layer
        :param dW: Current gradient of layer's weights
        :param db: Current gradient of layer's biases
        :return: Updated weights and biases
        """
        raise NotImplementedError

    def __str__(self):
        lines = [
            f"Learning Rate: {self.learning_rate}"
        ]
        return "\n".join(lines)

class Standard(LearningStrategy):
    def initialise_adaptive(self, unit_count: int, input_shape: Tuple[int,]) -> None:
        pass

    def update_parameters(self, W: np.ndarray, b: np.ndarray, dW: np.ndarray, db: np.ndarray) \
            -> Tuple[np.ndarray, np.ndarray]:
        W -= dW * self.learning_rate
        b -= db * self.learning_rate

        return W, b

class Adaptive(LearningStrategy):
    def __init__(self, learning_rate: float = 0.01, scale: float = 1.1, switch_value: float = 0.5):
        super().__init__(learning_rate)
        self.scale: float = scale
        self.switch_value: float = switch_value

        self._initialised: bool = False

    def initialise_adaptive(self, unit_count: int, input_shape: Tuple[int,]) -> None:
        self.adaptive_alpha_W = np.full((unit_count, *input_shape), self.learning_rate)
        self.adaptive_alpha_b = np.full((unit_count, 1), self.learning_rate)

        self._initialised = True

    def update_parameters(self, W: np.ndarray, b: np.ndarray, dW: np.ndarray, db: np.ndarray) \
            -> Tuple[np.ndarray, np.ndarray]:
        if not self._initialised: raise AttributeError("Adaptive learning rates have not been initialised")

        self.adaptive_alpha_W *= np.where(self.adaptive_alpha_W * dW > 0,
                                          self.scale,
                                          -self.switch_value)

        self.adaptive_alpha_b *= np.where(self.adaptive_alpha_b * db > 0,
                                          self.scale,
                                          -self.switch_value)
        W -= self.adaptive_alpha_W
        b -= self.adaptive_alpha_b

        return W, b

    def __str__(self):
        lines = [f"Adaptive Strategy"]
        lines += [super().__str__()]
        lines += [
            f"Adaptive Scale: {self.scale}",
            f"Adaptive Switch: {self.switch_value}"
        ]

        return "\n".join([lines[0]] + [indent_string(line) for line in lines[1:]])