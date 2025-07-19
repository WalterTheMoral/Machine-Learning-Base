import numpy as np
from typing import Tuple
from Util.Util import *

class LearningStrategy:
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate

    def initialise_adaptive(self, unit_count: int, input_shape: Tuple[int,]) -> None:
        raise NotImplementedError

    def update_parameters(self, W: np.ndarray, b: np.ndarray, dW: np.ndarray, db: np.ndarray) \
            -> Tuple[np.ndarray, np.ndarray]:
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
        self.scale = scale
        self.switch_value = switch_value

    def initialise_adaptive(self, unit_count: int, input_shape: Tuple[int,]) -> None:
        self.adaptive_alpha_W = np.full((unit_count, *input_shape), self.learning_rate)
        self.adaptive_alpha_b = np.full((unit_count, 1), self.learning_rate)

    def update_parameters(self, W: np.ndarray, b: np.ndarray, dW: np.ndarray, db: np.ndarray) \
            -> Tuple[np.ndarray, np.ndarray]:
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