import numpy as np
from typing import Tuple


class Regularisation():
    def __str__(self) -> str:
        lines = [
            f"Regularisation: {self.__class__.__name__}"
        ]
        return "\n".join(lines)

    def dropped_layer(self, layers: np.ndarray):
        return np.copy(layers)

    def calculate_l2(self, W: np.ndarray) -> float:
        return 0

class L2(Regularisation):
    def __init__(self, l2_lambda: float = 0.1) -> None:
        self.l2_lambda = l2_lambda

    def caclulate_l2(self, W: np.ndarray) -> float:
        return self.l2_lambda * np.sum(np.square(W))

    def __str__(self) -> str:
        lines = [super().__str__()]

        lines += [
            f"\tLambda: {self.l2_lambda}"
        ]

        return "\n".join(lines)


class Dropout(Regularisation):
    def __init__(self, keep_prob: float = 0.6) -> None:
        self.keep_prob = keep_prob

    def keep_vector(self, num_units: Tuple[int]) -> np.ndarray:
        probabilities: np.ndarray = np.random.rand(*num_units)
        return probabilities < self.keep_prob

    def dropped_layer(self, original_layer: np.ndarray):
        self.kept = self.keep_vector(original_layer.shape)

        dropped = np.multiply(original_layer, self.kept)
        dropped /= self.keep_prob

        return dropped

    def __str__(self) -> str:
        lines = [super().__str__()]

        lines += [
            f"\tDropout Keep Probability: {self.keep_prob}"
        ]

        return "\n".join(lines)
