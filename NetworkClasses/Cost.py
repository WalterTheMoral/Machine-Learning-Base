import numpy as np
from numpy import floating


class Cost:
    def calculate(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def gradient(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def compute_cost(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        error = self.calculate(predictions, targets)
        return (1 / predictions.shape[1]) * np.sum(error)

    def __str__(self):
        return f"Cost Function: {self.__class__.__name__}"

class SquaredMean(Cost):
    def calculate(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        return np.square(predictions - targets)

    def gradient(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        return 2 * (predictions - targets)

class CrossEntropy(Cost):
    def calculate(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        return np.where(targets == 1,
                        -np.log(predictions),
                        -np.log(1 - predictions))

    def gradient(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        return np.where(targets == 1,
                        -1 / predictions,
                        1 / (1 - predictions))

class CategoricalCrossEntropy(Cost):
    def calculate(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        return np.where(targets == 1,
                        - np.log(predictions),
                        0)

    def gradient(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        return predictions - targets
