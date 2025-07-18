from typing import Tuple
import numpy as np
import h5py

class Initialisation:
    def _initialise_W(self, unit_count: int, input_shape: Tuple[int,]) -> np.ndarray:
        raise NotImplementedError
    def _initialise_b(self, unit_count: int) -> np.ndarray:
        return np.zeros((unit_count, 1), dtype=float)

    def initialise_weights(self, unit_count: int, input_shape: Tuple[int,]) \
            -> Tuple[np.ndarray, np.ndarray]:
        return self._initialise_W(unit_count, input_shape), self._initialise_b(unit_count)

class Zero(Initialisation):
    def _initialise_W(self, unit_count: int, input_shape: Tuple[int,]) -> np.ndarray:
        return np.zeros( (unit_count, *input_shape) )

class Random(Initialisation):
    def __init__(self, random_scale: float = 0.01):
        self.random_scale = random_scale

    def _initialise_W(self, unit_count: int, input_shape: Tuple[int,]) -> np.ndarray:
        return np.random.randn(unit_count, *input_shape) * self.random_scale

class Xaviar(Initialisation):
    def _initialise_W(self, unit_count: int, input_shape: Tuple[int,]) -> np.ndarray:
        normalisation_scale = np.sqrt(1.0 / sum(input_shape))
        return np.random.randn(unit_count, *input_shape) * normalisation_scale

class He(Initialisation):
    def _initialise_W(self, unit_count: int, input_shape: Tuple[int,]) -> np.ndarray:
        normalisation_scale = np.sqrt(2.0 / sum(input_shape))
        return np.random.randn(unit_count, *input_shape) * normalisation_scale

class File(Initialisation):
    def __init__(self, file_name: str):
        self.file_name = file_name

    def initialise_weights(self, unit_count: int, input_shape: Tuple[int,]) \
            -> Tuple[np.ndarray, np.ndarray]:
        try:
            with h5py.File(self.file_name, 'r') as hf:
                W = hf['W'][:]
                b = hf['b'][:]

        except FileNotFoundError:
            raise FileNotFoundError(f"Weight file '{self.file_name}' not found.")
        except (KeyError, OSError, ValueError) as e:
            raise ValueError(f"Failed to load weights from '{self.file_name}': {e}")

        return W, b


def initialise_adaptive(unit_count, input_shape, learning_rate=0.01) -> Tuple[np.ndarray, np.ndarray]:
    adaptive_alpha_W = np.full((unit_count, *input_shape), learning_rate)
    adaptive_alpha_b = np.full((unit_count, 1), learning_rate)

    return adaptive_alpha_W, adaptive_alpha_b
