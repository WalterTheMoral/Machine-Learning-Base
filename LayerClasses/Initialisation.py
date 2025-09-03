from typing import Tuple
import numpy as np
import h5py

class Initialisation:
    def _initialise_W(self, unit_count: int, input_shape: Tuple[int,]) -> np.ndarray:
        """
        Initialises layer weights using chosen initialised method

        :param unit_count: Number of neurons in layer
        :param input_shape: Number of neurons in previous layer

        :return: Initialised weights of layer
        """
        raise NotImplementedError

    def _initialise_b(self, unit_count: int) -> np.ndarray:
        """
        Initialises layer biases using chosen initialised method

        :param unit_count: Number of neurons in layer

        :return: Initialised biases of layer
        """
        return np.zeros((unit_count,1), dtype=float)

    def initialise_weights(self, unit_count: int, input_shape: Tuple[int,]) \
            -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialises layer weights and biases using chosen initialised method

        :param unit_count: Number of neurons in layer
        :param input_shape: Number of neurons in previous layer

        :return: Initialised weights and biases of layer
        """
        return self._initialise_W(unit_count, input_shape), self._initialise_b(unit_count)

    def __str__(self) -> str:
        lines = [
            f"Inititialisation Method: {self.__class__.__name__}"
        ]
        return "\n".join(lines)


class Zero(Initialisation):
    def _initialise_W(self, unit_count: int, input_shape: Tuple[int,]) -> np.ndarray:
        return np.zeros( (unit_count, *input_shape) )

class Random(Initialisation):
    def __init__(self, random_scale: float = 0.01) -> None:
        self.random_scale = random_scale

    def _initialise_W(self, unit_count: int, input_shape: Tuple[int,]) -> np.ndarray:
        return np.random.randn(unit_count, *input_shape) * self.random_scale

    def __str__(self):
        lines = [super().__str__()]

        lines += [
            f"\tRandom Scale: {self.random_scale}"
        ]

        return "\n".join(lines)

class Xaviar(Initialisation):
    def _initialise_W(self, unit_count: int, input_shape: Tuple[int,]) -> np.ndarray:
        normalisation_scale = np.sqrt(1.0 / sum(input_shape))
        return np.random.randn(unit_count, *input_shape) * normalisation_scale

class He(Initialisation):
    def _initialise_W(self, unit_count: int, input_shape: Tuple[int,]) -> np.ndarray:
        normalisation_scale = np.sqrt(2.0 / sum(input_shape))
        return np.random.randn(unit_count, *input_shape) * normalisation_scale

class File(Initialisation):
    def __init__(self, file_name: str) -> None:
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

    def __str__(self) -> str:
        lines = [super().__str__()]

        lines += [
            f"\tSource File Name: {self.file_name}"
        ]

        return "\n".join(lines)