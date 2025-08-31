import numpy as np

class Activation:
    def calculate(self, Z: np.ndarray) -> np.ndarray:
        """
        Calculates activation function of parameter Z

        :param Z: Input to activation function

        :return: Output of activation function with input Z
        """
        raise NotImplementedError

    def gradient(self, Z: np.ndarray) -> np.ndarray:
        """
        Calculates gradient of activation function at input Z

        :param Z: Input at which to calculate gradient.

        :return: Gradient of activation function at input Z
        """
        raise NotImplementedError

    def __str__(self) -> str:
        lines = [
            f"Activation Function: {self.__class__.__name__}"
        ]
        return "\n".join(lines)

class TrimActivation(Activation):
    def __init__(self, trim=1e-10) -> None:
        self.trim = trim

    def __str__(self) -> str:
        lines = [super().__str__()]

        lines += [
            f"\tTrim Value: {self.trim}"
        ]

        return "\n".join(lines)

class Relu(Activation):
    def calculate(self, Z: np.ndarray) -> np.ndarray:
        return np.maximum(0, Z)

    def gradient(self, Z: np.ndarray) -> np.ndarray:
        return np.where(Z < 0, 0, 1)

class LeakyRelu(Activation):
    def __init__(self, leaky_relu_d=0.01) -> None:
        self.leaky_relu_d = leaky_relu_d

    def calculate(self, Z: np.ndarray) -> np.ndarray:
        return np.where(Z > 0, Z, self.leaky_relu_d * Z)

    def gradient(self, Z: np.ndarray) -> np.ndarray:
        return np.where(Z > 0, 1, self.leaky_relu_d)

    def __str__(self) -> str:
        lines = [super().__str__()]

        lines += [
            f"\tLeaky Reklu Gradient: {self.leaky_relu_d}"
        ]

        return "\n".join(lines)

class Sigmoid(Activation):
    def calculate(self, Z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-Z))

    def gradient(self, Z: np.ndarray) -> np.ndarray:
        sig = self.calculate(Z)
        return sig * (1 - sig)

class TanH(Activation):
    def calculate(self, Z: np.ndarray) -> np.ndarray:
        return np.tanh(Z)

    def gradient(self, Z: np.ndarray) -> np.ndarray:
        return 1 - np.square( self.calculate(Z) )

class Softmax(Activation):
    def calculate(self, Z: np.ndarray) -> np.ndarray:
        return np.exp(Z) / np.sum(np.exp(Z), axis=0)

    def gradient(self, Z: np.ndarray) -> np.ndarray:
        # return self.calculate(Z) * (1 - self.calculate(Z)) #TODO: Check Gradient
        return 1


class TrimSigmoid(TrimActivation):
    def calculate(self, Z: np.ndarray) -> np.ndarray:
        clipped = np.clip(Z, -100, 100)
        calculated = 1 / (1 + np.exp(-clipped))
        return np.clip(calculated, self.trim, 1 - self.trim)

    def gradient(self, Z: np.ndarray) -> np.ndarray:
        calculated = self.calculate(Z)
        return calculated * (1 - calculated)

class TrimTanH(TrimActivation):
    def calculate(self, Z: np.ndarray) -> np.ndarray:
        calculated = np.tanh(Z)
        return np.clip(calculated, -1 + self.trim, 1 - self.trim)

    def gradient(self, Z: np.ndarray) -> np.ndarray:
        return 1 - np.square( self.calculate(Z) )

class TrimSoftmax(Activation):
    def calculate(self, Z: np.ndarray) -> np.ndarray:
        clipped = np.min(Z, 100)
        return np.exp(clipped) / np.max(np.exp(Z), axis=0)

    def gradient(self, Z: np.ndarray) -> np.ndarray:
        return self.calculate(Z) * (1 - self.calculate(Z)) #TODO: Check Gradient
