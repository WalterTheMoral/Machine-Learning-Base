import numpy as np

def indent_string(text: str, indent: str = "\t") -> str:
    """
    Indents every line in a multi-line string with the given indent string.

    :param text: The input string, possibly containing multiple lines separated by '\n'.
    :param indent: The string to prepend to each line (default: tab character).

    :return: A new string where every line is prefixed with `indent`.
    """
    return "\n".join(f"{indent}{line}" for line in text.splitlines())

def parameters_to_vector(W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Converts parameters W and b (weights and biases) into a single flattened vector.

    :param W: Weights of model
    :param b: Biases of model

    :return: A 1D array containing the concatenated and flattened weights (self.W) and biases (self.b).
    """
    return np.concatenate((np.reshape(W,(-1,)), np.reshape(b, (-1,))), axis=0)

def derivatives_to_vector(dW: np.ndarray, db: np.ndarray) -> np.ndarray:
    """
    Concatenates and flattens the derivatives dW and db into a single vector.

    :param dW: Gradient of the weights, can be of any shape.
    :param db: Gradient of the biases, can be of any shape.

    :return: A 1D array containing the flattened and concatenated derivatives dW and db.
    """
    return np.concatenate((np.reshape(dW,(-1,)), np.reshape(db, (-1,))), axis=0)
