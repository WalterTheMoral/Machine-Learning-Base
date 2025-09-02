import numpy as np


def indent_string(text: str, indent: str = "\t") -> str:
    """
    Indents every line in a multi-line string with the given indent string.

    :param text: Text to indent
    :return: Indented text
    """
    return "\n".join(f"{indent}{line}" for line in text.splitlines())

def num_to_one_hot(categories: int, Y: np.ndarray) -> np.ndarray:
    """
    Convert an integer index into a one-hot encoded vector.

    :param categories: Number of possible categories (length of output vector)
    :param Y: Vector of indexes to set as 1 (0-indexed)
    :return: Matrix of vectors in which all elements are 0 with the exception of hot_digit
    """

    Y = Y.flatten().astype(int)
    return np.eye(categories)[Y].T