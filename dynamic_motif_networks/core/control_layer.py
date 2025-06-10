"""Control layers for modulating SNNs."""
from dataclasses import dataclass
import numpy as np

@dataclass
class LinearANNControl:
    weight: np.ndarray
    bias: np.ndarray

    def __init__(self, input_dim: int, output_dim: int):
        self.weight = np.zeros((output_dim, input_dim))
        self.bias = np.zeros(output_dim)

    def step(self, x: np.ndarray) -> np.ndarray:
        return self.weight @ x + self.bias
