"""Spiking neural network layer."""
from dataclasses import dataclass, field
from typing import List
import numpy as np

from .neuron import LIFNeuron

@dataclass
class SNNLayer:
    width: int
    height: int
    neighbors: int = 4
    neurons: List[LIFNeuron] = field(init=False)

    def __post_init__(self):
        self.neurons = [LIFNeuron() for _ in range(self.width * self.height)]

    def step(self, inputs: np.ndarray) -> np.ndarray:
        outputs = np.zeros_like(inputs)
        for i, (n, inp) in enumerate(zip(self.neurons, inputs.flatten())):
            outputs.flat[i] = n.step(inp)
        return outputs.reshape(inputs.shape)
