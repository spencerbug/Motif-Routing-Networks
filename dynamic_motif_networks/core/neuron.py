# TODO: Add more biologically realistic neuron models
"""Neuron implementations for Dynamic Motif Networks."""

from dataclasses import dataclass, field
import numpy as np

@dataclass
class LIFNeuron:
    """Leaky integrate-and-fire neuron."""
    tau: float = 10.0
    threshold: float = 1.0
    v: float = 0.0

    def step(self, input_current: float) -> int:
        """Integrate input and produce a spike if threshold crossed."""
        self.v += (-self.v + input_current) / self.tau
        spike = int(self.v >= self.threshold)
        if spike:
            self.v = 0.0
        return spike


@dataclass
class ANNNeuron:
    """Simple artificial neuron with ReLU activation."""
    weight: float = 1.0
    bias: float = 0.0

    def step(self, x: float) -> float:
        return max(0.0, self.weight * x + self.bias)


@dataclass
class LiquidNeuron:
    """Neuron with internal state for liquid networks."""
    state: float = 0.0
    alpha: float = 0.9

    def step(self, input_current: float) -> float:
        self.state = self.alpha * self.state + (1 - self.alpha) * input_current
        return self.state
