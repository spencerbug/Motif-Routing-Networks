"""Buffer utilities for reinforcement learning."""
from collections import deque
from dataclasses import dataclass, field
from typing import Deque
import numpy as np

@dataclass
class CircularBuffer:
    size: int
    data: Deque[np.ndarray] = field(init=False)

    def __post_init__(self):
        self.data = deque(maxlen=self.size)

    def push(self, x: np.ndarray):
        self.data.append(x)

    def contents(self):
        return list(self.data)
