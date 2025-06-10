"""Reinforcement learning loop."""
import numpy as np
from .buffers import CircularBuffer

class ReinforcementLearner:
    def __init__(self, buffer_size: int = 10):
        self.buffer = CircularBuffer(buffer_size)
        self.rewards = []

    def step(self, embedding: np.ndarray, reward: float):
        self.buffer.push(embedding)
        self.rewards.append(reward)

    def buffer_contents(self):
        return self.buffer.contents()
