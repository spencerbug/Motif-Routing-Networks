"""Loss functions and surrogate gradients."""
import numpy as np


def mse_loss(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))
