"""3D Discrete Wavelet Transform utilities."""
import numpy as np


def dwt3d(x: np.ndarray) -> np.ndarray:
    """Placeholder 3D-DWT using simple averaging."""
    return x.mean(axis=0, keepdims=True)


def idwt3d(coeffs: np.ndarray, shape) -> np.ndarray:
    """Inverse placeholder DWT."""
    return np.repeat(coeffs, shape[0], axis=0)
