"""Motif embedding utilities."""
import numpy as np
from .dwt import dwt3d


def encode_motif(activity: np.ndarray) -> np.ndarray:
    coeffs = dwt3d(activity)
    return coeffs.flatten()
