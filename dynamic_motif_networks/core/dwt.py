"""3D Discrete Wavelet Transform utilities."""
import numpy as np
import importlib
import os

# Try to import the C++ extension
try:
    from . import dwt3d_cpp
except ImportError:
    dwt3d_cpp = None
    # Optionally, could trigger build here


def dwt3d(x: np.ndarray, depth: int = 1, boundary: str = 'periodic') -> np.ndarray:
    """3D Discrete Wavelet Transform using C++ backend (Haar only)."""
    if dwt3d_cpp is None:
        raise ImportError("dwt3d_cpp extension not available.")
    return dwt3d_cpp.dwt3d_cpp(x, depth, boundary)


def idwt3d(coeffs: np.ndarray, shape, depth: int = 1, boundary: str = 'periodic') -> np.ndarray:
    """Inverse 3D Discrete Wavelet Transform using C++ backend (Haar only)."""
    if dwt3d_cpp is None:
        raise ImportError("dwt3d_cpp extension not available.")
    return dwt3d_cpp.idwt3d_cpp(coeffs, shape, depth, boundary)
