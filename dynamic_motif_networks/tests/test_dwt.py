import numpy as np
from dynamic_motif_networks.core.dwt import dwt3d, idwt3d


def test_dwt_roundtrip():
    x = np.random.rand(2, 2, 2)
    coeffs = dwt3d(x)
    recon = idwt3d(coeffs, x.shape)
    assert recon.shape == x.shape
