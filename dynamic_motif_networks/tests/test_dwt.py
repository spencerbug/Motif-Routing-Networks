import numpy as np
from dynamic_motif_networks.core.dwt import dwt3d, idwt3d


def test_dwt_roundtrip_default():
    x = np.random.rand(4, 4, 4)
    coeffs = dwt3d(x)
    recon = idwt3d(coeffs, x.shape)
    assert recon.shape == x.shape


def test_dwt_roundtrip_args():
    x = np.random.rand(8, 8, 8)
    for depth in [1, 2]:
        for boundary in ['periodic']:
            coeffs = dwt3d(x,  depth=depth, boundary=boundary)
            recon = idwt3d(coeffs, x.shape, depth=depth, boundary=boundary)
            assert recon.shape == x.shape


def test_dwt_invalid_backend(monkeypatch):
    import sys
    import builtins
    import importlib
    # Remove both modules from sys.modules before patching
    sys.modules.pop('dynamic_motif_networks.core.dwt3d_cpp', None)
    sys.modules.pop('dynamic_motif_networks.core.dwt', None)
    real_import = builtins.__import__
    def fake_import(name, *args, **kwargs):
        if name == 'dynamic_motif_networks.core.dwt3d_cpp' or name.endswith('.dwt3d_cpp'):
            raise ImportError('Simulated missing dwt3d_cpp')
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', fake_import)
    dwt_mod = importlib.import_module('dynamic_motif_networks.core.dwt')
    # Remove dwt3d_cpp from dwt_mod namespace if present (simulate missing symbol)
    if hasattr(dwt_mod, 'dwt3d_cpp'):
        delattr(dwt_mod, 'dwt3d_cpp')
    try:
        dwt_mod.dwt3d(np.zeros((2,2,2)))
    except (ImportError, NameError):
        pass
    else:
        assert False, "Expected ImportError or NameError when backend is missing"
