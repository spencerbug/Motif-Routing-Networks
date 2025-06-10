import numpy as np
from dynamic_motif_networks.core.motif_encoder import encode_motif


def test_encode_shape():
    x = np.random.rand(2, 2, 2)
    emb = encode_motif(x)
    assert emb.ndim == 1
