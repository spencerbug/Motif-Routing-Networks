import numpy as np
from dynamic_motif_networks.training.reinforcement import ReinforcementLearner


def test_buffer_append():
    r = ReinforcementLearner(buffer_size=2)
    r.step(np.array([1.0]), 1.0)
    assert len(r.buffer_contents()) == 1
