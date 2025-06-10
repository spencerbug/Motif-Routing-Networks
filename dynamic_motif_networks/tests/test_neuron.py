import numpy as np
from dynamic_motif_networks.core.neuron import LIFNeuron, ANNNeuron, LiquidNeuron


def test_lif_spike():
    n = LIFNeuron(tau=1.0, threshold=0.5)
    spike = n.step(1.0)
    assert spike == 1


def test_ann_relu():
    n = ANNNeuron(weight=1.0, bias=0.0)
    assert n.step(-1.0) == 0.0


def test_liquid_state():
    n = LiquidNeuron(alpha=0.5)
    out = n.step(1.0)
    assert 0 < out <= 1.0
