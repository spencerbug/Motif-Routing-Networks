"""Baseline DMN example."""
import numpy as np
from dynamic_motif_networks.core.neuron import LIFNeuron
from dynamic_motif_networks.core.motif_encoder import encode_motif
from dynamic_motif_networks.training.reinforcement import ReinforcementLearner


def run():
    activity = np.random.rand(4, 4, 4)
    embedding = encode_motif(activity)
    learner = ReinforcementLearner()
    learner.step(embedding, reward=1.0)
    print("Buffer size:", len(learner.buffer_contents()))


if __name__ == "__main__":
    run()
