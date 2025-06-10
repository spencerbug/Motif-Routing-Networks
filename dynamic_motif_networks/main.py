"""Entrypoint for running DMNs."""
import yaml
import numpy as np
from dynamic_motif_networks.core.snn_layer import SNNLayer
from dynamic_motif_networks.core.motif_encoder import encode_motif
from dynamic_motif_networks.training.reinforcement import ReinforcementLearner


def main():
    cfg = yaml.safe_load(open('dynamic_motif_networks/configs/hyperparams.yaml'))
    layer = SNNLayer(width=cfg['snn']['width'], height=cfg['snn']['height'])
    activity = np.random.rand(4, cfg['snn']['height'], cfg['snn']['width'])
    embedding = encode_motif(activity)
    learner = ReinforcementLearner(buffer_size=cfg['buffer_size'])
    learner.step(embedding, reward=1.0)
    print('Ran DMN with buffer size', len(learner.buffer_contents()))


if __name__ == '__main__':
    main()
