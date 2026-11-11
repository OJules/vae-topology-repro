"""Data: we reuse the authors' generator and split verbatim, so any difference
in results comes from training or evaluation, not from the data.

The cylinder S^1 x [0, 1]: an angle theta and a height h, embedded as
(cos theta, sin theta, h) and pushed through a fixed random 3-layer map into
R^50 with Gaussian noise (std 0.1). Topologically this is an annulus: one
connected component (H0) and exactly one loop (H1).
"""

import numpy as np
from experiments.synthetic.data import generate_cylinder_data, train_test_split


def load_cylinder(n_samples=2000, obs_dim=50, data_seed=42, test_ratio=0.2):
    data = generate_cylinder_data(n_samples=n_samples, obs_dim=obs_dim, seed=data_seed)
    train, test = train_test_split(data, test_ratio=test_ratio, seed=data_seed)
    return data, train, test


def ground_truth_embedding(d):
    """The noise-free cylinder in R^3, used as the topological reference."""
    return np.column_stack([np.cos(d["theta"]), np.sin(d["theta"]), d["h"]])
