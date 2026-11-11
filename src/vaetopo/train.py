"""Training: the authors' models and training loop, with their hyperparameters.

Gaussian baseline (latent R^2, beta = 1.2) against the cylinder VAE
(latent S^1 x [0, 1], beta = 1.0). The authors tune the two betas so that the
unweighted KL matches across models; we keep their values.
"""

import numpy as np
import torch
from topovae import CylinderVAE, GaussianVAE, get_latent_codes, train_vae
from topovae.evaluate import compute_all_metrics

MODELS = {"gaussian": GaussianVAE, "cylinder": CylinderVAE}


def fit(kind, x_train, x_test, beta, n_epochs, lr=1e-3, hidden_dim=128, seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = MODELS[kind](input_dim=x_train.shape[1], hidden_dim=hidden_dim)
    train_vae(model, x_train, n_epochs=n_epochs, lr=lr, beta=beta,
              beta_anneal_epochs=n_epochs // 4, x_test=x_test,
              print_every=max(1, n_epochs // 4))
    return model


def latent_points(model, x):
    """Latent codes as a point cloud in the space the decoder actually reads:
    R^2 for the Gaussian VAE, (cos theta, sin theta, h) in R^3 for the cylinder VAE."""
    z = get_latent_codes(model, x)
    if "z" in z:
        return z["z"]
    return np.column_stack([np.cos(z["theta"]), np.sin(z["theta"]), z["h"]])


def paper_metrics(model, train, test, beta, seed=0):
    """The paper's own metrics: reconstruction RMS, KL, geodesic stress."""
    np.random.seed(seed)
    m = compute_all_metrics(model, train["x"], test["x"], beta=beta,
                            true_factors={"theta": test["theta"], "h": test["h"]},
                            data_manifold="cylinder")
    return {k: float(v) for k, v in m.items()}
