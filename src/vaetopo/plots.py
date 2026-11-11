"""Figures for the presentation."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from persim import plot_diagrams


def latent_scatter(theta, gaussian_z, cylinder_pts, path):
    """Latent codes coloured by the true angle: a smooth colour wheel = circle kept."""
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.2))
    ax[0].scatter(gaussian_z[:, 0], gaussian_z[:, 1], c=theta, cmap="twilight", s=6)
    ax[0].set_title("Gaussian VAE  (latent $\\mathbb{R}^2$)")
    ang = np.arctan2(cylinder_pts[:, 1], cylinder_pts[:, 0])
    ax[1].scatter(ang, cylinder_pts[:, 2], c=theta, cmap="twilight", s=6)
    ax[1].set_title("Cylinder VAE  (latent $S^1\\times[0,1]$)")
    ax[1].set_xlabel("latent angle"); ax[1].set_ylabel("latent height")
    ax[1].set_xlim(-np.pi, np.pi)
    for a in ax:
        a.set_aspect("auto")
    fig.suptitle("Latent codes coloured by the true angle $\\theta$")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def diagrams(dgms, path):
    """H1 persistence diagrams side by side, same axes."""
    names = list(dgms)
    fig, ax = plt.subplots(1, len(names), figsize=(3.6 * len(names), 3.6))
    hi = max(float(d[:, 1].max()) for d in dgms.values() if len(d)) * 1.05
    for a, n in zip(ax, names):
        plot_diagrams([np.zeros((0, 2)), dgms[n]], ax=a, show=False,
                      xy_range=[0, hi, 0, hi], legend=False)
        a.set_title(n)
    fig.suptitle("$H_1$ persistence diagrams (far from the diagonal = real loop)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
