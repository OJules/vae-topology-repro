"""Figures for the presentation."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from persim import plot_diagrams

# Two series, fixed order: blue = cylinder VAE, orange = Gaussian VAE.
COLORS = {"Cylinder VAE": "#2a78d6", "Gaussian VAE": "#eb6834"}
INK, MUTED, GRID = "#1f2328", "#6b7280", "#e5e7eb"


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


def beta_sweep(df, ref_bar, path):
    """Four metrics against beta: one dot per seed, line = median over seeds.

    With few seeds and one outlying seed, a mean +/- std band is misleading
    (it can even cross zero for a distance), so every run is shown.
    """
    panels = [
        ("test_rms", "Reconstruction error (lower is better)"),
        ("geodesic_stress", "Geodesic stress (lower is better)"),
        ("h1_top1", "Longest loop in the latent space"),
        ("bottleneck_to_truth", "Bottleneck distance to the true cylinder (lower is better)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, (col, title) in zip(axes.ravel(), panels):
        for k, (model, color) in enumerate(COLORS.items()):
            sub = df[df["model"] == model]
            shift = 1.04 if k == 0 else 1 / 1.04  # nudge the two models apart on the log axis
            ax.scatter(sub["beta"] * shift, sub[col], s=22, color=color, alpha=0.45,
                       linewidths=0, zorder=2)
            med = sub.groupby("beta")[col].median()
            ax.plot(med.index * shift, med.values, color=color, linewidth=2, marker="o",
                    markersize=7, markeredgecolor="white", markeredgewidth=1.5, label=model,
                    zorder=3)
        if col == "h1_top1":
            ax.axhline(ref_bar, color=MUTED, linewidth=1.2, linestyle="--", zorder=1)
            ax.text(df["beta"].min() * 0.95, ref_bar * 1.02, "true cylinder", color=MUTED,
                    fontsize=9, va="bottom", ha="left")
        ax.set_xscale("log", base=2)
        betas = sorted(df["beta"].unique())
        ax.set_xticks(betas); ax.set_xticklabels([f"{b:g}" for b in betas])
        ax.set_title(title, fontsize=10.5, color=INK, loc="left")
        ax.set_xlabel("β (weight of the KL term)", fontsize=9, color=MUTED)
        ax.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.text(0.5, 0.985, "Both models trained with the same β · small dots = one seed each, line = median",
             ha="center", va="top", fontsize=9, color=MUTED)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.96), ncol=2,
               frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=150); plt.close(fig)
