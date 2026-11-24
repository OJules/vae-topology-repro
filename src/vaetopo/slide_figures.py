"""Large, minimal figures for the presentation slides.

The figures in plots.py are made for a report: titles, axes and several panels.
These ones are made to be read from the back of a room: one idea each, big marks,
almost no text in the image (the slide carries the words).
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1f2328", "#6b7280", "#e5e7eb"


def _square(path, draw):
    fig, ax = plt.subplots(figsize=(6, 6))
    draw(ax)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def as_ring(cyl_pts):
    """Flatten cylinder codes (cos phi, sin phi, h) into a ring: angle = phi, radius = 1 + 0.6 h.

    A cylinder cut open along its height and laid flat is a ring, so this view keeps
    both the loop and the height, and it can be compared with the Gaussian VAE's plane.
    """
    phi = np.arctan2(cyl_pts[:, 1], cyl_pts[:, 0])
    r = 1 + 0.6 * cyl_pts[:, 2]
    return np.column_stack([r * np.cos(phi), r * np.sin(phi)])


def latent_view(points2d, theta, path):
    """One model's codes, coloured by the true angle with a cyclic colour map."""
    def draw(ax):
        ax.scatter(points2d[:, 0], points2d[:, 1], c=theta, cmap="twilight",
                   s=34, linewidths=0)
    _square(path, draw)


def bottleneck_vs_beta(df, path):
    """Distance to the true loop shape against beta: one dot per seed, line = median."""
    fig, ax = plt.subplots(figsize=(11, 5.6))
    labels = {"Cylinder VAE": ("Cylinder rule", BLUE), "Gaussian VAE": ("Usual rule", ORANGE)}
    betas = sorted(df["beta"].unique())
    for model, (label, color) in labels.items():
        sub = df[df["model"] == model]
        ax.scatter(sub["beta"], sub["bottleneck_to_truth"], s=60, color=color, alpha=0.35,
                   linewidths=0, zorder=2)
        med = sub.groupby("beta")["bottleneck_to_truth"].median()
        ax.plot(med.index, med.values, color=color, linewidth=4, marker="o", markersize=12,
                markeredgecolor="white", markeredgewidth=2, zorder=3)
        ax.annotate(label, (med.index[-1], med.values[-1]), textcoords="offset points",
                    xytext=(16, 0), va="center", fontsize=17, color=color, fontweight="bold")
    ax.axhline(0.566, color=MUTED, linewidth=1.5, linestyle="--", zorder=1)
    ax.text(betas[0], 0.585, "no loop at all", color=MUTED, fontsize=14, va="bottom")
    ax.axhline(0, color=MUTED, linewidth=1.5, zorder=1)
    ax.text(betas[0], 0.02, "perfect shape", color=MUTED, fontsize=14, va="bottom")
    ax.set_xscale("log", base=2)
    ax.set_xticks(betas)
    ax.set_xticklabels([f"{b:g}" for b in betas])
    ax.set_xlim(betas[0] / 1.3, betas[-1] * 2.6)
    ax.set_ylim(-0.03, 0.66)
    ax.set_xlabel("β: how strictly the storage rule is enforced", fontsize=15, color=MUTED)
    ax.set_ylabel("Distance to the real loop shape", fontsize=15, color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=14)
    ax.grid(True, axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
