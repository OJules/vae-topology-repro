"""Persistent-homology evaluation of a point cloud (our contribution).

Idea: grow a ball around every point (Vietoris-Rips filtration) and record when
loops appear and disappear. A loop that lives over a long range of radii is real
structure; a short-lived one is noise. The cylinder has exactly one real loop,
so a latent space that preserves its topology should show one long H1 bar.

To compare clouds living at different scales (R^50, R^2, a circle x interval),
each cloud is first rescaled so that its mean pairwise distance is 1.
"""

import numpy as np
from persim import bottleneck
from ripser import ripser
from scipy.spatial.distance import pdist


def subsample(points, n, seed=0):
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(points), size=min(n, len(points)), replace=False)
    return points[idx], idx


def normalise(points):
    return points / pdist(points).mean()


def h1_diagram(points):
    """H1 persistence diagram (birth, death) of a normalised point cloud."""
    dgm = ripser(normalise(points), maxdim=1)["dgms"][1]
    return dgm[np.isfinite(dgm[:, 1])]


def h1_summary(dgm, threshold=0.2):
    """Top persistences and the number of loops longer than `threshold`.

    `threshold` is in normalised units (mean pairwise distance = 1). We report
    the two longest bars as well, so the choice of threshold can be checked.
    """
    pers = np.sort(dgm[:, 1] - dgm[:, 0])[::-1] if len(dgm) else np.zeros(0)
    top = np.pad(pers[:2], (0, max(0, 2 - len(pers))))
    return {
        "h1_top1": float(top[0]),
        "h1_top2": float(top[1]),
        "n_loops": int((pers > threshold).sum()),
    }


def compare(reference_dgm, dgm):
    """Bottleneck distance between two H1 diagrams: 0 = same topology and scale."""
    return float(bottleneck(reference_dgm, dgm))
