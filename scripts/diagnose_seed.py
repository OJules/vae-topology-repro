"""Diagnose individual seeds of the cylinder VAE: is the loop right, are the distances right?

Persistent homology only says whether the latent codes carry one loop. This script
looks at how the codes sit on that loop, to explain why one seed (seed 2) has the
right topology but a high geodesic stress.

For each seed, the cylinder VAE is retrained with the paper's settings and compared
with the true angle and height of the 400 test points:

    turns              how many times the latent angle goes round when the true angle
                       goes round once (+1 or -1 = one clean turn; the sign is only
                       the direction)
    angle_off_30deg    share of points whose latent angle is more than 30 degrees away
                       from the true angle, after the best rotation and reflection
    height_corr        correlation between true height and latent height
                       (+1 = same order, -1 = reversed)
    geodesic_stress    the paper's distance-preservation metric (0 = perfect)
    h1_top1            lifetime of the longest loop (true cylinder: 1.13)

    uv run python scripts/diagnose_seed.py                 # seeds 0 and 2, about 3 minutes
    uv run python scripts/diagnose_seed.py --seeds 0 1 2   # any seeds

Output: results/diagnosis/seed_diagnosis.csv
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from vaetopo.data import load_cylinder
from vaetopo.topology import h1_diagram, h1_summary, subsample
from vaetopo.train import fit, latent_points, paper_metrics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def wrap(a):
    """Bring angles back into [-pi, pi)."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def diagnose(theta, h, codes):
    """Compare latent codes (cos phi, sin phi, latent height) with the true angle and height."""
    phi = np.arctan2(codes[:, 1], codes[:, 0])

    # Turns: walk once around the true circle and add up the latent angle steps.
    order = np.argsort(theta)
    steps = wrap(np.diff(np.concatenate([phi[order], phi[order][:1]])))
    turns = steps.sum() / (2 * np.pi)

    # Angle error: allow the latent circle to be rotated and/or mirrored, keep the best fit.
    best = None
    for sign in (1, -1):
        z = np.exp(1j * (sign * phi - theta))
        fit_quality = abs(z.mean())
        if best is None or fit_quality > best[0]:
            best = (fit_quality, np.angle(z / z.mean()))
    angle_error = np.abs(best[1])

    return {
        "turns": round(float(turns), 2),
        "angle_off_30deg": float((angle_error > np.radians(30)).mean()),
        "median_angle_error_deg": float(np.degrees(np.median(angle_error))),
        "height_corr": float(np.corrcoef(h, codes[:, 2])[0, 1]),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.join(ROOT, "configs", "cylinder.json"))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 2])
    args = p.parse_args()
    cfg = json.load(open(args.config))
    beta = cfg["beta"]["cylinder"]
    out = os.path.join(ROOT, "results", "diagnosis")
    os.makedirs(out, exist_ok=True)

    data, train, test = load_cylinder(cfg["n_samples"], cfg["obs_dim"],
                                      cfg["data_seed"], cfg["test_ratio"])
    pts_idx = subsample(test["x"], cfg["ph_subsample"], seed=0)[1]

    rows = []
    for seed in args.seeds:
        print(f"\n=== cylinder VAE, seed {seed}, {cfg['n_epochs']} epochs ===", flush=True)
        model = fit("cylinder", train["x"], test["x"], beta, cfg["n_epochs"],
                    lr=cfg["lr"], hidden_dim=cfg["hidden_dim"], seed=seed)
        codes = latent_points(model, test["x"])
        metrics = paper_metrics(model, train, test, beta, seed)
        rows.append({"seed": seed,
                     **diagnose(test["theta"], test["h"], codes),
                     "geodesic_stress": metrics["geodesic_stress"],
                     "h1_top1": h1_summary(h1_diagram(codes[pts_idx]))["h1_top1"]})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "seed_diagnosis.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_columns", None,
                           "display.precision", 3):
        print("\n", df.to_string(index=False))
    print(f"\nResults written to {out}")


if __name__ == "__main__":
    sys.exit(main())
