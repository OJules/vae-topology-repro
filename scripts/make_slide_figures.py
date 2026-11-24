"""Make the large figures used in the presentation slides.

Retrains three models with the paper's settings (Gaussian VAE seed 0, cylinder VAE
seeds 0 and 2) and reads the beta sweep results already on disk.

    uv run python scripts/make_slide_figures.py     # about 5 minutes on a laptop CPU

Outputs in results/slides/:
    latent_gaussian.png, latent_cylinder.png   slide 10: torn spiral vs kept circle
    seed0.png, seed2.png                       slide 14: a clean cylinder vs a twisted one
    beta.png                                   slide 12: distance to the true loop shape vs beta
    codes.npz                                  the latent codes behind these figures
"""

import json
import os
import sys

import numpy as np
import pandas as pd

from vaetopo import slide_figures as sf
from vaetopo.data import load_cylinder
from vaetopo.train import fit, latent_points

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    cfg = json.load(open(os.path.join(ROOT, "configs", "cylinder.json")))
    out = os.path.join(ROOT, "results", "slides")
    os.makedirs(out, exist_ok=True)
    data, train, test = load_cylinder(cfg["n_samples"], cfg["obs_dim"],
                                      cfg["data_seed"], cfg["test_ratio"])

    codes = {}
    for kind, seed in [("gaussian", 0), ("cylinder", 0), ("cylinder", 2)]:
        print(f"\n=== {kind} VAE, seed {seed}, {cfg['n_epochs']} epochs ===", flush=True)
        model = fit(kind, train["x"], test["x"], cfg["beta"][kind], cfg["n_epochs"],
                    lr=cfg["lr"], hidden_dim=cfg["hidden_dim"], seed=seed)
        codes[f"{kind}_seed{seed}"] = latent_points(model, test["x"])
    np.savez(os.path.join(out, "codes.npz"), theta=test["theta"], h=test["h"], **codes)

    theta = test["theta"]
    sf.latent_view(codes["gaussian_seed0"], theta, os.path.join(out, "latent_gaussian.png"))
    sf.latent_view(sf.as_ring(codes["cylinder_seed0"]), theta, os.path.join(out, "latent_cylinder.png"))
    sf.latent_view(sf.as_ring(codes["cylinder_seed0"]), theta, os.path.join(out, "seed0.png"))
    sf.latent_view(sf.as_ring(codes["cylinder_seed2"]), theta, os.path.join(out, "seed2.png"))

    sweep = os.path.join(ROOT, "results", "beta_sweep", "metrics.csv")
    if os.path.exists(sweep):
        sf.bottleneck_vs_beta(pd.read_csv(sweep), os.path.join(out, "beta.png"))
    else:
        print("results/beta_sweep/metrics.csv not found: run scripts/run_beta_sweep.py first")
    print(f"\nFigures written to {out}")


if __name__ == "__main__":
    sys.exit(main())
