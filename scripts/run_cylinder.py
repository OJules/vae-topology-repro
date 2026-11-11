"""Cylinder experiment: Gaussian VAE vs cylinder VAE, evaluated with persistent homology.

    uv run python scripts/run_cylinder.py            # full run (paper settings, 3 seeds)
    uv run python scripts/run_cylinder.py --quick    # smoke test, ~2 min on a laptop CPU

Outputs in results/cylinder[_quick]/:
    metrics.csv   one row per (model, seed): paper metrics + our H1 metrics
    summary.csv   mean and std over seeds
    fig_latent.png, fig_h1_diagrams.png   figures for seed 0
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from vaetopo import plots
from vaetopo.data import ground_truth_embedding, load_cylinder
from vaetopo.topology import compare, h1_diagram, h1_summary, subsample
from vaetopo.train import fit, latent_points, paper_metrics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.join(ROOT, "configs", "cylinder.json"))
    p.add_argument("--quick", action="store_true", help="1500 epochs, one seed")
    args = p.parse_args()
    cfg = json.load(open(args.config))
    if args.quick:
        cfg.update(n_epochs=1500, seeds=[0])
    out = os.path.join(ROOT, "results", "cylinder_quick" if args.quick else "cylinder")
    os.makedirs(out, exist_ok=True)

    data, train, test = load_cylinder(cfg["n_samples"], cfg["obs_dim"],
                                      cfg["data_seed"], cfg["test_ratio"])
    pts_idx = subsample(test["x"], cfg["ph_subsample"], seed=0)[1]
    thr = cfg["ph_loop_threshold"]

    # Reference topology: the true cylinder and the observed 50-D data.
    ref = h1_diagram(ground_truth_embedding(test)[pts_idx])
    dgms = {"True cylinder": ref, "Observations (R^50)": h1_diagram(test["x"][pts_idx])}
    rows = []
    for name, d in list(dgms.items()):
        rows.append({"model": name, "seed": -1, **h1_summary(d, thr),
                     "bottleneck_to_truth": compare(ref, d)})

    for seed in cfg["seeds"]:
        for kind in ["gaussian", "cylinder"]:
            t0 = time.time()
            print(f"\n=== {kind} VAE, seed {seed}, {cfg['n_epochs']} epochs ===", flush=True)
            model = fit(kind, train["x"], test["x"], cfg["beta"][kind], cfg["n_epochs"],
                        lr=cfg["lr"], hidden_dim=cfg["hidden_dim"], seed=seed)
            z = latent_points(model, test["x"])
            d = h1_diagram(z[pts_idx])
            rows.append({"model": f"{kind} VAE", "seed": seed,
                         **paper_metrics(model, train, test, cfg["beta"][kind], seed),
                         **h1_summary(d, thr), "bottleneck_to_truth": compare(ref, d),
                         "train_seconds": round(time.time() - t0, 1)})
            if seed == cfg["seeds"][0]:
                dgms[f"{kind.capitalize()} VAE latent"] = d
                if kind == "gaussian":
                    z_gauss = z
                else:
                    plots.latent_scatter(test["theta"], z_gauss, z,
                                         os.path.join(out, "fig_latent.png"))

    plots.diagrams(dgms, os.path.join(out, "fig_h1_diagrams.png"))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "metrics.csv"), index=False)
    cols = ["test_rms", "kl_divergence", "geodesic_stress",
            "h1_top1", "h1_top2", "n_loops", "bottleneck_to_truth"]
    summary = df.groupby("model", sort=False)[[c for c in cols if c in df]].agg(["mean", "std"])
    summary.to_csv(os.path.join(out, "summary.csv"))
    with pd.option_context("display.width", 200, "display.precision", 3):
        print("\n", summary)
    print(f"\nResults written to {out}")


if __name__ == "__main__":
    sys.exit(main())
