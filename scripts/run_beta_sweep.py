"""Beta sweep: does the prescribed topology help across the useful range of beta?

Both models are trained with the SAME beta, for each beta in the config and each
seed, and evaluated with the paper's metrics and our persistent-homology metrics.

    uv run python scripts/run_beta_sweep.py            # full sweep, 5 betas x 3 seeds x 2 models
    uv run python scripts/run_beta_sweep.py --quick    # smoke test: 3 betas, 1 seed, 800 epochs

Outputs in results/beta_sweep[_quick]/:
    metrics.csv         one row per (model, beta, seed)
    summary.csv         mean and std over seeds, per (model, beta)
    fig_beta_sweep.png  the four metrics against beta
"""

import argparse
import json
import os
import sys
import time

import pandas as pd

from vaetopo import plots
from vaetopo.data import ground_truth_embedding, load_cylinder
from vaetopo.topology import compare, h1_diagram, h1_summary, subsample
from vaetopo.train import fit, latent_points, paper_metrics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS = {"gaussian": "Gaussian VAE", "cylinder": "Cylinder VAE"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.join(ROOT, "configs", "beta_sweep.json"))
    p.add_argument("--quick", action="store_true", help="3 betas, one seed, 800 epochs")
    args = p.parse_args()
    cfg = json.load(open(args.config))
    if args.quick:
        cfg.update(n_epochs=800, seeds=[0], betas=[0.5, 1.0, 2.0])
    out = os.path.join(ROOT, "results", "beta_sweep_quick" if args.quick else "beta_sweep")
    os.makedirs(out, exist_ok=True)

    data, train, test = load_cylinder(cfg["n_samples"], cfg["obs_dim"],
                                      cfg["data_seed"], cfg["test_ratio"])
    pts_idx = subsample(test["x"], cfg["ph_subsample"], seed=0)[1]
    ref = h1_diagram(ground_truth_embedding(test)[pts_idx])
    ref_bar = h1_summary(ref)["h1_top1"]

    rows = []
    n_runs = len(cfg["betas"]) * len(cfg["seeds"]) * 2
    run = 0
    for beta in cfg["betas"]:
        for seed in cfg["seeds"]:
            for kind in ["gaussian", "cylinder"]:
                run += 1
                t0 = time.time()
                print(f"\n=== run {run}/{n_runs}: {kind} VAE, beta {beta}, seed {seed}, "
                      f"{cfg['n_epochs']} epochs ===", flush=True)
                model = fit(kind, train["x"], test["x"], beta, cfg["n_epochs"],
                            lr=cfg["lr"], hidden_dim=cfg["hidden_dim"], seed=seed)
                d = h1_diagram(latent_points(model, test["x"])[pts_idx])
                rows.append({"model": LABELS[kind], "beta": beta, "seed": seed,
                             **paper_metrics(model, train, test, beta, seed),
                             **h1_summary(d, cfg["ph_loop_threshold"]),
                             "bottleneck_to_truth": compare(ref, d),
                             "train_seconds": round(time.time() - t0, 1)})
                # Save after every run, so an interrupted sweep keeps what it has done.
                pd.DataFrame(rows).to_csv(os.path.join(out, "metrics.csv"), index=False)

    df = pd.DataFrame(rows)
    cols = ["test_rms", "kl_divergence", "geodesic_stress", "h1_top1", "bottleneck_to_truth"]
    summary = df.groupby(["model", "beta"], sort=False)[cols].agg(["mean", "std"])
    summary.to_csv(os.path.join(out, "summary.csv"))
    plots.beta_sweep(df, ref_bar, os.path.join(out, "fig_beta_sweep.png"))

    with pd.option_context("display.width", 250, "display.max_columns", None,
                           "display.precision", 3):
        print("\n", summary)
    print(f"\nResults written to {out}")


if __name__ == "__main__":
    sys.exit(main())
