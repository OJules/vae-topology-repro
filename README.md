# VAE latent spaces with prescribed topology: a reproduction

Final project for *Geometry and Topology in Machine Learning* (University of Fribourg, fall 2026), Project 4, research direction 1.

We reproduce the **periodic-data (cylinder) experiment** of

> J. S. van Hulst, J. M. Tomczak, W. P. M. H. Heemels, D. J. Antunes.
> *Constructing VAE Latent Spaces with Prescribed Topology.* arXiv:2606.07058 (2026).

and evaluate the learned latent spaces with **persistent homology**, a tool the paper itself does not use.

## Question

Data sampled on a cylinder S¹ × [0, 1] (one loop) is embedded in R⁵⁰. Does a VAE whose latent space *is* a cylinder recover that loop, where a standard Gaussian VAE (latent R²) does not? The paper measures this with reconstruction error and geodesic stress; we add a topological check: the H₁ persistence diagram of the latent codes, compared with the diagram of the true cylinder.

## Install and run

Requires [uv](https://docs.astral.sh/uv/). Everything runs on a laptop CPU.

```bash
git clone <this repository>
cd vae-topology-repro
uv sync                                          # creates .venv, installs pinned deps
uv run python scripts/run_cylinder.py --quick    # smoke test, about 1 minute
uv run python scripts/run_all.py                 # full experiment, about 10 minutes
```

Results are written to `results/cylinder/` (`results/cylinder_quick/` for the smoke test).

## What is ours and what is the authors'

| Part | Source |
|---|---|
| Data generator, train/test split, models, training loop, paper metrics | Authors' `topovae` library, installed from [JvHulst/VAE-Topology](https://github.com/JvHulst/VAE-Topology) at commit `08ed7cc`, unchanged |
| Hyperparameters (2000 samples, 8000 epochs, lr 1e-3, β = 1.2 Gaussian / 1.0 cylinder) | Authors' `experiments/synthetic/experiment.py` |
| Persistent-homology evaluation, normalisation, multi-seed protocol, figures | This repository (`src/vaetopo/topology.py`, `plots.py`, `scripts/`) |

We train the unanchored models only (the anchor loss fixes the coordinate system, not the topology).

## Repository layout

```
configs/cylinder.json     all settings of the experiment
src/vaetopo/
    data.py               authors' cylinder data + ground-truth embedding
    train.py              trains Gaussian / cylinder VAE, extracts latent point clouds
    topology.py           Vietoris-Rips H1 diagrams (ripser), bottleneck distance (persim)
    plots.py              latent scatter and persistence-diagram figures
scripts/
    run_cylinder.py       the experiment
    run_all.py            re-runs everything
results/                  metrics and figures
```

## Evaluation protocol

1. Encode the 400 test points with the posterior mean.
2. Turn the codes into a point cloud in the space the decoder reads: R² for the Gaussian VAE, (cos θ, sin θ, h) ∈ R³ for the cylinder VAE.
3. Rescale every cloud to mean pairwise distance 1, so clouds of different dimension and scale are comparable.
4. Compute the H₁ Vietoris-Rips diagram and report: the two longest bars (`h1_top1`, `h1_top2`), the number of bars longer than 0.2 (`n_loops`), and the bottleneck distance to the diagram of the true cylinder.

Seeds vary model initialisation only; data and split are fixed (as in the paper).

## Results (paper settings, 3 seeds, mean ± std)

| Model | Test RMS ↓ | KL | Geodesic stress ↓ | Longest H₁ bar | Bottleneck to truth ↓ |
|---|---|---|---|---|---|
| True cylinder (reference) | – | – | – | 1.13 | 0 |
| Observations in R⁵⁰ | – | – | – | 0.73 | 0.27 |
| Gaussian VAE (R²) | 0.194 ± 0.002 | 3.14 ± 0.04 | 0.323 ± 0.004 | 0.30 ± 0.04 | 0.566 ± 0.000 |
| Cylinder VAE (S¹ × [0,1]) | **0.185 ± 0.006** | 3.22 ± 0.04 | **0.161 ± 0.125** | **1.12 ± 0.03** | **0.057 ± 0.009** |

What we observe:

1. **Reproduced:** at a comparable KL, the cylinder VAE reconstructs slightly better and halves the geodesic stress on average, as the paper claims.
2. **Topology:** the cylinder VAE's latent codes carry one long H₁ bar of the same length as the true cylinder (bottleneck 0.06). The Gaussian VAE's longest bar is about four times shorter: the latent scatter shows the circle torn into a spiral, so the loop never closes.
3. **The Gaussian bottleneck is exactly 0.566 for every seed.** This is half the true bar's length: the true loop is matched to the diagonal, i.e. the Gaussian latent has no loop worth matching. It is a ceiling, not a coincidence.
4. **One cylinder seed (seed 2) has a high geodesic stress (0.305) while its H₁ is still correct.** Persistent homology and geodesic stress measure different things: the loop exists, but distances along it are distorted. To investigate next.
5. **`n_loops` does not separate the models** (1 for both): the 0.2 threshold is too low, since the Gaussian's torn ring still produces a 0.3 bar. The bar length and bottleneck distance are the informative metrics.

Figures: `results/cylinder/fig_latent.png`, `results/cylinder/fig_h1_diagrams.png` (seed 0).

## Caveats

- For the cylinder VAE, the latent space is a circle by construction, so a loop is guaranteed *if the codes cover the whole circle*. Persistent homology still tests something: a posterior that collapses to an arc would show no long H₁ bar. The fairer comparison is the bottleneck distance, which also measures how large and clean the loop is.
- `n_loops` depends on the 0.2 threshold; in 50 dimensions, noise alone produces a second bar around 0.2. The raw bar lengths are reported so the threshold can be checked.
