"""Redraw the four-variant latent figures from the stored checkpoints, without training.

Same configurations, seed and held-out split as latent_space_characterization.main().
The geometry scalars are recomputed from the noise-free codes and must agree with
artifacts/latent_characterization.json. The reconstruction scalars come from a
stochastic forward pass whose random state is not stored, so they differ from the stored
ones; latent_reconstruction.png needs the per-recipe errors and is not redrawn here, and
latent_summary.png is drawn from the stored scalars.

    PYTHONPATH=. python scripts/latent_figures_ckpt.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import latent_space_characterization as lsc
from src import benchmark, config

STORED = config.ARTIFACT_DIR / "latent_characterization.json"
EPOCHS = config.EPOCHS
SCALARS = ("hole_fraction", "raw_hole_fraction", "var_top2_pca", "var_top2_raw", "bandwidth",
           "ks", "scale_ratio", "pr")


def restore(cfg):
    config.EPOCHS = 0
    run, store, loader, _ = lsc.train_variant(cfg)
    state = torch.load(run.checkpoint, map_location=run.device, weights_only=True)
    run.model.load_state_dict(state)
    tau = max(lsc.TAU_FLOOR, 0.95 ** EPOCHS)
    return run, store, loader, tau


def main():
    config.FIGURE_DIR = config.ARTIFACT_DIR / "figures/latent"
    torch.save = lambda *a, **k: None
    characterizations = []
    for cfg in lsc.best_configs():
        run, store, loader, tau = restore(cfg)
        characterizations.append(lsc.characterize(cfg, run, store, loader, tau))

    stored = json.load(open(STORED))
    worst = 0.0
    for new, old in zip(characterizations, stored):
        assert new["label"] == old["label"] and new["active"] == old["active"], (new["label"], old["label"])
        for key in SCALARS:
            worst = max(worst, abs(new[key] - old[key]) / max(abs(old[key]), 1e-9))
    print(f"largest relative difference to the stored geometry scalars: {worst:.2e}")
    assert worst < 1e-3

    d_z = characterizations[0]["d_z"]
    benchmark.plot_latent_manifold(characterizations)
    benchmark.plot_spectrum_comparison(characterizations, d_z)
    benchmark.plot_prior_marginals(characterizations)
    benchmark.plot_summary(stored)
    benchmark.plot_hole_robustness(characterizations)
    benchmark.plot_projection_comparison(characterizations)
    benchmark.plot_latent_manifold(
        characterizations, hole_key="raw_holes",
        filename="latent_manifold__raw_dims.png",
        dim_labels=("top raw dim 1", "top raw dim 2"))
    benchmark.plot_latent_clusters_grid(characterizations, n_dims=lsc.N_CLUSTER_DIMS)
    plt.close("all")


if __name__ == "__main__":
    main()
