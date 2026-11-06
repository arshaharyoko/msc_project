"""Per-direction KS against the prior for the four stored latent checkpoints.

Same configurations, seed, held-out split and projection as
latent_space_characterization.characterize(); the statistic is the scipy kstest call
benchmark.plot_prior_marginals prints in the latent_prior_fit legend. No training, no
file under artifacts/ other than the output is written.

    PYTHONPATH=. python scripts/latent_ks_directions.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import kstest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import latent_space_characterization as lsc
from src import config

OUT = config.ARTIFACT_DIR / "latent_ks_directions.json"
STORED = config.ARTIFACT_DIR / "latent_characterization.json"
PLOTTED = 2


def restore(cfg):
    epochs = config.EPOCHS
    config.EPOCHS = 0
    run, store, loader, _ = lsc.train_variant(cfg)
    config.EPOCHS = epochs
    state = torch.load(run.checkpoint, map_location=run.device, weights_only=True)
    run.model.load_state_dict(state)
    return run, store, loader, max(lsc.TAU_FLOOR, 0.95 ** epochs)


def main():
    torch.save = lambda *a, **k: None
    stored = {r["label"]: r for r in json.load(open(STORED))}

    variants = {}
    for cfg in lsc.best_configs():
        run, store, loader, tau = restore(cfg)
        item = lsc.characterize(cfg, run, store, loader, tau)
        projection = np.asarray(item["projection"])
        ks = [float(kstest(projection[:, d], "norm").statistic)
              for d in range(projection.shape[1])]
        variants[cfg["label"]] = {
            "n_directions": len(ks),
            "ks_by_direction": ks,
            "ks_plotted": ks[:PLOTTED],
            "ks_mean": float(np.mean(ks)),
            "ks_mean_characterize": item["ks"],
            "ks_mean_stored": stored[cfg["label"]]["ks"],
        }

    with open(OUT, "w") as f:
        json.dump(variants, f, indent=2)
    for label, v in variants.items():
        print(f"{label:<12} plotted {[round(x, 3) for x in v['ks_plotted']]} "
              f"mean {v['ks_mean']:.6f} stored {v['ks_mean_stored']:.6f}")


if __name__ == "__main__":
    main()
