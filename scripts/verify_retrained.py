"""Check retrained checkpoints against the deployed ones.

    PYTHONPATH=. python scripts/verify_retrained.py --retrained-dir artifacts/retrained --seeds 0 1 2
    PYTHONPATH=. python scripts/verify_retrained.py --retrained-dir A --second-dir B --seeds 0

Decision rule, fixed before any run. A seed reproduces when (i) every tensor of the retrained
state_dict equals the deployed one bitwise, or (ii) it does not, a second retrain of the same
seed (--second-dir) differs from the first by at least as much (max abs difference over all
tensors) as the first differs from the deployed one, and every downstream metric of the first
retrain is within the run to run spread of the second, or within 1e-6 of the deployed value when
that spread is zero. Without --second-dir a non-bitwise seed fails. Exit status 1 when any seed
fails.

Downstream metrics are computed with latent_space_characterization.characterize on the test
split of fold 0, the same call scripts/latent_curves.py makes: mean held-out reconstruction
error per recipe (recon_all), live latent directions, hole fraction, mean KS against the prior.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import latent_space_characterization as lsc                       # noqa: E402
from src.ablation import AblationSpec, _build, seed_everything, split_dataset  # noqa: E402
from src.modules.dataset import collate_fn                        # noqa: E402
from src.vector_store import VectorStore                          # noqa: E402

NAME = "ablation__selected__728320fe947b__s{}__graded.pth"
METRICS = ("recon", "active", "hole_fraction", "ks_mean")
ABS_TOL = 1e-6


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def tensor_gap(a, b):
    if a.keys() != b.keys():
        raise SystemExit("state_dict keys differ")
    gaps = [(a[k].double() - b[k].double()).abs().max().item() if a[k].numel() else 0.0 for k in a]
    return {"tensors": len(gaps), "equal": sum(g == 0 for g in gaps), "max_abs": max(gaps)}


def downstream(payload):
    spec = AblationSpec(**payload["spec"])
    seed_everything(spec.seed)
    run = _build(spec)
    run.model.load_state_dict(payload["state_dict"])
    run.model.eval()
    store = VectorStore.from_run(run)
    test = split_dataset(run.dataset, spec)["test"]
    loader = DataLoader(test, run.batch_size, shuffle=False, collate_fn=collate_fn)
    cfg = {"label": f"s{spec.seed}", "ablate_mmd": "1" if run.ablate_mmd else "0",
           "ablate_svd": "1" if run.ablate_svd else "0"}
    item = lsc.characterize(cfg, run, store, loader)
    return {"recon": float(np.mean(item["recon_per_recipe"])), "active": float(item["active"]),
            "hole_fraction": float(item["hole_fraction"]), "ks_mean": float(item["ks"])}


def judge(seed, stored, first, second, metrics):
    gap = tensor_gap(first["state_dict"], stored["state_dict"])
    row = {"seed": seed, "vs_stored": gap}
    if gap["equal"] == gap["tensors"]:
        row["pass"], row["reason"] = True, "bitwise equal"
        return row
    if second is None:
        row["pass"], row["reason"] = False, "not bitwise equal and no second retrain given"
        return row
    spread = tensor_gap(second["state_dict"], first["state_dict"])
    row["run_to_run"] = spread
    if spread["max_abs"] < gap["max_abs"]:
        row["pass"], row["reason"] = False, "gap to deployed exceeds run to run spread"
        return row
    m_first, m_stored, m_second = metrics
    bad = [m for m in METRICS
           if abs(m_first[m] - m_stored[m]) > max(abs(m_second[m] - m_first[m]), ABS_TOL)]
    row["pass"] = not bad
    row["reason"] = "within run to run spread" if not bad else f"metrics outside spread: {bad}"
    return row


def main():
    p = argparse.ArgumentParser(description="verify retrained checkpoints against the deployed ones")
    p.add_argument("--retrained-dir", required=True)
    p.add_argument("--second-dir", help="independent retrain of the same seeds")
    p.add_argument("--deployed-dir", default=str(ROOT / "artifacts"))
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    args = p.parse_args()

    failed = False
    print(f"{'seed':>4} {'tensors':>7} {'equal':>5} {'max_abs':>10}  " +
          "  ".join(f"{m:>13}" for m in METRICS) + "  result")
    for seed in args.seeds:
        stored = load(Path(args.deployed_dir) / NAME.format(seed))
        first = load(Path(args.retrained_dir) / NAME.format(seed))
        second = load(Path(args.second_dir) / NAME.format(seed)) if args.second_dir else None
        m_stored, m_first = downstream(stored), downstream(first)
        m_second = downstream(second) if second else None
        row = judge(seed, stored, first, second, (m_first, m_stored, m_second))
        failed |= not row["pass"]
        g = row["vs_stored"]
        diffs = "  ".join(f"{abs(m_first[m] - m_stored[m]):13.3g}" for m in METRICS)
        print(f"{seed:>4} {g['tensors']:>7} {g['equal']:>5} {g['max_abs']:>10.3g}  {diffs}  "
              f"{'PASS' if row['pass'] else 'FAIL'}: {row['reason']}", flush=True)
        if "run_to_run" in row:
            print(f"     run to run: {row['run_to_run']}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
