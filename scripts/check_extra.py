"""Checks for scripts/run_extra.sh (ADR-043). Each subcommand exits 0 on success, 1 with a reason on stderr.

  unit               patched functions on synthetic arrays, no checkpoint, no GPU
  validate FILE      FILE is a complete 7-weight sweep
  alpha1             the alpha = 1 sweeps of seed 0 equal the files in artifacts/optimizer; writes artifacts/optimizer_sweeps/check_alpha1.ok
"""

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

OPTIMIZER_DIR = ROOT / "artifacts" / "optimizer"
SWEEP_DIR = ROOT / "artifacts" / "optimizer_sweeps"
PAIRS = {"free": "latent_anch__s0.json", "prot": "latent_prot__s0.json"}
GRID = [0.01, 0.03162277660168379, 0.1, 0.31622776601683794, 1.0, 3.1622776601683795, 10.0]


def fail(message):
    print(message, file=sys.stderr)
    raise SystemExit(1)


def load_sweep(path):
    try:
        with open(path) as f:
            payload = json.load(f)
    except (OSError, ValueError) as err:
        fail(f"{path}: {err}")
    runs = payload.get("runs")
    if not isinstance(runs, list) or len(runs) != len(GRID):
        fail(f"{path}: {None if runs is None else len(runs)} runs, expected {len(GRID)}")
    got = [r.get("cost_weight") for r in runs]
    if not all(math.isclose(g, w, rel_tol=1e-9) for g, w in zip(got, GRID)):
        fail(f"{path}: cost weights {got}")
    return payload


def cmd_validate(args):
    load_sweep(args.file)


def cmd_alpha1(args):
    problems = []
    for arm, ref in PAIRS.items():
        new = load_sweep(SWEEP_DIR / f"check_alpha1_{arm}__s0.json")["runs"]
        old = load_sweep(OPTIMIZER_DIR / ref)["runs"]
        for n, o in zip(new, old):
            w = o["cost_weight"]
            if n["round_trip"]["feasible"] != o["round_trip"]["feasible"]:
                problems.append(f"{arm} w={w}: feasible {n['round_trip']['feasible']} vs {o['round_trip']['feasible']}")
            if n["realized_cost"] != o["realized_cost"]:
                problems.append(f"{arm} w={w}: realized_cost {n['realized_cost']!r} vs {o['realized_cost']!r}")
    marker = SWEEP_DIR / "check_alpha1.ok"
    if problems:
        marker.unlink(missing_ok=True)
        fail("alpha = 1 differs from artifacts/optimizer:\n  " + "\n  ".join(problems))
    marker.write_text("alpha=1 matches artifacts/optimizer on seed 0, both arms, all 7 weights\n")
    print(f"ok: 14 runs match, wrote {marker}")


def cmd_unit(args):
    import numpy as np
    import predict_extra as wrap

    rng = np.random.default_rng(0)
    x1, x2 = rng.normal(size=(1, 9)), rng.normal(size=(1, 4))
    anchor = (rng.normal(size=(1, 9)), rng.normal(size=(1, 4)))
    xh = (rng.normal(size=(1, 9)), rng.normal(size=(1, 4)))

    def decode(alpha):
        def original(run, ctx):
            ctx["anchor"] = anchor
        ctx = {"x1": x1, "x2": x2}
        wrap.with_anchor_strength(original, alpha)(None, ctx)
        return tuple(x + h - a for x, h, a in zip((x1, x2), xh, ctx["anchor"]))

    for alpha in (1.0, 0.0, 0.25):
        want = tuple(h + alpha * (x - a) for x, h, a in zip((x1, x2), xh, anchor))
        got = decode(alpha)
        if alpha == 1.0:
            ref = tuple(x + h - a for x, h, a in zip((x1, x2), xh, anchor))
            assert all(np.array_equal(g, r) for g, r in zip(got, ref)), "alpha 1 not exact"
        if alpha == 0.0:
            assert all(np.allclose(g, h, atol=1e-12) for g, h in zip(got, xh)), "alpha 0 not raw"
        assert all(np.allclose(g, w, atol=1e-12) for g, w in zip(got, want)), f"alpha {alpha}"
        print(f"alpha {alpha}: ok")

    import predict
    from src import config

    with open(config.DATABASE_JSON) as f:
        base = wrap.base_doses(json.load(f), predict.BASE_SOURCE)
    print(f"base doses from {config.DATABASE_JSON} record BASE_SOURCE: {base}")
    orig = predict.protein_floors()
    assert wrap.with_floor_scale(predict.protein_floors, "1", base)() == orig, "scale 1 differs"
    for scale in ("2", "5", "base"):
        got = wrap.with_floor_scale(predict.protein_floors, scale, base)()
        factor = float("inf") if scale == "base" else float(scale)
        assert set(got) == set(orig)
        for acc, low in orig.items():
            want = min(factor * low, base[acc]) if acc in base else low
            assert got[acc] == want, (scale, acc, got[acc], want)
            assert got[acc] <= base.get(acc, low)
        print(f"scale {scale}: " + ", ".join(f"{a}={got[a]:g}" for a in base))

    orig_fn, orig_dec = predict.protein_floors, predict.anchor_decode
    wrap.install("0.5", "2")
    assert predict.anchor_decode is not orig_dec and predict.protein_floors is not orig_fn
    print("install: both names replaced on the predict module")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("unit").set_defaults(fn=cmd_unit)
    v = sub.add_parser("validate")
    v.add_argument("file")
    v.set_defaults(fn=cmd_validate)
    sub.add_parser("alpha1").set_defaults(fn=cmd_alpha1)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
