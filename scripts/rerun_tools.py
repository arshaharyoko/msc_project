"""Checks used by scripts/run_rerun.sh. Each subcommand exits 0 on success, 1 with a reason on stderr."""

import argparse
import contextlib
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SETTINGS = {"fold": 1, "readout": "softmax", "negatives": "graded"}
CONFIG_KEYS = ("label", "seed", "pathway_alpha", "div_beta", "gate_gamma", "tsvd_k",
               "recon_w2", "cost_delta")


def fail(message):
    print(message, file=sys.stderr)
    raise SystemExit(1)


def shard_rows(total, n, i):
    return len(range(i, total, n))


def load_stage(path, stage, rows):
    try:
        with open(path) as f:
            payload = json.load(f)
    except (OSError, ValueError) as err:
        fail(f"{path}: {err}")
    if payload.get("stage") != stage:
        fail(f"{path}: stage {payload.get('stage')!r}, expected {stage!r}")
    results = payload.get("results")
    if not isinstance(results, list) or len(results) != rows:
        fail(f"{path}: {None if results is None else len(results)} rows, expected {rows}")
    for key, want in SETTINGS.items():
        bad = {r.get(key) for r in results} - {want}
        if bad:
            fail(f"{path}: {key} {sorted(bad, key=str)}, expected {want!r}")
    return payload


def cmd_fingerprint(args):
    import torch
    from src import config
    from src.constraints import _fingerprint
    from src.modules.dataset import GrowthMediumDataset

    dataset = GrowthMediumDataset(json_file=config.DATA_DIR / "database_graded.json")
    if torch.cuda.is_available():
        device = "cuda:" + torch.cuda.get_device_name(0).replace(" ", "_")
    else:
        device = "cpu"
    print(f"{device} {_fingerprint(dataset)} {len(dataset)}")


def cmd_rows(args):
    import numpy as np
    import hipo

    if args.stage == "screen":
        print(len(hipo.sample_configs()))
        return
    source = json.load(open(args.source))
    with contextlib.redirect_stdout(sys.stderr):
        if args.stage == "refine":
            fixed = {k: v for k, v in hipo.select_shared(source["results"]).items()
                     if k in ("recon_w2", "cost_delta")}
            count = len(hipo.sample_configs(np.random.default_rng(hipo.SEED + 1), fixed,
                                            hipo.REFINE_PER_VARIANT))
        else:
            count = len(hipo.confirm_configs(source["results"]))
    print(count)


def cmd_smallest_k_index(args):
    import hipo

    configs = hipo.sample_configs()
    best = min(range(len(configs)), key=lambda j: (configs[j]["tsvd_k"], j))
    print(len(configs), best)


def cmd_shards(args):
    print(math.ceil(args.total / args.max_rows))


def cmd_plan(args):
    """Pin a stage's (total, n); a later run with other numbers must not reuse its shards."""
    path = Path(args.dir) / f"hipo_{args.stage}_plan.json"
    plan = {"total": args.total, "n": args.n, "max_rows": args.max_rows}
    if path.exists():
        found = json.load(open(path))
        if found != plan:
            fail(f"{path} pins {found}, this run computed {plan}; shards written under one "
                 f"plan cannot be merged with another (move the old shards and plan aside)")
        return
    path.write_text(json.dumps(plan))


def cmd_validate(args):
    load_stage(args.path, args.stage, shard_rows(args.total, args.n, args.i))


def stage_shards(args):
    pattern = re.compile(rf"hipo_{args.stage}__(\d+)\.json$")
    found = {}
    for path in Path(args.dir).glob(f"hipo_{args.stage}__*.json"):
        match = pattern.match(path.name)
        if match:
            found[int(match.group(1))] = path
    return found


def cmd_missing(args):
    found = stage_shards(args)
    bad = []
    for i in range(args.n):
        if i not in found:
            bad.append(i)
            continue
        try:
            load_stage(found[i], args.stage, shard_rows(args.total, args.n, i))
        except SystemExit:
            bad.append(i)
    print(" ".join(map(str, bad)))


def cmd_checkset(args):
    """Every shard 0..n-1 present and valid, nothing else under the stage's shard glob."""
    found = stage_shards(args)
    extra = sorted(set(found) - set(range(args.n)))
    if extra:
        fail(f"{args.stage}: shard files outside 0..{args.n - 1}: {extra} "
             f"(another shard count was run; move them aside)")
    missing = [i for i in range(args.n) if i not in found]
    if missing:
        fail(f"{args.stage}: missing shards {missing}")
    seen = set()
    for i, path in sorted(found.items()):
        payload = load_stage(path, args.stage, shard_rows(args.total, args.n, i))
        for r in payload["results"]:
            key = json.dumps({k: r[k] for k in CONFIG_KEYS}, sort_keys=True)
            if key in seen:
                fail(f"{path}: duplicate configuration across shards: {key}")
            seen.add(key)
    if len(seen) != args.total:
        fail(f"{args.stage}: {len(seen)} distinct rows, expected {args.total}")


def cmd_validate_merged(args):
    load_stage(args.path, args.stage, args.total)


def cmd_partition(args):
    """Arms dealt round-robin to the processes, so each gets the same legacy/graded mix."""
    import ablation_study

    labels = [spec.label for spec in ablation_study.ARMS]
    if len(set(labels)) != len(labels):
        fail("ablation arm labels are not unique")
    out = Path(args.dir)
    out.mkdir(parents=True, exist_ok=True)
    for k in range(args.procs):
        mine = labels[k::args.procs]
        (out / f"arms_p{k}.txt").write_text("\n".join(mine) + "\n")
        print(k, len(mine))


# gate_auc is undefined in folds where the held-out gate labels hold a single class
OPTIONAL_METRICS = {"gate_auc"}


def read_cells(path):
    """Rows of the long-format log keyed by (label, seed, split, fold, metric)."""
    keys = []
    with open(path) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                keys.append((row["label"], row["seed"], row.get("fold", 0), row["split"], row["metric"]))
    return keys


def check_cells(path, keys):
    """Fails on a repeated row or a (label, seed, fold) cell that lacks a split or a metric; returns the cells."""
    if len(keys) != len(set(keys)):
        fail(f"{path}: {len(keys) - len(set(keys))} duplicate rows")
    groups = {}
    for label, seed, fold, split, metric in keys:
        groups.setdefault((label, seed, fold, split), set()).add(metric)
    metrics = {}
    for (label, _, _, split), m in groups.items():
        metrics.setdefault((label, split), set()).update(m - OPTIONAL_METRICS)
    cells = {g[:3] for g in groups}
    short = [(c, sp) for c in cells for sp in ("test", "val") if groups.get(c + (sp,), set()) - OPTIONAL_METRICS != metrics.get((c[0], sp))]
    if short:
        fail(f"{path}: {len(short)} (cell, split) groups lack a split or a metric, e.g. {short[0]}")
    return cells


def cmd_validate_ablation(args):
    arms = [a for a in Path(args.arms).read_text().split("\n") if a]
    want = {(a, s, f) for a in arms for f in args.folds for s in args.seeds}
    try:
        keys = read_cells(args.path)
    except (OSError, ValueError, KeyError) as err:
        fail(f"{args.path}: {err}")
    cells = check_cells(args.path, keys)
    if cells != want:
        fail(f"{args.path}: {len(cells & want)} of {len(want)} expected cells, "
             f"{len(cells - want)} unexpected")


def cmd_validate_ablation_merged(args):
    try:
        keys = read_cells(args.path)
    except (OSError, ValueError, KeyError) as err:
        fail(f"{args.path}: {err}")
    cells = check_cells(args.path, keys)
    if len(cells) != args.cells:
        fail(f"{args.path}: {len(cells)} distinct cells, expected {args.cells}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add(name, fn, *positional, **optional):
        p = sub.add_parser(name)
        for arg in positional:
            if isinstance(arg, tuple):
                p.add_argument(arg[0], **arg[1])
            else:
                p.add_argument(arg)
        for flag, kw in optional.items():
            p.add_argument("--" + flag.replace("_", "-"), **kw)
        p.set_defaults(fn=fn)
        return p

    integer = {"type": int}
    add("fingerprint", cmd_fingerprint)
    add("rows", cmd_rows, "stage", ("source", {"nargs": "?"}))
    add("smallest-k-index", cmd_smallest_k_index)
    add("shards", cmd_shards, ("total", integer), ("max_rows", integer))
    add("plan", cmd_plan, "dir", "stage", ("total", integer), ("n", integer),
        ("max_rows", integer))
    add("validate", cmd_validate, "path", "stage", ("total", integer), ("n", integer),
        ("i", integer))
    add("missing", cmd_missing, "dir", "stage", ("total", integer), ("n", integer))
    add("checkset", cmd_checkset, "dir", "stage", ("total", integer), ("n", integer))
    add("validate-merged", cmd_validate_merged, "path", "stage", ("total", integer))
    add("partition", cmd_partition, "dir", ("procs", integer))
    p = add("validate-ablation", cmd_validate_ablation, "path", "arms")
    p.add_argument("--folds", type=int, nargs="+", required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    add("validate-ablation-merged", cmd_validate_ablation_merged, "path", ("cells", integer))

    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    args.fn(args)


if __name__ == "__main__":
    main()
