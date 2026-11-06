#!/usr/bin/env python3
"""AC-D1 to AC-D8 as measurements over data/database_graded.json (ADR-019).

Usage: PYTHONPATH=. python scripts/check_negatives.py [--data PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from src.ablation import AblationSpec, split_dataset
from src.modules import negatives
from src.modules.dataset import GrowthMediumDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data" / "database_graded.json"
SPEC = REPO_ROOT / "data" / "negatives_spec.yaml"


def _auc(scores, labels):
    """Rank AUC with ties averaged."""
    pairs = sorted(zip(scores, labels))
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[k] = (i + j) / 2.0 + 1.0
        i = j + 1
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if not n_pos or not n_neg:
        return None
    ranked = [ranks[i] for i, (_, label) in enumerate(pairs) if label]
    return (sum(ranked) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _modalities(record):
    return {c.get("component_source") for c in record["components"]}


def _pool(records):
    return {c["component"] for r in records if r.get("legacy")
            for c in r["components"]}


def check(records, data_path=DATA, spec_path=SPEC):
    spec = yaml.safe_load(open(spec_path))
    fold_of = {}
    splits = split_dataset(GrowthMediumDataset(json_file=data_path), AblationSpec())
    for name, subset in splits.items():
        for i in subset.indices:
            fold_of[i] = name

    pool = _pool(records)
    results = []

    train = [r for i, r in enumerate(records) if fold_of[i] == "train"]
    labels = [r["p"] < 0.5 for r in train]
    scores = [1.0 if pool & {c["component"] for c in r["components"]} else 0.0
              for r in train]
    auc = _auc(scores, labels)
    results.append(("AC-D1", f"pool-presence AUC on train = {auc:.3f} "
                             f"(n={len(train)}, positives={sum(labels)})", auc <= 0.75))

    hi = Counter()
    lo = Counter()
    for r in records:
        present = pool & {c["component"] for c in r["components"]}
        for comp in present:
            (hi if r["p"] >= 0.5 else lo)[comp] += 1
    worst_hi = min(hi.get(c, 0) for c in pool)
    worst_lo = min(lo.get(c, 0) for c in pool)
    results.append(("AC-D2", f"{len(pool)} pool compounds; minimum recipes with p>=0.5 is "
                             f"{worst_hi}, with p<0.5 is {worst_lo}",
                    worst_hi >= 1 and worst_lo >= 1))

    synthetic = [r for r in records if r.get("synthetic")]
    both = sum(1 for r in synthetic if {"pubchem", "uniprot"} <= _modalities(r))
    share = both / len(synthetic) if synthetic else 0.0
    results.append(("AC-D3", f"{both}/{len(synthetic)} synthetic recipes have both "
                             f"modalities ({100 * share:.1f}%)", share >= 0.90))

    exact = sum(1 for r in records if r["p"] in (0.0, 1.0))
    share_exact = exact / len(records)
    profiles = {(round(r["p"], 6), round(r["s"], 6), round(r["d"], 6))
                for r in records if r["p"] < 0.5}
    results.append(("AC-D4", f"{exact}/{len(records)} recipes at p exactly 0 or 1 "
                             f"({100 * share_exact:.1f}%), {len(profiles)} distinct "
                             f"(p,s,d) profiles among negatives",
                    share_exact <= 0.60 and len(profiles) >= 3))

    misplaced = [i for i, r in enumerate(records)
                 if r.get("parent") is not None and fold_of[i] != fold_of[r["parent"]]]
    folds_per_source = defaultdict(set)
    for i, r in enumerate(records):
        if r.get("source"):
            folds_per_source[r["source"]].add(fold_of[i])
    straddling = {k: v for k, v in folds_per_source.items() if len(v) > 1}
    results.append(("AC-D5", f"{len(misplaced)} children outside their parent's fold; "
                             f"{len(straddling)} of {len(folds_per_source)} source "
                             f"publications span folds",
                    not misplaced and not straddling))

    bad_rows = [k for k, v in spec["toxicants"].items()
                if not v.get("cite") or not v.get("class") or not v.get("ic50")]
    missing = sorted(pool - set(spec["toxicants"]))
    bad_rules = [k for k, v in spec["depletion"].items() if len(v.get("targets", [])) != 3]
    if len(spec["dilution"].get("targets", [])) != len(spec["dilution"].get("scales", [])):
        bad_rules.append("dilution")
    if len(spec["osmotic"].get("targets", [])) != len(spec["osmotic"].get("targets_mosm_kg", [])):
        bad_rules.append("osmotic")
    results.append(("AC-D6", f"{len(spec['toxicants'])} toxicant rows, {len(bad_rows)} "
                             f"without a citation or 'assumed'; {len(missing)} pool "
                             f"compounds absent from the spec; {len(bad_rules)} rules "
                             f"without a target vector",
                    not bad_rows and not missing and not bad_rules))

    parent_ids = {r["parent"] for r in synthetic if r.get("parent") is not None}
    contaminated = [r for r in synthetic if r.get("parent") is not None
                    and float(records[r["parent"]]["proliferation"]) < 0.5]
    per_generator = Counter(r.get("generator") for r in synthetic)
    breakdown = ", ".join(f"{name} {count}" for name, count in sorted(per_generator.items()))
    results.append(("AC-D7", f"{len(parent_ids)} positive parents, {len(synthetic)} "
                             f"children ({breakdown}); {len(contaminated)} children from "
                             f"a parent with p<0.5", not contaminated))

    classes = list(spec["depletion"])
    siblings = defaultdict(list)
    for r in synthetic:
        if r.get("parent") is not None:
            siblings[r["parent"]].append(r)
    # The same chemistry under two target triples: the dedupe keeps both, so this is the
    # intervention collision it deliberately exposes rather than resolves.
    collisions = 0
    for batch in siblings.values():
        labels = defaultdict(set)
        for child in batch:
            labels[negatives.component_signature(child["components"])].add(
                (child["p"], child["s"], child["d"]))
        collisions += sum(len(t) - 1 for t in labels.values())

    emitted, skipped = Counter(), Counter()
    removed_sizes = defaultdict(Counter)
    removed_sets = defaultdict(set)
    partial = misreported = 0
    for r in synthetic:
        if r.get("generator") != "depletion":
            continue
        name = negatives.depletion_name(r)
        emitted[name] += 1
        entities = negatives.class_entities(spec, name)
        parent_components = {c["component"] for c in records[r["parent"]]["components"]}
        child_components = {c["component"] for c in r["components"]}
        present = {entity for entity, ids in entities.items() if ids & parent_components}
        removed = set().union(*(entities[entity] for entity in present)) if present else set()
        if not present or child_components != parent_components - removed:
            partial += 1
            continue
        if set(r.get("depleted_entities", ())) != present:
            misreported += 1
        removed_sizes[name][len(present)] += 1
        removed_sets[name].add(frozenset(present))

    for index, parent in enumerate(records):
        if (parent.get("parent") is not None or fold_of[index] != "train"
                or float(parent["proliferation"]) < 0.5
                or not {"pubchem", "uniprot"} <= _modalities(parent)):
            continue
        present = {c["component"] for c in parent["components"]}
        for name in classes:
            entities = negatives.class_entities(spec, name)
            if not any(ids & present for ids in entities.values()):
                skipped[name] += 1

    rows = []
    for name in classes:
        sizes = ", ".join(f"{n} {count}"
                          for n, count in sorted(removed_sizes[name].items())) or "-"
        distinct = len(removed_sets[name])
        rows.append(f"{name} emitted {emitted[name]} skipped {skipped[name]}, entities "
                    f"removed per child {{{sizes}}}, {distinct} distinct set(s)"
                    + (" FLAG" if distinct > 3 else ""))
    results.append(("AC-D8", f"{collisions} extra labels on an identical component "
                             f"signature; {partial} children whose removal does not match "
                             f"their present entities; {misreported} with a recorded set "
                             f"that does not match; " + "; ".join(rows),
                    partial == 0 and misreported == 0))

    for name, text, ok in results:
        print(f"{name} {'PASS' if ok else 'FAIL'}  {text}")
    return all(ok for _, _, ok in results)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(DATA))
    args = p.parse_args()
    with open(args.data) as f:
        records = json.load(f)
    if not check(records, args.data):
        sys.exit(1)


if __name__ == "__main__":
    main()
