#!/usr/bin/env python3
"""Write the float32 sidecar GrowthMediumDataset loads in place of the JSON.

The JSON stores every 1024-float component vector as a list of text floats, which costs
~22 s and several GB of RSS to parse per process. The sidecar holds the same records with
``component_vec`` as float32 tensors, so a process maps or reads binary instead.

Usage:
    PYTHONPATH=. python scripts/compact_dataset.py data/database_graded.json [--out PATH]
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch


def sidecar_path(json_path):
    return Path(json_path).with_suffix(".pt")


def vector_digest(records):
    digest = hashlib.sha256()
    n = 0
    for record in records:
        for item in record["components"]:
            vec = item["component_vec"]
            digest.update(vec.detach().contiguous().numpy().tobytes())
            n += 1
    return digest.hexdigest(), n


def to_sidecar(json_path, out_path):
    started = time.perf_counter()
    with open(json_path) as f:
        records = json.load(f)
    parsed = time.perf_counter()

    for record in records:
        for item in record["components"]:
            item["component_vec"] = torch.tensor(item["component_vec"], dtype=torch.float32)
    converted = time.perf_counter()

    torch.save(records, out_path)
    written = time.perf_counter()

    digest, n_vectors = vector_digest(records)
    print(f"input   {json_path}  {Path(json_path).stat().st_size / 1e6:.1f} MB")
    print(f"output  {out_path}  {Path(out_path).stat().st_size / 1e6:.1f} MB")
    print(f"records {len(records)}  vectors {n_vectors}")
    print(f"sha256(vectors) {digest}")
    print(f"build   parse {parsed - started:.1f}s  convert {converted - parsed:.1f}s  "
          f"write {written - converted:.1f}s  total {written - started:.1f}s")
    return records, digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    out_path = Path(args.out) if args.out else sidecar_path(args.json_path)
    to_sidecar(args.json_path, out_path)


if __name__ == "__main__":
    main()
