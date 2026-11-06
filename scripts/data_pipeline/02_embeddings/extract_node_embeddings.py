"""
Supposed to be run once to pull 21 pathway-node ProtT5 embeddings out of per-protein.h5.

Writes the JSON given by --out.
"""

import argparse
import json
from pathlib import Path

import h5py

from src import config
from src.modules.graph import NODE_ACCESSIONS


def main():
    parser = argparse.ArgumentParser(description="Extract the pathway-node ProtT5 embeddings from per-protein.h5.")
    parser.add_argument("--out", type=Path, required=True, help="output JSON; data/graph_nodes_embedded.json is the tracked copy")
    args = parser.parse_args()
    final_products = {}

    with h5py.File(config.PER_PROTEIN_H5, "r") as h5:
        for key in NODE_ACCESSIONS:
            final_products[key] = None

            if key in h5:
                final_products[key] = h5[key][:].tolist()
                print(f"Matched: {key[:10]}...")

            else:
                print(f"Missing: '{key[:20]}...' (Length: {len(key)})")
                break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(final_products, f, indent=2)

    print(f"Final entries: {len(final_products)}")


if __name__ == "__main__":
    main()
