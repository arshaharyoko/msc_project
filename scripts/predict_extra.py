"""predict.main() with an anchor strength and a protein-floor scale read from the environment (ADR-043).

ANCHOR_ALPHA  anchor strength: 1 is the anchored decode, 0 the raw decode. Unset leaves predict.py alone.
FLOOR_SCALE   multiple of each protein floor, capped at the Wu base dose; "base" is the base dose itself.
              Unset leaves predict.py alone.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import predict
from src import config


def with_anchor_strength(original, alpha):
    """x + x_hat - anchor' equals x_hat + alpha * (x - anchor) when anchor' = alpha * anchor + (1 - alpha) * x."""
    def anchor_decode(run, ctx):
        original(run, ctx)
        ctx["anchor"] = tuple(alpha * a + (1 - alpha) * x
                              for a, x in zip(ctx["anchor"], (ctx["x1"], ctx["x2"])))
    return anchor_decode


def base_doses(records, source):
    doses = {}
    for record in records:
        if record.get("source") != source:
            continue
        for c in record["components"]:
            if c["component_source"] == "uniprot" and c.get("concentration"):
                doses.setdefault(c["component"], c["concentration"])
    return doses


def with_floor_scale(original, scale, base):
    """Proteins the base recipe lacks keep their corpus floor: there is no base dose to cap a multiple."""
    factor = float("inf") if scale == "base" else float(scale)
    floors = {acc: min(factor * low, base[acc]) if acc in base else low
              for acc, low in original().items()}

    def protein_floors():
        return dict(floors)
    return protein_floors


def install(alpha, scale):
    if alpha is not None:
        predict.anchor_decode = with_anchor_strength(predict.anchor_decode, float(alpha))
    if scale is not None:
        with open(config.DATABASE_JSON) as f:
            base = base_doses(json.load(f), predict.BASE_SOURCE)
        predict.protein_floors = with_floor_scale(predict.protein_floors, scale, base)


if __name__ == "__main__":
    alpha = os.environ.get("ANCHOR_ALPHA") or None
    scale = os.environ.get("FLOOR_SCALE") or None
    install(alpha, scale)
    print(f"alpha={alpha} floor_scale={scale}")
    predict.main()
