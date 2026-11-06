"""Build data/database_graded.json: legacy negatives relabelled plus synthetic children.

Every target assigned here comes from data/negatives_spec.yaml and is a declared prior
(ADR-019); none of it is a measurement. Children are generated from the training positives
only, so a synthetic twin never appears in a fold its parent is not in.

Run from the repo root:
    PYTHONPATH=. python -m src.modules.negatives [--out data/database_graded.json]
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import yaml

from src.modules import priors

ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = ROOT / "data" / "negatives_spec.yaml"
BASE_PATH = ROOT / "data" / "database_preprocessed_cost.json"
OUT_PATH = ROOT / "data" / "database_graded.json"

POSITIVE = (1.0, 1.0, 0.0)
NACL = "[Na+].[Cl-]"
SPIKE_DOSES_PER_PARENT = 4
DEPLETIONS_PER_PARENT = 4

# Molar mass (g/mol) and dissolved particles per formula unit. Only the osmotic child uses
# this table; anything absent from it is left out of the estimate rather than guessed.
OSMOLYTES = {
    NACL: (58.44, 2),
    "[Cl-].[K+]": (74.55, 2),
    "C(=O)(O)[O-].[Na+]": (84.01, 2),
    "OP(=O)(O)[O-].[Na+]": (119.98, 2),
    "OP(=O)([O-])[O-].[Na+].[Na+]": (141.96, 3),
    "[Cl-].[Cl-].[Ca+2]": (110.98, 3),
    "[O-]S(=O)(=O)[O-].[Mg+2]": (120.37, 2),
    "CC(=O)C(=O)[O-].[Na+]": (110.04, 2),
    "C([C@@H]1[C@H]([C@@H]([C@H](C(O1)O)O)O)O)O": (180.16, 1),
    "C(CC(=O)N)[C@@H](C(=O)O)N": (146.15, 1),
}


def load_spec(path=SPEC_PATH):
    with open(path) as f:
        return yaml.safe_load(f)


def hill_fraction(dose_over_ic50, hill):
    """Occupancy of a single-site Hill curve at ``dose_over_ic50`` times the IC50/EC50."""
    r = float(dose_over_ic50) ** float(hill)
    return r / (1.0 + r)


def effect_targets(class_entry, fraction, base=POSITIVE):
    """Parent targets moved along the class effect by the Hill fraction."""
    dp, ds, dd = class_entry["effect"]
    return tuple(min(1.0, max(0.0, base[i] + fraction * delta))
                 for i, delta in enumerate((dp, ds, dd)))


def toxicant_targets(toxicant, dose_multiple, spec):
    fraction = hill_fraction(dose_multiple, toxicant["hill"])
    return effect_targets(spec["classes"][toxicant["class"]], fraction)


def legacy_targets(record, spec):
    """Effect at the IC50 of the class that contributes most components.

    The legacy recipes quote incidental concentrations rather than a titrated dose, so the
    class reference effect is used instead of the Hill curve at those concentrations.
    """
    counts = {}
    for comp in record.get("components", []):
        toxicant = spec["toxicants"].get(comp["component"])
        if toxicant:
            counts[toxicant["class"]] = counts.get(toxicant["class"], 0) + 1
    if not counts:
        return POSITIVE
    order = {name: i for i, name in enumerate(spec["classes"])}
    dominant = max(counts, key=lambda k: (counts[k], -order[k]))
    return effect_targets(spec["classes"][dominant], spec["legacy_fraction"])


def record_targets(record, spec):
    return POSITIVE if record.get("proliferation", 1) else legacy_targets(record, spec)


def _unit_cost(component):
    cost, conc = component.get("cost"), component.get("concentration")
    if cost is None or not conc:
        return None
    return cost / conc


def _with_concentration(component, concentration):
    c = copy.deepcopy(component)
    unit = _unit_cost(component)
    c["concentration"] = concentration
    c["cost"] = unit * concentration if unit is not None else None
    return c


def _scaled(components, k):
    out = []
    for comp in components:
        conc = comp.get("concentration")
        c = copy.deepcopy(comp)
        if isinstance(conc, (int, float)):
            unit = _unit_cost(comp)
            c["concentration"] = conc * k
            c["cost"] = unit * conc * k if unit is not None else None
        out.append(c)
    return out


def _both_modalities(record):
    sources = {c.get("component_source") for c in record.get("components", [])}
    return {"pubchem", "uniprot"} <= sources


def _child(parent, components, targets, generator, tag):
    p, s, d = targets
    return {
        "product": f"{parent['product']} [{tag}]",
        "flag": parent.get("flag"),
        "source": parent.get("source"),
        "p": p, "s": s, "d": d,
        "proliferation": p,
        "components": components,
        "synthetic": True,
        "generator": generator,
        "legacy": False,
    }


def class_entities(spec, name):
    """Entity name -> the alternative component encodings that stand for it."""
    return {entity: frozenset(ids)
            for entity, ids in spec["depletion"][name]["entities"].items()}


def _dose(component):
    conc = component.get("concentration")
    return repr(float(conc) if isinstance(conc, (int, float)) else conc)


def component_signature(components):
    """Identity of a recipe for deduplication: component ids with the dose each carries."""
    return tuple(sorted((c["component"], _dose(c)) for c in components))


def depletion_name(record):
    """The depletion class a child was built for, read off its tag."""
    product = record["product"]
    return product[product.rindex("[deplete ") + len("[deplete "):-1]


def osmolarity_mosm(components):
    """mOsm per litre from mg/L: mg/L / (mg/mmol) gives mmol/L, times particles."""
    total = 0.0
    for comp in components:
        entry = OSMOLYTES.get(comp.get("component"))
        conc = comp.get("concentration")
        if entry and isinstance(conc, (int, float)):
            total += conc / entry[0] * entry[1]
    return total


def _spike_children(parent, toxicant_id, toxicant, spec, templates, drop):
    doses = [d for i, d in enumerate(spec["doses"]) if i != drop]
    children = []
    for multiple in doses[:SPIKE_DOSES_PER_PARENT]:
        components = copy.deepcopy(parent["components"])
        components.append(_with_concentration(templates[toxicant_id],
                                              toxicant["ic50"] * multiple))
        tag = f"spike {toxicant['name']} {multiple:g}x"
        children.append(_child(parent, components,
                              toxicant_targets(toxicant, multiple, spec), "spike", tag))
    return children


def _depletion_children(parent, spec, rank, skipped):
    parent_ids = {c["component"] for c in parent["components"]}
    present = []
    for name in spec["depletion"]:
        entities = class_entities(spec, name)
        if any(ids & parent_ids for ids in entities.values()):
            present.append(name)
        else:
            skipped[name] += 1
    children = []
    for i in range(min(DEPLETIONS_PER_PARENT, len(present))):
        name = present[(rank + i) % len(present)]
        entities = class_entities(spec, name)
        removed = sorted(entity for entity, ids in entities.items() if ids & parent_ids)
        ids = set().union(*(entities[entity] for entity in removed))
        components = [c for c in parent["components"] if c["component"] not in ids]
        targets = spec["depletion"][name]["targets"]
        child = _child(parent, components, targets, "depletion", f"deplete {name}")
        child["depleted_entities"] = removed
        children.append(child)
    return children


def _dilution_children(parent, spec):
    ladder = priors.dilution_ladder({"components": parent["components"]})
    children = []
    for record in ladder:
        if record["scale"] >= 1.0:
            continue
        i = spec["dilution"]["scales"].index(record["scale"])
        components = _scaled(parent["components"], record["scale"])
        children.append(_child(parent, components, spec["dilution"]["targets"][i],
                               "dilution", f"dilute {record['scale']:g}x"))
    return children


def _osmotic_children(parent, spec):
    nacl = next((c for c in parent["components"] if c["component"] == NACL), None)
    if nacl is None or not isinstance(nacl.get("concentration"), (int, float)):
        return []
    others = osmolarity_mosm([c for c in parent["components"] if c is not nacl])
    children = []
    for value, targets in zip(spec["osmotic"]["targets_mosm_kg"],
                              spec["osmotic"]["targets"]):
        concentration = max(0.0, (value - others) / 2.0 * 58.44)
        components = [_with_concentration(nacl, concentration) if c is nacl else copy.deepcopy(c)
                      for c in parent["components"]]
        children.append(_child(parent, components, targets, "osmotic", f"osmotic {value}"))
    return children


def generate(base, spec, train_indices, stats=None):
    templates = {}
    for record in base:
        for comp in record.get("components", []):
            templates.setdefault(comp["component"], comp)

    if stats is None:
        stats = {}
    duplicates = stats.setdefault("duplicates", Counter())
    emitted = stats.setdefault("depletion_emitted", Counter())
    skipped = stats.setdefault("depletion_skipped", Counter())

    toxicants = list(spec["toxicants"])
    # A child's targets are a prior anchored on a positive parent, so a negative parent
    # would carry a label its targets contradict.
    eligible = [i for i in train_indices
                if base[i].get("parent") is None
                and float(base[i]["proliferation"]) >= 0.5
                and _both_modalities(base[i])]
    children = []
    for rank, index in enumerate(eligible):
        parent = base[index]
        toxicant_id = toxicants[rank % len(toxicants)]
        dropped_dose = rank % len(spec["doses"])
        batch = _spike_children(parent, toxicant_id, spec["toxicants"][toxicant_id],
                                spec, templates, dropped_dose)
        batch += _depletion_children(parent, spec, rank, skipped)
        batch += _dilution_children(parent, spec)
        batch += _osmotic_children(parent, spec)
        for child in batch:
            child["parent"] = index
        # Two generators can remove the same set of components from one parent, leaving
        # children that differ only in their targets; the first label stands. Keying on the
        # target triple as well keeps a depletion collision visible instead of dropping it:
        # the same chemistry under two labels is reported by AC-D8, not hidden here.
        seen = set()
        for child in batch:
            signature = (component_signature(child["components"]),
                         (child["p"], child["s"], child["d"]))
            if signature in seen:
                duplicates[child["generator"]] += 1
                continue
            seen.add(signature)
            children.append(child)
            if child["generator"] == "depletion":
                emitted[depletion_name(child)] += 1
    return children


def build(out=OUT_PATH, base_path=BASE_PATH, spec_path=SPEC_PATH):
    from src.ablation import AblationSpec, parent_folds

    with open(base_path) as f:
        base = json.load(f)
    spec = load_spec(spec_path)

    for i, record in enumerate(base):
        legacy = record.get("proliferation", 1) == 0
        p, s, d = record_targets(record, spec)
        record["p"], record["s"], record["d"] = p, s, d
        record["proliferation"] = p
        record["legacy"] = legacy
        record["parent"] = None
        record["synthetic"] = False

    folds = parent_folds(base, AblationSpec())
    stats = {}
    children = generate(base, spec, folds["train"], stats)
    records = base + children

    with open(out, "w") as f:
        # Compact separators: the embedding vectors dominate the file and indentation
        # triples it without adding information.
        json.dump(records, f, separators=(",", ":"))

    counts = {name: len(idx) for name, idx in folds.items()}
    print(f"Wrote {len(records)} records ({len(base)} real, {len(children)} synthetic) "
          f"to {out}")
    print(f"Training parents: {counts['train']} of {len(base)} "
          f"(val {counts['val']}, test {counts['test']}); parent folds are grouped by "
          f"source publication")
    dropped = stats["duplicates"]
    detail = ", ".join(f"{name} {count}" for name, count in sorted(dropped.items()))
    print(f"Duplicate children dropped within their parent batch: {sum(dropped.values())}"
          + (f" ({detail})" if detail else ""))
    emitted, skipped = stats["depletion_emitted"], stats["depletion_skipped"]
    print("Depletion emitted/skipped per class: " + ", ".join(
        f"{name} {emitted[name]}/{skipped[name]}" for name in spec["depletion"]))
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(OUT_PATH))
    p.add_argument("--base", default=str(BASE_PATH))
    p.add_argument("--spec", default=str(SPEC_PATH))
    args = p.parse_args()
    build(Path(args.out), Path(args.base), Path(args.spec))


if __name__ == "__main__":
    main()
