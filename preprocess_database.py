"""
Merge data/database.json and data/dataPoison.json into one labelled set with a
1024-D embedding per component: ProtT5 for UniProt accessions, the MolecularRGCN
for PubChem SMILES.

``cost`` is USD per litre: the unit price in data/price_table.csv times the
concentration in mg/L. A component quoted without a concentration is priced at
1 mg/L, the vectorizer's own fallback; one with no price stays None and is masked
out of the cost loss (ADR-001). A run also regenerates the component_prices
artifact and the ln(cost) statistics from that table (ADR-022).

    python preprocess_database.py
    python preprocess_database.py --merged data/database_with_cost.json \
        --out data/database_preprocessed_cost.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import h5py
import torch
import torch.nn.functional as F
from rdkit import Chem

from src import config
from src.cost_model import CostModel
from src.modules.featurization import collate, featurize_smiles
from src.modules.mol_encoder import MolecularRGCN

__all__ = ["main"]

DEFAULT_DATABASE = config.DATA_DIR / "database.json"
DEFAULT_DATAPOISON = config.DATA_DIR / "dataPoison.json"

UNIPROT_ACCESSION = re.compile(
    r"^([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})$"
)


def standardize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True) if mol else None


def infer_source(key):
    """dataPoison.json omits component_source; an accession never parses as SMILES."""
    return "uniprot" if UNIPROT_ACCESSION.match(key or "") else "pubchem"


def load_recipes(path, proliferation):
    with open(path) as f:
        products = json.load(f)

    for product in products:
        product["proliferation"] = proliferation
        for comp in product.get("components", []):
            comp.setdefault("component_source", infer_source(comp.get("component")))

    print(f"Loaded {len(products)} products from {path} (proliferation={proliferation})")
    return products


def load_merged(path):
    with open(path) as f:
        products = json.load(f)

    for product in products:
        product.setdefault("proliferation", 1)
        for comp in product.get("components", []):
            comp.setdefault("component_source", infer_source(comp.get("component")))

    alive = sum(product["proliferation"] for product in products)
    print(f"Loaded {len(products)} merged products from {path} "
          f"({alive} proliferation=1, {len(products) - alive} proliferation=0)")
    return products


def impute_costs(products, prices):
    """Fill missing costs from the component's unit price at the 1 mg/L fallback."""
    filled, missing = 0, 0
    for product in products:
        for comp in product.get("components", []):
            if comp.get("cost") is not None:
                comp["cost_imputed"] = False
                continue
            price = prices.get(comp["component"])
            if price is None:
                comp["cost"] = None
                missing += 1
            else:
                # Same fallback the vectorizer uses: log1p(concentration or 1)
                comp["cost"] = price * float(comp.get("concentration") or 1)
                filled += 1
            comp["cost_imputed"] = True

    print(f"Imputed {filled} missing costs from known unit prices, "
          f"{missing} left without a price")
    return filled, missing


def cost_stats(products):
    """Mean and std of ln(cost) over the known costs, the cost channel's standardization."""
    logs = [math.log(comp["cost"]) for product in products
            for comp in product.get("components", []) if comp.get("cost")]
    mean = sum(logs) / len(logs)
    std = math.sqrt(sum((v - mean) ** 2 for v in logs) / len(logs))
    # Key order matches the published artifact so a rerun is byte-identical
    return {"log_mean": mean, "log_std": std, "n": len(logs),
            "log_min": min(logs), "log_max": max(logs)}


def components(products, source):
    for product in products:
        for comp in product.get("components", []):
            if comp.get("component_source") == source:
                yield comp


def load_encoder(path, device):
    """Encoder restored at the geometry it was trained with. The property head is unused."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    trained = ckpt["args"]

    encoder = MolecularRGCN(
        d_h=trained["d_hidden"], d_out=trained["d_out"], num_layers=trained["num_layers"],
        dropout=trained["dropout"], normalize_output=False
    ).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()

    return encoder, trained["d_out"]


def molecule_vectors(smiles, encoder, reference, stored, device, batch_size):
    """1024-D embedding per SMILES, reused from the QM9 cloud where it is already in there."""
    vectors = {s: reference[stored[s]] for s in smiles if s in stored}
    pending = [s for s in smiles if s not in vectors]

    skipped = 0
    for start in range(0, len(pending), batch_size):
        chunk = [(s, featurize_smiles(s)) for s in pending[start:start + batch_size]]
        usable = [(s, feats) for s, feats in chunk if feats is not None]
        skipped += len(chunk) - len(usable)
        if not usable:
            continue

        packed = collate([feats for _, feats in usable])
        with torch.no_grad():
            z = encoder(*[t.to(device) for t in packed]).cpu()

        for (s, _), row in zip(usable, z):
            vectors[s] = row

    print(f"Reused {len(smiles) - len(pending)} QM9 embeddings, encoded "
          f"{len(pending) - skipped} molecules ({skipped} failed to featurize)")
    return vectors


def report_coverage(vectors, reference, device, block=8192):
    """QM9 stops at 9 heavy atoms, so most medium components are extrapolations."""
    query = F.normalize(torch.stack(list(vectors.values())).to(device), dim=-1)
    best = torch.full((query.shape[0],), -1.0)
    for start in range(0, reference.shape[0], block):
        neighbours = F.normalize(reference[start:start + block].to(device), dim=-1)
        best = torch.maximum(best, (query @ neighbours.T).max(dim=1).values.cpu())

    print(f"Nearest QM9 neighbour cosine: median {best.median():.3f}, "
          f"min {best.min():.3f}, below 0.9 for {int((best < 0.9).sum())}/{len(best)}")


def resolve(product, canonical, molecule_vecs, protein_vecs):
    vectors = []
    for comp in product.get("components", []):
        source, key = comp.get("component_source"), comp.get("component")

        if source == "pubchem":
            vec = molecule_vecs.get(canonical.get(key))
        elif source == "uniprot":
            vec = protein_vecs.get(key)
        else:
            vec = None

        if vec is None:
            return None, f"{source}:{str(key)[:24]}"
        vectors.append(vec)

    return vectors, None


def attach_targets(products):
    """Per-recipe (p, s, d) from the declared prior, with ``proliferation`` kept equal to p.

    Positives are (1, 1, 0). A negative is relabelled from the effect class of its pool
    compounds (data/negatives_spec.yaml, ADR-019) and keeps its scalar reader working
    through ``proliferation``.
    """
    from src.modules.negatives import POSITIVE, legacy_targets, load_spec

    spec = load_spec()
    for product in products:
        p, s, d = (POSITIVE if product.get("proliferation", 1)
                   else legacy_targets(product, spec))
        product["p"], product["s"], product["d"] = p, s, d
        product["proliferation"] = p


def run(args):
    device = torch.device(args.device)

    products = []
    if args.merged:
        products = load_merged(args.merged)
    else:
        for path, proliferation in ((args.database, 1), (args.datapoison, 0)):
            if Path(path).exists():
                products += load_recipes(path, proliferation)
            else:
                print(f"Skipping {path} (not found)")

    if not products:
        raise SystemExit("no recipes found")

    attach_targets(products)

    has_cost = any("cost" in comp for product in products
                   for comp in product.get("components", []))
    prices = CostModel.from_table(args.price_table).price_dict() if has_cost else {}
    if has_cost:
        impute_costs(products, prices)

    encoder, d_out = load_encoder(args.encoder, device)
    qm9 = torch.load(args.embeddings, map_location="cpu", weights_only=False)
    reference = qm9["embeddings"]
    stored = {s: i for i, s in enumerate(qm9["smiles"])}
    print(f"Embedding pubchem components into the {d_out}-D space of "
          f"{tuple(reference.shape)} QM9 molecules")

    canonical = {}
    for comp in components(products, "pubchem"):
        key = comp["component"]
        if key not in canonical:
            canonical[key] = standardize_smiles(key)

    unique_smiles = sorted({s for s in canonical.values() if s})
    molecule_vecs = molecule_vectors(unique_smiles, encoder, reference, stored, device, args.batch_size)
    report_coverage(molecule_vecs, reference, device)

    protein_vecs = {}
    with h5py.File(args.per_protein, "r") as h5:
        for comp in components(products, "uniprot"):
            key = comp["component"]
            if key not in protein_vecs and key in h5:
                protein_vecs[key] = h5[key][:]
    print(f"Matched {len(protein_vecs)} accessions in {args.per_protein}")

    kept = []
    for product in products:
        vectors, failed = resolve(product, canonical, molecule_vecs, protein_vecs)
        if vectors is None:
            print(f"Dropping '{product['product']}': unresolved {failed}")
            continue

        for comp, vec in zip(product["components"], vectors):
            comp["component_vec"] = vec.tolist()
        kept.append(product)

    alive = sum(1 for product in kept if product["proliferation"] >= 0.5)
    with open(args.out, "w") as f:
        json.dump(kept, f, indent=2)
    print(f"Wrote {len(kept)}/{len(products)} products to {args.out} "
          f"({alive} with p>=0.5, {len(kept) - alive} below)")

    torch.save({"y_mean": reference.mean(0), "y_std": reference.std(0).clamp(min=1e-6)},
               args.scaling_stats)
    print(f"Wrote {d_out}-D pubchem scaling stats to {args.scaling_stats}")

    if has_cost:
        stats = cost_stats(kept)
        torch.save(stats, args.cost_stats)
        print(f"Wrote ln(cost) stats mean={stats['log_mean']:.3f} std={stats['log_std']:.3f} "
              f"over {stats['n']} components to {args.cost_stats}")
        with open(args.prices, "w") as f:
            json.dump(prices, f, indent=2)
        print(f"Wrote {len(prices)} unit prices (USD/mg) to {args.prices}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--database", default=str(DEFAULT_DATABASE), help="raw recipes, proliferation=1")
    p.add_argument("--datapoison", default=str(DEFAULT_DATAPOISON), help="raw recipes, proliferation=0")
    p.add_argument("--merged", default=None,
                   help="one labelled file replacing --database/--datapoison, "
                        "e.g. data/database_with_cost.json")
    p.add_argument("--per-protein", default=str(config.PER_PROTEIN_H5))
    p.add_argument("--encoder", default=str(config.ARTIFACT_DIR / "qm9_gnn_encoder.pt"))
    p.add_argument("--embeddings", default=str(config.ARTIFACT_DIR / "qm9_gnn_embeddings.pt"),
                   help="QM9 embedding cloud: reference space and scaling stats")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=str(config.DATABASE_JSON))
    p.add_argument("--scaling-stats", default=str(config.SCALING_STATS))
    p.add_argument("--cost-stats", default=str(config.COST_STATS))
    p.add_argument("--price-table", default=str(config.PRICE_TABLE),
                   help="reviewed price table (ADR-022); component prices are never derived here")
    p.add_argument("--prices", default=str(config.COMPONENT_PRICES),
                   help="component -> USD/mg artifact, regenerated from --price-table")
    run(p.parse_args())


if __name__ == "__main__":
    main()
