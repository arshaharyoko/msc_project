"""
match_coprinet.py

Matches the dataset's 181 components against CoPriNet's publicly released
Mcule catalog price data (https://github.com/oxpig/CoPriNet, MIT license,
real prices in $/g, no scraping/API key involved) by canonicalizing SMILES
with RDKit so different-notation SMILES for the same molecule still match.

Usage:
    python match_coprinet.py
"""

import csv
import json
from pathlib import Path
from rdkit import Chem

BASE = Path(__file__).resolve().parent.parent
DB_PATH = BASE / "database_preprocessed.json"
SCRATCH = Path(
    r"C:\Users\user\AppData\Local\Temp\claude\d--msc-modified\ae877b55-51cd-4dc0-8300-3799eabc4103\scratchpad"
)
MCULE_FILES = [SCRATCH / "test_set_PC.csv", SCRATCH / "npnp_dataset.csv"]
OUT_CSV = BASE / "coprinet_matches.csv"


def canon(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def main():
    db = json.loads(DB_PATH.read_text(encoding="utf-8"))

    # Collect unique pubchem (small-molecule) components
    components = {}
    for product in db:
        for c in product.get("components", []):
            if c.get("component_source") == "pubchem":
                components[c["component"]] = True

    print(f"Unique pubchem components: {len(components)}")

    canon_to_orig = {}
    failed = []
    for smi in components:
        c = canon(smi)
        if c is None:
            failed.append(smi)
        else:
            canon_to_orig[c] = smi
    print(f"Canonicalized OK: {len(canon_to_orig)}, failed to parse: {len(failed)}")

    # Build a lookup of canonical Mcule SMILES -> price ($/g)
    mcule_price = {}
    total_rows = 0
    for f in MCULE_FILES:
        if not f.exists():
            continue
        with open(f, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                total_rows += 1
                smi = row.get("SMILES")
                price = row.get("price")
                if not smi or not price:
                    continue
                c = canon(smi)
                if c is None:
                    continue
                # keep the lowest price if the same molecule appears more than once
                p = float(price)
                if c not in mcule_price or p < mcule_price[c]:
                    mcule_price[c] = p
    print(f"Mcule rows scanned: {total_rows}, unique canonical molecules: {len(mcule_price)}")

    # Match
    matches = []
    for c, orig_smi in canon_to_orig.items():
        if c in mcule_price:
            matches.append((orig_smi, c, mcule_price[c]))

    print(f"\nMATCHES FOUND: {len(matches)} / {len(components)}")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["component_smiles", "canonical_smiles", "mcule_price_usd_per_g"])
        for row in matches:
            w.writerow(row)

    print(f"Wrote {OUT_CSV}")
    for orig_smi, c, price in matches:
        print(f"  {orig_smi}  ->  ${price}/g")


if __name__ == "__main__":
    main()
