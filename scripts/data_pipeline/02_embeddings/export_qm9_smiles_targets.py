"""
One-time migration: pull `smiles` + the 8 QM9 scalar targets out of the
already-cached Mordred-descriptor chunks and write a plain CSV.

This is the ONLY file in the GNN pipeline that imports legacy_pipeline's
qm9_dataset.py (and therefore mordred) -- and even here it's pure I/O against
a cache that already exists (data/qm9_descriptor_chunks/*.pkl), so no Mordred
descriptor is ever recomputed. Every thousand-column descriptor block gets
read into memory as part of that cache and then thrown away; only `smiles`
and the 8 targets below are written out. Nothing downstream of this script
(qm9_smiles_dataset.py, pretrain_mol_encoder.py, train_qm9_gnn.py) touches
legacy_pipeline/ or mordred.

Needs legacy_pipeline/ and data/qm9_descriptor_chunks/, neither of which is in this repository.
"""

from rdkit import Chem

from src import config
from legacy_pipeline.qm9_dataset import QM9

__all__ = ["main"]

TARGET_COLUMNS = ["mu", "alpha", "homo", "u0", "u", "h", "g", "cv"]


def main():
    print("Loading cached QM9 chunks (Mordred columns included, but discarded below)...")
    ds = QM9("unused", str(config.QM9_CHUNK_DIR))
    print(f"Loaded {ds.total_data} cached rows")

    df = ds._df[["smiles"] + TARGET_COLUMNS].copy()

    before = len(df)
    df = df.dropna(subset=["smiles"] + TARGET_COLUMNS)
    dropped_na = before - len(df)

    before = len(df)
    df = df.drop_duplicates(subset=["smiles"])
    dropped_dupe = before - len(df)

    valid_mask = df["smiles"].apply(lambda s: Chem.MolFromSmiles(s) is not None)
    dropped_invalid = int((~valid_mask).sum())
    df = df[valid_mask]

    out_path = config.DATA_DIR / "qm9_smiles_targets.csv"
    df.to_csv(out_path, index=False)

    print(f"Dropped {dropped_na} rows with missing values, "
          f"{dropped_dupe} duplicate SMILES, {dropped_invalid} unparseable SMILES")
    print(f"Wrote {len(df)} rows (smiles + {len(TARGET_COLUMNS)} targets, "
          f"no Mordred columns) to {out_path}")


if __name__ == "__main__":
    main()
