# Data pipeline

These scripts build the recipe dataset from raw recipes up to `data/database_with_cost.json` and `data/database_preprocessed_cost.json`. They contain code only; the data files live in `data/`.

## Order

```
raw recipes (component names)
   |  01  names to SMILES or UniProt accessions, units to mg/L
   v
database.json (proliferation=1) and dataPoison.json (proliferation=0)
   |  02  merge, label, attach embedding vectors (root preprocess_database.py)
   v
database_preprocessed.json
   |  03  add a price per component (USD per liter of medium)
   v
data/database_with_cost.json
   |  root preprocess_database.py --merged, then python -m src.modules.negatives
   v
data/database_preprocessed_cost.json, data/database_graded.json
```

Run everything from the repository root with `PYTHONPATH=.`.

## 01_names_to_smiles/

| Script | Produces | Notes |
|---|---|---|
| `rename_components_pubchem_uniprot.py` | `database.json` (115 recipes) | Looks each name up in PubChem and replaces it with a SMILES. A name that fails and looks like a protein is searched in UniProt. Adds `component_source` (`pubchem` or `uniprot`). Rows with a unit other than mg/L and rows that are not found are dropped. Example: `python rename_components_pubchem_uniprot.py -i copied.json -o out.json --sleep 0.8`. Its docstring still uses the old file name `..._smilesonly.py`. |
| `convert_poison.py` | `dataPoison.json` (50 recipes) | Same for the poison data: SMILES from PubChem, accessions from UniProt, and concentrations in nM, uM, ng/mL and ug/mL converted to mg/L with the molecular weight. Usage: `python convert_poison.py input.json output.json`. It does not write `component_source`; the merge step fills it. |

Both scripts call PubChem and UniProt, so they need network access. Their raw inputs (`copied.json`, `input.json`) and the earlier step, an export from MongoDB through the Django application, are not in this repository. Nothing in this stage was run here.

## 02_embeddings/

The merge and embedding step is the root `preprocess_database.py`. It merges `database.json` and `dataPoison.json`, fills `component_source` for the poison data, and attaches `component_vec` to every component: ProtT5 (1024-D) from `data/per-protein.h5` for proteins, `MolecularRGCN` (1024-D) for compounds. It also writes the scaling and cost statistics and `artifacts/component_prices.json`. `--merged data/database_with_cost.json` starts from a file that already holds the labelled recipes.

| Script | Purpose |
|---|---|
| `export_qm9_smiles_targets.py` | One-off export of the SMILES and the 8 QM9 targets to `data/qm9_smiles_targets.csv`, used to train the molecular encoder. It imports `legacy_pipeline`, which is not in this repository, so it cannot run here. It has no argparse. |
| `extract_node_embeddings.py` | One-off extraction of the 21 pathway-node embeddings from `data/per-protein.h5`. `--out` is required, so a run never overwrites `data/graph_nodes_embedded.json`. |

The 8-D `database_preprocessed.json` of the earlier project (compound vectors of 8 dimensions from a Mordred and FFN pipeline) is not in this repository, and the script that made it was not found. The current `preprocess_database.py` produces 1024-D vectors.

## 03_cost/

`add_component_cost.py` is the current price step. It queries Mcule and Chemspace and needs two API keys; see `03_cost/README.md`. `legacy_manual_js/` holds the earlier Node.js route with hand-entered prices.

Path defaults: `DATA` is the repository `data/` directory, `REPO` the repository root.

- `add_component_cost.py`: `--input` and `--output` are required, so a run cannot overwrite a tracked file by default. `--env` defaults to the repository `.env`, `--manual-cost` to `data/manual_component_cost.csv`, and `--cache` to `.price_cache.json` next to `--output`.
- `export_missing_cost.py` and `prepare_manual_cost_csv.py`: `--input` defaults to `data/database_with_cost.json`; `--output` is required.

Copy `.env.example` from the repository root to `.env` and fill in the keys. `.env` is not tracked.

`test_add_component_cost.py` runs offline with fake clients: `python scripts/data_pipeline/03_cost/test_add_component_cost.py`, or pytest on the same path.

## What reproduces here

Offline, `bash scripts/reproduce.sh data` rebuilds the embeddings, statistics and prices from `data/database_with_cost.json` into a temp directory and compares them with the stored files. `python -m src.modules.negatives` on the stored `data/database_preprocessed_cost.json` gives a byte identical `data/database_graded.json`. The rebuilt preprocessed JSON differs from the stored one (see the Dataset section of the main README), so `data/database_preprocessed_cost.json` stays an input.

Not reproduced here: stage 01, the live Mcule and Chemspace price calls, the MongoDB export, and the 8-D `database_preprocessed.json`. The raw recipes and the API prices are inputs. `data/price_table.csv` was recovered from the costs in `data/database_with_cost.json`, not from a vendor source, apart from the rows that cite a URL.
