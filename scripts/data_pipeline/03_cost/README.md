# Component cost estimate

Set `MCULE_API_KEY` and `CHEMSPACE_API_KEY` in `.env` at the repository root (start from `.env.example`), then run:

```bash
python scripts/data_pipeline/03_cost/add_component_cost.py --input <preprocessed.json> --output <out.json>
```

The script reads the input recipes and writes the output without changing the source data. Every component gets a numeric `cost` property in **USD per liter of medium**: `concentration (mg/L) x cheapest price (USD/mg)` over the mass offers available on Mcule and Chemspace. The price per mg is a proportional estimate; the cost of buying a whole pack, shipping, tax and purity are not counted.

`--input` and `--output` are required so that a run never overwrites a tracked data file. `--cache` defaults to `.price_cache.json` next to `--output`, `--env` to the repository `.env`, and `--manual-cost` to `data/manual_component_cost.csv`.

`cost` is `null` when the concentration or unit is missing, when the component is a UniProt ID without a manual price, or when neither API has a USD mass offer. Prices are looked up once per unique SMILES and saved in the cache so that an interrupted run can resume. `--refresh` requests every price again.

`CHEMSPACE_SHIP_TO_COUNTRY` in `.env` is a two-letter country code (default `ID`, Indonesia). Change it if your shipping destination differs; the Chemspace cache is kept per country.

The script uses only the Python 3.10+ standard library. Both APIs limit the request rate, so fetching prices can take several minutes. If authentication or an API call fails, the script stops without writing partial output.

If the Mcule endpoint is opened directly in a browser and shows `Authentication credentials were not provided`, the browser did not send the `Authorization: Token ...` header. The script sends it from `.env`. Three structures in the dataset exceed the 500 character limit of the Mcule search; the script looks up their prices through Chemspace.

API references: [Mcule](https://doc.mcule.com/api), [Chemspace](https://chem-space.com/purchasing-saas/chemspace-api).

To export a CSV of every component whose `cost` is `null`:

```bash
python scripts/data_pipeline/03_cost/export_missing_cost.py --output components_without_cost.csv
```

The CSV has one row per occurrence of a component in a product. The `reason` column is `no_quote`, `unsupported` or `no_concentration`. `--input` defaults to `data/database_with_cost.json`.

After rerunning `add_component_cost.py`, run `export_missing_cost.py` again so the CSV follows the new result. The output of `add_component_cost.py` prints the total number of components, how many were priced, and how many were not, by reason.

## Manual prices for unique components

`prepare_manual_cost_csv.py --output <file>` creates a CSV of the unique components with reason `no_quote` or `unsupported` in `data/database_with_cost.json`. It refuses to overwrite an existing file, so prices you entered are not lost. The first CSV had 24 rows: 3 `no_quote` and 21 `unsupported`.

Enter the price in the **`cost` column in USD/mg**. The `cost_unit` column already says `USD/mg` and must stay that way. For example, a pack of USD 100 for 10 mg means `cost` is `10`; at 5 mg/L the JSON then gets 50 USD/L. Leave `cost` empty if there is no price yet. `price_reference` and `notes` record the source and the kind of product used for comparison. For a UniProt ID, make sure the manual price refers to the matching protein or formulation, not just a similar gene name.

Run `add_component_cost.py` again after filling the CSV. It reads `data/manual_component_cost.csv` automatically (or the file given with `--manual-cost`), uses the manual price for an exact match on `component_source` and `component`, and multiplies the USD/mg price by the concentration in mg/L. A filled manual price takes priority for that component; all other components still use Mcule and Chemspace. A row without a concentration still gives `cost: null`.

## legacy_manual_js/

The earlier Node.js route with hand-entered prices (September 2026). The scripts expect to sit in `<project>/data/scripts/` because their paths are relative (`..`). Order: `enrich_pubchem.js`, `regen_csv.js`, fill in prices in the CSV, `join_prices.js`. `match_coprinet.py` matches components against the Mcule catalog prices released with CoPriNet. `fill_llm_estimates.js` fills the CSV with estimated prices, not vendor prices, and none of its output is used for a number in the paper.
