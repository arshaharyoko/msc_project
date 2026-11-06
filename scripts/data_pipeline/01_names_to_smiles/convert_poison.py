"""
convert_poison.py
-----------------
Reads a compound JSON file and outputs a new JSON with:
  - Component names replaced by PubChem canonical SMILES (small molecules)
    or UniProt accession IDs (proteins).
  - All concentrations converted to mg/L.

Supports BOTH input formats:
  - Single record:  { "flag": ..., "product": ..., "components": [...] }
  - Array of records: [ { ... }, { ... }, ... ]

Unit conversion rules (all → mg/L):
  nM       : nmol/L  → mg/L = value × MW(g/mol) × 1e-6
  µM / uM  : µmol/L  → mg/L = value × MW(g/mol) × 1e-3
  ng/mL    : 1 ng/mL = 0.001 mg/L  (× 1e-3)
  µg/mL    : 1 µg/mL = 1.000 mg/L  (× 1)

Usage:
  python convert_poison.py input.json output.json
"""

import json
import sys
import time
import urllib.request
import urllib.parse
import urllib.error

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RETRY_COUNT   = 4
RETRY_BACKOFF = [1, 2, 4, 8]
CALL_DELAY    = 0.5       # seconds between every API call

PROTEIN_NAMES: set[str] = set()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_protein(name: str) -> bool:
    suffixes = ("_HUMAN", "_MOUSE", "_RAT", "_BOVIN", "_YEAST", "_ECOLI")
    return name.upper().endswith(suffixes) or name in PROTEIN_NAMES


def http_get(url: str) -> dict | None:
    """GET a JSON URL with retry + exponential backoff."""
    for attempt in range(RETRY_COUNT):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=15) as r:
                time.sleep(CALL_DELAY)
                return json.loads(r.read().decode())

        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
            print(f"  [HTTP {e.code}] retry {attempt+1}/{RETRY_COUNT} in {wait}s…", file=sys.stderr)
            time.sleep(wait)

        except Exception as e:
            wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
            print(f"  [Error] {e} — retry {attempt+1}/{RETRY_COUNT} in {wait}s…", file=sys.stderr)
            time.sleep(wait)

    print(f"  [Error] all retries failed: {url}", file=sys.stderr)
    return None


# ---------------------------------------------------------------------------
# PubChem — NUCLEAR OPTION (Parse raw JSON record)
# ---------------------------------------------------------------------------

PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


def pubchem_cid(name: str) -> int | None:
    encoded = urllib.parse.quote(name)
    data = http_get(f"{PUBCHEM}/compound/name/{encoded}/cids/JSON")
    if not data:
        return None
    cids = data.get("IdentifierList", {}).get("CID", [])
    return cids[0] if cids else None


def resolve_small_molecule(name: str) -> tuple[str | None, float | None]:
    print(f"  [PubChem] Looking up '{name}' …")

    cid = pubchem_cid(name)
    if cid is None:
        print(f"  [PubChem] CID not found for '{name}'", file=sys.stderr)
        return None, None
    print(f"  [PubChem] CID={cid}")

    url = f"{PUBCHEM}/compound/cid/{cid}/JSON"
    data = http_get(url)
    
    if not data:
        return None, None

    smiles = None
    mw = None

    isomeric_smiles = None
    canonical_smiles = None
    fallback_smiles = None

    try:
        # FIXED: PubChem PUG REST puts the data in "PC_Compounds", not "Record"
        compounds = data.get("PC_Compounds", [])
        if not compounds:
            return None, None
            
        props = compounds[0].get("props", [])
        
        for prop in props:
            urn = prop.get("urn", {})
            val = prop.get("value", {})
            label = urn.get("label", "")
            name_val = urn.get("name", "")

            # Extract Molecular Weight
            if label == "Molecular Weight":
                mw_val = val.get("fval") or val.get("sval")
                if mw_val is not None:
                    mw = float(mw_val)

            # Extract SMILES
            if label == "SMILES":
                s_val = val.get("sval")
                if s_val:
                    if name_val == "Isomeric":
                        isomeric_smiles = s_val
                    elif name_val == "Canonical":
                        canonical_smiles = s_val
                    else:
                        fallback_smiles = s_val

        # Prefer Isomeric > Canonical > Unnamed
        smiles = isomeric_smiles or canonical_smiles or fallback_smiles

        print(f"  [PubChem] MW={mw}  SMILES={smiles[:60] if smiles else 'NOT FOUND'}")
        return smiles, mw

    except Exception as e:
        print(f"  [Error] parsing full JSON for '{name}': {e}", file=sys.stderr)
        return None, None


# ---------------------------------------------------------------------------
# UniProt
# ---------------------------------------------------------------------------

UNIPROT = "https://rest.uniprot.org/uniprotkb"


def resolve_protein(entry_name: str) -> str | None:
    print(f"  [UniProt] Looking up '{entry_name}' …")
    
    # UniProt's newer API syntax requires 'id:' for direct entry name querying
    params = urllib.parse.urlencode({
        "query":  f"id:{entry_name}",
        "fields": "accession",
        "format": "json",
        "size":   1,
    })
    
    data = http_get(f"{UNIPROT}/search?{params}")
    if not data:
        return None
        
    results = data.get("results", [])
    if not results:
        print(f"  [UniProt] No result for '{entry_name}'", file=sys.stderr)
        return None
        
    acc = results[0]["primaryAccession"]
    print(f"  [UniProt] Accession={acc}")
    return acc


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------

def to_mg_per_L(value: float, unit: str, mw: float | None) -> float:
    u = unit.strip().replace("μ", "µ")

    if u in ("ng/mL", "ng/ml"):   return value * 1e-3
    if u in ("µg/mL", "µg/ml",
             "ug/mL", "ug/ml"):   return value * 1.0
    if u == "nM":
        if mw is None: raise ValueError("MW required for nM → mg/L")
        return value * mw * 1e-6
    if u in ("µM", "uM"):
        if mw is None: raise ValueError("MW required for µM → mg/L")
        return value * mw * 1e-3

    raise ValueError(f"Unknown unit: '{unit}'")


# ---------------------------------------------------------------------------
# Record processing
# ---------------------------------------------------------------------------

def process_record(record: dict) -> dict:
    new_components = []

    for entry in record.get("components", []):
        name  = entry["component"]
        value = float(entry["concentration"])
        unit  = entry["unit"]

        print(f"\nProcessing: {name} ({value} {unit})")

        if is_protein(name):
            identifier, mw = resolve_protein(name), None
        else:
            identifier, mw = resolve_small_molecule(name)

        if identifier is None:
            print(f"  WARNING: keeping original name for '{name}'")
            identifier = name

        try:
            mg_L = to_mg_per_L(value, unit, mw)
        except ValueError as e:
            print(f"  ERROR unit conversion for '{name}': {e}", file=sys.stderr)
            mg_L = None

        new_components.append({
            "component":            identifier,
            "concentration":        round(mg_L, 8) if mg_L is not None else None,
            "unit":                 "mg/L",
        })

    return {
        "flag":       record.get("flag"),
        "product":    record.get("product"),
        "components": new_components,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def process(input_path: str, output_path: str) -> None:
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        print(f"Detected array input — processing {len(data)} record(s).\n")
        output = [process_record(r) for r in data]
    else:
        print("Detected single-record input.\n")
        output = process_record(data)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Written to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_poison.py <input.json> <output.json>")
        sys.exit(1)
    process(sys.argv[1], sys.argv[2])