#!/usr/bin/env python3
"""
rename_components_pubchem_uniprot_smilesonly.py

For your dataset format:
[
  {
    "product": "...",
    "flag": "...",
    "source": "...",
    "components": [
      {"component":"Glycine","concentration":10,"unit":"mg/L"},
      ...
    ]
  },
  ...
]

Rules:
- PubChem: replace component name -> SMILES ONLY (strictly "SMILES" field).
- UniProt: if PubChem fails AND name looks protein-ish -> replace component name -> amino-acid sequence.
- Adds component_source: "pubchem" | "uniprot"
- Drops component rows with non-mg/L units.
- Drops rows that cannot be resolved by PubChem or UniProt (when allowed).

Run:
  python rename_components_pubchem_uniprot_smilesonly.py -i copied.json -o out.json --sleep 0.8
  python rename_components_pubchem_uniprot_smilesonly.py -i copied.json -o out.json --debug
  python rename_components_pubchem_uniprot_smilesonly.py --test-name Glycine --debug --no-uniprot
"""

import argparse
import json
import re
import time
import requests
from typing import Any, Dict, Optional, Tuple, List
from urllib.parse import quote

# ---------------------- Endpoints ----------------------

PUBCHEM_DOMAIN = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

# STRICT: ask PubChem for SMILES only
PUBCHEM_NAME_PROP_URL = f"{PUBCHEM_DOMAIN}/compound/name/{{name}}/property/SMILES/JSON"
PUBCHEM_NAME_CID_URL  = f"{PUBCHEM_DOMAIN}/compound/name/{{name}}/cids/JSON"
PUBCHEM_CID_PROP_URL  = f"{PUBCHEM_DOMAIN}/compound/cid/{{cid}}/property/SMILES/JSON"

UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"

# ---------------------- Helpers ----------------------

def clean_name(name: str) -> str:
    if not name:
        return ""
    n = name.replace("\xa0", " ").strip()
    n = n.strip("\"'")
    n = re.sub(r"\s+", " ", n).strip()
    return n

_STEREO_PREFIX_RE = re.compile(r"^(?:[dDlL]{1,2}-|\(+\)-|\(-\)-|\(\+\)-)\s*", re.IGNORECASE)

def strip_stereo_prefix(name: str) -> str:
    return _STEREO_PREFIX_RE.sub("", name).strip()

def strip_hydrate_salt_words(name: str) -> str:
    n = name
    n = re.sub(r"\s*\[[^\]]*\]\s*", " ", n)
    n = re.sub(r"\s*\([^)]*\)\s*", " ", n)
    n = re.sub(
        r"\b(anhydrous|monohydrate|dihydrate|trihydrate|tetrahydrate|pentahydrate|hydrate|solution)\b",
        " ",
        n,
        flags=re.IGNORECASE,
    )
    n = re.sub(r"\b(salt|free\s*acid|free\s*base)\b", " ", n, flags=re.IGNORECASE)
    n = re.sub(r"\s+", " ", n).strip()
    return n

def get_name_variants(name: str) -> List[str]:
    """
    Variants to improve PubChem name hits without needing manual CID mapping.
    Examples:
      L-Alanine -> Alanine
      Dimethylsulfoxide -> Dimethyl sulfoxide, DMSO
    """
    raw = clean_name(name)
    if not raw:
        return []
    variants: List[str] = []

    def add(v: str):
        v = clean_name(v)
        if v and v not in variants:
            variants.append(v)

    add(raw)

    no_paren = re.sub(r"\s*\([^)]*\)", "", raw).strip()
    add(no_paren)

    no_comma = raw.split(",")[0].strip()
    add(no_comma)

    add(strip_stereo_prefix(raw))
    add(strip_stereo_prefix(no_comma))

    # Joined suffix fix: Dimethylsulfoxide -> Dimethyl sulfoxide
    suffixes = [
        "sulfoxide", "sulfate", "sulphate", "chloride", "phosphate", "nitrate",
        "selenite", "acetate", "citrate", "carbonate", "bicarbonate", "hydroxide",
        "hydrochloride", "tartrate", "lactate", "gluconate",
        "monohydrate", "dihydrate", "trihydrate", "tetrahydrate", "hydrate"
    ]
    low = raw.lower()
    if " " not in raw:
        for suf in suffixes:
            if low.endswith(suf):
                idx = len(raw) - len(suf)
                spaced = raw[:idx].rstrip() + " " + raw[idx:].lstrip()
                add(spaced)

    if raw.lower() in ("dimethyl sulfoxide", "dimethylsulfoxide"):
        add("DMSO")

    add(strip_hydrate_salt_words(raw))
    add(strip_hydrate_salt_words(strip_stereo_prefix(raw)))

    return variants

def is_mg_l(unit: Any) -> bool:
    if unit is None:
        return True
    u = str(unit).strip().lower().replace(" ", "")
    return u in ["", "none", "mg/l", "mg\\l", "mgl", "mg.l-1"]

def looks_like_protein(name: str) -> bool:
    """
    Gate UniProt so chemicals don't get hijacked.
    Only run UniProt for protein-ish strings.
    """
    low = clean_name(name).lower()
    if re.search(r"\b(protein|recombinant|growth factor|cytokine|insulin|transferrin|albumin|trypsin)\b", low):
        return True
    if re.search(r"\b(il-?\d+|interleukin|tgf[- ]?b|fgf[- ]?\d+|egf|bmp\d+|noggin|pdgf|vegf|ifn|tnf|csf\d*)\b", low):
        return True
    if re.match(r"^[A-Z0-9]{2,10}_[A-Z0-9]{3,10}$", clean_name(name)):
        return True
    return False

# ---------------------- Resolver ----------------------

class Resolver:
    def __init__(self, sleep: float = 0.8, debug: bool = False, allow_uniprot: bool = True):
        self.sleep = sleep
        self.debug = debug
        self.allow_uniprot = allow_uniprot
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "ResearchScript/SMILESOnly (mailto:biotech_research@example.com)",
            "Accept": "application/json"
        })
        self.cache: Dict[str, Optional[Tuple[str, str]]] = {}

    def _req_text(self, url: str, params=None) -> Tuple[Optional[int], Optional[str]]:
        for attempt in range(3):
            try:
                r = self.session.get(url, params=params, timeout=20)
                status = r.status_code
                text = r.text

                if self.debug:
                    snippet = (text or "")[:220].replace("\n", " ")
                    print(f"\n    [HTTP {status}] {url}\n    body: {snippet}")

                if status == 200:
                    time.sleep(self.sleep)
                    return status, text

                if status == 404:
                    time.sleep(self.sleep)
                    return status, text

                if status in (429, 500, 502, 503):
                    wait = self.sleep * (2 ** (attempt + 1))
                    if self.debug:
                        print(f"    [busy {status}] retry in {wait:.2f}s (attempt {attempt+1}/3)")
                    time.sleep(wait)
                    continue

                time.sleep(self.sleep)
                return status, text

            except Exception as e:
                if self.debug:
                    print(f"\n    [EXC] {type(e).__name__}: {e} | url={url}")
                time.sleep(1.0)

        return None, None

    def _req_json(self, url: str, params=None) -> Optional[dict]:
        status, text = self._req_text(url, params=params)
        if status != 200 or not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            return None

    def extract_smiles_strict(self, js: Optional[dict]) -> Optional[str]:
        """STRICT: only accept the 'SMILES' field."""
        if not js or not isinstance(js, dict):
            return None
        if "Fault" in js:
            return None

        props = js.get("PropertyTable", {}).get("Properties", [])
        if not props:
            return None
        return props[0].get("SMILES")

    def resolve_pubchem_name_smiles(self, name: str) -> Optional[str]:
        enc = quote(name, safe="")
        url = PUBCHEM_NAME_PROP_URL.format(name=enc)
        js = self._req_json(url)
        return self.extract_smiles_strict(js)

    def get_cid(self, name: str) -> Optional[int]:
        enc = quote(name, safe="")
        url = PUBCHEM_NAME_CID_URL.format(name=enc)
        js = self._req_json(url)
        if not js or "Fault" in js:
            return None
        cids = js.get("IdentifierList", {}).get("CID", [])
        return int(cids[0]) if cids else None

    def resolve_pubchem_cid_smiles(self, cid: int) -> Optional[str]:
        url = PUBCHEM_CID_PROP_URL.format(cid=cid)
        js = self._req_json(url)
        return self.extract_smiles_strict(js)

    def resolve_pubchem(self, raw_name: str) -> Optional[str]:
        variants = get_name_variants(raw_name)
        if self.debug:
            print(f"\n  variants={variants}")

        for variant in variants:
            # A) Direct name -> SMILES
            smi = self.resolve_pubchem_name_smiles(variant)
            if smi:
                return smi

            # B) CID fallback
            cid = self.get_cid(variant)
            if cid:
                smi2 = self.resolve_pubchem_cid_smiles(cid)
                if smi2:
                    return smi2

        return None

    def resolve_uniprot(self, name: str) -> Optional[str]:
        if not self.allow_uniprot:
            return None
        if not looks_like_protein(name):
            return None

        params = {
            "query": name,
            "format": "json",
            "fields": "accession",
            "size": 1
        }

        js = self._req_json(UNIPROT_SEARCH, params=params)

        if js and isinstance(js, dict):
            res = js.get("results", [])
            if res:
                return res[0].get("primaryAccession")

        return None

    def resolve(self, raw_name: str) -> Optional[Tuple[str, str]]:
        if not raw_name:
            return None
        raw_name = clean_name(raw_name)
        if raw_name in self.cache:
            return self.cache[raw_name]

        print(f"Resolving: {raw_name[:35]:<35} ... ", end="", flush=True)

        smi = self.resolve_pubchem(raw_name)
        if smi:
            print("[+] PubChem SMILES")
            self.cache[raw_name] = ("pubchem", smi)
            return ("pubchem", smi)

        seq = self.resolve_uniprot(raw_name)
        if seq:
            print("[+] UniProt Seq")
            self.cache[raw_name] = ("uniprot", seq)
            return ("uniprot", seq)

        print("[-] Missed")
        self.cache[raw_name] = None
        return None

# ---------------------- Dataset Transform ----------------------

def transform_dataset(data: list, resolver: Resolver) -> list:
    out = []
    for item in data:
        if not isinstance(item, dict):
            continue

        new_item = dict(item)
        comps = item.get("components", [])
        if not isinstance(comps, list):
            out.append(new_item)
            continue

        new_comps = []
        for row in comps:
            if not isinstance(row, dict):
                continue

            # unit filter
            if not is_mg_l(row.get("unit")):
                continue

            comp_name = row.get("component")
            if not isinstance(comp_name, str) or not comp_name.strip():
                continue

            res = resolver.resolve(comp_name)
            if not res:
                continue

            source, value = res
            new_row = dict(row)
            new_row["component"] = value
            new_row["component_source"] = source
            new_comps.append(new_row)

        new_item["components"] = new_comps
        out.append(new_item)

    return out

# ---------------------- Main ----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i")
    ap.add_argument("--output", "-o")
    ap.add_argument("--sleep", type=float, default=0.8)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--no-uniprot", action="store_true")
    ap.add_argument("--test-name", type=str, default=None)
    args = ap.parse_args()

    resolver = Resolver(sleep=args.sleep, debug=args.debug, allow_uniprot=(not args.no_uniprot))

    if args.test_name:
        print(f"TEST: {args.test_name}")
        res = resolver.resolve(args.test_name)
        print("\nRESULT:", res)
        return

    if not args.input or not args.output:
        raise SystemExit("Need --input and --output (or use --test-name).")

    print(f"Reading {args.input}...")
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit("Input must be a JSON list (your dataset format).")

    print("Starting...")
    new_data = transform_dataset(data, resolver)

    print(f"Saving to {args.output}...")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)

    print("Done.")

if __name__ == "__main__":
    main()
