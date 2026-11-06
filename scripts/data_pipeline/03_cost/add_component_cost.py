"""Add estimated USD/L cost to every component in database_preprocessed.json.

Prices are normalized to USD/mg from exact SMILES matches on Mcule and
Chemspace. The lowest available unit price is multiplied by concentration
(mg/L). This is a proportional estimate, not a purchasable pack quote.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import uuid
from decimal import Decimal, InvalidOperation
from pathlib import Path
from urllib import error, parse, request


REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "data"
MCULE_API = "https://mcule.com/api/v1"
CHEMSPACE_API = "https://api.chem-space.com"
MCULE_AMOUNTS_MG = [1, 5, 10, 100, 1000]
MASS_TO_MG = {
    "kg": Decimal("1000000"),
    "g": Decimal("1000"),
    "mg": Decimal("1"),
    "ug": Decimal("0.001"),
    "µg": Decimal("0.001"),
    "μg": Decimal("0.001"),
    "microg": Decimal("0.001"),
    "mcg": Decimal("0.001"),
}


class ApiError(RuntimeError):
    pass


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if not os.environ.get(key):
            os.environ[key] = value


def decimal_or_none(value: object) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return number if number.is_finite() else None


def mass_mg(amount: object, unit: object) -> Decimal | None:
    """Convert a vendor pack size to mg; reject volume/moles and unknown units."""
    if not isinstance(unit, str):
        return None
    amount_dec = decimal_or_none(amount)
    if amount_dec is None or amount_dec <= 0:
        return None
    normalized = unit.strip().lower().replace(" ", "")
    factor = MASS_TO_MG.get(normalized)
    if factor is not None:
        return amount_dec * factor
    # Some offers encode a multipack in uom, e.g. pack=1, uom="5x100mg".
    match = re.fullmatch(r"(\d+)x(\d+(?:\.\d+)?)(kg|g|mg|ug|µg|μg|microg|mcg)", normalized)
    if match:
        return amount_dec * Decimal(match[1]) * Decimal(match[2]) * MASS_TO_MG[match[3]]
    return None


def usd_per_mg(price: object, amount: object, unit: object, currency: object = "USD") -> Decimal | None:
    if currency != "USD":
        return None
    price_dec = decimal_or_none(price)
    pack_mg = mass_mg(amount, unit)
    if price_dec is None or price_dec < 0 or pack_mg is None:
        return None
    return price_dec / pack_mg


def min_price(prices: list[Decimal]) -> Decimal | None:
    return min(prices) if prices else None


def load_manual_costs(path: Path) -> dict[tuple[str, str], Decimal]:
    """Read manually entered USD/mg prices, ignoring rows with blank cost."""
    if not path.exists():
        return {}
    prices: dict[tuple[str, str], Decimal] = {}
    seen: set[tuple[str, str]] = set()
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"component_source", "component", "cost", "cost_unit"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Manual cost CSV must have columns {', '.join(sorted(required))}: {path}")
        for line_number, row in enumerate(reader, start=2):
            source = (row.get("component_source") or "").strip()
            identifier = (row.get("component") or "").strip()
            if not source or not identifier:
                raise ValueError(f"Manual cost CSV line {line_number}: component_source and component are required")
            key = (source, identifier)
            if key in seen:
                raise ValueError(f"Manual cost CSV line {line_number}: duplicate component {source} {identifier}")
            seen.add(key)
            raw_cost = (row.get("cost") or "").strip()
            if not raw_cost:
                continue
            if (row.get("cost_unit") or "").strip() != "USD/mg":
                raise ValueError(f"Manual cost CSV line {line_number}: cost_unit must be USD/mg")
            cost = decimal_or_none(raw_cost)
            if cost is None or cost < 0:
                raise ValueError(f"Manual cost CSV line {line_number}: cost must be a number in USD/mg, at least 0")
            prices[key] = cost
    return prices


class JsonClient:
    def __init__(self) -> None:
        self.next_request_at: dict[str, float] = {}

    def call(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        provider: str,
        min_interval: float,
        payload: dict | None = None,
        form: dict | None = None,
        timeout: float = 45,
        max_attempts: int = 5,
    ) -> dict:
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers = {**headers, "Content-Type": "application/json"}
        elif form is not None:
            boundary = f"----component-cost-{uuid.uuid4().hex}"
            parts = []
            for key, value in form.items():
                parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n')
            data = ("".join(parts) + f"--{boundary}--\r\n").encode("utf-8")
            headers = {**headers, "Content-Type": f"multipart/form-data; boundary={boundary}"}
        else:
            data = None

        for attempt in range(max_attempts):
            pause = self.next_request_at.get(provider, 0) - time.monotonic()
            if pause > 0:
                time.sleep(pause)
            self.next_request_at[provider] = time.monotonic() + min_interval
            req = request.Request(url, data=data, headers=headers, method=method)
            try:
                with request.urlopen(req, timeout=timeout) as response:
                    result = json.load(response)
                    if not isinstance(result, dict):
                        raise ApiError(f"{provider}: response is not a JSON object: {url}")
                    return result
            except error.HTTPError as exc:
                if exc.code == 429 or 500 <= exc.code <= 599:
                    if attempt < max_attempts - 1:
                        retry_after = exc.headers.get("Retry-After", "")
                        wait = float(retry_after) if retry_after.isdigit() else min(2 ** attempt * 5, 60)
                        self.next_request_at[provider] = time.monotonic() + max(wait, min_interval)
                        continue
                raise ApiError(f"{provider}: HTTP {exc.code} at {url}") from exc
            except (error.URLError, TimeoutError) as exc:
                if attempt < max_attempts - 1:
                    self.next_request_at[provider] = time.monotonic() + min(2 ** attempt * 5, 60)
                    continue
                raise ApiError(f"{provider}: connection failed at {url}: {exc}") from exc
            except json.JSONDecodeError as exc:
                raise ApiError(f"{provider}: response is not valid JSON at {url}") from exc
        raise ApiError(f"{provider}: all attempts failed at {url}")


class Cache:
    def __init__(self, path: Path, refresh: bool = False) -> None:
        self.path = path
        self.data: dict[str, dict[str, str | None]] = {} if refresh or not path.exists() else json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(self.data, dict):
            raise ValueError(f"Cache is not a JSON object: {path}")

    def get(self, provider: str, smiles: str) -> tuple[bool, Decimal | None]:
        entries = self.data.get(provider, {})
        if smiles not in entries:
            return False, None
        value = entries[smiles]
        return True, Decimal(value) if value is not None else None

    def put(self, provider: str, smiles: str, value: Decimal | None) -> None:
        self.data.setdefault(provider, {})[smiles] = str(value) if value is not None else None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(json.dumps(self.data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.path)


def mcule_prices(smiles_list: list[str], token: str, client: JsonClient, cache: Cache) -> None:
    pending = [smiles for smiles in smiles_list if not cache.get("mcule", smiles)[0]]
    if not pending:
        return
    too_long = [smiles for smiles in pending if len(smiles) > 500]
    for smiles in too_long:
        cache.put("mcule", smiles, None)
    if too_long:
        print(f"Mcule: {len(too_long)} SMILES melebihi batas 500 karakter; dilewati", file=sys.stderr)
    pending = [smiles for smiles in pending if len(smiles) <= 500]
    if not pending:
        return
    headers = {"Authorization": f"Token {token}", "Accept": "application/json"}
    ids: dict[str, list[str]] = {smiles: [] for smiles in pending}
    # Exact search accepts up to 1000 structures; 200 keeps requests manageable.
    for start in range(0, len(pending), 200):
        batch = pending[start:start + 200]
        body = client.call("POST", f"{MCULE_API}/search/exact/", headers=headers,
                           provider="mcule", min_interval=3.1, payload={"queries": batch})
        results = body.get("results")
        if not isinstance(results, list):
            raise ApiError("Mcule: exact search response has no results")
        for item in results:
            if isinstance(item, dict) and item.get("query") in ids and isinstance(item.get("mcule_id"), str):
                ids[item["query"]].append(item["mcule_id"])

    id_to_smiles: dict[str, set[str]] = {}
    for smiles, matches in ids.items():
        for compound_id in matches:
            id_to_smiles.setdefault(compound_id, set()).add(smiles)
    prices: dict[str, list[Decimal]] = {smiles: [] for smiles in pending}
    compound_ids = list(id_to_smiles)
    remaining_ids = []
    for compound_id in compound_ids:
        cached, unit_price = cache.get("mcule_compound", compound_id)
        if cached:
            if unit_price is not None:
                for smiles in id_to_smiles[compound_id]:
                    prices[smiles].append(unit_price)
        else:
            remaining_ids.append(compound_id)
    for start in range(0, len(remaining_ids), 10):
        batch = remaining_ids[start:start + 10]
        body = client.call("POST", f"{MCULE_API}/compounds/", headers=headers,
                           provider="mcule_details", min_interval=12.1,
                           payload={"mcule_ids": batch, "price_amounts": MCULE_AMOUNTS_MG,
                                    "components": False, "availability": False},
                           timeout=120, max_attempts=3)
        results = body.get("results")
        if not isinstance(results, list):
            raise ApiError("Mcule: compounds response has no results")
        batch_prices: dict[str, Decimal | None] = {compound_id: None for compound_id in batch}
        for item in results:
            if not isinstance(item, dict) or item.get("mcule_id") not in batch_prices:
                continue
            best_prices = item.get("best_prices")
            if not isinstance(best_prices, list):
                raise ApiError("Mcule: best_prices not found")
            candidates = []
            for quote in best_prices:
                if not isinstance(quote, dict):
                    continue
                unit_price = usd_per_mg(quote.get("price"), quote.get("amount"), quote.get("unit"), quote.get("currency"))
                if unit_price is not None:
                    candidates.append(unit_price)
            batch_prices[item["mcule_id"]] = min_price(candidates)
        for compound_id, unit_price in batch_prices.items():
            cache.put("mcule_compound", compound_id, unit_price)
            if unit_price is not None:
                for smiles in id_to_smiles[compound_id]:
                    prices[smiles].append(unit_price)
        print(f"Mcule prices: {min(start + 10, len(remaining_ids))}/{len(remaining_ids)} IDs", file=sys.stderr)
    for smiles in pending:
        cache.put("mcule", smiles, min_price(prices[smiles]))


def chemspace_token(api_key: str, client: JsonClient) -> str:
    body = client.call("GET", f"{CHEMSPACE_API}/auth/token",
                       headers={"Authorization": f"Bearer {api_key}",
                                "Accept": "application/json; version=2.6"},
                       provider="chemspace_auth", min_interval=1.6)
    token = body.get("access_token")
    if not isinstance(token, str) or not token:
        raise ApiError("Chemspace: access_token not found")
    return token


def chemspace_prices(smiles_list: list[str], api_key: str, client: JsonClient, cache: Cache, country: str = "ID") -> None:
    cache_group = f"chemspace_v4_{country}"
    pending = [smiles for smiles in smiles_list if not cache.get(cache_group, smiles)[0]]
    if not pending:
        return
    token = chemspace_token(api_key, client)
    for index, smiles in enumerate(pending, 1):
        prices: list[Decimal] = []
        page = 1
        while True:
            params = parse.urlencode({"count": 100, "page": page, "categories": "CSSB,CSSS,CSMB,CSMS",
                                      "shipToCountry": country})
            url = f"{CHEMSPACE_API}/v4/search/exact?{params}"
            try:
                body = client.call("POST", url,
                                   headers={"Authorization": f"Bearer {token}",
                                            "Accept": "application/json; version=2.6"},
                                   provider="chemspace", min_interval=1.6, form={"SMILES": smiles})
            except ApiError as exc:
                if "HTTP 401" in str(exc):
                    token = chemspace_token(api_key, client)
                    body = client.call("POST", url,
                                       headers={"Authorization": f"Bearer {token}",
                                                "Accept": "application/json; version=2.6"},
                                       provider="chemspace", min_interval=1.6, form={"SMILES": smiles})
                else:
                    raise
            items = body.get("items")
            if not isinstance(items, list):
                raise ApiError("Chemspace: exact search response has no items")
            for item in items:
                if not isinstance(item, dict):
                    continue
                offers = item.get("offers", [])
                if not isinstance(offers, list):
                    continue
                for offer in offers:
                    if not isinstance(offer, dict) or not isinstance(offer.get("prices"), list):
                        continue
                    for quote in offer["prices"]:
                        if isinstance(quote, dict):
                            if quote.get("packMg") is not None:
                                unit_price = usd_per_mg(quote.get("priceUsd"), quote.get("packMg"), "mg")
                            else:
                                unit_price = usd_per_mg(quote.get("priceUsd"), quote.get("pack"), quote.get("uom"))
                            if unit_price is not None:
                                prices.append(unit_price)
            if len(items) < 100:
                break
            page += 1
        cache.put(cache_group, smiles, min_price(prices))
        print(f"Chemspace: {index}/{len(pending)} unique components", file=sys.stderr)


def add_cost(dataset: list[dict], cache: Cache, country: str = "ID",
             manual_costs: dict[tuple[str, str], Decimal] | None = None) -> dict[str, int]:
    manual_costs = manual_costs or {}
    stats = {"priced": 0, "no_quote": 0, "unsupported": 0, "no_concentration": 0}
    for product in dataset:
        for component in product["components"]:
            component["cost"] = None
            concentration = decimal_or_none(component.get("concentration"))
            if concentration is None or component.get("unit") != "mg/L" or concentration < 0:
                stats["no_concentration"] += 1
                continue
            manual_price = manual_costs.get((component.get("component_source"), component["component"]))
            if manual_price is not None:
                component["cost"] = float(concentration * manual_price)
                stats["priced"] += 1
                continue
            if component.get("component_source") != "pubchem":
                stats["unsupported"] += 1
                continue
            smiles = component["component"]
            candidates = [cache.get(provider, smiles)[1] for provider in ("mcule", f"chemspace_v4_{country}")]
            candidates = [price for price in candidates if price is not None]
            if not candidates:
                stats["no_quote"] += 1
                continue
            component["cost"] = float(concentration * min(candidates))
            stats["priced"] += 1
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Add an estimated cost (USD/L) to every component.")
    parser.add_argument("--input", type=Path, required=True, help="preprocessed recipes without costs")
    parser.add_argument("--output", type=Path, required=True, help="never defaults to a tracked data file")
    parser.add_argument("--cache", type=Path, help="price cache; default .price_cache.json next to --output")
    parser.add_argument("--env", type=Path, default=REPO / ".env")
    parser.add_argument("--manual-cost", type=Path, default=DATA / "manual_component_cost.csv",
                        help="manual price CSV; the cost column is in USD/mg")
    parser.add_argument("--refresh", action="store_true", help="request every price from the APIs again")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("output must differ from input")
    load_env(args.env)
    mcule_key = os.getenv("MCULE_API_KEY", "").strip()
    chemspace_key = os.getenv("CHEMSPACE_API_KEY", "").strip()
    country = os.getenv("CHEMSPACE_SHIP_TO_COUNTRY", "ID").strip().upper()
    if not mcule_key or not chemspace_key:
        parser.error("set MCULE_API_KEY and CHEMSPACE_API_KEY in the .env file first")
    if not re.fullmatch(r"[A-Z]{2}", country):
        parser.error("CHEMSPACE_SHIP_TO_COUNTRY must be a two-letter country code, for example ID")
    dataset = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(dataset, list):
        parser.error("input must be a JSON array")
    manual_costs = load_manual_costs(args.manual_cost)
    smiles_list = sorted({component["component"] for product in dataset
                          for component in product["components"]
                          if component.get("component_source") == "pubchem"
                          and component.get("unit") == "mg/L"
                          and decimal_or_none(component.get("concentration")) is not None
                          and ("pubchem", component["component"]) not in manual_costs})
    cache = Cache(args.cache or args.output.with_name(".price_cache.json"), refresh=args.refresh)
    client = JsonClient()
    mcule_prices(smiles_list, mcule_key, client, cache)
    chemspace_prices(smiles_list, chemspace_key, client, cache, country)
    stats = add_cost(dataset, cache, country, manual_costs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(dataset, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    total = sum(len(product["components"]) for product in dataset)
    unpriced = total - stats["priced"]
    print(f"Saved: {args.output}")
    print(f"Total components: {total}")
    print(f"Priced: {stats['priced']}")
    print(f"Unpriced: {unpriced} "
          f"(no_quote: {stats['no_quote']}, unsupported: {stats['unsupported']}, "
          f"no_concentration: {stats['no_concentration']})")
    print(f"Manual prices filled: {len(manual_costs)} unique components")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ApiError, ValueError, KeyError) as exc:
        print(f"Failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
