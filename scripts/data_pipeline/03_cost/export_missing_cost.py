"""Export every component without a price from database_with_cost.json."""

import argparse
import csv
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path


DATA = Path(__file__).resolve().parents[3] / "data"
FIELDS = (
    "product_index",
    "component_index",
    "product",
    "component",
    "component_source",
    "concentration",
    "unit",
    "reason",
    "source",
)


def reason_for(component):
    try:
        concentration = Decimal(str(component.get("concentration")))
    except (InvalidOperation, TypeError, ValueError):
        concentration = None
    if concentration is None or not concentration.is_finite() or concentration < 0 or component.get("unit") != "mg/L":
        return "no_concentration"
    if component.get("component_source") != "pubchem":
        return "unsupported"
    return "no_quote"


def main():
    parser = argparse.ArgumentParser(description="Export components without a price to CSV.")
    parser.add_argument("--input", type=Path, default=DATA / "database_with_cost.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.input.open(encoding="utf-8") as file:
        products = json.load(file)

    counts = {"no_quote": 0, "unsupported": 0, "no_concentration": 0}
    with args.output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        for product_index, product in enumerate(products, start=1):
            for component_index, component in enumerate(product["components"], start=1):
                if component.get("cost") is not None:
                    continue
                reason = reason_for(component)
                counts[reason] += 1
                writer.writerow({
                    "product_index": product_index,
                    "component_index": component_index,
                    "product": product.get("product"),
                    "component": component.get("component"),
                    "component_source": component.get("component_source"),
                    "concentration": component.get("concentration"),
                    "unit": component.get("unit"),
                    "reason": reason,
                    "source": product.get("source"),
                })

    print(f"Saved {sum(counts.values())} rows to {args.output}")
    print(", ".join(f"{reason}: {count}" for reason, count in counts.items()))


if __name__ == "__main__":
    main()
