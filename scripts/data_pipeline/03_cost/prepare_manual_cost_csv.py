"""Create a unique-component CSV for manual prices missing from API results."""

import argparse
import csv
import json
from pathlib import Path

from export_missing_cost import reason_for


DATA = Path(__file__).resolve().parents[3] / "data"
FIELDS = (
    "component_source",
    "component",
    "reason",
    "occurrences",
    "example_product",
    "cost",
    "cost_unit",
    "price_reference",
    "notes",
)


def main():
    parser = argparse.ArgumentParser(description="Create a CSV of unique components that need a manual price.")
    parser.add_argument("--input", type=Path, default=DATA / "database_with_cost.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.input.open(encoding="utf-8") as file:
        products = json.load(file)

    unique = {}
    for product in products:
        for component in product["components"]:
            if component.get("cost") is not None:
                continue
            reason = reason_for(component)
            if reason not in ("no_quote", "unsupported"):
                continue
            key = (component["component_source"], component["component"])
            if key not in unique:
                unique[key] = {
                    "component_source": key[0],
                    "component": key[1],
                    "reason": reason,
                    "occurrences": 0,
                    "example_product": product.get("product", ""),
                    "cost": "",
                    "cost_unit": "USD/mg",
                    "price_reference": "",
                    "notes": ("Check the IMDM mapping: this SMILES is an isocyanate; "
                              "IMDM in the source article is a culture medium."
                              if key[1] == "CC1=CC=C(C=C1)CC2=CC=C(C=C2)N=C=O" else ""),
                }
            unique[key]["occurrences"] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("x", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(unique.values())
    except FileExistsError as exc:
        parser.error(f"{args.output} already exists; the manual price file is not overwritten")

    counts = {reason: sum(row["reason"] == reason for row in unique.values())
              for reason in ("no_quote", "unsupported")}
    print(f"Saved: {args.output} ({len(unique)} unique components; {counts})")


if __name__ == "__main__":
    main()
