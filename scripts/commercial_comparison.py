import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config
from src.cost_model import CostModel

BIBKEY = {
    "HiMesoXL": "HiMedia2026",
    "SCM015": "MilliporeSigma2026",
    "MSC-GRO": "VitroBiopharma2026",
    "TheraPEAK": "Lonza2026",
    "NutriStem": "Sartorius2026",
    "StemXVivo": "FisherRDSystems2026",
    "StemPro": "ThermoFisherStemPro2026",
}
STATUS_TEXT = {
    "serum-free": "Serum-free",
    "xeno-free": "Serum-free, xeno-free",
    "serum-containing": "Serum-containing",
    "not stated": "Not stated",
}
SERUM_FREE = {"serum-free", "xeno-free"}
BICARBONATE = "C(=O)(O)[O-].[Na+]"


def tex(text):
    return text.replace("&", "\\&")


def money(value, places=0):
    whole, _, frac = f"{value:.{places}f}".partition(".")
    whole = "{,}".join([whole[:-3], whole[-3:]]) if len(whole) > 3 else whole
    return whole + ("." + frac if frac else "")


def read_prices(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    for r in rows:
        r["price_per_l"] = round(float(r["pack_price_usd"]) / float(r["pack_volume_l"]))
    return rows


def cleaned_base_cost():
    model = CostModel.from_table(config.PRICE_TABLE)
    db = json.load(open(ROOT / "data" / "database_graded.json"))
    base = next(r for r in db if r["source"].startswith("Wu, X."))
    comps = list(base["components"])
    dup = [i for i, c in enumerate(comps) if c["component"] == BICARBONATE]
    del comps[dup[-1]]
    return model.recipe_cost(comps).total_usd_per_l


def optimizer_costs(directory):
    costs, bases = [], set()
    for k in range(10):
        d = json.load(open(directory / f"latent_prot__s{k}.json"))
        for r in d["runs"]:
            bases.add(r["realized_cost_base"])
            if r["round_trip"]["feasible"]:
                costs.append(r["realized_cost"])
    return costs, bases


def summary(name, values):
    print(f"{name}: n={len(values)} mean={statistics.mean(values):.2f} "
          f"median={statistics.median(values):.2f} min={min(values)} max={max(values)}")


def pct(ours, ref):
    return 100 * (ours - ref) / ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", type=Path, default=ROOT / "artifacts" / "commercial_media_prices.csv")
    ap.add_argument("--optimizer", type=Path, default=ROOT / "artifacts" / "optimizer")
    ap.add_argument("--out", type=Path, default=ROOT / "paper" / "commercial_media_rows.tex")
    args = ap.parse_args()

    rows = read_prices(args.prices)
    base = cleaned_base_cost()
    costs, bases = optimizer_costs(args.optimizer)
    cheapest, median = min(costs), statistics.median(costs)
    if len(bases) != 1 or abs(next(iter(bases)) - base) > 1e-6:
        raise SystemExit(f"cleaned base cost {base} does not match optimizer files {bases}")

    lines = []
    for r in rows:
        key = next(v for k, v in BIBKEY.items() if k in r["product"])
        name = f"{tex(r['product'])}, {tex(r['vendor'])}"
        if "converted from CNY" in r["note"]:
            name += " (converted from CNY)"
        lines.append(f"{name} & {STATUS_TEXT[r['serum_status']]} & {money(r['price_per_l'])} & \\cite{{{key}}} \\\\")
    lines += [
        "\\midrule",
        f"Cleaned base (\\texttt{{CostModel}}) & Serum-free & {money(base, 3)} & \\\\",
        f"This work, cheapest feasible recipe & Serum-free & {money(cheapest, 3)} & \\\\",
    ]
    args.out.write_text("\n".join(lines) + "\n")

    allp = [r["price_per_l"] for r in rows]
    sfp = [r["price_per_l"] for r in rows if r["serum_status"] in SERUM_FREE]
    print(f"cleaned base (CostModel, duplicate bicarbonate removed): {base:.3f}")
    print(f"feasible protein-floor runs: {len(costs)}")
    print(f"cheapest feasible: {cheapest:.3f}")
    print(f"median feasible: {median:.3f}")
    summary("all commercial", allp)
    summary("serum-free commercial", sfp)
    for label, ref in (("cheapest", cheapest), ("median", median)):
        print(f"commercial above our {label}: {sum(p > ref for p in allp)} of {len(allp)}"
              f" (serum-free {sum(p > ref for p in sfp)} of {len(sfp)})")
    print(f"commercial below cleaned base: {sum(p < base for p in allp)} of {len(allp)}")
    for label, ref in (("cheapest", cheapest), ("median", median)):
        print(f"{label} vs all-commercial mean: {pct(ref, statistics.mean(allp)):.1f} percent;"
              f" vs serum-free mean: {pct(ref, statistics.mean(sfp)):.1f} percent")
    print(f"saving to cleaned base: {base - cheapest:.3f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
