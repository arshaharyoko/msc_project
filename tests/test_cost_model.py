import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from src import config
from src.cost_model import CostModel

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def model():
    return CostModel.from_table(config.PRICE_TABLE)


def test_unit_prices_match_manual_table(model):
    rows = list(csv.DictReader(open(config.DATA_DIR / "manual_component_cost.csv",
                                   encoding="utf-8-sig")))
    assert len(rows) == 23
    for row in rows:
        price = model.unit_price(row["component"])
        assert price is not None
        assert price == pytest.approx(float(row["cost"]), rel=1e-12)
        assert model.is_verified(row["component"])


def test_insulin_worked_example(model):
    assert model.unit_price("P01308") == 1.5
    assert model.component_cost("P01308", 20.0) == 30.0
    cost = model.recipe_cost([{"component": "P01308", "concentration": 20.0}])
    assert cost.total_usd_per_l == 30.0
    assert cost.unpriced == 0
    assert cost.verified_share == 1.0


def test_standardize_round_trip(model):
    for usd_per_l in (1e-9, 3e-4, 30.0, 914.25, 3.0e6):
        z = model.standardize(usd_per_l)
        assert model.unstandardize(z) == pytest.approx(usd_per_l, rel=1e-12)
        assert math.isfinite(z)


def test_coverage_counts(model):
    coverage = model.coverage()
    assert coverage == {"components": 172, "verified": 23, "unverified": 149,
                        "verified_share": pytest.approx(23 / 172)}


def test_wu_reference_cost(model):
    with open(config.DATABASE_JSON) as f:
        base = next(r for r in json.load(f) if r.get("source", "").startswith("Wu, X."))
    cost = model.recipe_cost(base)
    assert cost.total_usd_per_l == pytest.approx(914.25124486532, rel=1e-12)
    assert round(cost.total_usd_per_l, 2) == 914.25
    assert cost.unpriced == 3
    assert cost.verified_share == pytest.approx(0.9965193978316226, rel=1e-12)


def test_module_runs_without_src_modules():
    code = ("import sys; sys.path.insert(0, 'src'); import cost_model; "
            "assert not [m for m in sys.modules if m.startswith('src.modules')]; "
            "print(cost_model.CostModel.from_table('data/price_table.csv').coverage())")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                         capture_output=True, text=True, check=True)
    assert "'components': 172" in out.stdout
