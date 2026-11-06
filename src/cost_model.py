"""
The single cost model: prices from data/price_table.csv, recipe pricing, and the
ln(USD/L) standardization the latent cost head uses (ADR-022). Importing it needs
only the standard library; pricing needs only the CSV.
"""

from __future__ import annotations

import csv
import math
from collections import namedtuple
from pathlib import Path

UNVERIFIED = "database_with_cost.json, unverified"
DEFAULT_STATS = Path(__file__).resolve().parents[1] / "artifacts" / "cost_stats.pt"

Price = namedtuple("Price", "usd_per_mg source grade")
RecipeCost = namedtuple("RecipeCost", "total_usd_per_l unpriced verified_share")


def _ingredients(recipe):
    return recipe.get("components", []) if isinstance(recipe, dict) else recipe


class CostModel:
    def __init__(self, prices, log_mean=None, log_std=None, stats_path=None):
        self.prices = prices
        self.log_mean = log_mean
        self.log_std = log_std
        self._stats_path = stats_path

    @classmethod
    def from_table(cls, path, log_mean=None, log_std=None, stats_path=DEFAULT_STATS):
        prices = {}
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                name = row["component"]
                if name in prices:
                    raise ValueError(f"duplicate price row for {name}")
                price = float(row["usd_per_mg"])
                if not price > 0:
                    raise ValueError(f"non-positive unit price for {name}: {price}")
                prices[name] = Price(price, row["source"], row["grade"])
        return cls(prices, log_mean, log_std, stats_path)

    def price_dict(self):
        return {name: entry.usd_per_mg for name, entry in self.prices.items()}

    @classmethod
    def from_prices(cls, prices, log_mean=None, log_std=None):
        """Bridge for the archived scripts that still pass a component -> USD/mg dict."""
        return cls({name: Price(float(p), UNVERIFIED, None) for name, p in prices.items()},
                   log_mean, log_std)

    def unit_price(self, component):
        entry = self.prices.get(component)
        return None if entry is None else entry.usd_per_mg

    def is_verified(self, component):
        entry = self.prices.get(component)
        return entry is not None and entry.source != UNVERIFIED

    def component_cost(self, component, mg_per_l):
        """USD/L this component adds; None when it has no price.

        A missing or zero dose is priced at the 1 mg/L fallback the vectorizer
        and ADR-001 imputation already use.
        """
        price = self.unit_price(component)
        if price is None:
            return None
        return price * float(mg_per_l or 1)

    def recipe_cost(self, recipe):
        """Total USD/L, unpriced-component count and verified-source cost share."""
        total, verified, unpriced = 0.0, 0.0, 0
        for item in _ingredients(recipe):
            name = item["component"]
            cost = self.component_cost(name, item.get("concentration"))
            if cost is None:
                unpriced += 1
                continue
            total += cost
            if self.is_verified(name):
                verified += cost
        return RecipeCost(total, unpriced, verified / total if total else 0.0)

    def reference_cost(self, base):
        """Priced total of a reference formulation such as the Wu 2016 base."""
        return self.recipe_cost(base).total_usd_per_l

    def coverage(self):
        verified = sum(1 for entry in self.prices.values() if entry.source != UNVERIFIED)
        n = len(self.prices)
        return {"components": n, "verified": verified, "unverified": n - verified,
                "verified_share": verified / n if n else 0.0}

    def _stats(self):
        if self.log_mean is None and self._stats_path is not None:
            if Path(self._stats_path).exists():
                import torch
                stats = torch.load(self._stats_path, map_location="cpu", weights_only=False)
                self.log_mean = float(stats["log_mean"])
                self.log_std = float(stats["log_std"])
        if self.log_mean is None:
            raise ValueError("cost statistics are not available; pass stats_path or log_mean")
        return self.log_mean, self.log_std

    def standardize(self, cost):
        """USD/L -> standardized ln(USD/L) for the latent cost head."""
        mean, std = self._stats()
        return (math.log(cost) - mean) / std

    def standardize_log(self, log_cost):
        """Standardize an already-logged cost, keeping a tensor's device and dtype."""
        mean, std = self._stats()
        return (log_cost - mean) / std

    def unstandardize(self, z):
        """Standardized ln(USD/L) -> USD/L."""
        mean, std = self._stats()
        log_cost = z * std + mean
        if isinstance(log_cost, float):
            return math.exp(log_cost)
        if hasattr(log_cost, "exp"):
            return log_cost.exp()
        import numpy as np
        return np.exp(log_cost)
