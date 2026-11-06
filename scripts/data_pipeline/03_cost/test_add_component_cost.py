import copy
import io
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from add_component_cost import JsonClient, add_cost, chemspace_prices, load_manual_costs, mass_mg, mcule_prices, usd_per_mg


class MemoryCache:
    def __init__(self):
        self.data = {}

    def get(self, provider, smiles):
        key = (provider, smiles)
        return key in self.data, self.data.get(key)

    def put(self, provider, smiles, value):
        self.data[(provider, smiles)] = value


class FakeClient:
    def __init__(self):
        self.calls = []

    def call(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        if url.endswith("/search/exact/"):
            return {"results": [{"query": "CCO", "mcule_id": "MCULE-123"}]}
        if url.endswith("/compounds/"):
            return {"results": [{"mcule_id": "MCULE-123", "best_prices": [
                {"price": 12, "amount": 10, "unit": "mg", "currency": "USD"},
                {"price": 50, "amount": 100, "unit": "mg", "currency": "USD"},
            ]}]}
        if url.endswith("/auth/token"):
            return {"access_token": "test-token"}
        if "/search/exact?" in url:
            return {"items": [{"offers": [{"prices": [
                {"priceUsd": 20, "packMg": 100},
                {"priceUsd": 3, "pack": 1, "uom": "mL"},
            ]}]}]}
        raise AssertionError(url)


class ComponentCostTests(unittest.TestCase):
    def test_chemspace_request_uses_multipart_form(self):
        captured = []

        def fake_urlopen(req, timeout):
            captured.append(req)
            return io.BytesIO(b'{"items": []}')

        with patch("add_component_cost.request.urlopen", side_effect=fake_urlopen):
            result = JsonClient().call("POST", "https://api.chem-space.com/v4/search/exact",
                                       headers={"Accept": "application/json; version=2.6"},
                                       provider="chemspace", min_interval=0,
                                       form={"SMILES": "C=C"})
        self.assertEqual(result, {"items": []})
        self.assertIn("multipart/form-data", captured[0].get_header("Content-type"))
        self.assertIn(b'name="SMILES"\r\n\r\nC=C\r\n', captured[0].data)

    def test_mass_conversion_rejects_non_mass(self):
        self.assertEqual(mass_mg("0.5", "g"), Decimal("500.0"))
        self.assertEqual(mass_mg(1, "5x100mg"), Decimal("500"))
        self.assertIsNone(mass_mg(1, "mL"))
        self.assertIsNone(usd_per_mg(10, 0, "mg"))
        self.assertIsNone(usd_per_mg(10, 1, "mg", "EUR"))

    def test_provider_prices_and_scaled_cost(self):
        dataset = [{"components": [
            {"component": "CCO", "component_source": "pubchem", "concentration": 10, "unit": "mg/L"},
            {"component": "CCO", "component_source": "pubchem", "concentration": 0.002, "unit": "mg/L"},
            {"component": "P12345", "component_source": "uniprot", "concentration": 2, "unit": "mg/L"},
            {"component": "CCO", "component_source": "pubchem", "concentration": None, "unit": None},
        ]}]
        original = copy.deepcopy(dataset)
        cache = MemoryCache()
        client = FakeClient()
        mcule_prices(["CCO"], "mcule-key", client, cache)
        chemspace_prices(["CCO"], "chemspace-key", client, cache)
        self.assertEqual(cache.get("mcule", "CCO"), (True, Decimal("0.5")))
        self.assertEqual(cache.get("chemspace_v4_ID", "CCO"), (True, Decimal("0.2")))
        calls = len(client.calls)
        mcule_prices(["CCO"], "mcule-key", client, cache)
        chemspace_prices(["CCO"], "chemspace-key", client, cache)
        self.assertEqual(len(client.calls), calls)
        stats = add_cost(dataset, cache)
        self.assertEqual(stats, {"priced": 2, "no_quote": 0, "unsupported": 1, "no_concentration": 1})
        self.assertEqual([entry["cost"] for entry in dataset[0]["components"]], [2.0, 0.0004, None, None])
        self.assertEqual([{k: v for k, v in item.items() if k != "cost"} for item in dataset[0]["components"]],
                         original[0]["components"])

    def test_mcule_skips_structures_above_server_limit(self):
        cache = MemoryCache()
        client = FakeClient()
        long_smiles = "C" * 501
        mcule_prices([long_smiles], "mcule-key", client, cache)
        self.assertEqual(cache.get("mcule", long_smiles), (True, None))
        self.assertEqual(client.calls, [])

    def test_manual_prices_override_api_and_scale_each_concentration(self):
        contents = ("component_source,component,cost,cost_unit\n"
                    "pubchem,CCO,0.75,USD/mg\n"
                    "uniprot,P01308,2,USD/mg\n"
                    "pubchem,CCC,,USD/mg\n")
        with patch.object(Path, "exists", return_value=True), patch.object(Path, "open", return_value=io.StringIO(contents)):
            manual = load_manual_costs(Path("manual.csv"))
        dataset = [{"components": [
            {"component": "CCO", "component_source": "pubchem", "concentration": 10, "unit": "mg/L"},
            {"component": "CCO", "component_source": "pubchem", "concentration": 0.5, "unit": "mg/L"},
            {"component": "P01308", "component_source": "uniprot", "concentration": 5, "unit": "mg/L"},
            {"component": "CCC", "component_source": "pubchem", "concentration": 1, "unit": "mg/L"},
            {"component": "P01308", "component_source": "uniprot", "concentration": None, "unit": None},
        ]}]
        cache = MemoryCache()
        cache.put("mcule", "CCO", Decimal("0.1"))
        stats = add_cost(dataset, cache, manual_costs=manual)
        self.assertEqual([x["cost"] for x in dataset[0]["components"]],
                         [7.5, 0.375, 10.0, None, None])
        self.assertEqual(stats, {"priced": 3, "no_quote": 1, "unsupported": 0, "no_concentration": 1})

    def test_manual_price_rejects_invalid_amount(self):
        contents = "component_source,component,cost,cost_unit\nuniprot,P01308,-1,USD/mg\n"
        with patch.object(Path, "exists", return_value=True), patch.object(Path, "open", return_value=io.StringIO(contents)):
            with self.assertRaisesRegex(ValueError, "line 2"):
                load_manual_costs(Path("manual.csv"))


if __name__ == "__main__":
    unittest.main()
