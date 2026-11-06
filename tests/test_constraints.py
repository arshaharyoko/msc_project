import json

import pytest
import torch

from src import constraints

TRYPTOPHAN = "C1=CC=C2C(=C1)C(=CN2)C[C@@H](C(=O)O)N"
UREA = "C(=O)(N)N"


def _add(recipe, component, concentration, source="pubchem"):
    return recipe + [{"component": component, "concentration": concentration,
                      "source": source}]


def _set(recipe, component, concentration, source="pubchem"):
    kept = [item for item in recipe if item["component"] != component]
    return kept + [{"component": component, "concentration": concentration, "source": source}]


def test_cleaned_base_is_feasible(base_recipe):
    assert len(base_recipe["removed"]) == 1
    assert base_recipe["removed"][0]["component"] == "C(=O)(O)[O-].[Na+]"
    result = constraints.evaluate(base_recipe["kept"])
    assert result["feasible"], result["violations"]


def test_osmolarity_bound(base_recipe):
    high = _add(base_recipe["kept"], UREA, 8000.0)
    assert constraints.recipe_violations(high)["osmolarity"] > 0
    low = [{**item, "concentration": None if item["concentration"] is None
            else item["concentration"] * 0.5} for item in base_recipe["kept"]]
    assert constraints.recipe_violations(low)["osmolarity"] > 0


@pytest.mark.parametrize("component,dose,ion", [
    ("[Na+].[Cl-]", 3300.0, "Na+"),
    ("[Cl-].[K+]", 320.0, "K+"),
    ("[Cl-].[Cl-].[Ca+2]", 130.0, "Ca2+"),
    ("[O-]S(=O)(=O)[O-].[Mg+2]", 100.0, "Mg2+"),
])
def test_ion_bound(base_recipe, component, dose, ion):
    assert constraints.recipe_violations(_add(base_recipe["kept"], component, dose))[ion] > 0
    assert constraints.recipe_violations(_add(base_recipe["kept"], component, dose / 100))[ion] == 0


def test_nutrient_class_bound(base_recipe):
    without = [item for item in base_recipe["kept"] if item["component"] != TRYPTOPHAN]
    key = "nutrient/essential_amino_acids/tryptophan"
    assert constraints.recipe_violations(without)[key] > 0
    assert constraints.recipe_violations(_add(without, TRYPTOPHAN, 16.0))[key] == 0


def test_protein_dose_bound(base_recipe):
    over = _set(base_recipe["kept"], "P01133", 5.90, source="uniprot")
    assert constraints.recipe_violations(over)["protein/P01133"] > 0
    maximum = constraints.reference()["protein_maxima"]["P01133"]
    assert constraints.recipe_violations(
        _set(base_recipe["kept"], "P01133", maximum, source="uniprot"))["protein/P01133"] == 0


def test_vocabulary_bound(base_recipe):
    biocide = "C1N(CN(CN1CCO)CC[O-])CCO.[Na+]"
    antibiotic = "CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C"
    isotopic = "C1=C(N[13CH]=N1)C[C@@H](C(=O)O)N.O.Cl"
    assert constraints.recipe_violations(
        _add(base_recipe["kept"], biocide, 0.1))["vocabulary/biocide"] == 1
    assert constraints.recipe_violations(
        _add(base_recipe["kept"], antibiotic, 50.0))["vocabulary/antibiotic"] == 1
    assert constraints.recipe_violations(
        _add(base_recipe["kept"], isotopic, 31.48))["vocabulary/isotopic_label"] == 1


def test_readout_bounds_flag_a_recipe():
    thresholds = {"tau_p": 7.2, "tau_s": 6.6, "tau_d": -4.6}
    below = torch.tensor([0.0] * 22 + [7.0, 6.0, -3.0])
    assert all(value > 0 for value in constraints.readout_violations(below, thresholds).values())
    above = torch.tensor([0.0] * 22 + [8.0, 7.0, -6.0])
    assert sum(constraints.readout_violations(above, thresholds).values()) == 0


def test_realism_bound_is_a_distance_to_the_nearest_training_recipes():
    training = torch.tensor([[float(i)] for i in range(5)])
    assert float(constraints.realism(torch.tensor([[10.0]]), training)[0]) == pytest.approx(8.0)
    assert constraints.realism_violation(1.2, {"realism_95": 0.9}) > 0
    assert constraints.realism_violation(0.8, {"realism_95": 0.9}) == 0


def test_cleaned_base_is_round_trip_feasible(base_recipe, model_state):
    """The base judged against thresholds fitted in the same frame (ADR-028)."""
    import predict

    run, store, ctx = model_state
    report = predict.round_trip(run, store, ctx, base_recipe["kept"])
    assert report["feasible"], report["violations"]
    assert not [k for k, v in report["violations"].items() if v > 0], report["violations"]
    assert report["readout_logit"][2] == pytest.approx(ctx["thresholds"]["tau_d"], abs=1e-6)


def test_threshold_pool_and_the_judge_share_one_frame(base_recipe, model_state):
    """The pool row for the base and the round trip agree, so both use one vector frame."""
    import predict

    run, store, ctx = model_state
    thresholds = ctx["thresholds"]
    rows = [row for row in thresholds["pool"] if row["source"] == predict.BASE_SOURCE]
    assert len(rows) == 1, [row["source"] for row in thresholds["pool"]]
    report = predict.round_trip(run, store, ctx, base_recipe["kept"])
    assert rows[0]["logit_d"] == pytest.approx(report["readout_logit"][2], abs=1e-6)
    assert rows[0]["R"] == pytest.approx(report["R"], abs=1e-6)
    assert thresholds["base_in_pool"] is True


def test_threshold_cache_rejects_a_foreign_estimator(tmp_path, monkeypatch, model_state):
    from src import config

    run, store, _ = model_state
    monkeypatch.setattr(config, "ARTIFACT_DIR", tmp_path)
    path = tmp_path / (
        f"constraint_thresholds__{run.checkpoint.stem}__{constraints.THRESHOLD_ESTIMATOR}.json")
    path.write_text(json.dumps({"estimator": "pooled-extremes-v0"}))
    with pytest.raises(ValueError) as raised:
        constraints.thresholds_for(run, store, run.checkpoint)
    assert constraints.THRESHOLD_ESTIMATOR in str(raised.value)
    assert "pooled-extremes-v0" in str(raised.value)


def test_concentration_bound_tolerance(base_recipe):
    ref = constraints.reference()
    floor = ref["nutrient_minima"]["essential_amino_acids"]["tryptophan"]
    key = "nutrient/essential_amino_acids/tryptophan"
    assert constraints.evaluate(_set(base_recipe["kept"], TRYPTOPHAN, floor * (1.0 - 1e-9)))["feasible"]
    assert constraints.recipe_violations(_set(base_recipe["kept"], TRYPTOPHAN, floor * (1.0 - 1e-3)))[key] > 0

    cap = ref["protein_maxima"]["P01133"]
    assert constraints.evaluate(_set(base_recipe["kept"], "P01133", cap * (1.0 + 1e-9), "uniprot"))["feasible"]
    assert constraints.recipe_violations(
        _set(base_recipe["kept"], "P01133", cap * (1.0 + 1e-3), "uniprot"))["protein/P01133"] > 0

    thresholds = {"tau_p": 7.2, "tau_s": 6.6, "tau_d": -4.6}
    logits = torch.tensor([0.0] * 22 + [7.2 - 1e-5, 6.6 - 1e-5, -4.6 + 1e-5], dtype=torch.float64)
    assert all(v > 0 for v in constraints.readout_violations(logits, thresholds).values())
