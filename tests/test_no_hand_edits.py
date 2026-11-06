import pytest

import predict
from src import constraints

TRYPTOPHAN = "C1=CC=C2C(=C1)C(=CN2)C[C@@H](C(=O)O)N"
HISTIDINE = "C1=C(NC=N1)C[C@@H](C(=O)O)N"
ISOTOPIC_HISTIDINE = "C1=C(N[13CH]=N1)C[C@@H](C(=O)O)N.O.Cl"
BIOCIDE = "C1N(CN(CN1CCO)CC[O-])CCO.[Na+]"
STREPTOMYCIN = "CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C"
PIPECOLIC_ACID = "C1CCNC(C1)C(=O)O"
HISTIDINOL_PHOSPHATE = "C1=C(NC=N1)CC(COP(=O)(O)O)N"


def _set(recipe, component, concentration, source="pubchem"):
    kept = [item for item in recipe if item["component"] != component]
    return kept + [{"component": component, "concentration": concentration, "source": source}]


def _add(recipe, component, concentration, source="pubchem"):
    return recipe + [{"component": component, "concentration": concentration, "source": source}]


def _bare_store():
    # Only the exclusion path is wired, so no faiss index is needed.
    from src.vector_store import VectorStore
    store = object.__new__(VectorStore)
    store.toxic_compounds = set()
    return store


def _fires(name, cls):
    _, exclusions, patterns = predict._decode_exclusions()
    fired = {}
    picked = _bare_store()._first_allowed([0, 1], [name, "allowed"], exclusions, patterns, fired)
    return picked, fired.get(cls, 0)


def test_egf_is_capped_by_its_maximum_observed_dose(base_recipe):
    """Was clamped by hand from 5.90 and 9.46 back to 0.01 mg/L."""
    maximum = constraints.reference()["protein_maxima"]["P01133"]
    for inflated in (5.90, 9.46):
        assert constraints.recipe_violations(
            _set(base_recipe["kept"], "P01133", inflated, "uniprot"))["protein/P01133"] > 0
    assert constraints.recipe_violations(
        _set(base_recipe["kept"], "P01133", maximum, "uniprot"))["protein/P01133"] == 0


def test_tryptophan_is_required_by_its_nutrient_class(base_recipe):
    """Was restored by hand at 16 mg/L."""
    without = [item for item in base_recipe["kept"] if item["component"] != TRYPTOPHAN]
    key = "nutrient/essential_amino_acids/tryptophan"
    assert constraints.recipe_violations(without)[key] > 0
    restored = _add(without, TRYPTOPHAN, 16.0)
    assert constraints.recipe_violations(restored)[key] == 0


def test_isotopic_histidine_is_excluded_at_decode(base_recipe):
    """Was replaced by hand, mole for mole."""
    picked, count = _fires(ISOTOPIC_HISTIDINE, "isotopic_label")
    assert picked == "allowed" and count == 1
    assert constraints.recipe_violations(
        _add(base_recipe["kept"], ISOTOPIC_HISTIDINE, 31.48))["vocabulary/isotopic_label"] == 1


def test_biocide_is_excluded_at_decode(base_recipe):
    """Tris(2-hydroxyethyl)triazine sodium salt was removed by hand."""
    picked, count = _fires(BIOCIDE, "biocide")
    assert picked == "allowed" and count == 1
    assert constraints.recipe_violations(
        _add(base_recipe["kept"], BIOCIDE, 0.1))["vocabulary/biocide"] == 1


def test_streptomycin_is_excluded_at_decode(base_recipe):
    """Streptomycin was removed by hand."""
    picked, count = _fires(STREPTOMYCIN, "antibiotic")
    assert picked == "allowed" and count == 1
    assert constraints.recipe_violations(
        _add(base_recipe["kept"], STREPTOMYCIN, 50.0))["vocabulary/antibiotic"] == 1


def test_nonstandard_metabolites_and_the_realism_bound(base_recipe, model_state):
    """
    Row: pipecolic acid and histidinol phosphate were removed by hand, on the expectation
    that R rejects them. Measured on the acceptance checkpoint this does not hold: adding
    both at their recorded doses moves R from 0.206 to 0.186, under R_95 = 0.900, so the
    realism bound leaves them in. The bound does fire on a recipe off the training
    manifold (pipecolic acid plus EGF alone, R = 1.063), which is the mechanism exercised.
    """
    run, store, ctx = model_state
    clean = predict.round_trip(run, store, ctx, base_recipe["kept"])
    loaded = _add(_add(base_recipe["kept"], PIPECOLIC_ACID, 0.0137),
                  HISTIDINOL_PHOSPHATE, 500.0)
    report = predict.round_trip(run, store, ctx, loaded)
    assert clean["violations"]["realism"] == 0
    assert report["violations"]["realism"] == 0
    assert report["R"] < clean["R"]

    far = [{"component": PIPECOLIC_ACID, "concentration": 500.0, "source": "pubchem"},
           {"component": "P01133", "concentration": 0.01, "source": "uniprot"}]
    assert predict.round_trip(run, store, ctx, far)["violations"]["realism"] > 0


def test_duplicate_histidine_slots_merge_by_summed_dose():
    decoded = [{"component": HISTIDINE, "concentration": 21.0, "source": "pubchem"},
               {"component": HISTIDINE, "concentration": 21.0, "source": "pubchem"}]
    ingredients, merges = predict.merge_slots(decoded)
    assert merges == 1 and len(ingredients) == 1
    assert ingredients[0]["concentration"] == pytest.approx(42.0)
