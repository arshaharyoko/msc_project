"""
Feasibility constraints of ADR-018 on a named medium recipe.

Osmolarity, ion and nutrient bounds are derived from the database positives except the
osmolarity and ion ranges, declared `assumed` (AC-O1). The exact forms judge a recipe;
the differentiable forms drive the optimizer.
"""

import hashlib
import json
import math
import re
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

from src import config
from src.modules.layers import gumbel_sigmoid

__all__ = [
    "BOUND_PROVENANCE", "ConstraintIndex", "ION_RANGES", "OSMOLARITY_RANGE",
    "THRESHOLD_ESTIMATOR", "clean_base", "component_facts", "encode_recipe", "evaluate",
    "fit_thresholds", "is_protein", "named_recipe", "positive_recipes",
    "readout_violations", "realism", "realism_violation", "recipe_embedding",
    "recipe_totals", "recipe_violations", "reference", "round_trip_vectors",
    "thresholds_for", "training_embeddings", "vocabulary_violations",
]

# Cache tag: a pool fitted in another frame answers a different question, so its
# file must not be read as this estimator's.
THRESHOLD_ESTIMATOR = "roundtrip-v2"

# Declared, not measured, and not confirmed by the authors (AC-O1). Never relax one to
# make a recipe feasible: report the binding constraint instead.
OSMOLARITY_RANGE = (260.0, 340.0)   # milliosmoles per litre of particles
ION_RANGES = {"Na+": (110.0, 170.0), "K+": (2.5, 8.0),
              "Ca2+": (0.2, 2.5), "Mg2+": (0.3, 1.5)}  # mM
BOUND_PROVENANCE = {"osmolarity": "assumed",
                    **{ion: "assumed" for ion in ION_RANGES}}

CLASSES_PATH = config.DATA_DIR / "nutrient_classes.json"
EXCLUSIONS_PATH = config.DATA_DIR / "decode_exclusions.json"

_IONS = {"Na+": 11, "K+": 19, "Ca2+": 20, "Mg2+": 12}
_ACCESSION = re.compile(r"[A-Z][A-Z0-9]{5,9}")
_UNCHARGER = rdMolStandardize.Uncharger()

# Violation scales, so one multiplier step means something comparable across families.
_OSM_SCALE = 10.0    # milliosmoles per litre
_ION_SCALE = 1.0     # mM
_CONC_TOL = 1e-6     # relative float32 projection tolerance on concentration bounds (ADR-031)
_TIE = 1e-6          # distance under which two vocabulary vectors are the same to the encoder
_SHARPNESS = 100.0   # softplus slope of the decoded dose channel, as in the price surrogate
_NEIGHBOURS = 5      # training recipes behind R
_READOUT = (22, 23, 24)


def _is_water(mol):
    return mol.GetNumAtoms() == 1 and mol.GetAtomWithIdx(0).GetAtomicNum() == 8


@lru_cache(maxsize=1)
def _declared(path=None):
    with open(path or CLASSES_PATH) as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _exclusions(path=None):
    with open(path or EXCLUSIONS_PATH) as f:
        return json.load(f)["classes"]


@lru_cache(maxsize=None)
def _skeleton(smiles):
    """Salt and hydrate free canonical form, so one nutrient matches all its forms."""
    mol = _parse(smiles)
    if mol is None:
        return None
    organic = [f for f in Chem.GetMolFrags(mol, asMols=True)
               if not _is_water(f) and any(a.GetAtomicNum() == 6 for a in f.GetAtoms())]
    if not organic:
        return None
    largest = max(organic, key=lambda m: m.GetNumHeavyAtoms())
    return Chem.MolToSmiles(_UNCHARGER.uncharge(largest))


def _parse(name):
    # Accessions are not SMILES and RDKit logs every failed parse; the caller wants None.
    with rdBase.BlockLogs():
        return Chem.MolFromSmiles(name)


@lru_cache(maxsize=None)
def component_facts(name):
    """
    MW, dissociated particles and ion counts of one component, or None for a name
    that is not a SMILES (a protein accession).

    Particles are the ions of a salt, water of crystallisation excluded; a covalent
    molecule is one particle. A declared polymer is one chain at its assumed molar
    mass, because its repeat-unit SMILES describes one unit, not the 9 g/L dose.
    """
    mol = _parse(name)
    if mol is None:
        return None
    ions = {ion: sum(a.GetAtomicNum() == z for a in mol.GetAtoms())
            for ion, z in _IONS.items()}
    polymer = _declared()["polymer_repeat_units"].get(name)
    if polymer is not None:
        return {"mw": float(polymer["molar_mass_g_per_mol"]), "particles": 1,
                "ions": ions, "provenance": polymer["provenance"]}
    fragments = [f for f in Chem.GetMolFrags(mol, asMols=True) if not _is_water(f)]
    return {"mw": float(Descriptors.MolWt(mol)), "particles": max(len(fragments), 1),
            "ions": ions, "provenance": "derived-from-data"}


def is_protein(name):
    return component_facts(name) is None and bool(_ACCESSION.fullmatch(name))


def recipe_totals(recipe):
    """
    Osmolarity and ion molarities of a named recipe.

    Protein and unparsable components are counted in ``n_unquantified`` and left out:
    the lightest protein in the vocabulary is insulin at 5.8 kDa, so even its highest
    observed dose adds under half a milliosmole, below the precision of the bound.
    """
    osmolarity, ions, missing = 0.0, {ion: 0.0 for ion in ION_RANGES}, 0
    for item in recipe:
        dose = item.get("concentration")
        if dose is None or not dose > 0:
            continue
        facts = component_facts(item["component"])
        if facts is None:
            missing += 1
            continue
        mmol = float(dose) / facts["mw"]
        osmolarity += mmol * facts["particles"]
        for ion, count in facts["ions"].items():
            ions[ion] += mmol * count
    return {"osmolarity": osmolarity, "ions": ions, "n_unquantified": missing}


@lru_cache(maxsize=1)
def positive_recipes(path=None):
    with open(str(path or config.DATABASE_JSON)) as f:
        records = json.load(f)
    return [{"components": [{"component": c["component"],
                             "component_source": c["component_source"],
                             "concentration": c["concentration"]}
                            for c in r["components"]]}
            for r in records if float(r.get("proliferation", 1)) >= 0.5]


@lru_cache(maxsize=1)
def _member_index(path=None):
    index = {}
    for cls, entry in _declared(path)["classes"].items():
        for member, forms in entry["members"].items():
            for smiles in forms:
                key = _skeleton(smiles)
                if key is None:
                    raise ValueError(f"nutrient class member does not parse: {smiles}")
                index[key] = (cls, member)
    return index


def _required(cls):
    return bool(_declared()["classes"][cls].get("required", False))


def _member_doses(recipe):
    index, doses = _member_index(), {}
    for item in recipe:
        key = index.get(_skeleton(item["component"]))
        dose = item.get("concentration")
        if key is None or dose is None:
            continue
        doses[key] = doses.get(key, 0.0) + float(dose)
    return doses


def nutrient_key(cls, member):
    return f"nutrient/{cls}/{member}"


def protein_key(accession):
    return f"protein/{accession}"


@lru_cache(maxsize=1)
def reference(path=None):
    """Bounds derived from the database positives: nutrient minima and protein maxima."""
    positives = positive_recipes(path)
    minima, maxima = {}, {}
    index = _member_index()
    for record in positives:
        for item in record["components"]:
            dose = item["concentration"]
            if dose is None:
                continue
            facts = component_facts(item["component"])
            if facts is None:
                if is_protein(item["component"]):
                    maxima[item["component"]] = max(
                        maxima.get(item["component"], 0.0), float(dose))
                continue
            key = index.get(_skeleton(item["component"]))
            if key is not None:
                minima.setdefault(key[0], {})
                minima[key[0]][key[1]] = min(
                    minima[key[0]].get(key[1], float("inf")), float(dose))
    return {"nutrient_minima": minima, "protein_maxima": maxima,
            "n_positives": len(positives)}


def _violation(key, value, ref):
    """Scaled violation h >= 0 of one constraint value, using the declared bounds."""
    if key == "osmolarity":
        lo, hi = OSMOLARITY_RANGE
        return max(0.0, (lo - value) / _OSM_SCALE) + max(0.0, (value - hi) / _OSM_SCALE)
    if key in ION_RANGES:
        lo, hi = ION_RANGES[key]
        if (lo - value) / lo > _CONC_TOL:
            return (lo - value) / _ION_SCALE
        if (value - hi) / hi > _CONC_TOL:
            return (value - hi) / _ION_SCALE
        return 0.0
    if key.startswith("nutrient/"):
        _, cls, member = key.split("/", 2)
        low = ref["nutrient_minima"][cls][member]
        h = (low - value) / low
        return h if h > _CONC_TOL else 0.0
    if key.startswith("protein/"):
        high = ref["protein_maxima"][key.split("/", 1)[1]]
        h = (value - high) / high
        return h if h > _CONC_TOL else 0.0
    return float(value)


def exclusion_class(name):
    """The declared vocabulary class a component name falls in, if any (ADR-021)."""
    classes = _exclusions()
    for cls, entry in classes.items():
        if name in entry.get("names", ()):
            return cls
    return next((cls for cls, entry in classes.items()
                 for pattern in entry.get("patterns", ()) if pattern in name), None)


def vocabulary_violations(recipe):
    out = []
    for item in recipe:
        cls = exclusion_class(item["component"])
        if cls is not None:
            out.append({"component": item["component"], "class": cls})
    return out


def evaluate(recipe, ref=None):
    """
    Every model-free constraint on one named recipe, raw values and scaled violations.

    ``feasible`` covers osmolarity, ions, required nutrients, protein doses and the
    vocabulary; the readout and realism bounds need a checkpoint and are added by
    ``predict`` on the round trip.
    """
    ref = reference() if ref is None else ref
    totals = recipe_totals(recipe)
    values = {"osmolarity": totals["osmolarity"], **totals["ions"]}
    values.update({nutrient_key(*key): dose for key, dose in _member_doses(recipe).items()})
    # An absent required nutrient is a violation of its floor, not a missing key.
    for cls, members in ref["nutrient_minima"].items():
        if _required(cls):
            for member in members:
                values.setdefault(nutrient_key(cls, member), 0.0)
    for item in recipe:
        if is_protein(item["component"]) and item.get("concentration") is not None:
            key = protein_key(item["component"])
            values[key] = values.get(key, 0.0) + float(item["concentration"])

    violations = {}
    for key, value in values.items():
        if key.startswith("nutrient/"):
            _, cls, member = key.split("/", 2)
            if not _required(cls) or member not in ref["nutrient_minima"].get(cls, {}):
                continue
        if key.startswith("protein/") and key.split("/", 1)[1] not in ref["protein_maxima"]:
            violations[key] = 1.0
            continue
        violations[key] = _violation(key, value, ref)
    for hit in vocabulary_violations(recipe):
        key = f"vocabulary/{hit['class']}"
        violations[key] = violations.get(key, 0.0) + 1.0
    return {"values": values, "violations": violations,
            "n_unquantified": totals["n_unquantified"],
            "feasible": not any(v > 0 for v in violations.values())}


def recipe_violations(recipe, ref=None):
    """The scaled violations only, in the keys the optimizer reports."""
    return evaluate(recipe, ref)["violations"]


def clean_base(recipe):
    """(kept, removed): exact duplicate rows dropped, the first listing kept."""
    seen, kept, removed = set(), [], []
    for item in recipe:
        key = (item["component"], item.get("concentration"), item.get("unit"))
        if key in seen:
            removed.append(dict(item))
            continue
        seen.add(key)
        kept.append(item)
    return kept, removed


def named_recipe(components):
    """A dataset's component rows as a named recipe, keeping an unrecorded dose as None."""
    return [{"component": c["component"], "concentration": c["concentration"],
             "unit": c.get("unit"),
             "source": "uniprot" if c["component_source"] == "uniprot" else "pubchem"}
            for c in components]


class ConstraintIndex:
    """Per-slot facts behind the differentiable constraints on a decoded recipe."""

    def __init__(self, run, store, ref=None):
        self.ref = reference() if ref is None else ref
        self.dec_conc = store.dec_conc
        self.prefer = None
        self.protein_floor = {}
        vectors, facts, groups, uniprot = {}, {}, {}, {}
        for item in run.dataset:
            for comp in item["components"]:
                name = comp["component"]
                if name in vectors or name in uniprot:
                    continue
                if comp["component_source"] == "uniprot":
                    uniprot[name] = store.std_uniprot(
                        comp["component_vec"].to(run.device, torch.float32))
                    continue
                if exclusion_class(name) is not None:
                    continue
                vectors[name] = comp["component_vec"]
                facts[name] = component_facts(name)
                key = _member_index().get(_skeleton(name))
                if (key is not None and _required(key[0])
                        and key[1] in self.ref["nutrient_minima"].get(key[0], {})):
                    groups.setdefault(key, len(groups))

        device = run.device
        self.names = list(vectors)
        self.row_of = {name: i for i, name in enumerate(self.names)}
        self.uniprot_vectors = uniprot
        self.vectors = torch.stack([
            (vectors[n].to(device=device, dtype=torch.float32) - store.targ_mean)
            / (store.targ_std + 1e-8) for n in self.names])
        self.mw = torch.tensor([facts[n]["mw"] if facts[n] else 1.0 for n in self.names],
                               device=device)
        self.particles = torch.tensor(
            [facts[n]["particles"] if facts[n] else 0.0 for n in self.names], device=device)
        self.ions = torch.tensor(
            [[facts[n]["ions"][ion] if facts[n] else 0.0 for n in self.names]
             for ion in ION_RANGES], device=device)
        self.group_of = torch.tensor(
            [groups.get(_member_index().get(_skeleton(n)), -1) for n in self.names],
            device=device)
        self.groups = [None] * len(groups)
        for key, g in groups.items():
            self.groups[g] = (key[0], key[1], self.ref["nutrient_minima"][key[0]][key[1]])

    def nearest_rows(self, flat):
        """Nearest vocabulary row per slot; ``prefer`` wins a tie the encoder cannot break."""
        d = torch.cdist(flat, self.vectors)
        nearest = d.argmin(-1)
        if self.prefer is not None and len(self.prefer) == len(flat):
            row = self.prefer.clamp_min(0)
            tied = (self.prefer >= 0) & (d.gather(1, row[:, None])[:, 0]
                                         <= d.min(-1).values + _TIE)
            nearest = torch.where(tied, row, nearest)
        return nearest

    def values(self, x1_hat, x2_hat, uniprot_names, n_scalars):
        """Raw constraint values for decoded halves, differentiable in the doses.

        Identity is the nearest vocabulary row and the dose is expm1(softplus(log1p)),
        the same two moves ``VectorStore.decoded_log_cost`` prices with. The
        channel is standardized, so it goes through ``dec_conc`` first (ADR-032).
        """
        dose1 = torch.expm1(F.softplus(self.dec_conc(x1_hat[..., -n_scalars]),
                                       beta=_SHARPNESS)).clamp_min(0.0)
        flat = x1_hat[..., :-n_scalars].reshape(-1, x1_hat.size(-1) - n_scalars)
        nearest = self.nearest_rows(flat).reshape(dose1.shape)
        mmol = dose1 / self.mw[nearest]
        out = {"osmolarity": (mmol * self.particles[nearest]).sum()}
        for i, ion in enumerate(ION_RANGES):
            out[ion] = (mmol * self.ions[i][nearest]).sum()
        groups = self.group_of[nearest]
        for g, (cls, member, _) in enumerate(self.groups):
            out[nutrient_key(cls, member)] = (dose1 * (groups == g)).sum()
        dose2 = torch.expm1(F.softplus(self.dec_conc(x2_hat[..., -n_scalars]),
                                       beta=_SHARPNESS)).clamp_min(0.0)
        for i, acc in enumerate(uniprot_names):
            out[protein_key(acc)] = dose2[0, i]
        return out

    def violations(self, x1_hat, x2_hat, uniprot_names, n_scalars):
        """Scaled violations h >= 0, same keys and bounds as ``evaluate``."""
        values = self.values(x1_hat, x2_hat, uniprot_names, n_scalars)
        out = {}
        for key, value in values.items():
            if key == "osmolarity":
                lo, hi = OSMOLARITY_RANGE
                out[key] = F.relu(lo - value) / _OSM_SCALE + F.relu(value - hi) / _OSM_SCALE
            elif key in ION_RANGES:
                lo, hi = ION_RANGES[key]
                out[key] = F.relu(lo - value) / _ION_SCALE + F.relu(value - hi) / _ION_SCALE
            elif key.startswith("nutrient/"):
                _, cls, member = key.split("/", 2)
                low = self.ref["nutrient_minima"][cls][member]
                out[key] = F.relu(low - value) / low
            else:
                acc = key.split("/", 1)[1]
                high = self.ref["protein_maxima"][acc]
                out[key] = F.relu(value - high) / high
        if self.protein_floor:
            # On the log1p channel, not the dose: below zero the softplus is flat and a
            # dose-space floor would have no gradient left to pull the protein back (ADR-032)
            channel = self.dec_conc(x2_hat[..., -n_scalars])
            for i, acc in enumerate(uniprot_names):
                low = self.protein_floor.get(acc)
                if low:
                    target = math.log1p(low)
                    out[f"floor/{acc}"] = F.relu(target - channel[0, i]) / target
        return out


def encode_recipe(run, x1, x2):
    """Readout logits, realism embedding and pre-gate code of one recipe, noise dropped."""
    model = run.model
    Y = model.f(model.encoder(x1, x2))
    if getattr(run, "ablate_svd", False):
        Yk = Y
    else:
        mu, Uk = model.truncated_svd(Y)
        Yk = torch.matmul(Uk, mu)
    pre = Yk
    gate = gumbel_sigmoid(model.gate_logit(Yk), 1.0, sample=False)
    Yk = Yk * gate
    logits = model.pathway_graph(model.z_c(Yk), model.graph_features_,
                                 run.adjacency_matrices, gate=gate)[0, :, 0]
    return logits, Yk.mean(1), pre


def recipe_embedding(run, x1, x2):
    """Mean of the noise-free gated component codes: where a recipe sits for R."""
    return encode_recipe(run, x1, x2)[1]


def realism(embedding, training):
    """Mean distance from a recipe embedding to its 5 nearest training recipes."""
    distances = torch.cdist(embedding, training)
    k = min(_NEIGHBOURS, training.size(0))
    return distances.topk(k, largest=False, dim=-1).values.mean(-1)


def _recipe_vectors(run, store, item):
    device = run.device
    x1 = [store.make_x1_vec(c, device) for c in item["components"]
          if c["component_source"] == "pubchem"]
    x2 = [store.make_x2_vec(c, device) for c in item["components"]
          if c["component_source"] == "uniprot"]
    if not x1 or not x2:
        return None
    return torch.stack(x1).unsqueeze(0), torch.stack(x2).unsqueeze(0)


def round_trip_vectors(store, index, ingredients, device):
    """Vectors of a named recipe, built exactly as store.make_x1_vec/make_x2_vec do.

    A component with no recorded dose is encoded at 1 mg/L, the convention
    make_x1_vec already uses; the chemistry totals leave it out.
    """
    x1, x2 = [], []
    for item in ingredients:
        channel = torch.tensor([store.enc_conc(math.log1p(float(item["concentration"] or 1)))],
                               device=device)
        if item.get("source") == "uniprot":
            if item["component"] not in index.uniprot_vectors:
                return None
            x2.append(torch.cat([index.uniprot_vectors[item["component"]], channel]))
        else:
            if item["component"] not in index.row_of:
                return None
            x1.append(torch.cat([index.vectors[index.row_of[item["component"]]], channel]))
    if not x1 or not x2:
        return None
    return torch.stack(x1).unsqueeze(0), torch.stack(x2).unsqueeze(0)


def _split(run, spec=None):
    from src.ablation import AblationSpec, split_dataset
    return split_dataset(run.dataset, AblationSpec() if spec is None else spec)


def training_embeddings(run, store, spec=None):
    """The realism reference: embeddings of the training split's recipes."""
    splits = _split(run, spec)
    out = []
    with torch.no_grad():
        for i in splits["train"].indices:
            vectors = _recipe_vectors(run, store, run.dataset[i])
            if vectors is not None:
                out.append(recipe_embedding(run, *vectors))
    return torch.cat(out, 0)


def _fingerprint(dataset):
    """Stable tag of the records a pool was fitted on, so a stale cache is visible."""
    digest = hashlib.sha256()
    for i in range(len(dataset)):
        item = dataset[i]
        digest.update(json.dumps([
            item["product"], item["flag"], item["source"],
            item["proliferation"], item["survival"], item["differentiation"],
            [[c["component"], c["concentration"], c["unit"], c["component_source"]]
             for c in item["components"]],
        ]).encode())
    return digest.hexdigest()[:16]


def _own_fold(checkpoint, spec):
    """The fold a checkpoint was fitted on: its tag first, then the spec."""
    name = checkpoint.stem if hasattr(checkpoint, "stem") else \
        str(checkpoint or "").rsplit("/", 1)[-1]
    match = re.search(r"__f(\d+)", name)
    if match:
        return int(match.group(1))
    return int(getattr(spec, "fold", 0) or 0) if spec is not None else 0


def _pool_bounds(rows):
    if not rows:
        return None
    return {"n": len(rows),
            "tau_p": min(row["logit_p"] for row in rows),
            "tau_s": min(row["logit_s"] for row in rows),
            "tau_d": max(row["logit_d"] for row in rows),
            "realism_95": float(np.percentile([row["R"] for row in rows], 95))}


def fit_thresholds(run, store, spec=None, checkpoint=None, out=None, index=None,
                   base_source=None):
    """
    tau_p, tau_s, tau_d and R_95 over every held-out real positive, pooled across the
    five grouped folds (ADR-028).

    Folding separates the graded corpus only: the legacy corpus has no synthetic parents, so
    every fold is the same split. The published base is appended by inclusion when its fold
    is not a test fold. Each recipe enters the pool in the judge's frame (kept ingredients
    named, re-vectorized through the constraint index, encoded by the model), and R is
    measured against that fold's training embeddings. The readout bounds are the pooled
    extremes (minimum for p and s, maximum for d), so a published medium is feasible by
    construction (AC-O0); R_95 is the 95th percentile of the same pool.
    """
    from dataclasses import replace

    from src.ablation import AblationSpec, N_FOLDS, split_dataset

    base = AblationSpec() if spec is None else replace(spec)
    if index is None:
        index = ConstraintIndex(run, store)

    # The judge round trip runs in eval mode; dropout would let two encodings of one
    # recipe disagree, and the pool would stop matching the bound it is meant to set.
    was_training = run.model.training
    run.model.eval()
    try:
        pool, skipped, seen, members, n_train = [], [], set(), set(), []
        trainings = {}
        for fold in range(N_FOLDS):
            splits = split_dataset(run.dataset, replace(base, fold=fold))
            training = []
            with torch.no_grad():
                for i in splits["train"].indices:
                    vectors = _recipe_vectors(run, store, run.dataset[i])
                    if vectors is not None:
                        training.append(recipe_embedding(run, *vectors))
            if not training:
                for i in splits["test"].indices:
                    item = run.dataset[i]
                    if float(item["proliferation"]) >= 0.5 and i not in seen:
                        seen.add(i)
                        skipped.append({"index": i, "fold": fold,
                                        "source": item.get("source", ""),
                                        "reason": "no-training"})
                continue
            training = torch.cat(training, 0)
            trainings[fold] = training
            n_train.append(training.size(0))
            with torch.no_grad():
                for i in splits["test"].indices:
                    item = run.dataset[i]
                    if float(item["proliferation"]) < 0.5 or i in seen:
                        continue
                    seen.add(i)
                    kept, _ = clean_base(named_recipe(item["components"]))
                    vectors = round_trip_vectors(store, index, kept, run.device)
                    if vectors is None:
                        skipped.append({"index": i, "fold": fold,
                                        "source": item.get("source", ""),
                                        "reason": "unencodable"})
                        continue
                    logits, embedding, _ = encode_recipe(run, *vectors)
                    pool.append({
                        "source": item.get("source", ""),
                        "fold": fold,
                        "n_components": len(kept),
                        "logit_p": float(logits[_READOUT[0]]),
                        "logit_s": float(logits[_READOUT[1]]),
                        "logit_d": float(logits[_READOUT[2]]),
                        "R": float(realism(embedding, training)[0]),
                    })
                    members.add(i)

        own = _own_fold(checkpoint, base)
        if base_source is not None:
            candidates = [i for i in range(len(run.dataset))
                          if run.dataset[i].get("source") == base_source]
            if not candidates:
                raise RuntimeError(f"no record in the corpus has base source {base_source!r}")
            base_index = candidates[0]
            if base_index not in members and own in trainings:
                item = run.dataset[base_index]
                kept, _ = clean_base(named_recipe(item["components"]))
                vectors = round_trip_vectors(store, index, kept, run.device)
                if vectors is not None:
                    with torch.no_grad():
                        logits, embedding, _ = encode_recipe(run, *vectors)
                        pool.append({
                            "source": item.get("source", ""),
                            "fold": own,
                            "n_components": len(kept),
                            "logit_p": float(logits[_READOUT[0]]),
                            "logit_s": float(logits[_READOUT[1]]),
                            "logit_d": float(logits[_READOUT[2]]),
                            "R": float(realism(embedding, trainings[own])[0]),
                        })
                    members.add(base_index)
    finally:
        run.model.train(was_training)

    if not pool:
        raise RuntimeError("constraint pool is empty: no held-out positive was encoded")
    base_in_pool = base_source is not None and \
        any(row["source"] == base_source for row in pool)
    values = [v for row in pool
              for v in (row["logit_p"], row["logit_s"], row["logit_d"], row["R"])]
    if not np.all(np.isfinite(values)):
        raise ValueError("constraint pool holds a non-finite logit or R")

    bounds = _pool_bounds(pool)
    thresholds = {
        "checkpoint": str(checkpoint or ""),
        "estimator": THRESHOLD_ESTIMATOR,
        "frame": "judge: kept ingredients re-vectorized through the constraint index "
                 "and re-encoded by the model",
        "dataset_fingerprint": _fingerprint(run.dataset),
        "n_pool": len(pool),
        "n_folds": N_FOLDS,
        "n_train": min(n_train) if n_train else 0,
        "n_skipped": len(skipped),
        "skipped": skipped,
        "base_in_pool": base_in_pool,
        "tau_p": bounds["tau_p"],
        "tau_s": bounds["tau_s"],
        "tau_d": bounds["tau_d"],
        "realism_95": bounds["realism_95"],
        "own_fold": _pool_bounds(
            [row for row in pool if row["fold"] == own]),
        "pool": pool,
        "provenance": "derived-from-data",
    }
    if out is not None:
        with open(out, "w") as f:
            json.dump(thresholds, f, indent=2)
    return thresholds


def _threshold_path(checkpoint):
    stem = checkpoint.stem if hasattr(checkpoint, "stem") else str(checkpoint).rsplit("/", 1)[-1]
    stem = stem[:-4] if stem.endswith(".pth") else stem
    return config.ARTIFACT_DIR / \
        f"constraint_thresholds__{stem or 'run'}__{THRESHOLD_ESTIMATOR}.json"


def thresholds_for(run, store, checkpoint=None, spec=None, index=None, base_source=None):
    """Per-checkpoint thresholds, fitted once and reused; the file is the archive."""
    path = _threshold_path(checkpoint) if checkpoint is not None else \
        config.ARTIFACT_DIR / f"constraint_thresholds__run__{THRESHOLD_ESTIMATOR}.json"
    if path.exists():
        with open(path) as f:
            cached = json.load(f)
        if cached.get("estimator") != THRESHOLD_ESTIMATOR:
            raise ValueError(f"{path} carries estimator {cached.get('estimator')!r}, "
                             f"not {THRESHOLD_ESTIMATOR!r}")
        if cached.get("dataset_fingerprint") != _fingerprint(run.dataset):
            print(f"{path} is stale for this dataset; refitting")
        else:
            return cached
    return fit_thresholds(run, store, spec=spec, checkpoint=checkpoint, out=path,
                          index=index, base_source=base_source)


def readout_violations(logits, thresholds):
    """Scaled violations of the three readout bounds, one nat at a time."""
    logits = logits.reshape(-1)
    p, s, d = (float(logits[22]), float(logits[23]), float(logits[24]))
    return {"readout/proliferation": max(0.0, thresholds["tau_p"] - p),
            "readout/survival": max(0.0, thresholds["tau_s"] - s),
            "readout/differentiation": max(0.0, d - thresholds["tau_d"])}


def realism_violation(r, thresholds):
    return max(0.0, (float(r) - thresholds["realism_95"]) / thresholds["realism_95"])
