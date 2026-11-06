"""
Constrained medium optimization on the MVRRAE latent (ADR-018).

Encodes the Wu 2016 base, then minimizes the priced cost of the decoded recipe subject
to readout, realism, osmolarity, ion, nutrient, protein-dose and vocabulary constraints.
The solver is an augmented Lagrangian with an L-BFGS inner loop, inside the trust region
of ADR-006. Feasibility is judged only on the round trip: the decoded recipe is named,
re-vectorized, re-encoded, and every constraint is evaluated on that.

Run:
    python predict.py                     # 20 multiplier updates at the default cost weight
    python predict.py --sweep             # the registered cost-weight grid, cheapest feasible kept
    python predict.py --steps 4           # shorter run while scouting the grid
    python predict.py --mode input        # scale the base recipe's concentrations (ADR-010)
"""

import argparse
from functools import lru_cache
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import LBFGS

from src import config, constraints, vae
from src.cost_model import CostModel
from src.modules.dataset import safe_stack
from src.modules.layers import gumbel_sigmoid
from src.modules.mvrrae_gv import recipe_log_cost
from src.vector_store import VectorStore

__all__ = ["BASE_SOURCE", "COST_WEIGHTS", "cost_detail", "encode_base", "input_mode",
           "load_cost_model", "load_prices", "main", "merge_slots", "optimize",
           "optimize_inputs", "optimize_weighted", "realized_cost", "report", "round_trip",
           "save_sweep", "select", "sweep"]

SEED = 67
OPTI_STEP = 100

# Nats the readout logit may drift against its target before the objective pushes
# back. The readout saturates (nodes sit at |logit| 10-31, where sigmoid' is ~1e-6),
# so an affine-squeezed BCE has no usable gradient and cannot tell a logit of
# 14 from one of 31. Holding the readout is a constraint, not something to maximize:
# at 0.0 a move toward target is free and a move away is paid for (ADR-011).
LOGIT_SLACK = 0.0

# Input-space optimizer.
# The decoder cannot reliably name proteins, so a code-space optimum does not start
# from the base recipe; this mode keeps the base components and scales each
# concentration by exp(s) in [s_min, 0].
INPUT_STEPS = 300
INPUT_LR = 0.05

BASE_SOURCE = (
    "Wu, X., Kang, H., Liu, X., Gao, J., Zhao, K., & Ma, Z. - 2016 - Serum and xeno-free, "
    "chemically defined, no-plate-coating-based culture system for mesenchymal stromal "
    "cells from the umbilical cord - DOI: 10.1111/cpr.12279"
)

# Largest move of a_z as a fraction of ||a_z0||. The cost readout is only
# calibrated near encoded codes; unbounded, it promises e^-400 savings (ADR-006).
# None restores the pre-cost unbounded search.
TRUST = 0.3

# Declared decode exclusions, applied in rank order (ADR-021)
DECODE_EXCLUSIONS = config.DATA_DIR / "decode_exclusions.json"

READOUT_NODES = [22, 23, 24]
TARGETS = [1.0, 1.0, 0.0]

# Augmented Lagrangian (ADR-018): 20 multiplier updates, each with a short L-BFGS run.
OUTER_UPDATES = 20
INNER_ITER = 10
MU_INIT = 1.0
MU_GROWTH = 2.0
# The penalty weight grows only while the summed violation stalls, and stops here: with a
# fixed doubling, a violation no dose can reach (an identity the decoder cannot name back)
# would swamp the cost term it is meant to trade against.
MU_MAX = 1e2

# Registered selection grid (ADR-018). Never change it after a cost is seen.
COST_WEIGHTS = np.geomspace(1e-2, 1e1, 7)

# Legacy weighted sum, still the objective benchmark_predict.py measures (ADR-006/011).
WEIGHTS = {
    "pathway_w": 2.0,       # hold the readout at target
    # Pre-cost legacy. Its only effect is to delete proteins, and at the base code it
    # carries 339x the pathway term's gradient (6.57e-2 against 1.94e-4), so the search
    # it drove was "use less protein", not "hold the readout" (ADR-011).
    "uniprot_w": 0.0,
    "anchor_w": 1e-3,       # keep a_z near the encoded latent
    "fidelity_w": 1.0,      # keep the PubChem half intact
    # Cost (ADR-006). The decoded term is the one measured to lower the priced
    # recipe; the latent readout term did not, inside the full objective.
    "cost_w": 0.0,          # hold the latent cost readout, ln total USD/L, down
    "decoded_w": 1.0,       # hold the priced decoded recipe, ln total USD/L, down
}


_COST_MODEL = None


def load_cost_model():
    """The one price source (ADR-022), built once from data/price_table.csv."""
    global _COST_MODEL
    if _COST_MODEL is None:
        _COST_MODEL = CostModel.from_table(config.PRICE_TABLE)
    return _COST_MODEL


def load_prices():
    """component -> USD/mg from CostModel; the vector store still wants a dict."""
    return load_cost_model().price_dict()


def realized_cost(ingredients, model=None):
    """
    USD/L of a named recipe at its concentrations, and how many were unpriced.

    A component without a quoted price adds nothing, so the figure is a floor. A ``model``
    that is a plain price dict is wrapped rather than left to price on its own.
    """
    if isinstance(model, dict):
        model = CostModel.from_prices(model)
    result = (load_cost_model() if model is None else model).recipe_cost(ingredients)
    return result.total_usd_per_l, result.unpriced


def cost_detail(ingredients, model=None):
    """(total USD/L, unpriced count, verified-source cost share) of a named recipe."""
    if isinstance(model, dict):
        model = CostModel.from_prices(model)
    return (load_cost_model() if model is None else model).recipe_cost(ingredients)


def encode_base(run, store):
    """The fixed starting point every optimization departs from."""
    model, device = run.model, run.device

    base = run.dataset[0]
    for item in run.dataset:
        if item["source"] == BASE_SOURCE:
            base = item
            break

    # The optimizer starts from the recorded base minus exact duplicate rows.
    components, removed = constraints.clean_base(base["components"])
    removed = [{"component": c["component"], "concentration": c["concentration"],
                "unit": c.get("unit")} for c in removed]
    base = dict(base, components=components)

    x1_vecs = [
        store.make_x1_vec(comp, device)
        for comp in components
        if comp["component_source"] == "pubchem"
    ]
    x2_vecs = [
        store.make_x2_vec(comp, device)
        for comp in components
        if comp["component_source"] == "uniprot"
    ]
    uniprot_names = [c["component"] for c in components
                     if c["component_source"] == "uniprot"]

    x1 = safe_stack([x1_vecs], dim=run.input_dim1)
    x2 = safe_stack([x2_vecs], dim=run.input_dim2)

    with torch.no_grad():
        Y = model.f(model.encoder(x1, x2))
        alpha, Uk = model.truncated_svd(Y)
        logvar = model.g_I(alpha)
        std = torch.exp(logvar * 0.5)

    a_z0 = (alpha + std * torch.randn_like(std)).detach()

    return {"base": base, "base_removed": removed, "x1": x1, "x2": x2, "Uk": Uk,
            "a_z0": a_z0, "uniprot_names": uniprot_names,
            # The concentration channel is last, or second to last under a cost channel
            "conc_index": -store.n_scalars,
            "cost_mean": store.cost_mean, "cost_std": store.cost_std,
            "decoded_log_cost": store.decoded_log_cost,
            "targets": torch.tensor([TARGETS], dtype=torch.float, device=device)}


def _forward(run, ctx, a_z):
    """Readout logits, both decoded halves and the cost readout for one a_z."""
    model = run.model
    Yk = torch.matmul(ctx["Uk"], a_z)
    gate = gumbel_sigmoid(model.gate_logit(Yk), 1.0, sample=False)
    cost_pred = model.read_cost(Yk if model.cost_pregate else Yk * gate)
    Yk = Yk * gate

    pathway_logit = model.pathway_graph(model.z_c(Yk), model.graph_features_,
                                        run.adjacency_matrices, gate=gate)
    x1_hat, x2_hat = model.decoder(Yk, ctx["x1"].size(1), ctx["x2"].size(1))
    if "anchor" in ctx:
        # The decoder's miss at a_z0 is a bias (ADR-032), so only its response to the move counts
        x1_hat = ctx["x1"] + x1_hat - ctx["anchor"][0]
        x2_hat = ctx["x2"] + x2_hat - ctx["anchor"][1]
    return pathway_logit, x1_hat, x2_hat, (cost_pred, gate)


def _terms(ctx, a_z, pathway_logit, x1_hat, x2_hat, cost_pred):
    """The five unweighted loss terms of the legacy weighted sum."""
    # Hinge on how far each readout logit has drifted from where the encoded recipe
    # started, in the direction that would take it away from its target. Measured on
    # logits because the probabilities are saturated: node 22 reads exactly 1.0 in
    # float32, so nothing downstream of the sigmoid can tell 14 nats from 31.
    logits = pathway_logit[:, READOUT_NODES, 0]
    sign = torch.where(ctx["targets"] > 0.5, 1.0, -1.0)
    drift = sign * (logits - ctx["logit0"])
    return {
        "pathway": F.relu(LOGIT_SLACK - drift).sum(),
        "uniprot": F.softplus(x2_hat[..., ctx["conc_index"]]).sum(),   # log1p(conc) -> 0
        "anchor": (a_z - ctx["a_z0"]).pow(2).sum(),
        "fidelity": F.mse_loss(x1_hat, ctx["x1"], reduction="mean"),
        # ln of the recipe's total USD/L as the latent reads it
        "cost": recipe_log_cost(*cost_pred[:1], ctx["cost_mean"], ctx["cost_std"],
                                cost_pred[1]).sum(),
        # The same quantity read off the decoded recipe instead of the latent
        "decoded": ctx["decoded_log_cost"](x1_hat, x2_hat).sum(),
    }


@lru_cache(maxsize=1)
def protein_floors():
    """Lowest dose of each UniProt component in a recipe that grew cells (ADR-032)."""
    with open(config.DATABASE_JSON) as f:
        records = json.load(f)
    floors = {}
    for record in records:
        if record.get("proliferation") != 1:
            continue
        for c in record["components"]:
            dose = c.get("concentration")
            if c["component_source"] == "uniprot" and dose:
                floors[c["component"]] = min(floors.get(c["component"], dose), dose)
    return floors


def anchor_decode(run, ctx):
    """The raw decoder output at a_z0, kept so ``_forward`` can subtract it (ADR-032)."""
    ctx.pop("anchor", None)
    with torch.no_grad():
        _, x1_hat, x2_hat, _ = _forward(run, ctx, ctx["a_z0"])
    ctx["anchor"] = (x1_hat, x2_hat)


def _displace(a_z0, v, radius):
    """
    a_z0 plus a step that never leaves a ball of ``radius`` around it.

    Radial tanh squashing rather than projection, so L-BFGS sees a smooth objective.
    ``radius`` None is an unbounded step (ADR-006).
    """
    if radius is None:
        return a_z0 + v
    norm = v.norm().clamp_min(1e-12)
    return a_z0 + v * (radius * torch.tanh(norm / radius) / norm)


def _total(weights, terms):
    # A weight missing from an older sweep file counts as off
    return sum(weights.get(f"{name}_w", 0.0) * value for name, value in terms.items())


def _decode_exclusions(path=None):
    """Name -> class map and (pattern, class) pairs from the declared exclusions."""
    with open(path or DECODE_EXCLUSIONS) as f:
        classes = json.load(f)["classes"]
    names, patterns = {}, []
    for cls, entry in classes.items():
        for name in entry.get("names", ()):
            names.setdefault(name, cls)
        patterns.extend((pattern, cls) for pattern in entry.get("patterns", ()))
    return classes, names, patterns


def _decode_slots(run, store, x1_hat, x2_hat, uniprot_names, verbose=False, index=None,
                  n_scalars=1):
    """Name every decoded slot by its rank-1 allowed neighbour (ADR-021)."""
    classes, exclusions, patterns = _decode_exclusions()
    fired = dict.fromkeys(classes, 0)
    decoded, eliminated = [], []
    tied = None
    if index is not None and index.prefer is not None:
        tied = index.nearest_rows(x1_hat[0, :, :-n_scalars])
    for i, vec in enumerate(x1_hat[0]):
        name, scalar = store.vec_lookup(vec, run.input_dim1, run.input_dim2,
                                        source="pubchem", exclusions=exclusions,
                                        patterns=patterns, fired=fired)
        if tied is not None:
            name = index.names[int(tied[i])]
        if name and scalar > 0:
            decoded.append({"component": name, "concentration": scalar, "source": "pubchem"})
            if verbose:
                print(f"  [PubChem]  {name} | concentration: {scalar:.4f} mg/L")

    for i, vec in enumerate(x2_hat[0]):
        name, scalar = store.vec_lookup(vec, run.input_dim1, run.input_dim2,
                                        source="uniprot", exclusions=exclusions,
                                        patterns=patterns, fired=fired)
        if tied is not None:
            name = uniprot_names[i]
        if name and scalar > 0:
            decoded.append({"component": name, "concentration": scalar, "source": "uniprot"})
            if verbose:
                print(f"  [UniProt]  {name} | concentration: {scalar:.4f} mg/L")
        else:
            eliminated.append({"component": uniprot_names[i], "concentration": scalar})
            if verbose:
                print(f"  [UniProt]  {uniprot_names[i]} eliminated "
                      f"(decoded {scalar:.4f} mg/L)")
    return decoded, eliminated, fired


def merge_slots(decoded):
    """One entry per name, doses summed, with the merge count (ADR-021)."""
    ingredients, by_name, merges = [], {}, 0
    for slot in decoded:
        entry = by_name.get(slot["component"])
        if entry is None:
            entry = {"component": slot["component"],
                     "concentration": slot["concentration"],
                     "source": slot["source"], "slots": 1}
            by_name[slot["component"]] = entry
            ingredients.append(entry)
        else:
            entry["concentration"] += slot["concentration"]
            entry["slots"] += 1
            merges += 1
    return ingredients, merges


def base_ingredients(ctx):
    """The base recipe as a named recipe; exact duplicate rows dropped."""
    return constraints.clean_base(constraints.named_recipe(ctx["base"]["components"]))


READOUT_KEYS = ("readout/proliferation", "readout/survival", "readout/differentiation")


def _readout_violations(logits, thresholds):
    """constraints.readout_violations, with a non-finite logit kept as a violation.

    ``max(0.0, tau - nan)`` is 0.0, so an unguarded NaN logit would read as satisfied.
    """
    out = constraints.readout_violations(logits, thresholds)
    flat = logits.reshape(-1)
    for key, node in zip(READOUT_KEYS, READOUT_NODES):
        if not math.isfinite(float(flat[node])):
            out[key] = float("inf")
    return out


def _realism_violation(r, thresholds):
    """constraints.realism_violation, with a non-finite R kept as a violation."""
    if not math.isfinite(float(r)):
        return float("inf")
    return constraints.realism_violation(r, thresholds)


def _verdict(violations):
    """Feasible only when every scaled violation is finite and <= 0 (ADR-018).

    A non-finite entry is renamed to a violation of its own key rather than left to
    compare false against every bound and pass.
    """
    for key, value in violations.items():
        if not math.isfinite(value):
            violations[key] = float("inf")
    return not any(value > 0 for value in violations.values())


def round_trip(run, store, ctx, ingredients, model=None):
    """
    Name -> vectorize -> encode, then the readout and every constraint on that recipe.

    This is the only place feasibility is decided (ADR-018). A recipe that cannot be
    vectorized or encoded is infeasible, not skipped.
    """
    vectors = constraints.round_trip_vectors(store, ctx["index"], ingredients, run.device)
    if vectors is None:
        cost, unpriced, verified = cost_detail(ingredients, model)
        return {"feasible": False, "violations": {"round_trip/unencodable": 1.0},
                "reason": "an encodable recipe needs both a PubChem and a UniProt half",
                "values": {}, "R": float("nan"),
                "realism_95": ctx["thresholds"]["realism_95"],
                "readout_logit": [float("nan")] * 3, "readout": [float("nan")] * 3,
                "realized_cost": cost, "n_unpriced": unpriced, "verified_share": verified,
                "n_components": len(ingredients), "n_unquantified": 0}
    with torch.no_grad():
        logits, embedding, _ = constraints.encode_recipe(run, *vectors)
        r = float(constraints.realism(embedding, ctx["training"])[0])
        rows = constraints.evaluate(ingredients, ctx["ref"])
        violations = dict(rows["violations"])
        violations.update(_readout_violations(logits, ctx["thresholds"]))
        violations["realism"] = _realism_violation(r, ctx["thresholds"])
    cost, unpriced, verified = cost_detail(ingredients, model)
    return {"feasible": _verdict(violations),
            "violations": violations, "values": rows["values"],
            "R": r, "realism_95": ctx["thresholds"]["realism_95"],
            "readout_logit": [float(v) for v in logits[READOUT_NODES]],
            "readout": [float(torch.sigmoid(v)) for v in logits[READOUT_NODES]],
            "realized_cost": cost, "n_unpriced": unpriced, "verified_share": verified,
            "n_components": len(ingredients), "n_unquantified": rows["n_unquantified"]}


def _constraint_state(run, store, spec=None, checkpoint=None, thresholds=None):
    """Bounds, the soft index, the per-checkpoint thresholds and the realism reference."""
    index = constraints.ConstraintIndex(run, store)
    return {"ref": constraints.reference(),
            "index": index,
            "thresholds": thresholds or constraints.thresholds_for(
                run, store, checkpoint, spec, index=index, base_source=BASE_SOURCE),
            "training": constraints.training_embeddings(run, store, spec)}


def _violation_tensors(run, ctx, a_z, thresholds):
    """Decoded-cost objective C and every scaled constraint violation at one a_z."""
    pathway_logit, x1_hat, x2_hat, _ = _forward(run, ctx, a_z)
    h = ctx["index"].violations(x1_hat, x2_hat, ctx["uniprot_names"],
                                -ctx["conc_index"])
    p, s, d = (pathway_logit[0, READOUT_NODES[0], 0], pathway_logit[0, READOUT_NODES[1], 0],
               pathway_logit[0, READOUT_NODES[2], 0])
    h["readout/proliferation"] = F.relu(thresholds["tau_p"] - p)
    h["readout/survival"] = F.relu(thresholds["tau_s"] - s)
    h["readout/differentiation"] = F.relu(d - thresholds["tau_d"])
    Yk = torch.matmul(ctx["Uk"], a_z)
    gate = gumbel_sigmoid(run.model.gate_logit(Yk), 1.0, sample=False)
    distance = constraints.realism((Yk * gate).mean(1), ctx["training"])[0]
    h["realism"] = F.relu(distance - thresholds["realism_95"]) / thresholds["realism_95"]
    return ctx["decoded_log_cost"](x1_hat, x2_hat).sum(), h


def _penalty(h, lam, mu):
    return torch.stack([lam[key] * h[key] + 0.5 * mu * h[key].pow(2) for key in h]).sum()


def optimize(run, store, ctx=None, cost_weight=1.0, steps=OUTER_UPDATES, verbose=True,
             trust=TRUST, state=None, spec=None, checkpoint=None, anchored=False,
             protect=False):
    """
    min C(a_z) + sum_j lambda_j h_j + (mu/2) sum_j h_j^2 over the ADR-018 constraint set.

    ``cost_weight`` scales C only, so the registered sweep moves the trade-off between
    the priced recipe and feasibility without touching a bound. The trust region is the
    one ADR-006 measured.
    """
    model = run.model
    model.eval()
    ctx = encode_base(run, store) if ctx is None else ctx
    if state is not None:
        ctx.update(state)
    elif "index" not in ctx:
        ctx.update(_constraint_state(run, store, spec, checkpoint))
    thresholds = ctx["thresholds"]
    radius = None if trust is None else float(trust * ctx["a_z0"].norm())
    index = ctx["index"]
    if anchored:
        anchor_decode(run, ctx)
        index.prefer = torch.tensor(
            [index.row_of.get(c["component"], -1) for c in ctx["base"]["components"]
             if c["component_source"] == "pubchem"], device=run.device)
    else:
        ctx.pop("anchor", None)
        index.prefer = None
    index.protein_floor = protein_floors() if protect else {}

    with torch.no_grad():
        _, h0 = _violation_tensors(run, ctx, ctx["a_z0"], thresholds)
    lam = {key: 0.0 for key in h0}
    mu = MU_INIT
    previous = sum(float(value) for value in h0.values())

    v = torch.nn.Parameter(torch.zeros_like(ctx["a_z0"]))
    optimizer_z = LBFGS([v], lr=0.1, max_iter=INNER_ITER, history_size=50,
                        tolerance_grad=1e-9, tolerance_change=1e-12)
    last = [None]

    def closure():
        optimizer_z.zero_grad()
        cost, h = _violation_tensors(run, ctx, _displace(ctx["a_z0"], v, radius), thresholds)
        loss = cost_weight * cost + _penalty(h, lam, mu)
        loss.backward()
        last[0] = loss.detach()
        return loss

    if verbose:
        print(f"\n=== cost weight {cost_weight:.3g}, {steps} multiplier updates ===")

    best = None
    start = time.perf_counter()
    for step in range(steps):
        optimizer_z.step(closure)
        with torch.no_grad():
            a_z = _displace(ctx["a_z0"], v, radius)
            cost_now, h = _violation_tensors(run, ctx, a_z, thresholds)
            total = sum(float(value) for value in h.values())
            # The multiplier loop oscillates, so the last iterate is not the best one (ADR-032)
            rank = (total > 1e-9, total if total > 1e-9 else float(cost_now))
            if best is None or rank < best[0]:
                best = (rank, v.detach().clone())
            for key, value in h.items():
                lam[key] = max(0.0, lam[key] + mu * float(value))
            if total > 0.5 * previous:
                mu = min(mu * MU_GROWTH, MU_MAX)
            previous = total
        if verbose and last[0] is not None:
            print(f"  outer [{step + 1:2d}/{steps}] loss {last[0].item():9.4f} "
                  f"mu {mu:9.3g} sum h {total:8.3f}")
    seconds = time.perf_counter() - start

    optimizer_z.zero_grad()
    with torch.no_grad():
        if best is not None:
            v.copy_(best[1])
        a_z = _displace(ctx["a_z0"], v, radius)
        pathway_logit, x1_final, x2_final, cost_final = _forward(run, ctx, a_z)
        logits = pathway_logit[0, READOUT_NODES, 0]
        readout = torch.sigmoid(logits)
        predicted_final = float(recipe_log_cost(cost_final[0], ctx["cost_mean"],
                                                ctx["cost_std"], cost_final[1]).exp())
        cost_start, gate_start = _forward(run, ctx, ctx["a_z0"])[3]
        predicted_start = float(recipe_log_cost(cost_start, ctx["cost_mean"],
                                               ctx["cost_std"], gate_start).exp())
        decoded, eliminated, fired = _decode_slots(run, store, x1_final, x2_final,
                                                   ctx["uniprot_names"], index=ctx["index"],
                                                   n_scalars=-ctx["conc_index"])
        decoded_cost = float(ctx["decoded_log_cost"](x1_final, x2_final).sum())

    ingredients, merges = merge_slots(decoded)
    cost_model = load_cost_model()
    base_items, _ = base_ingredients(ctx)
    start_report = round_trip(run, store, ctx, base_items, cost_model)
    result = round_trip(run, store, ctx, ingredients, cost_model)

    if verbose:
        print(f"  merges: {merges} duplicate name(s); exclusions fired: "
              + ", ".join(f"{cls} {count}" for cls, count in fired.items()))
        _print_constraints("Wu base (cleaned)", start_report)
        _print_constraints("optimized recipe", result)

    base_cost, _ = realized_cost(base_items, cost_model)
    present = {item["component"] for item in ingredients}
    doses = {item["component"]: item["concentration"] for item in ingredients}
    retained = all(doses.get(acc, 0.0) >= low for acc, low in protein_floors().items()
                   if acc in ctx["uniprot_names"])
    return {
        "trust": trust,
        "cost_weight": float(cost_weight),
        "outer": steps,
        "mu_final": mu,
        "step_norm": float((a_z - ctx["a_z0"]).norm()),
        "predicted_cost_start": predicted_start,
        "predicted_cost": predicted_final,
        "decoded_cost": decoded_cost,
        "realized_cost_base": base_cost,
        "realized_cost": result["realized_cost"],
        "n_unpriced": result["n_unpriced"],
        "verified_share": result["verified_share"],
        "readout": [float(p) for p in readout],
        # The probabilities saturate; the logits are where a readout move is visible.
        "readout_logit": [float(v) for v in logits],
        "readout_logit_start": [float(v) for v in start_report["readout_logit"]],
        "n_pubchem": sum(1 for i in ingredients if i["source"] == "pubchem"),
        "n_uniprot": sum(1 for i in ingredients if i["source"] == "uniprot"),
        "n_merges": merges,
        "protein_retained": retained,
        "n_rows_removed": len(ctx.get("base_removed", [])),
        "decode_exclusions": dict(fired),
        "components_removed": sorted({c["component"] for c in ctx["base"]["components"]}
                                     - present),
        "ingredients": ingredients,
        "eliminated": eliminated,
        "removed_duplicates": ctx.get("base_removed", []),
        "start": start_report,
        "round_trip": result,
        "thresholds": thresholds,
        "seconds": seconds,
    }


def _print_constraints(title, report):
    """Violation values of one round trip: raw where a bound is near, scaled if violated."""
    values = report.get("values", {})
    print(f"\n  --- {title} round trip ---")
    print(f"  osmolarity {values.get('osmolarity', float('nan')):8.2f} mOsm/L "
          f"in {constraints.OSMOLARITY_RANGE}")
    for ion, (lo, hi) in constraints.ION_RANGES.items():
        print(f"  {ion:<7} {values.get(ion, float('nan')):8.3f} mM in ({lo}, {hi})")
    pre = report.get("readout_logit_pre")
    if pre is not None:
        print(f"  readout pre-trip  prolif {pre[0]:8.3f}  surv {pre[1]:8.3f}  diff {pre[2]:8.3f}")
    logits = report.get("readout_logit", [float("nan")] * 3)
    print(f"  readout round trip prolif {logits[0]:8.3f}  surv {logits[1]:8.3f}  "
          f"diff {logits[2]:8.3f}")
    print(f"  realism  R {report.get('R', float('nan')):8.4f} <= "
          f"R_95 {report.get('realism_95', float('nan')):.4f}")
    violations = {k: v for k, v in report.get("violations", {}).items() if v > 0}
    if not violations:
        print("  violations none")
    for key, value in sorted(violations.items()):
        number = f"{value:8.4f}" if value >= 1e-3 else f"{value:8.2e}"
        print(f"  violation {key:<50s} {number}")
    note = f" ({report['n_unpriced']} unpriced)" if report.get("n_unpriced") else ""
    print(f"  feasible {report['feasible']} | priced {report.get('realized_cost', 0.0):.4f}"
          f" USD/L{note}")


def optimize_weighted(run, store, ctx=None, weights=None, steps=OPTI_STEP, verbose=True,
                      trust=TRUST):
    """
    The legacy weighted sum of ADR-006/011, kept because benchmark_predict.py measures it.

    Not the ADR-018 problem: it has no explicit constraint and judges nothing on the
    round trip.
    """
    model = run.model
    model.eval()
    ctx = encode_base(run, store) if ctx is None else ctx
    weights = dict(WEIGHTS) if weights is None else weights
    radius = None if trust is None else float(trust * ctx["a_z0"].norm())

    # The hinge is relative to the encoded recipe's own readout, so it is computed once
    # here rather than demanded of every caller that builds a ctx.
    with torch.no_grad():
        ctx["logit0"] = _forward(run, ctx, ctx["a_z0"])[0][:, READOUT_NODES, 0].detach()

    v = torch.nn.Parameter(torch.zeros_like(ctx["a_z0"]))
    optimizer_z = LBFGS([v], lr=0.1, max_iter=20, history_size=50,
                        tolerance_grad=1e-9, tolerance_change=1e-12)

    last_loss = [None]

    def closure():
        optimizer_z.zero_grad()
        a_z = _displace(ctx["a_z0"], v, radius)
        loss = _total(weights, _terms(ctx, a_z, *_forward(run, ctx, a_z)))
        loss.backward()
        last_loss[0] = loss.detach()
        return loss

    if verbose:
        print("\n=== Original Recipe ===")
        for comp in ctx["base"]["components"]:
            source_tag = "[PubChem]" if comp["component_source"] == "pubchem" else "[UniProt] "
            print(f"  {source_tag}  {comp['component']}: {comp['concentration']} mg/L")
        print("========================\n")

    start = time.perf_counter()
    for step in range(steps):
        optimizer_z.step(closure)
        if verbose and step % 10 == 0 and last_loss[0] is not None:
            print(f"Step [{step}], Loss: {last_loss[0].item():.4f}")
    seconds = time.perf_counter() - start

    optimizer_z.zero_grad()
    with torch.no_grad():
        a_z = _displace(ctx["a_z0"], v, radius)
        pathway_logit, x1_final, x2_final, cost_final = _forward(run, ctx, a_z)
        terms = _terms(ctx, a_z, pathway_logit, x1_final, x2_final, cost_final)
        logits = pathway_logit[0, READOUT_NODES, 0]
        readout = torch.sigmoid(logits)
        cost_start, gate_start = _forward(run, ctx, ctx["a_z0"])[3]
        predicted_start = float(recipe_log_cost(cost_start, ctx["cost_mean"],
                                                ctx["cost_std"], gate_start).exp())
        decoded, eliminated, fired = _decode_slots(run, store, x1_final, x2_final,
                                                   ctx["uniprot_names"], verbose=verbose)

    ingredients, merges = merge_slots(decoded)
    if verbose:
        print(f"  merges: {merges} duplicate name(s)")
        print("  exclusions fired: "
              + ", ".join(f"{cls} {count}" for cls, count in fired.items()))

    cost_model = load_cost_model()
    base_items, _ = base_ingredients(ctx)
    base_cost, _ = realized_cost(base_items, cost_model)
    final_cost, unpriced, verified = cost_detail(ingredients, cost_model)
    predicted_final = float(terms["cost"].exp())

    if verbose:
        print("========================")
        print(f"cost, latent readout   {predicted_start:12.4f} -> {predicted_final:12.4f} USD/L")
        print(f"cost, priced recipe    {base_cost:12.4f} -> {final_cost:12.4f} USD/L"
              f"{f'  ({unpriced} unpriced)' if unpriced else ''}")

    return {
        "trust": trust,
        "step_norm": float((a_z - ctx["a_z0"]).norm()),
        "predicted_cost_start": predicted_start,
        "predicted_cost": predicted_final,
        "realized_cost_base": base_cost,
        "realized_cost": final_cost,
        "n_unpriced": unpriced,
        "verified_share": verified,
        "weights": dict(weights),
        "steps": steps,
        "seconds": seconds,
        "loss": float(_total(weights, terms)),
        "terms": {name: float(value) for name, value in terms.items()},
        "readout": [float(p) for p in readout],
        # The probabilities saturate; the logits are where a readout move is visible.
        "readout_logit": [float(v) for v in logits],
        "readout_logit_start": [float(v) for v in ctx["logit0"][0]],
        "n_pubchem": sum(i["source"] == "pubchem" for i in ingredients),
        "n_uniprot": sum(i["source"] == "uniprot" for i in ingredients),
        "n_merges": merges,
        "decode_exclusions": dict(fired),
        "ingredients": ingredients,
        "eliminated": eliminated,
    }


def sweep(run, store, steps=OUTER_UPDATES, trust=TRUST, state=None, spec=None,
          checkpoint=None, anchored=False, protect=False):
    """One optimization per point of the registered cost-weight grid, same start."""
    ctx = encode_base(run, store)
    ctx.update(_constraint_state(run, store, spec, checkpoint) if state is None else state)

    header = (f"{'cost_w':>9} {'USD/L base':>11} {'USD/L':>10} {'feasible':>9} "
              f"{'violations':>10} {'R':>8} {'R_95':>8} {'components':>10}")
    print(f"\nsweeping {len(COST_WEIGHTS)} cost weights x {steps} multiplier updates"
          f"\n\n{header}\n" + "-" * len(header))

    records = []
    for weight in COST_WEIGHTS:
        record = optimize(run, store, ctx, float(weight), steps=steps, verbose=False,
                          trust=trust, anchored=anchored, protect=protect)
        records.append(record)
        result = record["round_trip"]
        print(f"{weight:9.3g} {record['realized_cost_base']:11.3f} "
              f"{record['realized_cost']:10.3f} {str(result['feasible']):>9} "
              f"{sum(v > 0 for v in result['violations'].values()):10d} "
              f"{result['R']:8.4f} {result['realism_95']:8.4f} "
              f"{record['n_pubchem'] + record['n_uniprot']:10d}")
    return records


def select(records):
    """
    The selection rule registered in ADR-018: per seed, the cheapest recipe whose
    round trip is feasible. None when no point of the grid is feasible; that is the
    result, and the bounds stay as they are.
    """
    feasible = [r for r in records if r["round_trip"]["feasible"]]
    return min(feasible, key=lambda r: r["realized_cost"]) if feasible else None


def save_sweep(records, filename="predict_sweep.json"):
    """Persist every optimized recipe and the selection, so the grid need not be rerun."""
    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = config.ARTIFACT_DIR / filename
    chosen = select(records)
    with open(path, "w") as f:
        json.dump({
            "seed": SEED,
            "base_source": BASE_SOURCE,
            "readout_nodes": READOUT_NODES,
            "targets": TARGETS,
            "cost_weights": COST_WEIGHTS.tolist(),
            "selection_rule": "cheapest round-trip-feasible recipe on the cost-weight grid "
                              "(ADR-018)",
            "selected_cost_weight": None if chosen is None else chosen["cost_weight"],
            "runs": records,
        }, f, indent=2)

    print(f"\nsaved {len(records)} optimized recipes to {path}")
    return path


def report(records, chosen=None):
    """The grid, the selection and, when nothing is feasible, the binding constraints."""
    print(f"\n{len(records)} cost weights; feasible "
          f"{sum(r['round_trip']['feasible'] for r in records)}")
    if chosen is None:
        binding = {}
        for record in records:
            for key, value in record["round_trip"]["violations"].items():
                if value > 0:
                    binding[key] = min(binding.get(key, value), value)
        print("no feasible recipe; binding constraints: "
              + (", ".join(f"{k} (best h {v:.4f})" for k, v in binding.items()) or "none"))
        return
    print(f"selected cost_weight={chosen['cost_weight']:.3g}: "
          f"{chosen['realized_cost']:.3f} USD/L of {chosen['realized_cost_base']:.3f}, "
          f"{len(chosen['ingredients'])} components")


def observed_minima(run):
    """Per-component minimum observed concentration over the dataset."""
    # the graded run.dataset holds synthetic depleted twins whose doses are not observations
    with open(config.DATABASE_JSON) as f:
        records = json.load(f)
    mins = {}
    for item in records:
        for c in item["components"]:
            v = c.get("concentration")
            if v:
                mins[c["component"]] = min(mins.get(c["component"], v), v)
    return mins


def input_floors(run, comps, protect=True):
    """
    Per-component lower bound on the scale ``exp(s)``.

    Each component's floor is its minimum observed dataset concentration. UniProt entries
    are protected (frozen at the base concentration) so the optimizer cannot reach the
    unconstrained "use water" optimum, whose proteins the decoder could not name back.
    """
    mins = observed_minima(run)
    lows = []
    for c in comps:
        base = float(c["concentration"] or 1)
        floor = mins.get(c["component"], base * 1e-3)
        s = min(math.log(max(floor / base, 1e-3)), 0.0)
        lows.append(0.0 if (protect and c["component_source"] == "uniprot") else s)
    return torch.tensor(lows, device=run.device)


def _input_vectors(store, comps, scale, device):
    """x1, x2 for the base components at concentration conc * scale (differentiable in scale)."""
    x1, x2 = [], []
    for comp, k in zip(comps, scale):
        cs = torch.as_tensor(store.enc_conc(torch.log1p(float(comp["concentration"] or 1) * k)),
                             device=device).reshape(1)
        base = comp["component_vec"].to(device=device, dtype=torch.float32)
        if comp["component_source"] == "pubchem":
            x1.append(torch.cat([(base - store.targ_mean) / (store.targ_std + 1e-8), cs]))
        else:
            x2.append(torch.cat([store.std_uniprot(base), cs]))
    return torch.stack(x1).unsqueeze(0), torch.stack(x2).unsqueeze(0)


def _input_forward(run, store, comps, scale):
    """(readout logits, realism embedding, latent-head ln USD/L) of the scaled recipe."""
    x1, x2 = _input_vectors(store, comps, scale, run.device)
    logits, embedding, pre = constraints.encode_recipe(run, x1, x2)
    head = recipe_log_cost(run.model.read_cost(pre), store.cost_mean, store.cost_std)
    return logits, embedding, head


def _exact_log_cost(comps, scale, model):
    terms = [math.log(model.unit_price(c["component"]))
             + math.log(float(c["concentration"] or 1)) + torch.log(k)
             for c, k in zip(comps, scale) if model.unit_price(c["component"]) is not None]
    return torch.logsumexp(torch.stack(terms), 0)


def _scaled_ingredients(comps, scale):
    return [{"component": c["component"],
             "concentration": float(c["concentration"] or 1) * float(k),
             "source": "uniprot" if c["component_source"] == "uniprot" else "pubchem"}
            for c, k in zip(comps, scale)]


def _input_cost_and_violations(run, store, comps, scale, model, state):
    """C of the scaled recipe and its scaled violations, differentiable in ``scale``."""
    logits, embedding, _ = _input_forward(run, store, comps, scale)
    x1, x2 = _input_vectors(store, comps, scale, run.device)
    names = [c["component"] for c in comps if c["component_source"] == "uniprot"]
    h = state["index"].violations(x1, x2, names, store.n_scalars)
    ths = state["thresholds"]
    p, s, d = logits[READOUT_NODES[0]], logits[READOUT_NODES[1]], logits[READOUT_NODES[2]]
    h["readout/proliferation"] = F.relu(ths["tau_p"] - p)
    h["readout/survival"] = F.relu(ths["tau_s"] - s)
    h["readout/differentiation"] = F.relu(d - ths["tau_d"])
    distance = constraints.realism(embedding, state["training"])[0]
    h["realism"] = F.relu(distance - ths["realism_95"]) / ths["realism_95"]
    return _exact_log_cost(comps, scale, model), h


def optimize_inputs(run, store, comps, model, state, cost_weight=1.0,
                    steps=OUTER_UPDATES, protect=True):
    """Augmented Lagrangian on the base recipe's scale factors, same constraints (ADR-018)."""
    device = run.device
    lo = input_floors(run, comps, protect=protect)
    # s is projected onto [lo, 0] after every step: clamping inside the forward would
    # zero the gradient exactly at the start, where every scale sits on the upper bound.
    s = torch.zeros(len(comps), device=device, requires_grad=True)
    opt = torch.optim.Adam([s], lr=INPUT_LR)
    lam, mu = None, MU_INIT
    inner = max(1, INPUT_STEPS // steps)

    for outer in range(steps):
        for _ in range(inner):
            opt.zero_grad()
            cost, h = _input_cost_and_violations(run, store, comps, torch.exp(s), model, state)
            if lam is None:
                lam = {key: 0.0 for key in h}
            (cost_weight * cost + _penalty(h, lam, mu)).backward()
            opt.step()
            with torch.no_grad():
                s.copy_(torch.maximum(torch.minimum(s, torch.zeros_like(s)), lo))
        with torch.no_grad():
            _, h = _input_cost_and_violations(run, store, comps, torch.exp(s), model, state)
            for key, value in h.items():
                lam[key] = max(0.0, lam[key] + mu * float(value))
        mu *= MU_GROWTH

    with torch.no_grad():
        scale = torch.exp(s)
        logits_pre, _, head = _input_forward(run, store, comps, scale)
        ingredients = _scaled_ingredients(comps, scale)
        rows = constraints.evaluate(ingredients, state["ref"])
        violations = dict(rows["violations"])
        vectors = constraints.round_trip_vectors(store, state["index"], ingredients, device)
        if vectors is None:
            violations["round_trip/unencodable"] = 1.0
            logits = torch.full_like(logits_pre, float("nan"))
            r = float("nan")
        else:
            # Same verdict as the latent mode (ADR-018): the scaled recipe is named,
            # re-vectorized and re-encoded, and readout and realism are read there.
            logits, embedding, _ = constraints.encode_recipe(run, *vectors)
            r = float(constraints.realism(embedding, state["training"])[0])
            violations.update(_readout_violations(logits, state["thresholds"]))
            violations["realism"] = _realism_violation(r, state["thresholds"])
    cost, unpriced, verified = cost_detail(ingredients, model)
    return {"cost_weight": cost_weight, "protect": protect,
            "readout": [float(torch.sigmoid(v)) for v in logits[READOUT_NODES]],
            "readout_logit": [float(v) for v in logits[READOUT_NODES]],
            "readout_logit_pre": [float(v) for v in logits_pre[READOUT_NODES]],
            "ln_cost0": float(_exact_log_cost(comps, torch.ones_like(scale), model)),
            "ln_cost": float(_exact_log_cost(comps, scale, model)),
            "head_ln_cost": float(head), "realized_cost": cost, "n_unpriced": unpriced,
            "verified_share": verified,
            "violations": violations,
            "values": rows["values"],
            "R": r,
            "realism_95": state["thresholds"]["realism_95"],
            "n_components": len(ingredients), "n_unquantified": rows["n_unquantified"],
            "feasible": _verdict(violations),
            "scale": scale.tolist()}


def input_sensitivity(run, store, comps):
    """Readout change when each UniProt component alone is cut to 1/1000."""
    device = run.device
    with torch.no_grad():
        p0 = torch.sigmoid(_input_forward(run, store, comps,
                                          torch.ones(len(comps), device=device))[0][READOUT_NODES])
        out = {}
        for i, c in enumerate(comps):
            if c["component_source"] != "uniprot":
                continue
            scale = torch.ones(len(comps), device=device)
            scale[i] = 1e-3
            p = torch.sigmoid(_input_forward(run, store, comps, scale)[0][READOUT_NODES])
            out[c["component"]] = [round(float(a - b), 4) for a, b in zip(p, p0)]
        scale = torch.tensor([1e-3 if c["component_source"] == "uniprot" else 1.0
                              for c in comps], device=device)
        p = torch.sigmoid(_input_forward(run, store, comps, scale)[0][READOUT_NODES])
        out["ALL_UNIPROT"] = [round(float(a - b), 4) for a, b in zip(p, p0)]
    return [float(v) for v in p0], out


def input_mode(run, store, model, state):
    """
    Base-recipe input optimization, reported under the same constraints as the latent mode.

    ``protect`` freezes the UniProt entries at their base dose; ``free`` lets them go to
    their observed floor, which is the unconstrained optimum the decoder could not name
    back. Neither row is reported alone (section 1C).
    """
    base = next(item for item in run.dataset if item["source"] == BASE_SOURCE)
    # Same starting recipe as the latent mode: the recorded base minus exact duplicates.
    comps, _ = constraints.clean_base(base["components"])
    # encoder ties, same prior as the anchored latent decode (ADR-032)
    state["index"].prefer = torch.tensor(
        [state["index"].row_of.get(c["component"], -1) for c in comps
         if c["component_source"] == "pubchem"], device=run.device)
    p0, sens = input_sensitivity(run, store, comps)
    row = {"base_source": BASE_SOURCE, "readout0": p0, "sensitivity": sens}
    start = round_trip(run, store, state, constraints.named_recipe(comps), model)
    _print_constraints("Wu base (cleaned)", start)
    for protect in (True, False):
        result = optimize_inputs(run, store, comps, model, state, protect=protect)
        key = "protect" if protect else "free"
        row[key] = {k: v for k, v in result.items() if k != "scale"}
        row[key]["moved"] = {c["component"][:24]: round(k, 4)
                             for c, k in zip(comps, result["scale"]) if k < 0.9}
        print(f"\n  === input {key}: ln cost {result['ln_cost']:.3f} "
              f"from {result['ln_cost0']:.3f} ===")
        _print_constraints(f"input {key}", result)
    print(f"  readout0 {np.round(p0, 3).tolist()} | "
          f"all-protein cut readout delta {sens['ALL_UNIPROT']}")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--sweep", action="store_true",
                        help="run the registered cost-weight grid instead of one recipe")
    parser.add_argument("--steps", type=int, default=OUTER_UPDATES,
                        help="augmented-Lagrangian multiplier updates")
    parser.add_argument("--trust", type=float, default=TRUST,
                        help="max move of a_z as a fraction of ||a_z0||, <= 0 for unbounded")
    parser.add_argument("--mode", choices=("latent", "input"), default="latent",
                        help="latent: decode a code; input: scale the base recipe's concentrations")
    parser.add_argument("--checkpoint", default=None,
                        help="state_dict or ablation checkpoint, default the run's own")
    parser.add_argument("--out", default="predict_sweep.json",
                        help="filename under artifacts/")
    parser.add_argument("--json", nargs="?", const="predict.json", default=None,
                        help="write the single-recipe record as JSON under artifacts/")
    parser.add_argument("--protect-proteins", action="store_true",
                        help="latent mode: keep each protein at or above its lowest effective dose")
    parser.add_argument("--decode", choices=("raw", "anchored"), default="raw",
                        help="latent mode: raw decoder output, or anchored at the base (ADR-032)")
    args = parser.parse_args()

    from src.ablation import AblationSpec, _build

    torch.manual_seed(SEED)
    payload = (torch.load(args.checkpoint, map_location="cpu", weights_only=False)
               if args.checkpoint else None)
    spec = AblationSpec(**payload["spec"]) if payload and "spec" in payload else None
    # A checkpoint is judged on the data and readout it was trained with (ADR-032)
    run = _build(spec) if spec is not None else vae.build()
    store = VectorStore.from_run(run)

    checkpoint = args.checkpoint or run.checkpoint
    if payload is None:
        payload = torch.load(checkpoint, map_location=run.device, weights_only=False)
    # ablation.py checkpoints wrap the state_dict with their spec
    run.model.load_state_dict(payload.get("state_dict", payload))
    if spec is None and "spec" in payload:
        spec = AblationSpec(**payload["spec"])
    trust = None if args.trust is not None and args.trust <= 0 else args.trust
    store.load()
    if not getattr(store, "oracle", None):
        store.fit_price_oracle(run.dataset, load_prices(), run.cost_bounds[0])

    state = _constraint_state(run, store, spec, checkpoint)
    print("thresholds " + ", ".join(f"{k} {v:.4f}" for k, v in state["thresholds"].items()
                                    if isinstance(v, float)))

    if args.mode == "input":
        config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        row = input_mode(run, store, load_cost_model(), state)
        path = config.ARTIFACT_DIR / args.out
        with open(path, "w") as f:
            json.dump(row, f, indent=1)
        print(f"\nsaved {path}")
        return row

    if not args.sweep:
        record = optimize(run, store, steps=args.steps, trust=trust, state=state,
                          anchored=args.decode == "anchored",
                          protect=args.protect_proteins)
        if args.json:
            config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
            path = config.ARTIFACT_DIR / args.json
            with open(path, "w") as f:
                json.dump(record, f, indent=2)
            print(f"\nsaved {path}")
        return record

    records = sweep(run, store, steps=args.steps, trust=trust, state=state,
                    anchored=args.decode == "anchored",
                    protect=args.protect_proteins)
    chosen = select(records)
    report(records, chosen)
    save_sweep(records, args.out)
    return records


if __name__ == "__main__":
    main()
