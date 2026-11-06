"""
Injected per-component biology, in one place.

Every prior the pipeline assumes rather than learns lives here and nowhere else: one
file, one registry, one provenance table. A reader asking "what did we assume rather
than measure?" reads this file.

These tables are a HELD-OUT YARDSTICK, not a constraint on the search. Nothing in the
model reads them except the recovery evaluation and the augmentation generators, which
draw only on the tables below. The search itself stays unconstrained and proposals are
ranked by distance FROM the table (handoff 2.3).

Accession-only. Names and receptor assignments are citable UniProt/ligand-receptor
facts; anything not citable is left out rather than guessed. Node indices are into
``src.modules.graph.NODE_ACCESSIONS``. Provenance: ``textbook`` (citable) or
``assumed`` (our inference, declared wherever it is used).
"""

__all__ = ["NODE_NAMES", "GROWTH_FACTORS", "POISONS", "CARRIERS", "provenance",
           "recovery_pairs", "depletion_pairs", "dilution_ladder"]

# 21 protein nodes, then Selenium, then the three readouts (graph.py comments).
NODE_NAMES = {
    0: "FGFR1", 1: "PDGFR", 2: "EGFR", 3: "IGF1R", 4: "Integrins",
    5: "TGFBR", 6: "Wnt ligand", 7: "FZD/LRP6", 8: "Grb2/SOS", 9: "Ras-GTP",
    10: "RAF", 11: "MEK", 12: "ERK", 13: "PI3K", 14: "Akt", 15: "mTOR",
    16: "FAK", 17: "Dsh", 18: "SMAD2/3", 19: "b-Catenin", 20: "Transferrin",
    21: "Selenium", 22: "Proliferation", 23: "Survival/Qui", 24: "Differentiation",
}

GROWTH_FACTORS = {
    "P01133": {"name": "EGF", "nodes": [2], "direction": "agonist", "provenance": "textbook",
               "cite": "EGF -> EGFR (P00533)"},
    "P05019": {"name": "IGF-1", "nodes": [3], "direction": "agonist", "provenance": "textbook",
               "cite": "IGF1 -> IGF1R (P08069)"},
    "P04085": {"name": "PDGF-A", "nodes": [1], "direction": "agonist", "provenance": "textbook",
               "cite": "PDGFA -> PDGFR (P09619)"},
    "P56704": {"name": "WNT3A", "nodes": [6, 7], "direction": "agonist", "provenance": "textbook",
               "cite": "WNT3A -> FZD/LRP6 (O75581); node 6 is the Wnt ligand itself"},
    "P01137": {"name": "TGF-beta-1", "nodes": [5], "direction": "agonist", "provenance": "textbook",
               "cite": "TGFB1 -> TGFBR (P36897)"},
    "P10600": {"name": "TGF-beta-3", "nodes": [5], "direction": "agonist", "provenance": "textbook",
               "cite": "TGFB3 -> TGFBR (P36897)"},
    "P04004": {"name": "Vitronectin", "nodes": [4], "direction": "agonist", "provenance": "textbook",
               "cite": "VTN is an integrin (P05556) ligand"},
    "Q02297": {"name": "NRG1", "nodes": [2], "direction": "agonist", "provenance": "assumed",
               "cite": "NRG1 signals via ERBB3/4, mapped to the EGFR node; family-level"},
    "P01308": {"name": "Insulin", "nodes": [3], "direction": "agonist", "provenance": "assumed",
               "cite": "INSR is not a graph node; IGF1R cross-reactivity, weak"},
}

# Carriers: no receptor node of their own, kept so provenance can say why.
CARRIERS = {
    "P02787": {"name": "Transferrin", "provenance": "textbook",
               "cite": "iron carrier; graph has a Transferrin node (20) but not a receptor"},
    "P02768": {"name": "Albumin", "provenance": "textbook", "cite": "bulk carrier"},
    "P17936": {"name": "IGFBP-3", "provenance": "textbook",
               "cite": "IGF carrier; modulates IGF1R indirectly"},
    "Q8TAT2": {"name": "FGFBP-3", "provenance": "textbook",
               "cite": "FGF carrier; releases FGF to FGFR, no receptor of its own"},
}

POISONS = {
    "O94907": {"name": "DKK1", "nodes": [7], "direction": "antagonist", "provenance": "textbook",
               "cite": "DKK1 antagonizes LRP6 (O75581)"},
    "P01375": {"name": "TNF", "nodes": [], "direction": "antagonist", "provenance": "textbook",
               "cite": "TNFR1 not a graph node"},
    "P01579": {"name": "IFN-gamma", "nodes": [], "direction": "antagonist", "provenance": "textbook",
               "cite": "IFNGR not a graph node"},
    "P01555": {"name": "Cholera toxin A", "nodes": [], "direction": "antagonist", "provenance": "textbook",
               "cite": "Gsalpha/adenylyl cyclase; not a graph node"},
}

_KNOWN = {**GROWTH_FACTORS, **POISONS, **CARRIERS}


def provenance():
    """The full provenance table: accession -> (name, provenance, cite)."""
    return {acc: (e["name"], e["provenance"], e["cite"]) for acc, e in _KNOWN.items()}


def _downstream(receptor_nodes, adj):
    """Nodes reachable from ``receptor_nodes`` over the union of the relation adjacencies."""
    seen = set(receptor_nodes)
    frontier = set(receptor_nodes)
    while frontier:
        nxt = set()
        for n in frontier:
            for r in range(len(adj)):
                for m, v in enumerate(adj[r][n]):
                    if v and m not in seen:
                        nxt.add(m)
        seen |= nxt
        frontier = nxt
    return sorted(seen - set(receptor_nodes))


def recovery_pairs(adj):
    """
    (accession, receptor_nodes, downstream_nodes) for every entry with a graph node.

    Receptors are excluded from downstream: recovery asks whether ablating the ligand
    moves what is DOWNSTREAM of where it binds, not the node it binds. Pairs with no
    downstream are dropped.
    """
    pairs = []
    for acc, e in {**GROWTH_FACTORS, **POISONS}.items():
        if not e["nodes"]:
            continue
        down = _downstream(e["nodes"], adj)
        if down:
            pairs.append((acc, list(e["nodes"]), down))
    return pairs


def _is_known(accession):
    return accession in _KNOWN


def depletion_pairs(recipe):
    """
    (full, depleted, control) component lists for each growth factor in ``recipe``.

    ``depleted`` drops one growth factor; ``control`` drops the cheapest count-matched
    PubChem row instead, so an arm that moves the control on the same statistic is
    exploiting slot count and not biology. No label is invented -- the pair is only for a
    paired ordering constraint on the logit (``logit(full) > logit(depleted)`` by a margin).
    """
    comps = recipe["components"]
    growth = [c for c in comps if c["component"] in GROWTH_FACTORS]
    pubchem = [c for c in comps if c["component_source"] == "pubchem" and not _is_known(c["component"])]
    cheap = (min(pubchem, key=lambda c: c["cost"] if c.get("cost") is not None else float("inf"))
             if pubchem else None)
    return [{"full": comps, "depleted": [c for c in comps if c is not gf],
             "control": ([c for c in comps if c is not cheap] if cheap is not None else None),
             "dropped": gf["component"]}
            for gf in growth]


def _scaled(comps, k):
    out = []
    for c in comps:
        c2 = dict(c)
        conc = c.get("concentration")
        c2["concentration"] = conc * k if isinstance(conc, (int, float)) else conc
        out.append(c2)
    return out


def dilution_ladder(recipe):
    """
    Concentration-scaled copies of ``recipe``.

    1e-3 and 1e-2 copies (``label`` 0, generator ``diluted``), plus an unscaled
    ``dil_ctrl`` twin (``label`` 1), so "the label" is separable from "three more
    records". Each record is tagged ``synthetic`` with its generator name.
    """
    comps = recipe["components"]
    ladder = [{"components": _scaled(comps, k), "label": 0, "scale": k, "generator": "diluted",
               "synthetic": True} for k in (1e-3, 1e-2)]
    ladder.append({"components": _scaled(comps, 1.0), "label": 1, "scale": 1.0,
                   "generator": "dil_ctrl", "synthetic": True})
    return ladder
