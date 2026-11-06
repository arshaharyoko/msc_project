"""
Filesystem layout, hyperparameters and ablation flags.

Application (tier 3), no modules may import.
"""

import os
from pathlib import Path

__all__ = [
    "ARTIFACT_DIR",
    "BATCH_SIZE",
    "BETA_END",
    "BETA_START",
    "COMPONENT_PRICES",
    "COST_DELTA",
    "COST_BOUNDED",
    "COST_CONSIST",
    "COST_CONSIST_MODE",
    "COST_INPUT",
    "COST_PREGATE",
    "COST_RECIPE",
    "COST_STATS",
    "DATABASE_JSON",
    "DATAPOISON_JSON",
    "DATA_DIR",
    "DEP_MARGIN",
    "DEP_W",
    "EPOCHS",
    "FIGURE_DIR",
    "GRAPH_NODES_JSON",
    "HIDDEN_DIM",
    "INPUT_DIM1",
    "INPUT_DIM2",
    "LATENT_DIM",
    "LR",
    "PER_PROTEIN_H5",
    "PRICE_TABLE",
    "PROJECT_ROOT",
    "PUBCHEM_INDEX",
    "QM9_CHUNK_DIR",
    "QM9_DIR",
    "RECON_T_START",
    "SCALING_STATS",
    "SCHEDULE",
    "T",
    "TIME_DIM",
    "TSVD_DIM",
    "TSVD_K",
    "TSVD_UNIPROT",
    "T_SAMPLES",
    "UNIPROT_INDEX",
    "checkpoint_path",
    "diffusion_run",
    "vae_run",
]

PROJECT_ROOT = Path(os.environ.get("GROWTHMEDIUM_ROOT", Path(__file__).resolve().parents[1]))

DATA_DIR = PROJECT_ROOT / "data"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
FIGURE_DIR = ARTIFACT_DIR / "figures"

# Recomputed with a per-component cost in USD/L (ADR-001); the cost-free set is at
# data/database_preprocessed.json.
DATABASE_JSON = DATA_DIR / "database_preprocessed_cost.json"
DATAPOISON_JSON = DATA_DIR / "datapoison_preprocessed.json"
GRAPH_NODES_JSON = DATA_DIR / "graph_nodes_embedded.json"
PER_PROTEIN_H5 = DATA_DIR / "per-protein.h5"
QM9_DIR = DATA_DIR / "qm9"
QM9_CHUNK_DIR = DATA_DIR / "qm9_descriptor_chunks"

SCALING_STATS = ARTIFACT_DIR / "scaling_stats.pt"
TSVD_UNIPROT = ARTIFACT_DIR / "tsvd_uniprot.pt"
PUBCHEM_INDEX = ARTIFACT_DIR / "pubchem_index.bin"
UNIPROT_INDEX = ARTIFACT_DIR / "uniprot_index.bin"
COST_STATS = ARTIFACT_DIR / "cost_stats.pt"                 # ln(cost) mean / std
# The single reviewed price source; component_prices.json is generated from it (ADR-022)
PRICE_TABLE = DATA_DIR / "price_table.csv"
COMPONENT_PRICES = ARTIFACT_DIR / "component_prices.json"   # component -> USD/mg

BATCH_SIZE = 1
LR = 1e-3
EPOCHS = 16
LATENT_DIM = 32
INPUT_DIM1 = 1024 + 1  # +1 for concentration
INPUT_DIM2 = 1024 + 1
HIDDEN_DIM = 128
TSVD_K = 32
TSVD_DIM = 32  # target UniProt projection dimension

PATHWAY_ALPHA = 1e+00
KL_BETA = 1e-01
MMD_BETA = 1e-05
GATE_GAMMA = 1e-02
RECON_W1 = 1
# 1 now: 1e3 compensated for rows of norm ~0.01 under the old encoding (ADR-012)
RECON_W2 = 1e+00
MMD_POOL = 16
# Cost readout off the latent, per component, in standardized ln(USD/L) (ADR-002)
# The cost loss now regresses ln USD/L rather than standardised units, which made
# the same weight about 19.2x heavier (ADR-029). Both cost weights are divided by
# that factor so the tuned balance between terms survives the unit change.
COST_UNIT_RESCALE = 19.2
COST_DELTA = 1e+01 / COST_UNIT_RESCALE
# Off: cost is a target only, so the latent must infer it (ADR-003)
COST_INPUT = False
# Head reads the pre-gate code and is squashed into the observed ln-cost range (ADR-002)
COST_PREGATE = True
COST_BOUNDED = True
# Weight of pricing what each sampled code decodes to; 0 turns it off (ADR-007)
COST_CONSIST = 5.0
# "level": head matches the decoded price; "delta": head change matches decoded change
COST_CONSIST_MODE = "delta"
# Weight of the recipe-total MSE next to the per-component MSE (ADR-008)
COST_RECIPE = 0.3 / COST_UNIT_RESCALE

# Task 1 augmentation (handoff 2.2): the depletion hinge's weight and margin on the
# proliferation logit. Read only when AblationSpec.augment is on.
DEP_W = 1.0
DEP_MARGIN = 3.0

T = 1000
BETA_START = 1e-5
BETA_END = 2e-2
SCHEDULE = "cosine"
TIME_DIM = 64
T_SAMPLES = 4
RECON_T_START = 100


def _flag(name):
    return os.environ.get(name, "0") == "1"


def _run_tag(active):
    tags = [tag for tag, on in active if on]
    return "+".join(tags) if tags else "baseline"


def checkpoint_path(prefix, run_tag):
    return ARTIFACT_DIR / f"{prefix}__{run_tag}.pth"


def vae_run():
    """Ablation flags, run tag and checkpoint path for the MVRRAE_GV track."""
    ablate_rgcn = _flag("ABLATE_RGCN")
    ablate_svd = _flag("ABLATE_SVD")  # reduced-rank SVD -> standard VAE
    ablate_mmd = _flag("ABLATE_MMD")
    ablate_gate = _flag("ABLATE_GATE")
    ablate_cost = _flag("ABLATE_COST")
    run_tag = _run_tag([("no_rgcn", ablate_rgcn), ("no_svd", ablate_svd), ("no_mmd", ablate_mmd),
                        ("no_gate", ablate_gate), ("no_cost", ablate_cost)])
    return {
        "ablate_rgcn": ablate_rgcn,
        "ablate_svd": ablate_svd,
        "ablate_mmd": ablate_mmd,
        "run_tag": run_tag,
        "checkpoint": checkpoint_path("mvrrae_gv", run_tag),
        "pathway_alpha": 0.0 if ablate_rgcn else PATHWAY_ALPHA,
        "div_beta": KL_BETA if ablate_mmd else MMD_BETA,
        "gate_gamma": 0.0 if ablate_gate else GATE_GAMMA,
        "recon_w1": RECON_W1,
        "recon_w2": RECON_W2,
        "ablate_cost": ablate_cost,
        "cost_delta": 0.0 if ablate_cost else COST_DELTA,
        "cost_input": COST_INPUT,
    }


def diffusion_run():
    """Ablation flags, run tag and checkpoint path for the MDDPM_GV track."""
    ablate_rgcn = _flag("ABLATE_RGCN")
    run_tag = _run_tag([("no_rgcn", ablate_rgcn)])
    return {
        "ablate_rgcn": ablate_rgcn,
        "run_tag": run_tag,
        "checkpoint": checkpoint_path("mddpm_gv", run_tag),
        "pathway_alpha": 0.0 if ablate_rgcn else 1e-2,
    }
