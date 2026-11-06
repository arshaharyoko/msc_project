"""
Composition root for the MVRRAE_GV track with MMD loss.
"""

from dataclasses import dataclass

import torch
import torch.linalg as la
from torch.optim import Adam
from torch.utils.data import ConcatDataset

from src import config
from src.modules.dataset import GrowthMediumDataset
from src.modules.graph import build_adjacency_matrices, load_node_features
from src.modules.mvrrae_gv import MVRRAE_GV

__all__ = ["VaeRun", "build"]


@dataclass
class VaeRun:
    device: torch.device
    dataset: object
    model: MVRRAE_GV
    optimizer: Adam
    ablate_svd: bool
    ablate_mmd: bool
    adjacency_matrices: torch.Tensor
    targ_mean: torch.Tensor
    targ_std: torch.Tensor
    input_dim1: int
    input_dim2: int
    batch_size: int
    epochs: int
    pathway_alpha: float
    div_beta: float
    gate_gamma: float
    recon_w1: float
    recon_w2: float
    run_tag: str
    checkpoint: object
    # Cost readout (ADR-002/003). cost_mean / cost_std standardize ln(USD/L).
    cost_delta: float = 0.0
    cost_input: bool = False
    ablate_cost: bool = False
    cost_mean: float = None
    cost_std: float = None
    cost_bounds: tuple = None
    cost_consist: float = 0.0
    cost_consist_mode: str = "level"
    cost_recipe: float = 0.0


def build():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    run = config.vae_run()
    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    stats = torch.load(config.SCALING_STATS, map_location=device, weights_only=False)
    targ_mean = stats["y_mean"].to(device)
    targ_std = stats["y_std"].to(device)

    graph_protein_features, graph_node_features = load_node_features(
        config.GRAPH_NODES_JSON, device
    )
    adjacency_matrices = build_adjacency_matrices(device)
    graph_n_relations = adjacency_matrices.size(0)

    dataset = GrowthMediumDataset(json_file=config.DATABASE_JSON)
    if config.DATAPOISON_JSON.exists():
        dataset = ConcatDataset([dataset, GrowthMediumDataset(json_file=config.DATAPOISON_JSON)])
        print(
            f"Merged {config.DATAPOISON_JSON.name} into training set ({len(dataset)} total entries)"
        )


    cost_mean = cost_std = cost_bounds = None
    if config.COST_STATS.exists():
        cost_stats = torch.load(config.COST_STATS, map_location="cpu", weights_only=False)
        cost_mean, cost_std = float(cost_stats["log_mean"]), float(cost_stats["log_std"])
        # Observed standardized range, half a std of slack either side
        cost_bounds = ((cost_stats["log_min"] - cost_mean) / cost_std - 0.5,
                       (cost_stats["log_max"] - cost_mean) / cost_std + 0.5)
    elif run["cost_delta"] > 0 or run["cost_input"]:
        raise FileNotFoundError(f"{config.COST_STATS} missing, rerun preprocess_database.py --merged")

    # The ln-cost channel sits after the concentration channel in both modalities
    input_dim1 = config.INPUT_DIM1 + int(run["cost_input"])
    input_dim2 = config.INPUT_DIM2 + int(run["cost_input"])

    graph_n_static = graph_protein_features.size(0)
    prots = graph_protein_features
    prots = (prots - prots.mean(0)) / (prots.std(0) + 1e-8)

    trainables = graph_node_features[graph_n_static:]
    trainables = trainables / trainables.std()

    graph_node_features = torch.cat([prots, trainables], dim=0)

    model = MVRRAE_GV(
        input_dim1,
        input_dim2,
        config.LATENT_DIM,
        config.HIDDEN_DIM,
        config.TSVD_K,
        graph_node_features,
        graph_n_relations,
        graph_n_static,
    ).to(device)

    # toggles standard-VAE vs reduced-rank reparameterization in forward()
    model.ablate_svd = run["ablate_svd"]
    model.ablate_mmd = run["ablate_mmd"]
    model.cost_pregate = config.COST_PREGATE
    if config.COST_BOUNDED and cost_bounds is not None:
        model.cost_range = cost_bounds
    optimizer = Adam(model.parameters(), lr=config.LR)

    return VaeRun(
        device=device,
        dataset=dataset,
        model=model,
        optimizer=optimizer,
        ablate_svd=run["ablate_svd"],
        ablate_mmd=run["ablate_mmd"],
        adjacency_matrices=adjacency_matrices,
        targ_mean=targ_mean,
        targ_std=targ_std,
        input_dim1=input_dim1,
        input_dim2=input_dim2,
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS,
        pathway_alpha=run["pathway_alpha"],
        div_beta=run["div_beta"],
        gate_gamma=run["gate_gamma"],
        recon_w1=run["recon_w1"],
        recon_w2=run["recon_w2"],
        run_tag=run["run_tag"],
        checkpoint=run["checkpoint"],
        cost_delta=run["cost_delta"],
        cost_input=run["cost_input"],
        ablate_cost=run["ablate_cost"],
        cost_mean=cost_mean,
        cost_std=cost_std,
        cost_bounds=cost_bounds,
        cost_consist=config.COST_CONSIST,
        cost_consist_mode=config.COST_CONSIST_MODE,
        cost_recipe=config.COST_RECIPE,
    )
