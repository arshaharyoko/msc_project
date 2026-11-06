"""
Composition root for the MDDPM_GV track.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.linalg as la
from torch.optim import Adam
from torch.utils.data import ConcatDataset

from src import config
from src.modules.dataset import GrowthMediumDataset
from src.modules.graph import build_adjacency_matrices, load_node_features
from src.modules.mddpm_gv import MDDPM_GV

__all__ = ["DiffusionRun", "build"]

NUM_RELATIONS = 3


@dataclass
class DiffusionRun:
    device: torch.device
    dataset: object
    model: MDDPM_GV
    optimizer: Adam
    adjacency_matrices: torch.Tensor
    targ_mean: torch.Tensor
    targ_std: torch.Tensor
    tsvd_Vh: torch.Tensor | None
    tsvd_x2_mean: torch.Tensor | None
    tsvd_x2_std: torch.Tensor | None
    input_dim1: int
    input_dim2: int
    batch_size: int
    epochs: int
    pathway_alpha: float
    run_tag: str
    checkpoint: object
    T: int
    t_samples: int
    recon_t_start: int


def build():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    run = config.diffusion_run()
    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    stats = torch.load(config.SCALING_STATS, map_location=device, weights_only=False)
    targ_mean = stats["y_mean"].to(device)
    targ_std = stats["y_std"].to(device)

    graph_protein_features, graph_node_features = load_node_features(
        config.GRAPH_NODES_JSON, device
    )
    adjacency_matrices = build_adjacency_matrices(device)

    dataset = GrowthMediumDataset(json_file=config.DATABASE_JSON)
    if config.DATAPOISON_JSON.exists():
        dataset = ConcatDataset([dataset, GrowthMediumDataset(json_file=config.DATAPOISON_JSON)])
        print(
            f"Merged {config.DATAPOISON_JSON.name} into training set ({len(dataset)} total entries)"
        )

    tsvd_Vh = None
    tsvd_x2_mean = None
    tsvd_x2_std = None

    if config.TSVD_UNIPROT.exists():
        ckpt = torch.load(config.TSVD_UNIPROT, map_location=device, weights_only=False)
        if isinstance(ckpt, dict):
            tsvd_Vh = ckpt["Vh"]
            tsvd_x2_mean = ckpt["x2_mean"].to(device)
            tsvd_x2_std = ckpt["x2_std"].to(device)
        else:
            tsvd_Vh = ckpt  # legacy: raw tensor, no norm stats
        print(f"Loaded tSVD projection ({tsvd_Vh.shape[0]}D) from {config.TSVD_UNIPROT}")
    else:
        all_uniprot_vecs = []
        for item in dataset:
            for comp in item["components"]:
                if comp["component_source"] == "uniprot":
                    all_uniprot_vecs.append(
                        comp["component_vec"].numpy()
                        if isinstance(comp["component_vec"], torch.Tensor)
                        else np.array(comp["component_vec"], dtype=np.float32)
                    )

        for feature in graph_protein_features:
            all_uniprot_vecs.append(feature)

        if all_uniprot_vecs:
            X = torch.tensor(np.array(all_uniprot_vecs, dtype=np.float32), device=device)
            X_centered = X - X.mean(dim=0, keepdim=True)
            _, _, Vh = la.svd(X_centered, full_matrices=False)
            tsvd_Vh = Vh[: config.TSVD_DIM].detach()
            X_proj = X @ tsvd_Vh.T  # project raw (uncentered) vectors
            tsvd_x2_mean = X_proj.mean(dim=0).detach()
            tsvd_x2_std = X_proj.std(dim=0).detach()
            torch.save(
                {"Vh": tsvd_Vh, "x2_mean": tsvd_x2_mean, "x2_std": tsvd_x2_std},
                config.TSVD_UNIPROT,
            )
            print(
                f"Fitted tSVD on {len(all_uniprot_vecs)} UniProt vectors to "
                f"{config.TSVD_DIM}D. Saved to {config.TSVD_UNIPROT}"
            )
        else:
            print("No UniProt vectors found; tSVD disabled")

    input_dim2 = (tsvd_Vh.shape[0] if tsvd_Vh is not None else 1024) + 1

    # Fit the log1p(concentration) channel stats the diffusion process standardizes against
    conc_log = [
        np.log1p(comp["concentration"] or 1) for item in dataset for comp in item["components"]
    ]
    conc_mean = float(np.mean(conc_log))
    conc_std = float(np.std(conc_log)) or 1.0
    print(f"Concentration channel: mean={conc_mean:.4f}, std={conc_std:.4f} "
          f"over {len(conc_log)} components")

    # OPEN BUG, preserved deliberately (summary7_8_2026.md §12 bug 7, §14 item 1):
    #   * graph_n_total is passed into the graph_n_relations slot, so the RGCN is
    #     built with 25 relations against a (3, 25, 25) adjacency and raises
    #     `IndexError: index 3 is out of bounds` on the first forward pass. The
    #     VAE track fixes this with `adjacency_matrices.size(0)`; NUM_RELATIONS
    #     below is the value it should receive.
    #   * the tSVD projection block from vae.build() was never ported, so raw
    #     1024-D node features reach the model.
    # Both are modelling changes that invalidate existing checkpoints, so they
    # are left for the author rather than applied during a restructure.
    graph_n_total = graph_node_features.size(0)
    graph_n_static = graph_protein_features.size(0)
    model = MDDPM_GV(
        config.INPUT_DIM1,
        input_dim2,
        config.LATENT_DIM,
        config.HIDDEN_DIM,
        graph_node_features,
        graph_n_total,
        graph_n_static,
        T=config.T,
        beta_start=config.BETA_START,
        beta_end=config.BETA_END,
        schedule=config.SCHEDULE,
        time_dim=config.TIME_DIM,
        conc_mean=conc_mean,
        conc_std=conc_std,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=config.LR)

    return DiffusionRun(
        device=device,
        dataset=dataset,
        model=model,
        optimizer=optimizer,
        adjacency_matrices=adjacency_matrices,
        targ_mean=targ_mean,
        targ_std=targ_std,
        tsvd_Vh=tsvd_Vh,
        tsvd_x2_mean=tsvd_x2_mean,
        tsvd_x2_std=tsvd_x2_std,
        input_dim1=config.INPUT_DIM1,
        input_dim2=input_dim2,
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS,
        pathway_alpha=run["pathway_alpha"],
        run_tag=run["run_tag"],
        checkpoint=run["checkpoint"],
        T=config.T,
        t_samples=config.T_SAMPLES,
        recon_t_start=config.RECON_T_START,
    )
