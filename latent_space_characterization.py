"""
Characterize the latent space of the four best MVRRAE_GV variants.

Takes the winning configuration for each of MMD/KL x with/without SVD, retrains it,
and measures holes in the aggregate posterior, surviving width, what the code still
reconstructs, the drift of q(z) from the prior, and the robustness of the hole count.

Best configurations come from artifacts/hipo_sweep.json when hipo.py has been run,
ranked by surviving width and broken by reconstruction; otherwise the config.py
defaults.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from src import benchmark, config, vae
from src.modules.dataset import collate_fn, safe_stack
from src.modules.mvrrae_gv import kl, loss_reg, mmd
from src.vector_store import VectorStore

__all__ = ["main", "best_configs", "characterize", "train_variant"]

# (label, ABLATE_MMD, ABLATE_SVD)
VARIANTS = (
    ("MMD + tSVD", "0", "0"),
    ("MMD", "0", "1"),
    ("KL + tSVD", "1", "0"),
    ("KL", "1", "1"),
)

SEED = 67
TAU_FLOOR = 0.1
SIGMA_REF = 16.0            # loss_reg defaults sigma=1, where the MMD gradient underflows

SWEEP_JSON = "hipo_sweep.json"
CLUSTER_JSON = "latent_cluster_recipes.json"

N_CLUSTER_DIMS = 4          # dimensions drawn in the corner plot
K_RANGE = range(2, 13)      # k-means candidates, picked by silhouette


def best_configs(path=None):
    """
    Winning coefficients per variant, from the hipo sweep when it exists.

    RECON_W2 is picked first and held common to every arm, on modality balance alone: the
    grid value whose held out nmse_x1 and nmse_x2 sit closest together. Picking it per arm
    on `active` would let a nuisance knob move the collapse metric independently of the
    divergence. Within that weight the arms are ranked on surviving width and broken by
    reconstruction. Falls back to the config.py defaults per variant.
    """
    path = (config.ARTIFACT_DIR / SWEEP_JSON) if path is None else path
    fallback = {
        "pathway_alpha": config.PATHWAY_ALPHA,
        "gate_gamma": config.GATE_GAMMA,
        "tsvd_k": config.TSVD_K,
        "recon_w2": config.RECON_W2,
    }

    if not path.exists():
        print(f"no {path.name}, falling back to config.py defaults "
              f"(run hipo.py to pick these properly)")
        return [
            {"label": label, "ablate_mmd": ablate_mmd, "ablate_svd": ablate_svd,
             "div_beta": config.KL_BETA if ablate_mmd == "1" else config.MMD_BETA,
             **fallback}
            for label, ablate_mmd, ablate_svd in VARIANTS
        ]

    rows = json.load(open(path))["results"]

    weights = sorted({r["recon_w2"] for r in rows if "recon_w2" in r and "balance" in r})
    if weights:
        def imbalance(w):
            at_w = [abs(np.log(max(r["balance"], 1e-12)))
                    for r in rows if r["recon_w2"] == w]
            return float(np.median(at_w)) if at_w else float("inf")

        matched_w2 = min(weights, key=imbalance)
        print("RECON_W2 selected on modality balance, common to all arms:")
        for w in weights:
            mark = "  <- chosen" if w == matched_w2 else ""
            print(f"    w2={w:9.3e}   median |log nmse_x2/nmse_x1| = {imbalance(w):.4f}{mark}")
        rows = [r for r in rows if r["recon_w2"] == matched_w2]
    else:
        matched_w2 = config.RECON_W2
        print(f"sweep predates the recon_w2 axis, holding config default {matched_w2:.3e}")

    chosen = []
    for label, ablate_mmd, ablate_svd in VARIANTS:
        arm = [r for r in rows if r["label"] == label]
        if not arm:
            print(f"{label}: absent from the sweep, using defaults")
            chosen.append({"label": label, "ablate_mmd": ablate_mmd,
                           "ablate_svd": ablate_svd,
                           "div_beta": config.KL_BETA if ablate_mmd == "1"
                           else config.MMD_BETA, **fallback, "recon_w2": matched_w2})
            continue

        best = max(arm, key=lambda r: (r["active"], -r["recon"]))
        chosen.append({"label": label, "ablate_mmd": ablate_mmd, "ablate_svd": ablate_svd,
                       "pathway_alpha": best["pathway_alpha"], "div_beta": best["div_beta"],
                       "gate_gamma": best["gate_gamma"], "tsvd_k": best["tsvd_k"],
                       "recon_w2": matched_w2})
    return chosen


def train_variant(cfg):
    """Build and train one winning configuration."""
    os.environ["ABLATE_MMD"] = cfg["ablate_mmd"]
    os.environ["ABLATE_SVD"] = cfg["ablate_svd"]
    config.TSVD_K = cfg["tsvd_k"]
    config.PATHWAY_ALPHA = cfg["pathway_alpha"]
    config.GATE_GAMMA = cfg["gate_gamma"]
    config.RECON_W2 = cfg["recon_w2"]
    config.MMD_BETA = config.KL_BETA = cfg["div_beta"]
    config.COST_DELTA = cfg.get("cost_delta", config.COST_DELTA)

    print(f"\n===== {cfg['label']} =====")
    print(f"  alpha={cfg['pathway_alpha']:.3e} beta={cfg['div_beta']:.3e} "
          f"gamma={cfg['gate_gamma']:.3e} k={cfg['tsvd_k']}")

    torch.manual_seed(SEED)
    run = vae.build()
    store = VectorStore.from_run(run)
    try:
        # Names for decoded centroids. The index maps vectors to compounds and does not depend on the model
        store.load()
    except Exception as error:
        print(f"  no vector index loaded ({error}), centroids will decode unnamed")
    model, optimizer, device = run.model, run.optimizer, run.device

    n_test = max(5, int(0.1 * len(run.dataset)))
    gen = torch.Generator().manual_seed(SEED)
    train_set, test_set = random_split(
        run.dataset, [len(run.dataset) - n_test, n_test], generator=gen
    )
    train_loader = DataLoader(
        train_set, batch_size=run.batch_size, shuffle=True, collate_fn=collate_fn
    )
    # Held out, unshuffled so the characterization is reproducible
    test_loader = DataLoader(test_set, run.batch_size, shuffle=False, collate_fn=collate_fn)

    tau = 1.0
    pooled_loss, pooled_codes, pooled_prior, n_pooled = 0.0, [], [], 0

    def flush_pool():
        nonlocal pooled_loss, pooled_codes, pooled_prior, n_pooled
        total = pooled_loss
        if pooled_codes:
            total = total + run.div_beta * mmd(
                torch.cat(pooled_codes), torch.cat(pooled_prior))

        (total / n_pooled).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        optimizer.zero_grad()
        step_loss = total.item()
        pooled_loss, pooled_codes, pooled_prior, n_pooled = 0.0, [], [], 0
        return step_loss

    for epoch in range(run.epochs):
        model.train()
        running_loss = 0.0
        batch_count = 0.0
        optimizer.zero_grad()

        for batch in train_loader:
            x1_batch, x2_batch = store.vectorize_batch(batch, device)

            if not any(x1_batch) or not any(x2_batch):
                continue

            x1_in = safe_stack(x1_batch, dim=run.input_dim1)
            x2_in = safe_stack(x2_batch, dim=run.input_dim2)
            real_prob = float(batch["proliferation"][0])
            x1, x2, divergence_args, pathway_prob, gate, cost_pred = model(
                x1_in, x2_in, run.adjacency_matrices, tau=tau
            )

            # beta=0.0 defers the divergence to the pooled step below
            pooled_loss = pooled_loss + loss_reg(
                x1, x2, x1_in, x2_in, divergence_args, real_prob, pathway_prob, gate,
                alpha=run.pathway_alpha, beta=0.0, gamma=run.gate_gamma,
                sigma=SIGMA_REF, recon_w1=run.recon_w1, recon_w2=run.recon_w2,
                ablate_mmd=run.ablate_mmd, cost_pred=cost_pred,
                **store.cost_kwargs(batch, x1, x2, run, x1_in, x2_in, tau))

            if run.ablate_mmd:
                pooled_loss = pooled_loss + run.div_beta * kl(*divergence_args)
            else:
                Yk, Yk_pz = divergence_args
                pooled_codes.append(Yk.reshape(-1, Yk.size(-1)))
                pooled_prior.append(Yk_pz.reshape(-1, Yk_pz.size(-1)))

            batch_count += 1
            n_pooled += 1
            if n_pooled < config.MMD_POOL:
                continue

            running_loss += flush_pool()

        if n_pooled:
            running_loss += flush_pool()

        tau = max(TAU_FLOOR, tau * 0.95)
        avg_loss = running_loss / batch_count if batch_count > 0 else 0.0
        print(f"Epoch [{epoch + 1}/{run.epochs}], Loss: {avg_loss:.4f}  [tau={tau:.4f}]")

    torch.save(model.state_dict(), run.checkpoint)

    return run, store, test_loader, tau


def cluster_latents(Z, k_range=K_RANGE, seed=SEED):
    """
    k-means over the full code, k chosen by silhouette.

    Clustering runs on every dimension, not just the plotted ones, so a centroid
    is a real point in the latent space and decodes to a genuine recipe. The
    four dimension corner plot is only a window onto the same labels.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    best = None
    for k in k_range:
        if k >= len(Z):
            break
        model = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)
        score = silhouette_score(Z, model.labels_,
                                 sample_size=min(2000, len(Z)), random_state=seed)
        if best is None or score > best[0]:
            best = (score, k, model)
    return best


def decode_centroid(run, store, centroid):
    """
    Read a latent point back as a recipe entry.

    The code is per component, so one centroid is one ingredient. It is decoded
    through both heads, giving the nearest PubChem compound and the nearest
    UniProt protein for that region, and the cluster's own modality mix says
    which of the two to believe.
    """
    z = torch.as_tensor(np.asarray(centroid), dtype=torch.float32, device=run.device)
    z = z.view(1, 1, -1).repeat(1, 2, 1)

    with torch.no_grad():
        x1_hat, x2_hat = run.model.decoder(z, 1, 1)

    name1, conc1 = store.vec_lookup(x1_hat[0, 0], run.input_dim1, run.input_dim2, source="pubchem")
    name2, conc2 = store.vec_lookup(x2_hat[0, 0], run.input_dim1, run.input_dim2, source="uniprot")
    return {
        "pubchem": {"component": name1, "concentration": float(conc1)},
        "uniprot": {"component": name2, "concentration": float(conc2)},
    }


def characterize(cfg, run, store, loader, tau=TAU_FLOOR):
    """Collect everything the six views need from one trained model."""
    from scipy.stats import kstest

    model, device = run.model, run.device
    codes, sources, recon_all, recon_x1, recon_x2 = [], [], [], [], []
    scale_x1, scale_x2 = [], []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            x1_batch, x2_batch = store.vectorize_batch(batch, device)

            if not any(x1_batch) or not any(x2_batch):
                continue

            x1_in = safe_stack(x1_batch, dim=run.input_dim1)
            x2_in = safe_stack(x2_batch, dim=run.input_dim2)

            x1_hat, x2_hat, _, _, _, _ = model(x1_in, x2_in, run.adjacency_matrices, tau=tau)
            e1 = F.mse_loss(x1_hat, x1_in).item()
            e2 = F.mse_loss(x2_hat, x2_in).item()
            recon_x1.append(e1)
            recon_x2.append(e2)
            recon_all.append(e1 + e2)
            scale_x1.append(x1_in.pow(2).mean().item())
            scale_x2.append(x2_in.pow(2).mean().item())

            # Noise free, the only version collapse is visible in
            Zc = benchmark.mean_code(run, x1_in, x2_in, tau)
            codes.append(Zc.reshape(-1, Zc.size(-1)).cpu())
            
            # The code is per component. The encoder concatenates [pubchem, uniprot] along that axis.
            n1 = x1_in.size(1)
            sources.append(np.array([True] * n1 + [False] * (Zc.size(1) - n1)))

    Z = torch.cat(codes)
    is_pubchem = np.concatenate(sources)
    lam = benchmark.latent_spectrum(Z)
    codes_np = Z.numpy()
    projection, eigenvalues, scales = benchmark.project_active(codes_np)
    holes = benchmark.find_holes(projection)

    # The same hole count in the frame top_variance_dims would have picked. The
    # prior is rotation invariant so both frames are legal, and this says what
    # the rotation is worth rather than asserting it.
    top_dims = benchmark.top_variance_dims(codes_np, N_CLUSTER_DIMS)
    raw_projection, raw_scales = benchmark.project_raw(codes_np, top_dims[:2])
    raw_holes = benchmark.find_holes(raw_projection)
    var_top2_pca, var_top2_raw = benchmark.variance_share(
        codes_np, eigenvalues, top_dims[:2])

    score, k, kmeans = cluster_latents(codes_np)

    return {
        "label": cfg["label"],
        "divergence": "KL" if cfg["ablate_mmd"] == "1" else "MMD",
        "codes": codes_np,
        "is_pubchem": is_pubchem,
        "top_dims": top_dims,
        "cluster_labels": kmeans.labels_,
        "centroids": kmeans.cluster_centers_,
        "k": int(k),
        "silhouette": float(score),
        "svd": cfg["ablate_svd"] == "0",
        "projection": projection,
        "eigenvalues": eigenvalues,
        "holes": holes,
        "hole_fraction": holes["hole_fraction"],
        "bandwidth": holes["bandwidth"],
        "raw_projection": raw_projection,
        "raw_holes": raw_holes,
        "raw_hole_fraction": raw_holes["hole_fraction"],
        "var_top2_pca": var_top2_pca,
        "var_top2_raw": var_top2_raw,
        "active": benchmark.active_directions(lam),
        "pr": float(lam.sum().pow(2) / lam.pow(2).sum().clamp_min(1e-12)),
        "d_z": int(lam.numel()),
        "recon_per_recipe": np.array(recon_all),
        "recon_median": float(np.median(recon_all)),
        "recon_x1": float(np.mean(recon_x1)),
        "recon_x2": float(np.mean(recon_x2)),
        "nmse_x1": float(np.mean(recon_x1) / max(np.mean(scale_x1), 1e-12)),
        "nmse_x2": float(np.mean(recon_x2) / max(np.mean(scale_x2), 1e-12)),
        # Mean KS against the prior over the two active directions
        "ks": float(np.mean([kstest(projection[:, d], "norm").statistic
                             for d in range(projection.shape[1])])),
        # The prior has unit width, this says how far off the code's scale is
        "scale_ratio": float(np.mean(scales) if np.mean(scales) >= 1
                             else 1 / max(np.mean(scales), 1e-9)),
    }


def report(characterizations):
    print("\n============================")
    header = (f"{'variant':<14} {'live':>5} {'PR':>6} {'holes':>7} {'KS':>6} "
              f"{'recon':>8} {'x1':>7} {'x2':>7} {'nx1':>6} {'nx2':>6}")
    print(header)
    print("-" * len(header))
    for item in characterizations:
        print(f"{item['label']:<14} {item['active']:>2d}/{item['d_z']:<2d} {item['pr']:6.2f} "
              f"{item['hole_fraction']:6.0%} {item['ks']:6.3f} "
              f"{item['recon_median']:8.3f} {item['recon_x1']:7.3f} {item['recon_x2']:7.3f} "
              f"{item['nmse_x1']:6.3f} {item['nmse_x2']:6.3f}")

    print("\nclustering (k by silhouette, over every dimension)")
    for item in characterizations:
        print(f"  {item['label']:<12} k={item['k']:<2d} silhouette={item['silhouette']:.3f}  "
              f"top dims {[int(d) for d in item['top_dims']]}")

    print("\nprojection choice (top-2 PCs vs top-2 raw dims)")
    for item in characterizations:
        print(f"  {item['label']:<12} variance {item['var_top2_pca']:5.1%} vs "
              f"{item['var_top2_raw']:5.1%}   holes {item['hole_fraction']:5.1%} vs "
              f"{item['raw_hole_fraction']:5.1%}   kernel sd {item['bandwidth']:.3f}")

    fewest = min(characterizations, key=lambda r: r["hole_fraction"])
    widest = max(characterizations, key=lambda r: r["active"])
    tightest = min(characterizations, key=lambda r: r["recon_median"])
    print(f"\nfewest holes:   {fewest['label']} at {fewest['hole_fraction']:.0%}")
    print(f"widest latent:  {widest['label']} at {widest['active']}/{widest['d_z']}")
    print(f"best recon:     {tightest['label']} at {tightest['recon_median']:.3f}")
    print("============================")


def save_results(characterizations, filename="latent_characterization.json"):
    """Persist the scalars so the six views can be redrawn without retraining."""
    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = config.ARTIFACT_DIR / filename
    keep = ("label", "divergence", "svd", "active", "d_z", "pr", "hole_fraction",
            "raw_hole_fraction", "var_top2_pca", "var_top2_raw", "bandwidth",
            "ks", "recon_median", "recon_x1", "recon_x2", "nmse_x1", "nmse_x2",
            "scale_ratio")
    with open(path, "w") as f:
        json.dump([{k: item[k] for k in keep} for item in characterizations], f, indent=2)

    print(f"saved scalars to {path}")
    return path


def save_cluster_recipes(characterizations, run_lookup, filename=CLUSTER_JSON):
    """
    Latent regions written out as recipes.

    Per variant gives every k-means centroid, its coordinate on every dimension, the
    share of its members that came from each modality, and what the decoder
    reads the centroid back as. That is the map from a region of the latent
    space to an actual ingredient.
    """
    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = config.ARTIFACT_DIR / filename
    payload = {
        "note": ("The code is per component, so each centroid is one ingredient, "
                 "not a whole medium. share_pubchem says which decode head to "
                 "trust for that cluster."),
        "seed": SEED,
        "variants": {},
    }

    for item in characterizations:
        run, store = run_lookup[item["label"]]
        clusters = []
        for cid, centroid in enumerate(item["centroids"]):
            members = item["cluster_labels"] == cid
            clusters.append({
                "id": int(cid),
                "size": int(members.sum()),
                "share_pubchem": float(item["is_pubchem"][members].mean()),
                "centroid_by_dim": {f"z{d}": float(v) for d, v in enumerate(centroid)},
                "recipe": decode_centroid(run, store, centroid),
            })

        payload["variants"][item["label"]] = {
            "d_z": int(item["d_z"]),
            "k": item["k"],
            "silhouette": item["silhouette"],
            "top_dims": [int(d) for d in item["top_dims"]],
            "clusters": clusters,
        }

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"saved centroid recipes for {len(characterizations)} variants to {path}")
    return path


def main():
    characterizations, run_lookup = [], {}
    for cfg in best_configs():
        run, store, test_loader, tau = train_variant(cfg)
        characterizations.append(characterize(cfg, run, store, test_loader, tau))
        run_lookup[cfg["label"]] = (run, store)

    report(characterizations)
    save_results(characterizations)
    save_cluster_recipes(characterizations, run_lookup)

    d_z = characterizations[0]["d_z"]
    benchmark.plot_latent_manifold(characterizations)
    benchmark.plot_spectrum_comparison(characterizations, d_z)
    benchmark.plot_reconstruction(characterizations)
    benchmark.plot_prior_marginals(characterizations)
    benchmark.plot_summary(characterizations)
    benchmark.plot_hole_robustness(characterizations)
    benchmark.plot_projection_comparison(characterizations)
    benchmark.plot_latent_manifold(
        characterizations, hole_key="raw_holes",
        filename="latent_manifold__raw_dims.png",
        dim_labels=("top raw dim 1", "top raw dim 2"))

    benchmark.plot_latent_clusters_grid(characterizations, n_dims=N_CLUSTER_DIMS)
    plt.show()

    return characterizations


if __name__ == "__main__":
    main()
