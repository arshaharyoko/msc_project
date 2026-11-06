"""
Train MVRRAE_GV with MMD loss and plot the loss decomposition.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from src import config, vae
from src.modules.dataset import collate_fn, safe_stack
from src.modules.mvrrae_gv import kl, loss_reg, mmd
from src.vector_store import VectorStore

__all__ = ["main"]

# Init, shuffling and the Gumbel/reparameterization noise. Without it every run
# builds a different model, and predict.py cannot be reproduced from a rerun.
SEED = 67

# Weighted per-term contributions to the total, averaged per epoch
COMPONENTS = ("reconstruction", "pathway", "gate", "div", "cost")
RECON_PARTS = ("recon_x1", "recon_x2")
TRACKED = COMPONENTS + RECON_PARTS


def style(ax, title, ylabel):
    ax.grid(False)
    ax.set_title(title, fontsize=13, pad=14, loc="left")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9)
    for spine in ax.spines.values():
        spine.set_linewidth(1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def endpoint_label(ax, x, y, text, color):
    ax.annotate(text, xy=(x, y), xytext=(8, 0), textcoords="offset points",
                color=color, fontsize=9, va="center", clip_on=False)


def plot(run, train_losses, test_losses, comp_history):
    epochs_range = list(range(1, len(train_losses) + 1))
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    okabe_ito = ['#E69F00', '#56B4E9', '#009E73', '#F0E442',
             '#0072B2', '#D55E00', '#CC79A7', '#000000']
    series = {
        "reconstruction": (okabe_ito[0], "Reconstruction (MSE)"),
        "pathway": (okabe_ito[2], f"Pathway BCE x {run.pathway_alpha:g}"),
        "gate": (okabe_ito[4], f"Gate x {run.gate_gamma:g}"),
        "div": (okabe_ito[6], f"Divergence x {run.div_beta:g}"),
        "cost": (okabe_ito[5], f"Cost readout x {run.cost_delta:g}"),
    }

    fig, ax = plt.subplots(figsize=(11, 5.5))
    plotted = 0
    for key in COMPONENTS:
        color, label = series[key]
        y = np.array(comp_history[key], dtype=float)
        if not np.any(y > 0):
            continue  # term switched off (e.g. PATHWAY_ALPHA=0 under ABLATE_RGCN)
        ax.plot(epochs_range, y, color=color, linewidth=2, label=label, solid_capstyle="round")
        endpoint_label(ax, epochs_range[-1], y[-1], label, color)
        plotted += 1

    ax.set_yscale("log")
    style(ax, f"Loss components per epoch  -  {run.run_tag}", "Contribution to total (log scale)")
    if plotted >= 2:
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13),
                  ncol=min(plotted, 4), fontsize=9,
                  handlelength=1.6, columnspacing=2.0)
    fig.subplots_adjust(right=0.76, bottom=0.20)
    fig.savefig(config.FIGURE_DIR / f"loss_components__{run.run_tag}.png",
                dpi=150, bbox_inches="tight")

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(epochs_range, train_losses, color="#2a78d6", linewidth=2, label="Training")
    ax2.plot(epochs_range, test_losses, color="#eb6834", linewidth=2, linestyle="--",
             label="Validation")
    endpoint_label(ax2, epochs_range[-1], train_losses[-1], "Training", "#2a78d6")
    endpoint_label(ax2, epochs_range[-1], test_losses[-1], "Validation", "#eb6834")
    style(ax2, f"Total loss per epoch  -  {run.run_tag}", "Average loss per recipe")
    ax2.legend(frameon=False, loc="upper right", fontsize=9)
    fig2.subplots_adjust(right=0.82)
    fig2.savefig(config.FIGURE_DIR / f"loss_graph__{run.run_tag}.png",
                 dpi=150, bbox_inches="tight")

    plt.show()


def main():
    torch.manual_seed(SEED)
    run = vae.build()
    store = VectorStore.from_run(run)
    model, optimizer, device = run.model, run.optimizer, run.device

    n_test = max(5, int(0.1 * len(run.dataset)))
    gen = torch.Generator().manual_seed(420)

    train_set, test_set = random_split(
        run.dataset, [len(run.dataset) - n_test, n_test], generator=gen
    )
    train_loader = DataLoader(
        train_set, batch_size=run.batch_size, shuffle=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(test_set, run.batch_size, shuffle=True, collate_fn=collate_fn)

    train_losses = []
    test_losses = []
    comp_history = {k: [] for k in TRACKED}
    tau = 1.0
    pooled_loss, pooled_codes, pooled_prior, n_pooled = 0.0, [], [], 0

    def flush_pool():
        nonlocal pooled_loss, pooled_codes, pooled_prior, n_pooled
        total = pooled_loss
        div_part = 0.0
        if pooled_codes:
            div_term = run.div_beta * mmd(
                torch.cat(pooled_codes), torch.cat(pooled_prior))
            total = total + div_term
            div_part = div_term.item()

        (total / n_pooled).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        optimizer.zero_grad()
        step_loss = total.item()
        pooled_loss, pooled_codes, pooled_prior, n_pooled = 0.0, [], [], 0
        return step_loss, div_part

    for epoch in range(run.epochs):
        print(f"Epoch: {epoch}")
        model.train()
        running_loss = 0.0
        test_running_loss = 0.0
        batch_count = 0.0
        test_batch_count = 0.0
        running_parts = {k: 0.0 for k in TRACKED}
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
            loss, parts = loss_reg(x1, x2, x1_in, x2_in, divergence_args, real_prob, pathway_prob, gate,
                                   alpha=run.pathway_alpha, beta=0.0, gamma=run.gate_gamma,
                                   recon_w1=run.recon_w1, recon_w2=run.recon_w2,
                                   ablate_mmd=run.ablate_mmd, return_parts=True,
                                   cost_pred=cost_pred, **store.cost_kwargs(batch, x1, x2, run, x1_in, x2_in, tau))

            for k in TRACKED:
                running_parts[k] += parts[k]

            pooled_loss = pooled_loss + loss
            if run.ablate_mmd:
                kl_term = run.div_beta * kl(*divergence_args)
                pooled_loss = pooled_loss + kl_term
                running_parts["div"] += kl_term.item()
            else:
                Yk, Yk_pz = divergence_args
                pooled_codes.append(Yk.reshape(-1, Yk.size(-1)))
                pooled_prior.append(Yk_pz.reshape(-1, Yk_pz.size(-1)))

            batch_count += 1
            n_pooled += 1

            store.update_db_from_batch(batch, run.input_dim1, run.input_dim2, device)

            if n_pooled < config.MMD_POOL:
                continue

            step_loss, div_part = flush_pool()
            running_loss += step_loss
            running_parts["div"] += div_part
            print(f"Loss: {step_loss:.4f}")

        if n_pooled:
            step_loss, div_part = flush_pool()
            running_loss += step_loss
            running_parts["div"] += div_part

        avg_loss = running_loss / batch_count if batch_count > 0 else 0.0
        train_losses.append(avg_loss)
        for k in TRACKED:
            comp_history[k].append(running_parts[k] / batch_count if batch_count > 0 else 0.0)
        print(f"Epoch [{epoch + 1}/{run.epochs}], Loss: {avg_loss:.4f}")

        r1, r2 = comp_history["recon_x1"][-1], comp_history["recon_x2"][-1]
        print(f"  recon  x1 {r1:.4f}  x2 {r2:.4f}  x1/x2 {r1 / r2:.1f}" if r2 > 0
              else f"  recon  x1 {r1:.4f}  x2 {r2:.4f}  x1/x2 n/a")

        val_i = 1
        model.eval()
        with torch.no_grad():
            for batch_step, batch in enumerate(test_loader):
                x1_batch, x2_batch = store.vectorize_batch(batch, device)

                if not any(x1_batch) or not any(x2_batch):
                    continue

                x1_in = safe_stack(x1_batch, dim=run.input_dim1)
                x2_in = safe_stack(x2_batch, dim=run.input_dim2)
                real_prob = float(batch["proliferation"][0])
                x1, x2, divergence_args, pathway_prob, gate, cost_pred = model(
                    x1_in, x2_in, run.adjacency_matrices, tau=tau
                )
                loss = loss_reg(x1, x2, x1_in, x2_in, divergence_args, real_prob, pathway_prob, gate,
                                alpha=run.pathway_alpha, beta=run.div_beta, gamma=run.gate_gamma,
                                recon_w1=run.recon_w1, recon_w2=run.recon_w2, ablate_mmd=run.ablate_mmd,
                                cost_pred=cost_pred, **store.cost_kwargs(batch, x1, x2, run, x1_in, x2_in, tau))

                test_running_loss += loss.item()
                test_batch_count += 1

                print("============================")
                print(f"Test {val_i}: {loss}")
                print(f"Original Recipe Length: {len(x1_in) + len(x2_in)}")
                print(f"Original Recipe {val_i}:")
                print(x1_in)
                print(x2_in)
                print(f"Reconstructed Recipe {val_i}:")
                print(x1)
                print(x2)
                print("============================")
                val_i += 1

            avg_test_loss = test_running_loss / test_batch_count if test_batch_count > 0 else 0.0
            test_losses.append(avg_test_loss)

        tau = max(0.1, tau * 0.95)
        print(f"  [tau={tau:.4f}]")

    torch.save(model.state_dict(), run.checkpoint)
    store.save()

    plot(run, train_losses, test_losses, comp_history)


if __name__ == "__main__":
    main()
