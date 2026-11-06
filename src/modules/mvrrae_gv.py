"""
Multimodal rank-reducing autoencoder with graph regularization.

The per-component latent code also carries a linear cost readout, one
standardized ln(USD/L) per component, so recipe cost is a direction in the
latent that the latent optimizer in predict.py can push down (ADR-002). It
reads the pre-gate code, is squashed into the observed price range, and is
trained to follow the decoder's priced output near each code (ADR-007).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.linalg as la

from src.modules.layers import MultimodalEncoder, MultimodalDecoder, RGCN, gumbel_sigmoid

__all__ = ["MVRRAE_GV", "cost_loss", "depletion_hinge", "loss_reg", "kl", "mmd",
           "recipe_cost_loss", "recipe_log_cost"]


def _gauss_ker(z, zr, sigma):
    sq_euc_dist = (torch.cdist(z, zr, p=2))**2
    return torch.exp(-sq_euc_dist/(2 * sigma**2))


def _median_heuristic(X, Y):
    """Median heuristic over the pooled sample to set pairwise median at gradient peak as bandwith"""
    # Detached, no grad
    Z = torch.cat([X, Y], dim=0)
    d = torch.cdist(Z, Z, p=2)
    mask = torch.eye(d.size(0), dtype=torch.bool, device=d.device)
    off_diag = d[~mask]
    return off_diag.median().clamp_min(1e-6).detach()


def mmd(X, Y, sigma=None):
    # sigma := bandwith
    # Prevent in-batch exclusivity
    if X.dim() != 2 or Y.dim() != 2:
        raise ValueError(f"MMD needs 2-D (n, d), got {tuple(X.shape)} and {tuple(Y.shape)}")

    sigma = _median_heuristic(X, Y) if sigma is None else sigma
    K_xx = _gauss_ker(X, X, sigma).mean()
    K_yy = _gauss_ker(Y, Y, sigma).mean()
    K_xy = _gauss_ker(X, Y, sigma).mean()
    return K_xx + K_yy - 2*K_xy


def kl(mu, logvar):
    div_kullback_leibler =  0.5 * (logvar.exp() + mu.pow(2) - 1 - logvar)
    return div_kullback_leibler.sum(dim=1).mean()


def cost_loss(cost_pred, cost_target, cost_mask):
    """Masked MSE in standardized ln(cost), averaged over the components that have a price."""
    sq = (cost_pred - cost_target).pow(2) * cost_mask
    return sq.sum() / cost_mask.sum().clamp_min(1.0)


def recipe_cost_loss(cost_pred, cost_target, cost_mask, cost_mean=0.0, cost_std=1.0):
    """
    MSE between predicted and true ln recipe totals in USD/L, over priced components.

    Per-component MSE alone regresses every component toward the mean and overprices recipes
    made of many cheap ones; supervising the total holds the ranking (ADR-008). Both totals
    are destandardized before the logsumexp, because the logsumexp of standardized values is
    not the ln of the summed cost that ``recipe_log_cost`` reports.
    """
    keep = cost_mask > 0
    has = keep.any(dim=-1)
    if not has.any():
        return torch.zeros((), device=cost_pred.device)
    neg = torch.finfo(cost_pred.dtype).min
    pred = torch.logsumexp(torch.where(keep, cost_pred * cost_std + cost_mean, neg), dim=-1)
    true = torch.logsumexp(torch.where(keep, cost_target * cost_std + cost_mean, neg), dim=-1)
    return (pred - true)[has].pow(2).mean()


def recipe_log_cost(cost_pred, cost_mean, cost_std, gate=None):
    """
    ln of the recipe's total USD/L, from the per-component standardized readout.

    Total cost is a sum over components, so in log space it is a logsumexp. With
    ``gate`` the components are weighted by how open their gate is.
    """
    log_cost = cost_pred * cost_std + cost_mean                 # (batch, n_comp)
    if gate is not None:
        log_cost = log_cost + gate.squeeze(-1).clamp_min(1e-12).log()
    return torch.logsumexp(log_cost, dim=-1)                    # (batch,)


def depletion_hinge(logit_full, logit_depleted, margin=3.0):
    """
    Hinge on the proliferation logit: the full recipe must beat its depletion by margin.

    The ordering constraint of the depletion prior (handoff 2.2). No label is invented;
    it asserts only that removing a growth factor lowers the proliferation readout, which
    a convex-average readout cannot express on its own.
    """
    return F.relu(margin - (logit_full - logit_depleted)).mean()


def loss_reg(
        output1, output2,
        target1, target2,
        divergence_args,
        real_prob,
        pathway_logit,
        gate_values,
        alpha=1e-2,
        beta=1e-4,
        gamma=1e-2,
        sigma=1,
        recon_w1=1.0,
        recon_w2=1.0,
        ablate_mmd=False,
        return_parts=False,
        cost_pred=None,
        cost_target=None,
        cost_mask=None,
        cost_mean=0.0,
        cost_std=1.0,
        delta=0.0,
        cost_dec_target=None,
        cost_dec_mask=None,
        consist=0.0,
        cost_anchor_pred=None,
        cost_anchor_target=None,
        recipe_w=0.0,
        node_targets=None,
        ident_w2=0.0,
        ident_tables=None,
        ident_temp=0.1):

    recon_x1 = recon_w1 * F.mse_loss(output1, target1, reduction="mean")
    recon_x2 = recon_w2 * F.mse_loss(output2, target2, reduction="mean")
    reconstruction_loss = recon_x1 + recon_x2

    if node_targets is None:
        # A scalar label is the legacy (p, p, 1-p) task, which is what the graded
        # targets reduce to on a binary record.
        node_targets = torch.tensor([[real_prob, real_prob, 1.0 - real_prob]],
                                    dtype=torch.float, device=pathway_logit.device)
    else:
        node_targets = torch.as_tensor(node_targets, dtype=torch.float,
                                       device=pathway_logit.device)
    output_logits = pathway_logit[:, [22, 23, 24], 0]
    if node_targets.size(0) == 1 and output_logits.size(0) != 1:
        node_targets = node_targets.expand(output_logits.size(0), -1)
    pathway_loss = F.binary_cross_entropy_with_logits(
        output_logits, node_targets, reduction="sum")

    if ident_w2 > 0.0 and ident_tables is not None:
        # Cosine logits of the decoded identity block against the UniProt table.
        # Labels come from the target code under no_grad (handoff 1B); off by default.
        table = F.normalize(ident_tables[-1], dim=-1)
        d_id = table.size(1)
        logits = F.normalize(output2[..., :d_id], dim=-1) @ table.t() / ident_temp
        with torch.no_grad():
            labels = (F.normalize(target2[..., :d_id], dim=-1) @ table.t()).argmax(-1)
        ident_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
    else:
        ident_loss = torch.zeros((), device=pathway_logit.device)

    if beta == 0.0:
        div_loss = torch.zeros((), device=pathway_logit.device)
    elif ablate_mmd:
        div_loss = kl(*divergence_args)
    else:
        Yk, Yk_pz = divergence_args
        div_loss = mmd(Yk.reshape(-1, Yk.size(-1)), Yk_pz.reshape(-1, Yk_pz.size(-1)))

    gate_loss = gate_values.mean()

    if delta == 0.0 or cost_pred is None:
        c_loss = torch.zeros((), device=pathway_logit.device)
    else:
        c_loss = cost_loss(cost_pred, cost_target, cost_mask)
        if recipe_w > 0.0:
            c_loss = c_loss + recipe_w * recipe_cost_loss(
                cost_pred, cost_target, cost_mask, cost_mean, cost_std)
        # Decoder consistency: the head should also price what this sampled code
        # decodes to, so its gradient agrees with the decoder off the data (ADR-007)
        if consist > 0.0 and cost_dec_target is not None:
            if cost_anchor_pred is None:
                # level: the head prices the decoded recipe outright
                c_loss = c_loss + consist * cost_loss(cost_pred, cost_dec_target, cost_dec_mask)
            else:
                # difference: only the head's change from the noise-free code has
                # to follow the decoded price's change, its level stays the data's
                c_loss = c_loss + consist * cost_loss(cost_pred - cost_anchor_pred,
                                                      cost_dec_target - cost_anchor_target,
                                                      cost_dec_mask)

    total = (reconstruction_loss + pathway_loss * alpha + div_loss * beta + gate_loss * gamma
             + c_loss * delta + ident_loss * ident_w2)

    if return_parts:
        return total, {
            "reconstruction": reconstruction_loss.item(),
            "recon_x1": recon_x1.item(),
            "recon_x2": recon_x2.item(),
            "pathway": (pathway_loss * alpha).item(),
            "div": (div_loss * beta).item(),
            "gate": (gate_loss * gamma).item(),
            "cost": (c_loss * delta).item(),
            "ident": (ident_loss * ident_w2).item()
        }

    return total


class MVRRAE_GV(nn.Module):
    def __init__(
            self, input_dim1, input_dim2,
            latent_dim, hidden_dim, tsvd_k,
            graph_features, graph_n_relations, graph_n_static):
        super().__init__()
        self.encoder = MultimodalEncoder(input_dim1, input_dim2, input_dim1*2, input_dim2*2, hidden_dim)
        self.decoder = MultimodalDecoder(input_dim1, input_dim2, input_dim1*2, input_dim2*2, hidden_dim, latent_dim)

        feature_dim = graph_features.size(1)
        self.z_c = nn.Linear(latent_dim, feature_dim)
        # Binary range RCGN, l: z ->[0,1]
        self.pathway_graph = RGCN(feature_dim, hidden_dim, graph_n_relations, d_out=1)
        self.register_buffer("graph_features_static", graph_features[:graph_n_static].clone().float()) # Minus trainables
        self.graph_features_trainable = nn.Parameter(graph_features[graph_n_static:].clone().float())

        self.f = nn.Linear(hidden_dim, latent_dim)
        self.g = nn.Linear(hidden_dim, latent_dim)
        self.g_I = nn.Linear(latent_dim, 1)
        self.gate_logit = nn.Linear(latent_dim, 1)
        # Linear on purpose: cost is then one direction of the per-component code,
        # which the latent optimizer can follow without leaving the data manifold
        # through a nonlinear head it could exploit (ADR-002).
        self.cost_head = nn.Linear(latent_dim, 1)
        # Where the head reads (pre-gate or gated code) and whether its output is
        # squashed into the observed standardized range (lo, hi); set by vae.build
        self.cost_pregate = True
        self.cost_range = None
        self.k = tsvd_k

    def read_cost(self, code):
        """Per-component standardized ln(USD/L) off a (batch, n_comp, d_z) code."""
        u = self.cost_head(code).squeeze(-1)
        if self.cost_range is None:
            return u
        lo, hi = self.cost_range
        return lo + (hi - lo) * torch.sigmoid(u)

    def mean_readout(self, x1, x2, tau=1.0):
        """
        Cost readout and decode of the noise-free code, forward() always samples.

        The anchor of the difference consistency term (ADR-007): the head's
        change from here to the sampled code is matched to the change in the
        decoded recipe's price.
        """
        Y = self.f(self.encoder(x1, x2))
        if getattr(self, "ablate_svd", False):
            Yk = Y
        else:
            mu, Uk = self.truncated_svd(Y)
            Yk = torch.matmul(Uk, mu)
        gate = gumbel_sigmoid(self.gate_logit(Yk), tau, sample=False)
        cost = self.read_cost(Yk if self.cost_pregate else Yk * gate)
        x1_hat, x2_hat = self.decoder(Yk * gate, x1.size(1), x2.size(1))
        return cost, x1_hat, x2_hat

    @property
    def graph_features_(self):
        return torch.cat([self.graph_features_static, self.graph_features_trainable], dim=0)

    def pad_vector(self, tensor, target_size, pad_value=0):
        current_size = tensor.size(2)
        if current_size >= target_size:
            return tensor

        pad_amount = target_size - current_size
        padded_vector = F.pad(tensor, (0, pad_amount), mode="constant", value=pad_value)

        return padded_vector

    def truncated_svd(self, Y):
        U, S, Vh = la.svd(Y)

        S = S + 1e-6

        k_cap = min(self.k, S.size(-1))

        Uk = U[:, :, :k_cap]                            # (batch, n_comp, k_cap)
        Sk = torch.diag_embed(S[:, :k_cap])             # (batch, k_cap, k_cap)
        Vkh = Vh[:, :k_cap, :]                          # (batch, k_cap, d_z)
 
        alpha = torch.matmul(Sk, Vkh)                   # (batch, k_cap, d_z), the "mean"
        return alpha, Uk

    def forward(self, x1, x2, adjancency_mats, tau=1.0, deterministic=False):
        """
        ``deterministic`` drops every draw, so one input maps to one output.

        Eval mode alone is not enough: the posterior sample, the gate's Gumbel noise
        and the divergence's prior draw all keep running under ``eval()``. Scoring
        and the constraint judge pass it, training does not.
        """
        Y = self.encoder(x1, x2)                        # (batch, n_comp_1+n_comp_2, d_h)
        if getattr(self, "ablate_svd", False):
            logvar = self.g(Y)                          # (batch, n_comp, d_z)
            mu = self.f(Y)                              # f = I
            std = torch.exp(logvar*0.5)
            eps_sample = torch.zeros_like(std) if deterministic else torch.randn_like(std)
            Yk = mu + std * eps_sample                  # (batch, n_comp, d_z)
    
        else:
            Y = self.f(Y)                               # f = I
            mu, Uk = self.truncated_svd(Y)              # (batch, k_cap, d_z) | (batch, n_comp, k_cap)

            # Linear layer to learn combined logvar of two distributions from truncated SVD coefs.
            logvar = self.g_I(mu)

            # Cholesky decomp for a diagonal matrix is the sqrt of each diagonal element
            # logvar -> var -> std
            std = torch.exp(logvar*0.5)
            # Samples from a normal distribution mu=0, sig=1 with same dim as input
            eps_sample = torch.zeros_like(mu) if deterministic else torch.randn_like(mu)

            a_z = mu + std * eps_sample                 # (batch, k_cap, d_z)
            Yk = torch.matmul(Uk, a_z)                  # (batch, n_comp, d_z)


        logits = self.gate_logit(Yk)                    # (batch, n_comp, 1)
        if self.training and not deterministic:
            gate = gumbel_sigmoid(logits, tau, sample=True)
        else:
            gate = gumbel_sigmoid(logits, tau, sample=False)

        # The MMD prior term has no noise-free value; replacing the prior draw with
        # the mean code forward already computed keeps the reported metric fixed
        # instead of tracking whichever draw it got.
        Yk_pz = Yk.clone() if deterministic else torch.randn_like(Yk)
        # A component's price is a property of what it is and how much of it,
        # which the gate does not change, so by default the head reads pre-gate
        cost_pred = self.read_cost(Yk if self.cost_pregate else Yk * gate)   # (batch, n_comp)
        Yk = Yk * gate                                  # (batch, n_comp, d_z)
        Yk_z = self.z_c(Yk)                             # (batch, n_comp, feature_dim)

        pathway_logit = self.pathway_graph(
            Yk_z, self.graph_features_, adjancency_mats, gate=gate)

        x1, x2 = self.decoder(
                            Yk,
                            x1.size(1),
                            x2.size(1))                 # (batch, n_comp_1, d_in1) | (batch, n_comp_2, d_in2)

        if getattr(self, "ablate_mmd", False):
            return x1, x2, (mu, logvar), pathway_logit, gate, cost_pred
        else:
            return x1, x2, (Yk, Yk_pz), pathway_logit, gate, cost_pred
