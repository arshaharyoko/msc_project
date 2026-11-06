"""
Multimodal denoising diffusion model with graph regularization, noise-prediction network, beta schedule and objective.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modules.layers import MultimodalEncoder, MultimodalDecoder, RGCN, gumbel_sigmoid

__all__ = [
    "MDDPM_GV",
    "NoisePredictor",
    "OUTPUT_NODES",
    "SinusoidalEmbedding",
    "diffusion_loss_reg",
    "make_beta_schedule",
    "pathway_bce",
]

# Pathway graph readout nodes: Proliferation, Survival/Quiescence, Differentiation
OUTPUT_NODES = [22, 23, 24]

def pathway_bce(pathway_prob, real_prob):
    node_targets = torch.tensor([[real_prob, real_prob, 1.0 - real_prob]], dtype=torch.float, device=pathway_prob.device)
    output_probs = pathway_prob[:, OUTPUT_NODES, 0]
    graph_real_prob = node_targets.expand(output_probs.size(0), -1)
    per_graph = F.binary_cross_entropy(output_probs, graph_real_prob, reduction="none")
    return per_graph.sum(dim=-1).mean()

def diffusion_loss_reg(output1, output2, target1, target2, real_prob, pathway_prob, alpha=1e-2, verbose=False):
    noise_loss = F.mse_loss(output1, target1, reduction="mean") + F.mse_loss(output2, target2, reduction="mean")
    pathway_loss = pathway_bce(pathway_prob, real_prob)

    if verbose:
        print(f"PATHWAY_PROB: {pathway_prob}")
        print(f"GRAPH_LOSS: {pathway_loss}")
        print(f"NOISE_LOSS: {noise_loss}")
        print(f"FINAL_LOSS: {noise_loss + pathway_loss * alpha}\n")

    return noise_loss + pathway_loss * alpha

def make_beta_schedule(T, beta_start, beta_end, schedule="cosine"):
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, T)

    # Cosine (Nichol & Dhariwal 2021) because linear schedule destroys structure well before t=T/2.
    s = 0.008
    steps = torch.arange(T + 1, dtype=torch.float32) / T
    f = torch.cos((steps + s) / (1 + s) * math.pi * 0.5) ** 2
    alpha_bar = f / f[0]
    return (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(1e-8, 0.999)

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0, "time_embed_dim must be even"
        self.dim = dim
        half = dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half).float() / half)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, t):
        t = t.float().view(-1, 1)                           # (batch, 1)
        args = t * self.freqs[None, :]                      # (batch, half)
        return torch.cat([args.sin(), args.cos()], dim=-1)  # (batch, dim)

class NoisePredictor(nn.Module):
    def __init__(self, d_in1, d_in2, d_h, d_z, d_z_time):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalEmbedding(d_z_time),
            nn.Linear(d_z_time, d_z_time),
            nn.SiLU(),
            nn.Linear(d_z_time, d_z_time),
        )

        # Timestep is concatenated per component, so the encoder sees d_in + d_z_time
        self.encoder = MultimodalEncoder(d_in1 + d_z_time, d_in2 + d_z_time,
                                        (d_in1 + d_z_time) * 2, (d_in2 + d_z_time) * 2,
                                         d_h, d_z)
        self.decoder = MultimodalDecoder(d_in1, d_in2, d_in1 * 2, d_in2 * 2, d_h, d_z)

    def encode(self, ut1, ut2, t):
        t_z = self.time_mlp(t)                              # (batch, d_z_time)
        t1 = t_z.unsqueeze(1).expand(-1, ut1.size(1), -1)   # (batch, n_comp_1, d_z_time)
        t2 = t_z.unsqueeze(1).expand(-1, ut2.size(1), -1)   # (batch, n_comp_2, d_z_time)

        return self.encoder(torch.cat([ut1, t1], dim=-1),
                            torch.cat([ut2, t2], dim=-1))   # (batch, n_comp_1+n_comp_2, d_z)

    def decode(self, Y, n_in_1, n_in_2):
        return self.decoder(Y, n_in_1, n_in_2)

class MDDPM_GV(nn.Module):
    def __init__(self, input_dim1, input_dim2, latent_dim, hidden_dim, graph_features, graph_n_relations, graph_n_static,
                 T=1000, beta_start=1e-5, beta_end=2e-2, schedule="cosine",
                 time_dim=64, conc_mean=0.0, conc_std=1.0):
        super().__init__()
        self.T = T
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2

        beta = make_beta_schedule(T, beta_start, beta_end, schedule)
        alpha = 1.0 - beta
        # So schedule follows .to(device) and is checkpointed
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", torch.cumprod(alpha, dim=0))

        # Standardization for the log1p(concentration) channel once it is decoupled
        self.register_buffer("conc_mean", torch.tensor(float(conc_mean)))
        self.register_buffer("conc_std", torch.tensor(float(conc_std)))

        self.model = NoisePredictor(input_dim1, input_dim2, hidden_dim, latent_dim, time_dim)

        feature_dim = graph_features.size(1)
        self.z_c = nn.Linear(latent_dim, feature_dim)
        # Binary range RGCN, l: z -> [0,1]
        self.pathway_graph = RGCN(feature_dim, hidden_dim, graph_n_relations, d_out=1)
        self.register_buffer("graph_features_static", graph_features[:graph_n_static].clone().float()) # Minus trainables
        self.graph_features_trainable = nn.Parameter(graph_features[graph_n_static:].clone().float())

        self.gate_logit = nn.Linear(latent_dim, 1)

    @property
    def graph_features_(self):
        return torch.cat([self.graph_features_static, self.graph_features_trainable], dim=0)

    def set_conc_stats(self, mean, std):
        self.conc_mean.fill_(float(mean))
        self.conc_std.fill_(float(std))

    def decouple(self, x):
        # Undo coupling between embedding vector and concentration
        # [descriptor * log1p(conc), log1p(conc)]
        s = x[..., -1:]
        d = x[..., :-1] / s.clamp(min=1e-3)
        s = (s - self.conc_mean) / self.conc_std
        return torch.cat([d, s], dim=-1)

    def recouple(self, u):
        # Component with concentration < 0 gives zero vector
        s = (u[..., -1:] * self.conc_std + self.conc_mean).clamp(min=0.0)
        return torch.cat([u[..., :-1] * s, s], dim=-1)

    def _sched(self, buf, t):
        # (batch,) schedule lookup, broadcast over (batch, n_comp, d)
        return buf[t].view(-1, 1, 1)

    def q_sample(self, u0, t, eps):
        sqrt_alpha_bar = torch.sqrt(self._sched(self.alpha_bar, t))
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self._sched(self.alpha_bar, t))
        return sqrt_alpha_bar * u0 + sqrt_one_minus_alpha_bar * eps

    def predict_u0(self, ut, t, eps_hat):
        sqrt_alpha_bar = torch.sqrt(self._sched(self.alpha_bar, t))
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self._sched(self.alpha_bar, t))
        return (ut - sqrt_one_minus_alpha_bar * eps_hat) / sqrt_alpha_bar

    def denoise(self, ut1, ut2, t, adjancency_mats, tau=1.0):
        Y = self.model.encode(ut1, ut2, t)      # (batch, n_comp_1+n_comp_2, d_z)

        logits = self.gate_logit(Y)             # (batch, n_comp, 1)
        if self.training:
            gate = gumbel_sigmoid(logits, tau, sample=True)
        else:
            gate = gumbel_sigmoid(logits, tau, sample=False)

        Yk = Y * gate
        Yk_z = self.z_c(Yk)

        pathway_prob = torch.sigmoid(
            self.pathway_graph(Yk_z, self.graph_features_, adjancency_mats))

        eps1_hat, eps2_hat = self.model.decode(Yk, ut1.size(1), ut2.size(1))

        return eps1_hat, eps2_hat, pathway_prob, gate

    def forward(self, x1, x2, adjancency_mats, t=None, tau=1.0, t_samples=4):
        u1 = self.decouple(x1)
        u2 = self.decouple(x2)

        # batch_size is 1 in this project, so a single random t per step gives a very
        # noisy gradient. Evaluate the same recipe at several timesteps instead.
        if t_samples > 1 and t is None:
            u1 = u1.repeat(t_samples, 1, 1)
            u2 = u2.repeat(t_samples, 1, 1)

        batch = u1.size(0)
        if t is None:
            t = torch.randint(0, self.T, (batch,), device=u1.device)

        eps1 = torch.randn_like(u1)
        eps2 = torch.randn_like(u2)

        ut1 = self.q_sample(u1, t, eps1)    # (batch, n_comp_1, d_in1)
        ut2 = self.q_sample(u2, t, eps2)    # (batch, n_comp_2, d_in2)

        eps1_hat, eps2_hat, pathway_prob, gate = self.denoise(ut1, ut2, t, adjancency_mats, tau=tau)

        return eps1_hat, eps2_hat, eps1, eps2, pathway_prob, gate

    def _ddpm_step(self, u, eps_hat, step):
        alpha_t = self.alpha[step]
        beta_t = self.beta[step]
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar[step])

        mu = (u - (beta_t / sqrt_one_minus_alpha_bar) * eps_hat) / torch.sqrt(alpha_t)

        if step > 0:
            return mu + torch.sqrt(beta_t) * torch.randn_like(u)
        return mu

    def _guided_eps(self, eps_hat, grad, step, guidance_scale, normalize):
        if normalize:
            # Normalized steering towards binary-constrained RCGN output
            direction = grad / (grad.norm() + 1e-12) * eps_hat.norm()
            return eps_hat + guidance_scale * direction
        return eps_hat + guidance_scale * torch.sqrt(1 - self.alpha_bar[step]) * grad

    def _reverse_loop(self, u1, u2, start_step, adjancency_mats, real_prob=1.0,
                      guidance_scale=0.0, tau=0.1, normalize_guidance=True):
        # Ancestral sampling from start_step down to 0. With guidance_scale > 0
        # pathway graph acts as the classifier in classifier guidance:
        #   eps <- eps + s * sqrt(1 - alpha_bar_t) * grad_ut BCE(readout, target)
        # counterpart of predict.py's LBFGS search over a_z.
        batch = u1.size(0)

        for step in reversed(range(start_step)):
            t = torch.full((batch,), step, device=u1.device, dtype=torch.long)

            if guidance_scale > 0:
                u1 = u1.detach().requires_grad_(True)
                u2 = u2.detach().requires_grad_(True)
                with torch.enable_grad():
                    eps1_hat, eps2_hat, pathway_prob, _ = self.denoise(u1, u2, t, adjancency_mats, tau=tau)
                    bce = pathway_bce(pathway_prob, real_prob)
                    grad1, grad2 = torch.autograd.grad(bce, [u1, u2])

                eps1_hat = self._guided_eps(eps1_hat, grad1, step, guidance_scale, normalize_guidance).detach()
                eps2_hat = self._guided_eps(eps2_hat, grad2, step, guidance_scale, normalize_guidance).detach()
                u1, u2 = u1.detach(), u2.detach()
            else:
                with torch.no_grad():
                    eps1_hat, eps2_hat, _, _ = self.denoise(u1, u2, t, adjancency_mats, tau=tau)

            u1 = self._ddpm_step(u1, eps1_hat, step)
            u2 = self._ddpm_step(u2, eps2_hat, step)

        return self.recouple(u1.detach()), self.recouple(u2.detach())

    def sample(self, n_comp_1, n_comp_2, adjancency_mats, batch=1, real_prob=1.0,
               guidance_scale=0.0, tau=0.1, normalize_guidance=True, device=None):
        # Unconditional generation. n_comp_* is an upper bound: components the model
        # assigns no concentration come back as zero vectors and are dropped downstream.
        device = device if device is not None else self.beta.device
        was_training = self.training
        self.eval()

        u1 = torch.randn(batch, n_comp_1, self.input_dim1, device=device)
        u2 = torch.randn(batch, n_comp_2, self.input_dim2, device=device)
        x1, x2 = self._reverse_loop(u1, u2, self.T, adjancency_mats, real_prob=real_prob,
                                    guidance_scale=guidance_scale, tau=tau,
                                    normalize_guidance=normalize_guidance)

        if was_training:
            self.train()
        return x1, x2

    def refine(self, x1, x2, adjancency_mats, t_start=None, real_prob=1.0,
               guidance_scale=0.0, tau=0.1, normalize_guidance=True):
        # SDEdit for guidance_scale > 0 steers the recipe toward the pathway target without concentration modifications.
        # guidance_scale left at 0 it is a plain reconstruction, for the validation print in train.py.
        was_training = self.training
        self.eval()

        t_start = t_start if t_start is not None else self.T // 2
        u1 = self.decouple(x1)
        u2 = self.decouple(x2)

        t = torch.full((u1.size(0),), t_start - 1, device=u1.device, dtype=torch.long)
        u1 = self.q_sample(u1, t, torch.randn_like(u1))
        u2 = self.q_sample(u2, t, torch.randn_like(u2))

        x1_out, x2_out = self._reverse_loop(u1, u2, t_start, adjancency_mats, real_prob=real_prob,
                                            guidance_scale=guidance_scale, tau=tau,
                                            normalize_guidance=normalize_guidance)

        if was_training:
            self.train()
        return x1_out, x2_out
