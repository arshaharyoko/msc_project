"""
Component vectors, name lookup and the cost channels for the multimodal encoder.

A row is [standardized embedding, standardized log1p(conc)]: identity at constant
norm, dose in its own channel (ADR-012). With ``cost_input`` a standardized
ln(cost) channel follows the dose channel (ADR-003).
"""

import json

import numpy as np
import torch
import torch.nn.functional as F

from src import config
from src.cost_model import CostModel
from src.modules.vec_db import VectorDatabase

__all__ = ["VectorStore"]


class VectorStore:
    def __init__(self, targ_mean, targ_std, cost_mean=None, cost_std=None,
                 cost_input=False, cost_live=True,
                 uni_mean=None, uni_std=None, conc_mean=0.0, conc_std=1.0):
        self.vec_db = VectorDatabase()
        self.targ_mean = targ_mean
        self.targ_std = targ_std
        # Per-dimension UniProt standardization and the concentration channel's own
        # standardization (ADR-012). Defaults are the identity, so a store built
        # without them reproduces the pre-change encoding.
        self.uni_mean = uni_mean
        self.uni_std = uni_std
        self.conc_mean = float(conc_mean)
        self.conc_std = float(conc_std)
        # ln(cost) standardization; None leaves the store cost-free. The run supplies
        # the published constants; pricing itself comes from the price table (ADR-022).
        self.cost_model = (CostModel.from_table(config.PRICE_TABLE, stats_path=None)
                           if config.PRICE_TABLE.exists() else CostModel({}))
        self.cost_model.log_mean = cost_mean
        self.cost_model.log_std = cost_std
        # Width of the cost channel is set by cost_input, its content by cost_live.
        # A dead channel stays at zero so an ablation keeps the input width.
        self.cost_input = cost_input
        self.cost_live = cost_live

        self.toxic_compounds = set()
        for path in (config.DATABASE_JSON, config.DATAPOISON_JSON):
            if path.exists():
                with open(path) as f:
                    for entry in json.load(f):
                        if entry.get("proliferation", 1) == 0:
                            for comp in entry["components"]:
                                self.toxic_compounds.add(comp["component"])
        print(
            f"Toxic compound blacklist: {len(self.toxic_compounds)} compounds "
            f"from proliferation=0 entries"
        )

    @classmethod
    def from_run(cls, run):
        store = cls._from_run(run)
        if getattr(run, "cost_consist", 0.0) > 0 and store.cost_model.prices:
            store.fit_price_oracle(run.dataset, None, run.cost_bounds[0])
        return store

    @staticmethod
    def encoding_stats(dataset):
        """
        Per-dimension UniProt stats over the DISTINCT proteins, and log1p(conc) stats
        over every component (ADR-012).

        The UniProt statistics come from 21 samples per dimension, and they are taken
        over the whole dataset rather than the training split: these are fixed input
        features, not labels, but it is an assumption and ADR-012 declares it.
        """
        vecs, concs = {}, []
        for item in dataset:
            for comp in item["components"]:
                concs.append(float(np.log1p(comp["concentration"] or 1)))
                if comp["component_source"] == "uniprot":
                    vecs.setdefault(comp["component"], comp["component_vec"])
        V = torch.stack([torch.as_tensor(v, dtype=torch.float32) for v in vecs.values()])
        c = torch.tensor(concs, dtype=torch.float32)
        return V.mean(0), V.std(0).clamp_min(1e-6), float(c.mean()), float(c.std().clamp_min(1e-6))

    @classmethod
    def _from_run(cls, run):
        uni_mean, uni_std, conc_mean, conc_std = cls.encoding_stats(run.dataset)
        return cls(
            run.targ_mean,
            run.targ_std,
            getattr(run, "cost_mean", None),
            getattr(run, "cost_std", None),
            getattr(run, "cost_input", False),
            getattr(run, "cost_delta", 0.0) > 0,
            uni_mean.to(run.targ_mean.device),
            uni_std.to(run.targ_mean.device),
            conc_mean,
            conc_std,
        )

    def _is_poisonous(self, name):
        return name in self.toxic_compounds

    def std_uniprot(self, base):
        """Per-dimension z-score of a UniProt embedding; identity if stats are absent."""
        if self.uni_mean is None:
            return base
        return (base - self.uni_mean.to(base.device)) / (self.uni_std.to(base.device) + 1e-8)

    def enc_conc(self, conc_scaled):
        """log1p(conc) -> the standardized value carried in the concentration channel."""
        return (conc_scaled - self.conc_mean) / self.conc_std

    def dec_conc(self, channel):
        """The concentration channel back to log1p(conc)."""
        return channel * self.conc_std + self.conc_mean

    @property
    def n_scalars(self):
        """Trailing scalar channels after the embedding: concentration, then cost."""
        return 2 if self.cost_input else 1

    @property
    def cost_mean(self):
        return self.cost_model.log_mean

    @property
    def cost_std(self):
        return self.cost_model.log_std

    def scaled_cost(self, comp):
        """Standardized ln(USD/L), or None where the cost is unknown."""
        cost = comp.get("cost")
        if cost is None or not cost > 0 or self.cost_model.log_mean is None:
            return None
        return np.float64(self.cost_model.standardize(cost))

    def _with_cost(self, vec, comp, device):
        if not self.cost_input:
            return vec
        scaled = self.scaled_cost(comp) if self.cost_live else None
        # Unknown cost sits at the standardized mean
        value = 0.0 if scaled is None else float(scaled)
        return torch.cat([vec, torch.tensor([value], dtype=torch.float32, device=device)])

    def cost_targets(self, batch, device):
        """
        (target, mask), each (batch, n_comp), in the encoder's component order.

        MultimodalEncoder concatenates pubchem before uniprot along the component
        axis, so the targets follow the same split as vectorize_batch.
        """
        targets, masks = [], []
        for components in batch["components"]:
            ordered = ([c for c in components if c["component_source"] == "pubchem"]
                       + [c for c in components if c["component_source"] == "uniprot"])
            scaled = [self.scaled_cost(c) for c in ordered]
            targets.append([0.0 if v is None else float(v) for v in scaled])
            masks.append([0.0 if v is None else 1.0 for v in scaled])
        return (torch.tensor(targets, dtype=torch.float32, device=device),
                torch.tensor(masks, dtype=torch.float32, device=device))

    def fit_price_oracle(self, dataset, prices=None, floor=0.0):
        """
        Nearest-neighbour price table for pricing decoded vectors (ADR-007).

        One row per distinct priced component, in the same space ``vec_lookup``
        searches: standardized embedding for PubChem, raw for UniProt. ``floor``
        is the standardized ln(cost) a zero-concentration decode gets.
        """
        if prices is None:
            prices = self.cost_model.price_dict()
        rows = {"pubchem": {}, "uniprot": {}}
        for item in dataset:
            for comp in item["components"]:
                name, source = comp["component"], comp["component_source"]
                if name in rows[source] or prices.get(name) is None:
                    continue
                base = comp["component_vec"].to(dtype=torch.float32, device=self.targ_mean.device)
                if source == "pubchem":
                    base = (base - self.targ_mean) / (self.targ_std + 1e-8)
                else:
                    base = self.std_uniprot(base)
                rows[source][name] = (base, float(prices[name]))

        self.oracle = {}
        for source, entries in rows.items():
            if entries:
                vecs = torch.stack([v for v, _ in entries.values()])
                price = torch.tensor([p for _, p in entries.values()], device=vecs.device)
                self.oracle[source] = (vecs, price)
        self.oracle_floor = float(floor)

    def _price_decoded(self, x_hat, source):
        """Standardized ln(USD/L) of each decoded vector, (batch, n) and its mask."""
        conc_log = self.dec_conc(x_hat[..., -self.n_scalars])
        base = x_hat[..., :-self.n_scalars]
        if source not in self.oracle or base.size(1) == 0:
            zeros = torch.zeros(conc_log.shape, device=x_hat.device)
            return zeros, zeros
        vecs, price = self.oracle[source]
        flat = base.reshape(-1, base.size(-1))
        nearest = torch.cdist(flat, vecs).argmin(dim=-1).reshape(conc_log.shape)
        conc = torch.expm1(conc_log)
        alive = conc > 0
        cost = price[nearest] * conc.clamp_min(1e-30)
        scaled = self.cost_model.standardize_log(cost.log())
        # A component decoded to no concentration is dropped from the recipe
        scaled = torch.where(alive, scaled.clamp_min(self.oracle_floor),
                             torch.full_like(scaled, self.oracle_floor))
        return scaled, torch.ones_like(scaled)

    def decoded_log_cost(self, x1_hat, x2_hat, sharpness=100.0):
        """
        ln total USD/L of the decoded recipe, differentiable in concentration.

        Identity is the nearest priced component (piecewise constant, no
        gradient); concentration is expm1 of a softplus-smoothed log1p channel,
        so a component driven below zero leaves the recipe smoothly.
        """
        terms = []
        for x_hat, source in ((x1_hat, "pubchem"), (x2_hat, "uniprot")):
            if source not in self.oracle or x_hat.size(1) == 0:
                continue
            vecs, price = self.oracle[source]
            conc_log = self.dec_conc(x_hat[..., -self.n_scalars])
            with torch.no_grad():
                base = x_hat[..., :-self.n_scalars]
                flat = base.reshape(-1, base.size(-1))
                nearest = torch.cdist(flat, vecs).argmin(dim=-1).reshape(conc_log.shape)
            conc = torch.expm1(F.softplus(conc_log, beta=sharpness))
            terms.append(price[nearest].log() + conc.clamp_min(1e-30).log())
        return torch.logsumexp(torch.cat(terms, dim=-1), dim=-1)

    @torch.no_grad()
    def decoded_cost_targets(self, x1_hat, x2_hat):
        """(target, mask) pricing the decoded recipe, in the encoder's component order."""
        s1, m1 = self._price_decoded(x1_hat, "pubchem")
        s2, m2 = self._price_decoded(x2_hat, "uniprot")
        return torch.cat([s1, s2], dim=1), torch.cat([m1, m2], dim=1)

    def cost_kwargs(self, batch, x1_hat, x2_hat, run, x1_in=None, x2_in=None, tau=1.0):
        """
        The cost arguments of ``loss_reg`` for one forward pass.

        With ``cost_consist`` on, the decoded recipe is priced too. In "delta"
        mode the inputs are re-encoded noise free so the term can compare the
        sampled code against it (ADR-007).
        """
        target, mask = self.cost_targets(batch, x1_hat.device)
        kwargs = {"cost_target": target, "cost_mask": mask, "delta": run.cost_delta,
                  "recipe_w": getattr(run, "cost_recipe", 0.0),
                  # The recipe term regresses ln USD/L, so it needs the scale the
                  # targets were standardized with.
                  "cost_mean": float(self.cost_mean or 0.0),
                  "cost_std": float(self.cost_std or 1.0)}
        consist = getattr(run, "cost_consist", 0.0)
        if consist > 0 and run.cost_delta > 0 and getattr(self, "oracle", None):
            dec_target, dec_mask = self.decoded_cost_targets(x1_hat, x2_hat)
            kwargs.update(cost_dec_target=dec_target, cost_dec_mask=dec_mask, consist=consist)
            if getattr(run, "cost_consist_mode", "level") == "delta" and x1_in is not None:
                anchor_pred, a1, a2 = run.model.mean_readout(x1_in, x2_in, tau)
                anchor_target, _ = self.decoded_cost_targets(a1, a2)
                kwargs.update(cost_anchor_pred=anchor_pred, cost_anchor_target=anchor_target)
        return kwargs

    def unscale_cost(self, scaled):
        """Standardized ln(cost) back to USD/L."""
        return self.cost_model.unstandardize(scaled)

    def make_x1_vec(self, comp, device):
        conc_scaled = np.log1p(comp["concentration"] or 1)
        base = comp["component_vec"].clone().detach().to(device=device, dtype=torch.float32)
        vec = torch.cat(
            [
                (base - self.targ_mean) / (self.targ_std + 1e-8),
                torch.tensor([self.enc_conc(conc_scaled)], dtype=torch.float32, device=device),
            ]
        )
        return self._with_cost(vec, comp, device)

    def make_x2_vec(self, comp, device):
        base = comp["component_vec"].clone().detach().to(device=device, dtype=torch.float32)
        conc = float(comp["concentration"] or 1)
        conc_scaled = np.log1p(conc)
        vec = torch.cat(
            [
                self.std_uniprot(base),
                torch.tensor([self.enc_conc(conc_scaled)], dtype=torch.float32, device=device),
            ]
        )
        return self._with_cost(vec, comp, device)

    def vectorize_batch(self, batch, device):
        """Split each recipe in the batch into its PubChem and UniProt vectors."""
        x1_batch = []
        x2_batch = []
        for components in batch["components"]:
            x1_batch.append(
                [
                    self.make_x1_vec(comp, device)
                    for comp in components
                    if comp["component_source"] == "pubchem"
                ]
            )
            x2_batch.append(
                [
                    self.make_x2_vec(comp, device)
                    for comp in components
                    if comp["component_source"] == "uniprot"
                ]
            )
        return x1_batch, x2_batch

    def update_db_from_batch(self, batch, dim1, dim2, device):
        pubchem_vectors, pubchem_names = [], []
        uniprot_vectors, uniprot_names = [], []

        for components in batch["components"]:
            for comp in components:
                if comp["component_source"] == "pubchem":
                    base_vec = (
                        comp["component_vec"]
                        .clone()
                        .detach()
                        .to(device=device, dtype=torch.float32)
                    )
                    normalized = (base_vec - self.targ_mean) / (self.targ_std + 1e-8)
                    pubchem_vectors.append(normalized.cpu().numpy())
                    pubchem_names.append(comp["component"])
                elif comp["component_source"] == "uniprot":
                    base_vec = (
                        comp["component_vec"]
                        .clone()
                        .detach()
                        .to(device=device, dtype=torch.float32)
                    )
                    uniprot_vectors.append(self.std_uniprot(base_vec).cpu().numpy())
                    uniprot_names.append(comp["component"])

        if pubchem_vectors:
            self.vec_db.add_pubchem_vectors(
                np.array(pubchem_vectors), dim=dim1 - 1, names=pubchem_names
            )
        if uniprot_vectors:
            self.vec_db.add_uniprot_vectors(
                np.array(uniprot_vectors), dim=dim2 - 1, names=uniprot_names
            )

    def _exclusion_class(self, name, exclusions, patterns):
        if exclusions and name in exclusions:
            return exclusions[name]
        for pattern, cls in patterns or ():
            if pattern in name:
                return cls
        if self._is_poisonous(name):
            return "toxicant_pool"
        return None

    def _first_allowed(self, indices, data, exclusions, patterns, fired, exclude=None):
        for idx in indices:
            name = data[idx]
            if exclude is not None and name in exclude:
                continue
            cls = self._exclusion_class(name, exclusions, patterns)
            if cls is not None:
                if fired is not None:
                    fired[cls] = fired.get(cls, 0) + 1
                continue
            return name
        return None

    def vec_lookup(self, item, x1_dim, x2_dim, exclude=None, source=None,
                   exclusions=None, patterns=None, fired=None):
        """
        Rank-1 allowed neighbour name and dose for one decoded slot.

        ``exclusions`` maps a name to its class and ``patterns`` carries (substring, class)
        pairs, both scanned in rank order; ``fired`` collects one count per skipped class
        (ADR-021). ``exclude`` is a caller-side set of names to skip.
        """
        vec = item.cpu().detach()
        if self.cost_input:
            vec = vec[:-1]
        log_scalar = self.dec_conc(vec[-1].item())
        # Identity is at constant norm, so there is nothing to un-multiply (ADR-012).
        vector = vec[:-1].numpy()
        actual_conc = float(np.expm1(log_scalar))

        # Both modalities share a width, so the shape does not identify one.
        if source == "pubchem" or (source is None and item.shape[-1] == x1_dim):
            if self.vec_db.pubchem_index is None or not self.vec_db.pubchem_data:
                return None, actual_conc
            k = min(10, len(self.vec_db.pubchem_data))
            indices, _ = self.vec_db.vec_lookup_pubchem(vector, k=k)
            name = self._first_allowed(indices, self.vec_db.pubchem_data,
                                       exclusions, patterns, fired, exclude)
            return name, actual_conc
        elif source == "uniprot" or (source is None and item.shape[-1] == x2_dim):
            if self.vec_db.uniprot_index is None or not self.vec_db.uniprot_data:
                return None, actual_conc
            k = min(10, len(self.vec_db.uniprot_data))
            indices, _ = self.vec_db.vec_lookup_uniprot(vector, k=k)
            name = self._first_allowed(indices, self.vec_db.uniprot_data,
                                       exclusions, patterns, fired, exclude)
            return name, actual_conc
        return None, actual_conc

    def save(self):
        self.vec_db.save(str(config.PUBCHEM_INDEX), str(config.UNIPROT_INDEX))

    def load(self):
        self.vec_db.load(str(config.PUBCHEM_INDEX), str(config.UNIPROT_INDEX))
