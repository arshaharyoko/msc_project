"""
Ablation harness for the MVRRAE_GV track.

AblationSpec fixes every flag and weight of a run, so two arms differ only where the
spec says. An ablated component is replaced by a substitute of matched width:

    use_svd     reduced-rank reparameterization -> per-component reparameterization
    use_mmd     MMD against a prior sample      -> closed-form KL
    use_rgcn    pathway RGCN readout            -> mean-pool + MLP readout
    use_gate    Gumbel gate                     -> identity, every component passes
    use_cost    latent cost readout             -> head left untrained (delta = 0),
                                                   and a zeroed channel under cost_input

``use_cost`` keeps the cost head and input channel at the same width and only stops
training them, so every arm has the same parameter count. Whether cost is still
readable off the latent without them is ``cost_probe_r2``, a ridge probe fit on the
frozen training codes (ADR-005).

``use_svd`` and ``use_mmd`` are branches inside ``MVRRAE_GV.forward``, so they travel as
environment flags the way ``config.vae_run`` reads them. ``use_rgcn`` and ``use_gate``
are module swaps applied after ``vae.build``; ``ABLATE_RGCN`` and ``ABLATE_GATE`` only
zero a loss weight, leaving both modules wired into the forward pass.

Every run writes long-format rows (one per metric) to a JSONL file keyed by
``config_id`` and ``seed``, so a sweep can be resumed and the table read back.
"""

import hashlib
import json
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import rankdata
from torch.utils.data import DataLoader, Subset, random_split

from src import benchmark, config, vae
from src.modules.dataset import GrowthMediumDataset, collate_fn, safe_stack
from src.modules.layers import MeanPoolReadout, OpenGate, RGCN, SumPoolReadout
from src.modules.mvrrae_gv import depletion_hinge, kl, loss_reg, mmd, recipe_log_cost

__all__ = [
    "AblationSpec",
    "RUN_LOG",
    "SIGMA_REF",
    "TAU_DECAY",
    "TAU_FLOOR",
    "TAU_START",
    "apply_spec",
    "checkpoint_path",
    "completed_runs",
    "cost",
    "cost_codes",
    "evaluate",
    "evaluate_checkpoint",
    "load_rows",
    "log_metrics",
    "parent_folds",
    "run_spec",
    "run_study",
    "seed_everything",
    "split_dataset",
    "train_spec",
]

RUN_LOG = config.ARTIFACT_DIR / "ablation_runs.jsonl"

# hipo.py reports the divergence at a fixed bandwidth so the number stays
# comparable across the beta axis. Training keeps the median heuristic, which is
# what train.py and hipo.py both optimize.
SIGMA_REF = 16.0

TAU_START = 1.0
TAU_DECAY = 0.95
TAU_FLOOR = 0.1

# The readout node the depletion hinge orders, and a cap on how many pairs are
# forward-passed per epoch: each pair is two extra forwards, and on CPU that is the
# binding cost.
PROLIF_NODE = 22
DEP_PAIRS_PER_EPOCH = 24

VAL_FRACTION = 0.1
TEST_FRACTION = 0.1
MIN_SPLIT = 5
N_FOLDS = 5


@dataclass(frozen=True)
class AblationSpec:
    """Everything that distinguishes one run from another"""

    label: str = "baseline"
    use_svd: bool = True
    use_mmd: bool = True
    use_rgcn: bool = True
    use_gate: bool = True

    seed: int = 420
    split_seed: int = 67
    # Which grouped fold is the test fold. Excluded from config_id; the checkpoint
    # tag carries it, because the five folds share an arm hash.
    fold: int = 0
    # Provenance only: ADR-012 made the dose split the only encoding, so this records
    # which one produced a checkpoint rather than switching behaviour (ADR-027).
    encoding: str = "dose_split"

    epochs: int = config.EPOCHS
    batch_size: int = config.BATCH_SIZE
    latent_dim: int = config.LATENT_DIM
    hidden_dim: int = config.HIDDEN_DIM
    tsvd_k: int = config.TSVD_K

    pathway_alpha: float = config.PATHWAY_ALPHA
    div_beta: float = None
    gate_gamma: float = config.GATE_GAMMA
    recon_w1: float = config.RECON_W1
    recon_w2: float = config.RECON_W2

    use_cost: bool = True
    cost_delta: float = config.COST_DELTA
    cost_input: bool = config.COST_INPUT
    cost_pregate: bool = config.COST_PREGATE
    cost_bounded: bool = config.COST_BOUNDED
    cost_consist: float = config.COST_CONSIST
    cost_consist_mode: str = config.COST_CONSIST_MODE
    cost_recipe: float = config.COST_RECIPE

    # None matches the substitute readout to the RGCN it replaces.
    readout_width: int = None
    # The blacklist censors decoded poison, which is the output an ablation is
    # trying to read. Off by default here; the toxic names are still kept, they
    # are what gate_auc is scored against.
    blacklist: bool = False

    # Task 1 augmentation (handoff 2.2): the depletion hinge only. Off by default, and
    # excluded from config_id, so every pre-existing run hash is unchanged.
    augment: bool = False
    dep_w: float = config.DEP_W
    dep_margin: float = config.DEP_MARGIN

    # Selects data/database_graded.json over the legacy database (ADR-019). Excluded from
    # config_id for the same reason as the augmentation fields; the checkpoint tag
    # separates the two datasets.
    negatives: str = "legacy"

    # Aggregation rule of the pathway readout (ADR-020). Excluded from config_id so the
    # pre-additive hashes hold; the checkpoint tag separates the arms.
    readout: str = "softmax"
    # Identity cross-entropy weight on the UniProt block (handoff 1B). Off, and excluded
    # from config_id because it is not adopted.
    ident_w2: float = 0.0

    def __post_init__(self):
        if not 0 <= self.fold < N_FOLDS:
            raise ValueError(f"fold must be in 0..{N_FOLDS - 1}, got {self.fold}")
        if self.readout not in ("softmax", "additive"):
            raise ValueError(f"readout must be 'softmax' or 'additive', got {self.readout!r}")
        if self.div_beta is None:
            beta = config.MMD_BETA if self.use_mmd else config.KL_BETA
            object.__setattr__(self, "div_beta", beta)
        if not self.use_gate:
            # The substitute gate is constant, so its penalty is a constant with no gradient
            object.__setattr__(self, "gate_gamma", 0.0)
        if not self.use_cost:
            object.__setattr__(self, "cost_delta", 0.0)

    @property
    def flags(self):
        return {
            "use_svd": self.use_svd,
            "use_mmd": self.use_mmd,
            "use_rgcn": self.use_rgcn,
            "use_gate": self.use_gate,
            "use_cost": self.use_cost,
        }

    @property
    def logvar_head(self):
        """Which head produces logvar in ``MVRRAE_GV.forward``."""
        return "g" if not self.use_svd else "g_I"

    @property
    def config_id(self):
        """Stable hash of the run definition, seed excluded."""
        # The augmentation, provenance, readout-variant and fold fields are excluded so
        # every pre-existing config_id is unchanged (handoff 2.2, ADR-027);
        # checkpoint_path carries their tags instead.
        fields = {k: v for k, v in asdict(self).items()
                  if k not in ("seed", "label", "augment", "dep_w", "dep_margin",
                               "encoding", "negatives", "readout", "ident_w2", "fold")}
        blob = json.dumps(fields, sort_keys=True, default=str)
        return hashlib.sha1(blob.encode()).hexdigest()[:12]

    def env(self):
        """The two flags ``config.vae_run`` reads out of the environment.

        RGCN and GATE are pinned off here and handled by module swap instead
        ``config.vae_run`` would answer ABLATE_RGCN by setting pathway_alpha to
        zero, which removes the loss term the substitute readout has to train on.
        Pinning them also stops a stray shell export from changing a run.
        """
        return {
            "ABLATE_SVD": "0" if self.use_svd else "1",
            "ABLATE_MMD": "0" if self.use_mmd else "1",
            "ABLATE_RGCN": "0",
            "ABLATE_GATE": "0",
            "ABLATE_COST": "0" if self.use_cost else "1",
        }


def apply_spec(run, spec):
    """Install the substitutes on a built run and make the spec authoritative.

    ``vae.build`` already sets ``model.ablate_svd`` / ``ablate_mmd`` this way;
    this extends the same convention to the two components that have no branch
    in ``forward``.
    """
    model = run.model

    if spec.use_rgcn:
        graph = model.pathway_graph
        if isinstance(graph, RGCN) and spec.readout != graph.readout:
            # Fresh stack, same shape: the run starts from scratch either way.
            model.pathway_graph = graph.with_readout(spec.readout).to(run.device)
    else:
        d_in = model.z_c.out_features
        n_nodes = model.graph_features_.size(0)
        source = model.pathway_graph
        if isinstance(source, RGCN) and spec.readout != "softmax":
            source = source.with_readout(spec.readout)
        target = sum(p.numel() for p in source.parameters())
        readout_cls = SumPoolReadout if spec.readout == "additive" else MeanPoolReadout
        readout = (readout_cls(d_in, n_nodes, spec.readout_width)
                   if spec.readout_width
                   else readout_cls.matched(d_in, n_nodes, target))
        model.pathway_graph = readout.to(run.device)
        # The abstract node rows are a graph input; without the graph they are
        # unreachable, so they stop being trainable rather than silently idling.
        model.graph_features_trainable.requires_grad_(False)

    if not spec.use_gate:
        model.gate_logit = OpenGate().to(run.device)

    # Rebuild the optimizer: the swaps changed which parameters exist.
    run.optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

    run.pathway_alpha = spec.pathway_alpha
    run.div_beta = spec.div_beta
    run.gate_gamma = spec.gate_gamma
    run.recon_w1 = spec.recon_w1
    run.recon_w2 = spec.recon_w2
    run.cost_delta = spec.cost_delta
    run.cost_input = spec.cost_input
    run.cost_consist = spec.cost_consist
    run.cost_consist_mode = spec.cost_consist_mode
    run.cost_recipe = spec.cost_recipe
    run.ablate_cost = not spec.use_cost
    run.epochs = spec.epochs
    run.run_tag = spec.label
    run.checkpoint = checkpoint_path(spec)
    return run


def seed_everything(seed):
    """One seed for init, shuffling and the Gumbel/reparameterization noise."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@contextmanager
def _config_override(**values):
    """Temporarily set module-level config values, then put them back.

    ``vae.build`` reads its dimensions off ``config`` at call time, so a per-run
    value has to go through the module. Restoring on exit keeps one run in a
    sweep from inheriting the previous run's numbers.
    """
    previous = {name: getattr(config, name) for name in values}
    try:
        for name, value in values.items():
            setattr(config, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(config, name, value)


def checkpoint_path(spec):
    tag = f"ablation__{spec.label}__{spec.config_id}__s{spec.seed}"
    # config_id excludes the augmentation and dataset fields (see config_id), so the tag is
    # what separates a variant from its twin at the same seed.
    suffix = "__aug" if spec.augment else ""
    suffix += "__graded" if spec.negatives != "legacy" else ""
    suffix += "__add" if spec.readout != "softmax" else ""
    suffix += f"__f{spec.fold}" if spec.fold else ""
    return config.ARTIFACT_DIR / f"{tag}{suffix}.pth"


def cost(run, x1, x2):
    """
    Trainable, gradient-carrying and per-example FLOP cost of a built model.

    ``params_live`` is the count that receives gradient and the one to compare across arms:
    every module is constructed unconditionally, so ``params_trainable`` is the same for
    every variant. FLOPs are counted on one reference recipe; ``torch.linalg.svd`` has no
    entry in the flop counter, so the SVD arm's number is a lower bound.
    """
    from torch.utils.flop_counter import FlopCounterMode

    model = run.model
    params_total = sum(p.numel() for p in model.parameters())
    params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    model.zero_grad(set_to_none=True)
    was_training = model.training
    model.train()
    out1, out2, div_args, pathway, gate, cost_pred = model(
        x1, x2, run.adjacency_matrices, tau=TAU_START)
    # Any target exercises the head's gradient path; only liveness is read here.
    loss = loss_reg(out1, out2, x1, x2, div_args, 1.0, pathway, gate,
                    alpha=run.pathway_alpha, beta=run.div_beta, gamma=run.gate_gamma,
                    recon_w1=run.recon_w1, recon_w2=run.recon_w2,
                    ablate_mmd=run.ablate_mmd, cost_pred=cost_pred,
                    cost_target=torch.zeros_like(cost_pred),
                    cost_mask=torch.ones_like(cost_pred), delta=run.cost_delta)
    loss.backward()
    params_live = sum(p.numel() for p in model.parameters()
                      if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0)
    model.zero_grad(set_to_none=True)

    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        model(x1, x2, run.adjacency_matrices, tau=TAU_START)
    flops = counter.get_total_flops()
    model.train(was_training)

    return {
        "params_total": int(params_total),
        "params_trainable": int(params_trainable),
        "params_live": int(params_live),
        "flops_per_example": int(flops),
        "flops_ref_n_comp": int(x1.size(1) + x2.size(1)),
    }


def _record(dataset, index):
    """The raw JSON entry; ``GrowthMediumDataset.__getitem__`` returns a sample instead."""
    data = getattr(dataset, "data", None)
    return data[index] if data is not None else dataset[index]


def _publication_groups(dataset):
    """Publication key -> real record indices, in first-occurrence order."""
    groups = {}
    for i in range(len(dataset)):
        record = _record(dataset, i)
        if record.get("parent") is not None:
            continue
        key = record.get("source") or record.get("product") or f"#{i}"
        groups.setdefault(key, []).append(i)
    return groups


def _fold_groups(groups, spec):
    """The five publication groups as key lists.

    The seeded order is the one the current split already consumes, so the first two
    groups are its test and validation groups and fold 0 reproduces it exactly. The
    remaining keys are sorted before being cut into thirds, which keeps the three
    chunk boundaries stable whatever the permutation did.
    """
    keys = list(groups)
    order = torch.randperm(len(keys),
                           generator=torch.Generator().manual_seed(spec.split_seed)).tolist()
    n = sum(len(indices) for indices in groups.values())
    n_test = max(MIN_SPLIT, int(TEST_FRACTION * n))
    n_val = max(MIN_SPLIT, int(VAL_FRACTION * n))

    test, val, rest = [], [], []
    n_test_seen = n_val_seen = 0
    for k in order:
        key = keys[k]
        if n_test_seen < n_test:
            test.append(key)
            n_test_seen += len(groups[key])
        elif n_val_seen < n_val:
            val.append(key)
            n_val_seen += len(groups[key])
        else:
            rest.append(key)
    rest.sort()
    third = len(rest) // 3
    return [test, val, rest[:third], rest[third:2 * third], rest[2 * third:]]


def parent_folds(dataset, spec):
    """Fold indices per real record, grouped by source publication, for ``spec.fold``.

    A publication split across folds lets a model memorise the paper, so each source is
    one group and children follow their parent. Group 0 is the held-out fold (AC-H2):
    fold 0 is the current split, and folds 1 to 4 rotate test and validation over groups
    1 to 4 only, so the held-out recipes train and validate in no stage.
    """
    groups = _publication_groups(dataset)
    key_groups = _fold_groups(groups, spec)

    if spec.fold == 0:
        test, val, train = 0, 1, [2, 3, 4]
    else:
        test = spec.fold
        # (fold + 1) mod 5 would be group 0 for fold 4, which AC-H2 keeps out of use.
        val = 1 + spec.fold % (N_FOLDS - 1)
        train = [g for g in range(1, N_FOLDS) if g not in (test, val)]
    if spec.fold:
        assert 0 not in train and val != 0

    chosen = {"test": [test], "val": [val], "train": train}
    return {name: [i for g in idx for key in key_groups[g] for i in groups[key]]
            for name, idx in chosen.items()}


def split_dataset(dataset, spec):
    """
    Three-way split, fixed by ``spec.split_seed`` and never by ``spec.seed``.

    train.py and hipo.py hold out one block for both tuning and the reported number; this
    split gives model selection somewhere else to live.

    A synthetic record (``parent`` set, ADR-019) is routed to its parent's split, otherwise
    a diluted or depleted twin of a held-out recipe lands in train. When no record carries a
    parent the split is the plain ``random_split``.
    """
    n = len(dataset)
    parents = [i for i in range(n) if _record(dataset, i).get("parent") is None]

    if len(parents) == n:
        n_test = max(MIN_SPLIT, int(TEST_FRACTION * n))
        n_val = max(MIN_SPLIT, int(VAL_FRACTION * n))
        # Ask test first to prevent overlap with hipo test split
        test_set, val_set, train_set = random_split(
            dataset, [n_test, n_val, n - n_val - n_test],
            generator=torch.Generator().manual_seed(spec.split_seed))
        return {"train": train_set, "val": val_set, "test": test_set}

    groups = parent_folds(dataset, spec)
    owner = {p: name for name, ps in groups.items() for p in ps}
    for i in range(n):
        par = _record(dataset, i).get("parent")
        if par is not None:
            groups[owner[par]].append(i)
    return {name: Subset(dataset, sorted(idx)) for name, idx in groups.items()}


def _depletion_pairs(spec, subset):
    """Depletion pairs drawn from a split's recipes, one per growth factor present."""
    from src.modules import priors

    return [p for i in range(len(subset)) for p in priors.depletion_pairs(subset[i])]


def _build(spec, dataset_path=None):
    """Build a run under the spec, with the config values restored afterwards."""
    os.environ.update(spec.env())
    overrides = {
        "LATENT_DIM": spec.latent_dim,
        "HIDDEN_DIM": spec.hidden_dim,
        "TSVD_K": spec.tsvd_k,
        "BATCH_SIZE": spec.batch_size,
        "EPOCHS": spec.epochs,
        "COST_INPUT": spec.cost_input,
        "COST_PREGATE": spec.cost_pregate,
        "COST_BOUNDED": spec.cost_bounded,
    }
    with _config_override(**overrides):
        run = vae.build()

    if dataset_path is not None:
        run.dataset = GrowthMediumDataset(json_file=dataset_path)
    elif spec.negatives != "legacy":
        run.dataset = GrowthMediumDataset(json_file=config.DATA_DIR / "database_graded.json")
    return apply_spec(run, spec)


def _toxic_labels(batch, n_pubchem, n_uniprot, toxic):
    """Per-component toxic flags in the gate's own order.

    ``vectorize_batch`` splits by source and ``MultimodalEncoder`` concatenates
    pubchem-then-uniprot along the component axis, so the gate axis is that same
    order and the two line up index for index.
    """
    labels = []
    for components in batch["components"]:
        pubchem = [c for c in components if c["component_source"] == "pubchem"]
        uniprot = [c for c in components if c["component_source"] == "uniprot"]
        labels.extend(c["component"] in toxic for c in pubchem[:n_pubchem])
        labels.extend(c["component"] in toxic for c in uniprot[:n_uniprot])
    return labels


def _auc(scores, labels):
    """Rank AUC with ties averaged. None when one class is absent."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = rankdata(scores)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _collate(batch):
    """``collate_fn`` plus the per-recipe (p, s, d) the graded objective reads."""
    collated = collate_fn(batch)
    collated["targets"] = [item["targets"] for item in batch]
    return collated


def train_spec(spec, dataset_path=None, verbose=True):
    """Train one arm from scratch. Returns the run, its store and the split loaders."""
    from src.vector_store import VectorStore

    seed_everything(spec.seed)
    run = _build(spec, dataset_path)
    store = VectorStore.from_run(run)
    toxic = set(store.toxic_compounds)
    if not spec.blacklist:
        # Kept as labels, dropped as a decode-time filter.
        store.toxic_compounds = set()

    model, optimizer, device = run.model, run.optimizer, run.device
    splits = split_dataset(run.dataset, spec)
    loaders = {
        name: DataLoader(subset, batch_size=run.batch_size,
                         shuffle=(name == "train"), collate_fn=_collate)
        for name, subset in splits.items()
    }

    pairs = _depletion_pairs(spec, splits["train"]) if spec.augment else []

    def prolif_logit(comps):
        x1 = [store.make_x1_vec(c, run.device) for c in comps if c["component_source"] == "pubchem"]
        x2 = [store.make_x2_vec(c, run.device) for c in comps if c["component_source"] == "uniprot"]
        if not x1 or not x2:
            return None
        x1, x2 = torch.stack(x1).unsqueeze(0), torch.stack(x2).unsqueeze(0)
        _, _, _, pathway, _, _ = model(x1, x2, run.adjacency_matrices, tau=tau)
        return pathway[:, PROLIF_NODE, 0]

    def augment_step(epoch):
        """One depletion-hinge step: logit(full) - logit(depleted) >= margin, on a sample."""
        rng = np.random.default_rng(spec.seed * 1000 + epoch)
        pick = (pairs if len(pairs) <= DEP_PAIRS_PER_EPOCH
                else [pairs[i] for i in rng.choice(len(pairs), DEP_PAIRS_PER_EPOCH, replace=False)])
        parts = []
        for p in pick:
            lf, ld = prolif_logit(p["full"]), prolif_logit(p["depleted"])
            if lf is not None and ld is not None:
                parts.append(depletion_hinge(lf, ld, spec.dep_margin))
        if not parts:
            return 0.0
        loss = spec.dep_w * torch.stack(parts).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        optimizer.zero_grad()
        return float(loss.item())

    tau = TAU_START
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
        running_loss, batch_count = 0.0, 0
        optimizer.zero_grad()

        for batch in loaders["train"]:
            x1_batch, x2_batch = store.vectorize_batch(batch, device)

            # An empty modality makes F.mse_loss average over zero elements and
            # return NaN; 44 of the 165 recipes are one-sided.
            if not any(x1_batch) or not any(x2_batch):
                continue

            x1_in = safe_stack(x1_batch, dim=run.input_dim1)
            x2_in = safe_stack(x2_batch, dim=run.input_dim2)
            real_prob = float(batch["proliferation"][0])
            out1, out2, div_args, pathway_logit, gate, cost_pred = model(
                x1_in, x2_in, run.adjacency_matrices, tau=tau
            )

            # beta=0.0 defers the divergence to the pooled step below
            pooled_loss = pooled_loss + loss_reg(
                out1, out2, x1_in, x2_in, div_args, real_prob, pathway_logit, gate,
                alpha=run.pathway_alpha, beta=0.0, gamma=run.gate_gamma,
                recon_w1=run.recon_w1, recon_w2=run.recon_w2,
                ablate_mmd=run.ablate_mmd, cost_pred=cost_pred,
                node_targets=batch["targets"], ident_w2=spec.ident_w2,
                **store.cost_kwargs(batch, out1, out2, run, x1_in, x2_in, tau))

            if run.ablate_mmd:
                pooled_loss = pooled_loss + run.div_beta * kl(*div_args)
            else:
                Yk, Yk_pz = div_args
                pooled_codes.append(Yk.reshape(-1, Yk.size(-1)))
                pooled_prior.append(Yk_pz.reshape(-1, Yk_pz.size(-1)))

            batch_count += 1
            n_pooled += 1

            # store.update_db_from_batch is deliberately not called: it writes a
            # vector index shared by every run, and no metric here reads it.

            if n_pooled < config.MMD_POOL:
                continue
            running_loss += flush_pool()

        if n_pooled:
            running_loss += flush_pool()

        if pairs:
            running_loss += augment_step(epoch)

        tau = max(TAU_FLOOR, tau * TAU_DECAY)
        if verbose:
            avg = running_loss / batch_count if batch_count else 0.0
            print(f"  epoch {epoch + 1:2d}/{run.epochs}  loss {avg:.4f}  tau {tau:.3f}")

    return run, store, loaders, toxic


def cost_codes(run, store, loader, tau=TAU_FLOOR):
    """
    Noise-free codes the head reads and standardized ln(cost) targets of priced components.

    Returns (Z, y, head, recipes): codes (n, d_z), targets (n,), the cost head's
    prediction (n,), and per recipe (predicted, true) ln total USD/L over the
    priced components only, so the two totals sum over the same set.
    """
    model, device = run.model, run.device
    codes, targets, heads, recipes = [], [], [], []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch in loader:
                x1_batch, x2_batch = store.vectorize_batch(batch, device)
                if not any(x1_batch) or not any(x2_batch):
                    continue
                x1_in = safe_stack(x1_batch, dim=run.input_dim1)
                x2_in = safe_stack(x2_batch, dim=run.input_dim2)
                target, mask = store.cost_targets(batch, device)

                # The code the head reads, pre-gate unless the model says otherwise
                Zc = benchmark.mean_code(run, x1_in, x2_in, tau,
                                         gated=not model.cost_pregate)  # (1, n_comp, d_z)
                head = model.read_cost(Zc)                              # (1, n_comp)
                keep = mask[0] > 0
                if keep.any():
                    codes.append(Zc[0][keep].cpu())
                    targets.append(target[0][keep].cpu())
                    heads.append(head[0][keep].cpu())
                    mean, std = store.cost_mean, store.cost_std
                    recipes.append((
                        float(recipe_log_cost(head[:, keep], mean, std)),
                        float(recipe_log_cost(target[:, keep], mean, std)),
                    ))
    finally:
        # Same reason as evaluate: callers keep training or measuring this model.
        model.train(was_training)
    if not codes:
        empty = torch.empty(0)
        return torch.empty(0, run.model.cost_head.in_features), empty, empty, []
    return torch.cat(codes), torch.cat(targets), torch.cat(heads), recipes


def evaluate(run, store, loader, toxic=frozenset(), tau=TAU_FLOOR, probe=None):
    """
    Held-out geometry, reconstruction, gate behaviour and cost readout of one model.

    ``probe`` is the (Z, y) of the training split from ``cost_codes``, the fit set for
    ``cost_probe_r2``. The model runs with ``deterministic=True``, so scoring one checkpoint
    twice gives one row; the mode is restored on exit.
    """
    model = run.model
    was_training = model.training
    model.eval()
    try:
        return _evaluate(run, store, loader, toxic, tau, probe)
    finally:
        # cost() and the training loop both run on this model afterwards; leaving it
        # in eval mode would silently change the gate and any dropout.
        model.train(was_training)


def _evaluate(run, store, loader, toxic, tau, probe):
    """``evaluate``'s body: eval mode, every draw dropped, so one run gives one row."""
    model, device = run.model, run.device
    recon1, recon2, scale1, scale2 = [], [], [], []
    # Slot sums beside the per-recipe means: a recipe weighs its component count,
    # not one, so a 58-component recipe does not count the same as a 5-component one.
    sq1 = sq2 = energy1 = energy2 = 0.0
    n1 = n2 = 0
    n_comp = 0
    ranks, gates, divergences, centres = [], [], [], []
    gate_scores, gate_labels = [], []
    n_recipes = 0

    with torch.no_grad():
        for batch in loader:
            x1_batch, x2_batch = store.vectorize_batch(batch, device)

            if not any(x1_batch) or not any(x2_batch):
                continue

            x1_in = safe_stack(x1_batch, dim=run.input_dim1)
            x2_in = safe_stack(x2_batch, dim=run.input_dim2)
            x1_hat, x2_hat, div_args, _, gate, _ = model(
                x1_in, x2_in, run.adjacency_matrices, tau=tau, deterministic=True
            )

            e1 = F.mse_loss(x1_hat, x1_in).item()
            e2 = F.mse_loss(x2_hat, x2_in).item()
            recon1.append(e1)
            recon2.append(e2)
            scale1.append(x1_in.pow(2).mean().item())
            scale2.append(x2_in.pow(2).mean().item())
            sq1 += e1 * x1_in.numel()
            sq2 += e2 * x2_in.numel()
            energy1 += x1_in.pow(2).sum().item()
            energy2 += x2_in.pow(2).sum().item()
            n1 += x1_in.numel()
            n2 += x2_in.numel()
            n_comp += x1_in.size(1) + x2_in.size(1)
            gates.append(gate.mean().item())
            n_recipes += 1

            gate_scores.extend(gate.reshape(-1).cpu().tolist())
            gate_labels.extend(_toxic_labels(batch, x1_in.size(1), x2_in.size(1), toxic))

            if run.ablate_mmd:
                divergences.append(kl(*div_args).item())
            else:
                Yk, Yk_pz = div_args
                divergences.append(mmd(Yk.reshape(-1, Yk.size(-1)),
                                       Yk_pz.reshape(-1, Yk_pz.size(-1)), SIGMA_REF).item())

            Zc = benchmark.mean_code(run, x1_in, x2_in, tau)        # noise free
            ranks.append(benchmark.effective_rank(Zc).mean().item())
            centres.append(Zc.reshape(-1, Zc.size(-1)).cpu())

    Z = torch.cat(centres)
    lam = benchmark.latent_spectrum(Z)
    share = lam / lam.sum().clamp_min(1e-12)
    active = benchmark.active_directions(lam)

    # Slot-weighted: every component of every recipe contributes one slot, so the
    # pooled MSE is the error per slot and the scale is the input energy per slot.
    recon_x1 = sq1 / max(n1, 1)
    recon_x2 = sq2 / max(n2, 1)
    nmse_x1 = recon_x1 / max(energy1 / max(n1, 1), 1e-12)
    nmse_x2 = recon_x2 / max(energy2 / max(n2, 1), 1e-12)
    recon_recipe_mean = float(np.mean(recon1) + np.mean(recon2))
    nmse2_recipe_mean = float(np.mean(recon2) / max(np.mean(scale2), 1e-12))

    Z_cost, y_cost, head_cost, recipes = cost_codes(run, store, loader, tau)
    trained_head = run.cost_delta > 0
    fit = benchmark.cost_fit(head_cost, y_cost, torch.ones_like(y_cost),
                             store.cost_std or 1.0) if trained_head else {}
    rho = (benchmark.recipe_cost_rank([p for p, _ in recipes], [t for _, t in recipes])
           if trained_head else None)
    probe_r2 = (benchmark.cost_probe(probe[0], probe[1], Z_cost, y_cost)
                if probe is not None else None)

    return {
        # Cost readout (ADR-005). The head metrics are None in a no_cost arm,
        # whose head never trained; the probe is comparable across every arm.
        "cost_r2": fit.get("r2"),
        "cost_mae_dec": fit.get("mae_decades"),
        "recipe_cost_rho": rho,
        "cost_probe_r2": probe_r2,
        "recon": float(recon_x1 + recon_x2),
        "recon_x1": float(recon_x1),
        "recon_x2": float(recon_x2),
        "nmse_x1": float(nmse_x1),
        "nmse_x2": float(nmse_x2),
        "balance": float(nmse_x2 / max(nmse_x1, 1e-12)),
        "recon_recipe_mean": recon_recipe_mean,
        "nmse2_recipe_mean": nmse2_recipe_mean,
        "comp_per_recipe": float(n_comp / max(n_recipes, 1)),
        "active": int(active),
        "d_z": int(lam.numel()),
        "collapsed": float(active < benchmark.COLLAPSE_FRACTION * lam.numel()),
        "pr": float(lam.sum().pow(2) / lam.pow(2).sum().clamp_min(1e-12)),
        "dims95": int((share.cumsum(0) < 0.95).sum()) + 1,
        "eff_rank": float(np.mean(ranks)),
        "gate_open": float(np.mean(gates)),
        "div_loss": float(np.mean(divergences)),
        # Whether the gate separates toxic components from the rest. Under
        # use_svd the code mixes across the whole recipe, so a high value there
        # can be recipe recognition; without the SVD each gate sees only its own
        # component, and the gap between the two arms is the thing to read.
        "gate_auc": _auc(gate_scores, gate_labels),
        "n_recipes": n_recipes,
    }


def evaluate_checkpoint(path, split="test", dataset_path=None, tau=TAU_FLOOR, fold=None):
    """
    Score a saved checkpoint without retraining it.

    The spec travels inside the checkpoint, so the architecture is rebuilt as it
    was trained. Every arm has identical ``state_dict`` keys, which means a
    mismatched pair would otherwise load cleanly and score silently wrong.
    """
    from src.vector_store import VectorStore

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "spec" not in payload:
        raise ValueError(f"{path} carries no spec; it predates the ablation harness")

    spec = AblationSpec(**payload["spec"])
    if fold is not None:
        spec = replace(spec, fold=fold)
    seed_everything(spec.seed)
    run = _build(spec, dataset_path)
    run.model.load_state_dict(payload["state_dict"])

    store = VectorStore.from_run(run)
    toxic = set(store.toxic_compounds)
    if not spec.blacklist:
        store.toxic_compounds = set()

    splits = split_dataset(run.dataset, spec)
    train = DataLoader(splits["train"], batch_size=run.batch_size, shuffle=False,
                       collate_fn=collate_fn)
    loader = DataLoader(splits[split], batch_size=run.batch_size, shuffle=False,
                        collate_fn=collate_fn)
    probe = cost_codes(run, store, train, tau)[:2]
    return spec, evaluate(run, store, loader, toxic, tau, probe)


def log_metrics(spec, split, metrics, costs, wall_clock, path=RUN_LOG):
    """Append one row per metric, flushed immediately so a crash keeps the rest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    base = {
        "config_id": spec.config_id,
        "label": spec.label,
        "seed": spec.seed,
        "split": split,
        "fold": spec.fold,
        "negatives": spec.negatives,
        "readout": spec.readout,
        **spec.flags,
        "logvar_head": spec.logvar_head,
        "tsvd_k": spec.tsvd_k,
        "pathway_alpha": spec.pathway_alpha,
        "div_beta": spec.div_beta,
        "gate_gamma": spec.gate_gamma,
        "recon_w1": spec.recon_w1,
        "recon_w2": spec.recon_w2,
        "latent_dim": spec.latent_dim,
        "hidden_dim": spec.hidden_dim,
        "batch_size": spec.batch_size,
        "readout_width": spec.readout_width,
        "cost_delta": spec.cost_delta,
        "cost_input": spec.cost_input,
        "cost_pregate": spec.cost_pregate,
        "cost_bounded": spec.cost_bounded,
        "cost_consist": spec.cost_consist,
        "cost_consist_mode": spec.cost_consist_mode,
        "cost_recipe": spec.cost_recipe,
        "augment": spec.augment,
        "dep_w": spec.dep_w,
        "dep_margin": spec.dep_margin,
        "ident_w2": spec.ident_w2,
        "epochs": spec.epochs,
        "split_seed": spec.split_seed,
        **costs,
        "wall_clock_s": round(wall_clock, 3),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    with open(path, "a") as f:
        for name, value in metrics.items():
            if value is None:
                continue
            f.write(json.dumps({**base, "metric": name, "value": float(value)}) + "\n")
        f.flush()

    return path


def load_rows(path=RUN_LOG):
    """The long-format table, as a list of dicts."""
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def completed_runs(path=RUN_LOG):
    """``(config_id, seed, fold)`` triples already on disk, so a sweep can skip them.

    Rows logged before folds existed carry no ``fold`` and count as fold 0.
    """
    return {(row["config_id"], row["seed"], row.get("fold", 0)) for row in load_rows(path)}


def run_spec(spec, dataset_path=None, save=True, path=RUN_LOG, verbose=True):
    """Train one arm, score val and test, log both, optionally checkpoint."""
    started = time.perf_counter()
    run, store, loaders, toxic = train_spec(spec, dataset_path, verbose)
    wall_clock = time.perf_counter() - started

    reference = _reference_batch(run, store, loaders["train"])
    costs = cost(run, *reference)

    # Unshuffled pass over the training split, the probe's fit set
    train = DataLoader(loaders["train"].dataset, batch_size=run.batch_size, shuffle=False,
                       collate_fn=collate_fn)
    probe = cost_codes(run, store, train)[:2]

    results = {}
    for split in ("val", "test"):
        metrics = evaluate(run, store, loaders[split], toxic, probe=probe)
        results[split] = metrics
        log_metrics(spec, split, metrics, costs, wall_clock, path)

    if save:
        torch.save({"state_dict": run.model.state_dict(),
                    "spec": asdict(spec),
                    "config_id": spec.config_id},
                   run.checkpoint)

    return run, results, costs


def _reference_batch(run, store, loader):
    """First usable recipe of the training split, as the FLOP reference input."""
    for batch in loader:
        x1_batch, x2_batch = store.vectorize_batch(batch, run.device)
        if any(x1_batch) and any(x2_batch):
            return (safe_stack(x1_batch, dim=run.input_dim1),
                    safe_stack(x2_batch, dim=run.input_dim2))
    raise RuntimeError("no recipe in the training split has both modalities")


def run_study(specs, seeds=(0,), dataset_path=None, path=RUN_LOG,
              resume=True, save=True, verbose=True):
    """Every spec at every seed, skipping whatever the log already holds."""
    done = completed_runs(path) if resume else set()
    planned = [replace(spec, seed=seed) for spec in specs for seed in seeds]
    results = []

    for i, spec in enumerate(planned, start=1):
        key = (spec.config_id, spec.seed, spec.fold)
        if key in done:
            if verbose:
                print(f"[{i}/{len(planned)}] {spec.label} seed={spec.seed}"
                      f" fold={spec.fold}  already logged, skipping")
            continue

        if verbose:
            print(f"[{i}/{len(planned)}] {spec.label} seed={spec.seed}"
                  f" fold={spec.fold}  ({spec.config_id})")
        _, metrics, costs = run_spec(spec, dataset_path, save, path, verbose)
        results.append((spec, metrics, costs))
        done.add(key)

    return results
