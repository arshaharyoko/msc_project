"""
SMILES -> tensors for the molecular RGCN.

Molecules differ from the signalling pathway in that every sample carries its
own graph, so adjacencies are emitted per-molecule as (R, N, N) and padded into
(B, R, N, N) by `collate`. MessagePassingLayer handles both ranks.
"""

from __future__ import annotations

import torch

try:
    from rdkit import Chem
except ImportError:
    Chem = None

__all__ = [
    "ATOM_TYPES",
    "BOND_TYPES",
    "NUM_RELATIONS",
    "NUM_ATOM_FEATS",
    "MASK_IDX",
    "featurize_smiles",
    "collate",
    "augment",
    "scaffold_key",
]

# "other" catches unseen elements; "mask" is the corruption token for the
# self-supervised objective. Freeze this vocabulary across pretraining and
# fine-tuning, or the atom embedding table stops meaning the same thing.
ATOM_TYPES = ["H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl",
              "Se", "Br", "I", "other", "mask"]
ATOM_TO_IDX = {s: i for i, s in enumerate(ATOM_TYPES)}
MASK_IDX = ATOM_TO_IDX["mask"]

BOND_TYPES = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
NUM_RELATIONS = len(BOND_TYPES)

NUM_ATOM_FEATS = 8


def featurize_smiles(smiles: str, add_hydrogens: bool = False):
    """
    Returns (atom_types [N], atom_feats [N, F], adj [R, N, N]) or None.

    Bonds are undirected, so each adjacency is symmetric and in-degree equals
    degree. No self-loops: MessagePassingLayer's self_weight covers those.
    """
    if Chem is None:
        raise ImportError("rdkit is required for featurization")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if add_hydrogens:
        mol = Chem.AddHs(mol)
    n = mol.GetNumAtoms()
    if n == 0:
        return None

    atom_types = torch.empty(n, dtype=torch.long)
    atom_feats = torch.zeros(n, NUM_ATOM_FEATS)
    for atom in mol.GetAtoms():
        i = atom.GetIdx()
        atom_types[i] = ATOM_TO_IDX.get(atom.GetSymbol(), ATOM_TO_IDX["other"])
        atom_feats[i] = torch.tensor([
            atom.GetFormalCharge() / 2.0,
            atom.GetDegree() / 4.0,
            atom.GetTotalNumHs() / 4.0,
            float(atom.GetIsAromatic()),
            float(atom.IsInRing()),
            float(atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED),
            atom.GetTotalValence() / 6.0,
            float(atom.GetNumRadicalElectrons() > 0),
        ])

    adj = torch.zeros(NUM_RELATIONS, n, n)
    for bond in mol.GetBonds():
        name = str(bond.GetBondType())
        if name not in BOND_TYPES:
            continue  # dative / unspecified bonds dropped
        r = BOND_TYPES.index(name)
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adj[r, i, j] = 1.0
        adj[r, j, i] = 1.0

    return atom_types, atom_feats, adj


def collate(items):
    """
    Pad featurized molecules into a batch.
    -> atom_types (B, N), atom_feats (B, N, F), adj (B, R, N, N), node_mask (B, N)
    """
    batch = len(items)
    max_n = max(item[0].shape[0] for item in items)

    atom_types = torch.zeros(batch, max_n, dtype=torch.long)
    atom_feats = torch.zeros(batch, max_n, NUM_ATOM_FEATS)
    adj = torch.zeros(batch, NUM_RELATIONS, max_n, max_n)
    node_mask = torch.zeros(batch, max_n)

    for b, (types, feats, a) in enumerate(items):
        n = types.shape[0]
        atom_types[b, :n] = types
        atom_feats[b, :n] = feats
        adj[b, :, :n, :n] = a
        node_mask[b, :n] = 1.0

    return atom_types, atom_feats, adj, node_mask


def augment(item, mask_rate: float = 0.15, edge_drop_rate: float = 0.15):
    """
    Corrupted view for contrastive pretraining: masked atoms + dropped bonds.

    atom_feats also carries RDKit's per-atom structure summary (degree, ring, aromatic,
    valence), so it has to be corrupted alongside the graph.
    """
    atom_types, atom_feats, adj = (t.clone() for t in item)

    if edge_drop_rate > 0:
        # Drop symmetrically so the graph stays undirected.
        upper = torch.triu(torch.ones_like(adj), diagonal=1)
        drop = (torch.rand_like(adj) < edge_drop_rate) * upper
        adj = adj * (1 - (drop + drop.transpose(-1, -2)))
        atom_feats[:, 1] = adj.sum(dim=(0, 2)) / 4.0  # degree must follow surviving bonds

    if atom_types.shape[0] > 1:
        masked = torch.rand(atom_types.shape[0]) < mask_rate
        atom_types[masked] = MASK_IDX
        atom_feats[masked] = 0.0  # a masked atom keeps no fingerprint

    return atom_types, atom_feats, adj


def scaffold_key(smiles):
    """
    Bemis-Murcko scaffold. No random splits because it
    leak analogs across the boundary and inflate every metric.
    """
    from rdkit.Chem.Scaffolds import MurckoScaffold

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
