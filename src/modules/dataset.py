"""
Growth-medium recipes dataset & collation helpers the loaders need.

A recipe is a variable-length list of components, each carrying a source
(``pubchem`` or ``uniprot``), a concentration in mg/L, a cost in USD/L (None
where no price is known, or for a database without costs), and a pre-computed
embedding. Because recipes differ in length, batches are kept as lists and
stacked per modality by :func:`safe_stack` once they have been vectorized.
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

__all__ = ["GrowthMediumDataset", "collate_fn", "safe_stack", "targets"]


def targets(entry):
    """
    (p, s, d) for a record.

    Graded records carry the three nodes outright; the legacy database carries one label,
    with survival taken equal to it and differentiation as its complement.
    """
    if all(key in entry for key in ("p", "s", "d")):
        return [float(entry["p"]), float(entry["s"]), float(entry["d"])]
    y = float(entry.get("proliferation", 1))
    return [y, y, 1.0 - y]


def safe_stack(tensors, dim=9, device=None):
    if len(tensors) == 0:
        return torch.empty(0, 0, dim, device=device)

    if len(tensors[0]) == 0:
        return torch.empty(len(tensors), 0, dim, device=device)

    return torch.stack([torch.stack(sample) for sample in tensors])


def collate_fn(batch):
    collated = {
        "product": [item["product"] for item in batch],
        "flag": [item["flag"] for item in batch],
        "proliferation": [item["proliferation"] for item in batch],
        "components": [item["components"] for item in batch],
    }
    for key in ("targets", "survival", "differentiation", "source"):
        if any(key in item for item in batch):
            collated[key] = [item.get(key) for item in batch]
    return collated


def _load_records(json_file):
    """Records from the float32 sidecar when it is at least as new as the JSON.

    Parsing the JSON costs ~22 s and several GB per process; the sidecar is written by
    scripts/compact_dataset.py and holds the same records with binary vectors.
    """
    json_file = Path(json_file)
    sidecar = json_file.with_suffix(".pt")
    if sidecar.is_file() and sidecar.stat().st_mtime >= json_file.stat().st_mtime:
        return torch.load(sidecar, map_location="cpu", weights_only=True)
    with open(json_file) as f:
        return json.load(f)


def _component_vec(value):
    """The sidecar already stores float32; the JSON path has to build the tensor."""
    return value if isinstance(value, torch.Tensor) else torch.tensor(value, dtype=torch.float)


class GrowthMediumDataset(Dataset):
    def __init__(self, json_file, transform=None):
        self.data = _load_records(json_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        components_processed = []
        for item in entry["components"]:
            components_processed.append(
                {
                    "component": item["component"],
                    "concentration": item["concentration"],
                    "unit": item["unit"],
                    "component_source": item["component_source"],
                    "component_vec": _component_vec(item["component_vec"]),
                    "cost": item.get("cost"),
                }
            )

        p, s, d = targets(entry)
        sample = {
            "product": entry["product"],
            "flag": entry["flag"],
            "source": entry.get("source", ""),
            "proliferation": p,
            "survival": s,
            "differentiation": d,
            "targets": [p, s, d],
            "components": components_processed,
        }
        if self.transform:
            sample = self.transform(sample)
        return sample
