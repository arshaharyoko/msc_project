"""
Generative models for growth-medium recipes.

Two independent model tracks share one data and pathway-graph:

* **MVRRAE_GV**: Rank reducing (truncated-SVD) variational autoencoder.
* **MDDPM_GV**: Denoising diffusion model over the same two modalities.

Both are regularized by a 25-node signalling-pathway R-GCN whose readout nodes
(Proliferation / Survival / Differentiation) are supervised against the
``proliferation`` label.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
