"""latent_space_characterization.main() with figures and scalars sent to separate paths, so paper/his_lsc and artifacts/figures stay as they are."""

import functools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import latent_space_characterization as lsc
from src import config

config.FIGURE_DIR = config.ARTIFACT_DIR / "figures/latent"
lsc.save_results = functools.partial(lsc.save_results,
                                     filename="latent_characterization.json")
lsc.save_cluster_recipes = functools.partial(lsc.save_cluster_recipes,
                                             filename="latent_cluster_recipes.json")

if __name__ == "__main__":
    lsc.main()
