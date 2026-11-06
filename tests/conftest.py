import json
import os

import pytest
import torch

from src import config, constraints

CHECKPOINT = os.environ.get(
    "CHECKPOINT", str(config.ARTIFACT_DIR / "ablation__baseline__a219542b9118__s0.pth"))
BASE_SOURCE_PREFIX = "Wu, X."


@pytest.fixture(scope="session")
def base_recipe():
    with open(config.DATABASE_JSON) as f:
        records = json.load(f)
    record = next(r for r in records if r.get("source", "").startswith(BASE_SOURCE_PREFIX))
    recipe = [{"component": c["component"],
               "concentration": c["concentration"],
               "unit": c.get("unit"),
               "source": "uniprot" if c["component_source"] == "uniprot" else "pubchem"}
              for c in record["components"]]
    kept, removed = constraints.clean_base(recipe)
    return {"kept": kept, "removed": removed}


@pytest.fixture(scope="session")
def model_state():
    if not os.path.exists(CHECKPOINT):
        pytest.skip(f"checkpoint not present: {CHECKPOINT}")
    import predict
    from src import vae
    from src.ablation import AblationSpec
    from src.vector_store import VectorStore

    run = vae.build()
    store = VectorStore.from_run(run)
    payload = torch.load(CHECKPOINT, map_location=run.device, weights_only=False)
    run.model.load_state_dict(payload["state_dict"])
    run.model.eval()
    store.load()
    spec = AblationSpec(**payload["spec"]) if "spec" in payload else None
    state = predict._constraint_state(run, store, spec, CHECKPOINT)
    ctx = predict.encode_base(run, store)
    ctx.update(state)
    return run, store, ctx
