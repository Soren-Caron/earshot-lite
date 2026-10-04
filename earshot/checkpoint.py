"""Save/load models as a flat .npz + config.json.

Keys are slash-joined module paths ("stem/conv/kernel", "blocks/0/layers/1/bn/mean"),
so the weights are easy to read from numpy-only tooling such as the C exporter.
"""

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from flax import nnx

from earshot.models import build_model

_SAVED = (nnx.Param, nnx.BatchStat)


def _key(path) -> str:
    return "/".join(str(p) for p in path)


def save(model: nnx.Module, run_dir: Path, config: dict, name: str = "model") -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    flat = nnx.to_flat_state(nnx.state(model, _SAVED))
    np.savez(run_dir / f"{name}.npz", **{_key(p): np.asarray(v[...]) for p, v in flat})
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))


def load_weights(model: nnx.Module, path: Path) -> nnx.Module:
    data = np.load(path)
    flat = nnx.to_flat_state(nnx.state(model, _SAVED))
    missing = {_key(p) for p, _ in flat} ^ set(data.files)
    if missing:
        raise KeyError(f"checkpoint/model key mismatch: {sorted(missing)}")
    for p, v in flat:
        v.set_value(jnp.asarray(data[_key(p)]))
    nnx.update(model, nnx.from_flat_state(flat))
    return model


def load(run_dir: Path, name: str = "model") -> tuple[nnx.Module, dict]:
    """Rebuild a model from run_dir/config.json and load its weights (eval mode)."""
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    model = build_model(config["model"], len(config["classes"]), **config.get("model_kwargs", {}))
    load_weights(model, run_dir / f"{name}.npz")
    model.eval()
    return model, config
