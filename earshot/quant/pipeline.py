"""Checkpoint -> folded float graph -> calibrated int8 QModel."""

from pathlib import Path

import numpy as np

from earshot.data.dataset import make_task
from earshot.quant.graph import fold_dscnn
from earshot.quant.quantize import QModel, calibrate, quantize_model


def normalize(logmel: np.ndarray, norm) -> np.ndarray:
    return ((np.asarray(logmel, np.float32) - norm["mean"]) / norm["std"])[..., None]


def quantize_run(run_dir: Path, n_calib: int = 2000, seed: int = 0, cache_dir: Path = Path("data/cache")):
    """Returns (float layers, QModel, norm). Calibrates on a fixed random subset of the training split."""
    run_dir = Path(run_dir)
    norm = dict(np.load(run_dir / "norm.npz"))
    layers = fold_dscnn(run_dir / "model.npz")
    train = make_task("train", cache_dir=cache_dir, seed=seed)
    idx = np.sort(np.random.default_rng(seed).choice(len(train.y), size=n_calib, replace=False))
    ranges = calibrate(layers, normalize(train.x[idx], norm))
    qm: QModel = quantize_model(layers, ranges, norm["mean"], norm["std"], train.classes)
    return layers, qm, norm
