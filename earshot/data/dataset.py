"""Turn the cached 35-word features into a classification task.

The default task is the standard 12-class KWS benchmark: 10 keywords +
silence + unknown (all other words). Silence and unknown are subsampled to the
mean keyword-class size so no class dominates. `keywords` is a parameter so
later experiments (continual learning) can carve out other word subsets.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from earshot.data.build_cache import SILENCE

KEYWORDS_12 = ("yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go")
UNKNOWN = "_unknown_"
DEFAULT_CACHE = Path("data/cache")


@dataclass
class Task:
    x: np.ndarray  # (N, 49, 40) float16
    y: np.ndarray  # (N,) int32 class index
    classes: list[str]


def load_meta(cache_dir: Path = DEFAULT_CACHE) -> dict:
    return json.loads((Path(cache_dir) / "meta.json").read_text())


def load_split(split: str, cache_dir: Path = DEFAULT_CACHE) -> tuple[np.ndarray, np.ndarray]:
    cache_dir = Path(cache_dir)
    return np.load(cache_dir / f"{split}_x.npy", mmap_mode="r"), np.load(cache_dir / f"{split}_y.npy")


def make_task(
    split: str,
    keywords=KEYWORDS_12,
    *,
    silence: bool = True,
    unknown: bool = True,
    cache_dir: Path = DEFAULT_CACHE,
    seed: int = 0,
) -> Task:
    words = load_meta(cache_dir)["words"]
    x_all, y_word = load_split(split, cache_dir)
    rng = np.random.default_rng(seed)

    classes = list(keywords) + ([SILENCE] if silence else []) + ([UNKNOWN] if unknown else [])
    kw_ids = [words.index(k) for k in keywords]
    sel_idx, sel_y = [], []
    for c, w in enumerate(kw_ids):
        idx = np.flatnonzero(y_word == w)
        sel_idx.append(idx)
        sel_y.append(np.full(len(idx), c))
    target = int(np.mean([len(i) for i in sel_idx]))

    def add_sampled(mask: np.ndarray, cls: str):
        idx = np.flatnonzero(mask)
        idx = np.sort(rng.choice(idx, size=min(target, len(idx)), replace=False))
        sel_idx.append(idx)
        sel_y.append(np.full(len(idx), classes.index(cls)))

    if silence:
        add_sampled(y_word == words.index(SILENCE), SILENCE)
    if unknown:
        add_sampled(~np.isin(y_word, kw_ids + [words.index(SILENCE)]), UNKNOWN)

    idx = np.concatenate(sel_idx)
    order = np.argsort(idx)  # read the memmap sequentially
    return Task(
        x=np.asarray(x_all[idx[order]]),
        y=np.concatenate(sel_y)[order].astype(np.int32),
        classes=classes,
    )


def norm_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-mel-bin mean/std over (examples, frames)."""
    x = x.astype(np.float32)
    return x.mean(axis=(0, 1)), x.std(axis=(0, 1)) + 1e-5


def batches(x, y, batch_size: int, mean, std, *, shuffle: bool, rng=None, drop_last: bool = False):
    """Yield normalized float32 (B, 49, 40, 1) batches and int32 labels."""
    n = len(y)
    order = rng.permutation(n) if shuffle else np.arange(n)
    stop = n - n % batch_size if drop_last else n
    for i in range(0, stop, batch_size):
        b = order[i : i + batch_size]
        xb = (x[b].astype(np.float32) - mean) / std
        yield xb[..., None], y[b]
