"""Stream the Speech Commands tarball into cached log-mel features.

Reads WAVs straight out of the .tar.gz (no extraction), so it never writes
100k+ tiny files to disk. Output in data/cache/:

    {train,val,test}_x.npy   (N, 49, 40) float16 log-mel
    {train,val,test}_y.npy   (N,) int16 index into meta["words"]
    meta.json                word list, split counts, front-end params

Splits follow the dataset's validation_list.txt / testing_list.txt. Silence
clips are random 1 s crops of _background_noise_ at random gain, cut from
disjoint time ranges per split (first 80% / next 10% / last 10% of each file).

Usage: python -m earshot.data.build_cache
"""

import argparse
import io
import json
import tarfile
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

from earshot.data import features as F

SILENCE = "_silence_"
NOISE_DIR = "_background_noise_/"
SPLITS = ("train", "val", "test")
NOISE_RANGES = {"train": (0.0, 0.8), "val": (0.8, 0.9), "test": (0.9, 1.0)}
N_SILENCE = {"train": 4000, "val": 500, "test": 500}
BATCH = 2048


def read_wav(data: bytes) -> np.ndarray:
    sr, x = wavfile.read(io.BytesIO(data))
    assert sr == F.SAMPLE_RATE, sr
    assert x.dtype == np.int16, x.dtype
    return F.pcm16_to_float(x)


def make_silence(noise: list[np.ndarray], split: str, n: int, rng: np.random.Generator) -> np.ndarray:
    lo, hi = NOISE_RANGES[split]
    clips = np.zeros((n, F.CLIP_SAMPLES), dtype=np.float32)
    for i in range(n):
        src = noise[rng.integers(len(noise))]
        a, b = int(lo * len(src)), int(hi * len(src)) - F.CLIP_SAMPLES
        start = rng.integers(a, b)
        # ~10% true digital silence, rest noise at random level
        gain = 0.0 if rng.random() < 0.1 else rng.uniform(0.0, 1.0)
        clips[i] = gain * src[start : start + F.CLIP_SAMPLES]
    return clips


def build(tar_path: Path, out_dir: Path, seed: int = 0) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    feats: list[np.ndarray] = []
    noise: list[np.ndarray] = []
    lists: dict[str, set[str]] = {}
    pending: list[np.ndarray] = []

    def flush():
        if pending:
            feats.append(F.log_mel(np.stack(pending)).astype(np.float16))
            pending.clear()

    with tarfile.open(tar_path, "r:gz") as tar, tqdm(desc="clips", unit="clip") as bar:
        for m in tar:
            if not m.isfile():
                continue
            name = m.name.removeprefix("./")
            if name in ("validation_list.txt", "testing_list.txt"):
                lists[name] = set(tar.extractfile(m).read().decode().split())
            elif not name.endswith(".wav"):
                continue
            elif name.startswith(NOISE_DIR):
                noise.append(read_wav(tar.extractfile(m).read()))
            else:
                paths.append(name)
                pending.append(F.fix_length(read_wav(tar.extractfile(m).read())))
                bar.update()
                if len(pending) == BATCH:
                    flush()
    flush()

    x_all = np.concatenate(feats)
    word_of = [p.split("/")[0] for p in paths]
    words = sorted(set(word_of)) + [SILENCE]
    word_idx = {w: i for i, w in enumerate(words)}
    y_all = np.array([word_idx[w] for w in word_of], dtype=np.int16)

    val, test = lists["validation_list.txt"], lists["testing_list.txt"]
    split_of = np.array(["val" if p in val else "test" if p in test else "train" for p in paths])

    rng = np.random.default_rng(seed)
    counts = {}
    for split in SPLITS:
        sel = split_of == split
        sil = F.log_mel(make_silence(noise, split, N_SILENCE[split], rng)).astype(np.float16)
        x = np.concatenate([x_all[sel], sil])
        y = np.concatenate([y_all[sel], np.full(len(sil), word_idx[SILENCE], np.int16)])
        np.save(out_dir / f"{split}_x.npy", x)
        np.save(out_dir / f"{split}_y.npy", y)
        with open(out_dir / f"{split}_paths.txt", "w") as f:
            f.write("\n".join([p for p, s in zip(paths, sel) if s] + [SILENCE] * len(sil)))
        counts[split] = int(len(y))

    meta = {
        "words": words,
        "counts": counts,
        "n_noise_files": len(noise),
        "features": {
            k: getattr(F, k)
            for k in ("SAMPLE_RATE", "CLIP_SAMPLES", "WIN", "HOP", "N_FFT", "N_MELS", "FMIN", "FMAX", "LOG_EPS", "N_FRAMES")
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(counts))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tar", type=Path, default=Path("data/raw/speech_commands_v0.02.tar.gz"))
    p.add_argument("--out", type=Path, default=Path("data/cache"))
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    build(a.tar, a.out, a.seed)
