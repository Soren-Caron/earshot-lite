import json

import numpy as np
import pytest

from earshot.data.build_cache import SILENCE
from earshot.data.dataset import UNKNOWN, batches, make_task, norm_stats

WORDS = ["bed", "cat", "go", "no", "stop", "yes", SILENCE]
# per-word counts: plenty of unknowns (bed, cat) and silence
COUNTS = {"bed": 50, "cat": 50, "go": 10, "no": 12, "stop": 8, "yes": 10, SILENCE: 40}


@pytest.fixture
def cache(tmp_path):
    y = np.concatenate([np.full(n, WORDS.index(w), np.int16) for w, n in COUNTS.items()])
    x = np.random.default_rng(0).normal(size=(len(y), 49, 40)).astype(np.float16)
    x[:, 0, 0] = np.arange(len(y))  # tag rows so we can check labels line up
    np.save(tmp_path / "train_x.npy", x)
    np.save(tmp_path / "train_y.npy", y)
    (tmp_path / "meta.json").write_text(json.dumps({"words": WORDS}))
    return tmp_path, y


def test_task_classes_and_balance(cache):
    path, _ = cache
    t = make_task("train", ("yes", "no", "go", "stop"), cache_dir=path)
    assert t.classes == ["yes", "no", "go", "stop", SILENCE, UNKNOWN]
    counts = np.bincount(t.y, minlength=6)
    assert counts[:4].tolist() == [10, 12, 10, 8]
    assert counts[4] == counts[5] == 10  # mean keyword count


def test_labels_line_up_with_rows(cache):
    path, y_word = cache
    t = make_task("train", ("yes", "no", "go", "stop"), cache_dir=path)
    rows = t.x[:, 0, 0].astype(int)
    for row, cls in zip(rows, t.y):
        word = WORDS[y_word[row]]
        name = t.classes[cls]
        assert name == word or (name == UNKNOWN and word in ("bed", "cat"))


def test_no_silence_no_unknown(cache):
    path, _ = cache
    t = make_task("train", ("yes", "no"), silence=False, unknown=False, cache_dir=path)
    assert t.classes == ["yes", "no"] and len(t.y) == 22


def test_batches_normalize(cache):
    path, _ = cache
    t = make_task("train", cache_dir=path, keywords=("yes", "no", "go", "stop"))
    mean, std = norm_stats(t.x)
    xb, yb = next(batches(t.x, t.y, 1000, mean, std, shuffle=False))
    assert xb.shape == (len(t.y), 49, 40, 1) and xb.dtype == np.float32
    np.testing.assert_allclose(xb[:, 1:].mean(axis=(0, 1, 3))[1:], 0, atol=0.2)
