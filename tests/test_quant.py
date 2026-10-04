"""Quantization + C engine tests.

Unit tests use a randomly initialized DS-CNN with randomized BatchNorm stats,
so they need neither the dataset nor a trained run. The integration test at
the bottom runs the real checkpoint on real test clips and is skipped when
runs/dscnn-m1 or data/cache are missing.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from earshot import checkpoint
from earshot.models import build_model
from earshot.quant.cbuild import build_runner, run_c
from earshot.quant.export_c import write_model_c
from earshot.quant.graph import fold_dscnn, forward_float, same_pad
from earshot.quant.quantize import (
    calibrate,
    forward_int8,
    qparams,
    quantize_input,
    quantize_model,
    quantize_multiplier,
    requantize,
)

RUN = Path("runs/dscnn-m1")
CACHE = Path("data/cache")


@pytest.fixture(scope="module")
def random_model(tmp_path_factory):
    """Untrained DS-CNN whose BN stats/affine params are randomized, saved as a checkpoint."""
    d = tmp_path_factory.mktemp("model")
    model = build_model("dscnn", 12, seed=3)
    rng = np.random.default_rng(0)
    for path, var in nnx.to_flat_state(nnx.state(model, (nnx.Param, nnx.BatchStat))):
        if path[-2] == "bn":
            v = var[...]
            if path[-1] == "var":
                var.set_value(jnp.asarray(rng.uniform(0.5, 2.0, v.shape), jnp.float32))
            else:
                var.set_value(jnp.asarray(rng.normal(0, 0.3, v.shape) + (path[-1] == "scale"), jnp.float32))
    model.eval()
    checkpoint.save(model, d, {"model": "dscnn", "classes": [f"c{i}" for i in range(12)]})
    return model, d


def _random_logmel(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(-3.0, 3.0, (n, 49, 40)).astype(np.float32)


@pytest.fixture(scope="module")
def random_qmodel(random_model):
    _, d = random_model
    layers = fold_dscnn(d / "model.npz")
    mean, std = np.full(40, -3.0, np.float32), np.full(40, 3.0, np.float32)
    calib = ((_random_logmel(64, seed=1) - mean) / std)[..., None]
    qm = quantize_model(layers, calibrate(layers, calib), mean, std, [f"c{i}" for i in range(12)])
    return layers, qm


# --- pieces -------------------------------------------------------------------


def test_same_pad_matches_xla():
    assert same_pad(49, 10, 2) == (4, 5, 25)
    assert same_pad(40, 4, 2) == (1, 1, 20)
    assert same_pad(25, 3, 1) == (1, 1, 25)


def test_qparams_represents_zero_exactly():
    for lo, hi in [(-2.09, 3.52), (0.0, 12.1), (-44.9, 33.7), (0.5, 3.0)]:
        scale, zp = qparams(lo, hi)
        assert -128 <= zp <= 127
        assert np.round(0.0 / scale) + zp == zp  # real 0 -> integer zp, so zero padding is exact
        assert (127 - zp) * scale >= max(hi, 0) - scale and (-128 - zp) * scale <= min(lo, 0) + scale


def test_quantize_multiplier_precision():
    m = np.geomspace(1e-6, 0.99, 200)
    m0, shift = quantize_multiplier(m)
    assert ((m0 >= 2**30) & (m0 < 2**31)).all()
    np.testing.assert_allclose(m0 / 2.0**shift, m, rtol=2**-30)


def test_requantize_is_round_half_up():
    rng = np.random.default_rng(0)
    acc = rng.integers(-(2**24), 2**24, 1000)
    m0, shift = quantize_multiplier(rng.uniform(1e-4, 0.5, 1000))
    got = requantize(acc, m0, shift)
    from fractions import Fraction
    from math import floor

    want = [floor(Fraction(int(a) * int(m), 2 ** int(s)) + Fraction(1, 2)) for a, m, s in zip(acc, m0, shift)]
    assert got.tolist() == want


# --- float graph --------------------------------------------------------------


def test_folded_graph_matches_jax(random_model):
    model, d = random_model
    x = jax.random.normal(jax.random.key(0), (8, 49, 40, 1))
    np.testing.assert_allclose(forward_float(fold_dscnn(d / "model.npz"), np.asarray(x)), model(x), rtol=1e-4, atol=1e-4)


def test_int8_reference_tracks_float(random_qmodel):
    layers, qm = random_qmodel
    logmel = _random_logmel(32, seed=2)
    xf = ((logmel - qm.norm_mean) / qm.norm_std)[..., None]
    f = forward_float(layers, xf)
    q = (forward_int8(qm, quantize_input(qm, logmel)).astype(np.float32) - qm.output_zp) * qm.output_scale
    rel = np.abs(q - f).mean() / np.abs(f).mean()
    assert rel < 0.1, rel


# --- C engine -----------------------------------------------------------------


def test_c_bit_exact_on_random_model(random_qmodel, tmp_path):
    _, qm = random_qmodel
    exe = build_runner(write_model_c(qm, tmp_path / "model.c"), tmp_path / "runner.exe")
    logmel = _random_logmel(64, seed=3)
    logmel[0] = np.log(1e-6)  # digital silence: saturates the input quantizer
    logmel[1] = 50.0  # out-of-range loud input: clips at +127
    c_logits, _ = run_c(exe, logmel, len(qm.classes))
    np.testing.assert_array_equal(c_logits, forward_int8(qm, quantize_input(qm, logmel)))


@pytest.mark.skipif(not (RUN / "model.npz").exists() or not (CACHE / "meta.json").exists(), reason="needs trained run + data cache")
def test_trained_model_c_parity_and_accuracy(tmp_path):
    from earshot.data.dataset import make_task
    from earshot.quant.pipeline import normalize, quantize_run

    layers, qm, norm = quantize_run(RUN)
    test = make_task("test")
    idx = np.random.default_rng(0).choice(len(test.y), 600, replace=False)
    x, y = test.x[idx].astype(np.float32), test.y[idx]

    ref = forward_int8(qm, quantize_input(qm, x))
    exe = build_runner(write_model_c(qm, tmp_path / "model.c"), tmp_path / "runner.exe")
    c_logits, _ = run_c(exe, x, len(qm.classes))
    np.testing.assert_array_equal(c_logits, ref)  # C == int8 spec, bit for bit

    f_pred = forward_float(layers, normalize(x, norm)).argmax(-1)
    q_pred = ref.argmax(-1)
    assert (q_pred == f_pred).mean() >= 0.98
    assert (f_pred == y).mean() - (q_pred == y).mean() < 0.015
