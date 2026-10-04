import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from earshot import checkpoint
from earshot.flops import count_macs, model_macs
from earshot.models import build_model, count_params
from earshot.train import augment, predict, train_step


def test_dscnn_shape_and_size():
    model = build_model("dscnn", 12)
    assert model(jnp.zeros((2, 49, 40, 1))).shape == (2, 12)
    assert 20_000 < count_params(model) < 30_000


def test_checkpoint_roundtrip(tmp_path):
    model = build_model("dscnn", 12, seed=1)
    # take a train step so BN stats differ from init
    opt = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    x = jax.random.normal(jax.random.key(0), (8, 49, 40, 1))
    train_step(model, opt, x, jnp.arange(8) % 12, jax.random.key(1))
    model.eval()
    checkpoint.save(model, tmp_path, {"model": "dscnn", "classes": list("abcdefghijkl")})
    loaded, _ = checkpoint.load(tmp_path)
    np.testing.assert_allclose(predict(model, x), predict(loaded, x), rtol=1e-6, atol=1e-6)


def test_overfits_tiny_batch():
    model = build_model("dscnn", 12, dropout=0.0)
    opt = nnx.Optimizer(model, optax.adam(3e-3), wrt=nnx.Param)
    x = jax.random.normal(jax.random.key(0), (24, 49, 40, 1))
    y = jnp.arange(24) % 12
    losses = [float(train_step(model, opt, x, y, jax.random.key(0))[0]) for _ in range(60)]
    assert losses[-1] < 0.3 * losses[0]


def test_augment_preserves_shape_and_masks_with_zero():
    x = jnp.ones((16, 49, 40, 1))
    out = augment(x, jax.random.key(0))
    assert out.shape == x.shape
    assert set(np.unique(np.asarray(out)).tolist()) <= {0.0, 1.0}


@pytest.mark.parametrize("name, lo, hi", [("gru", 40_000, 50_000), ("kwt", 90_000, 120_000)])
def test_other_models_shape_size_and_train_step(name, lo, hi):
    model = build_model(name, 12)
    assert lo < count_params(model) < hi
    opt = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    x = jax.random.normal(jax.random.key(0), (4, 49, 40, 1))
    loss, _ = train_step(model, opt, x, jnp.arange(4), jax.random.key(1))
    assert np.isfinite(float(loss))
    model.eval()
    assert predict(model, x).shape == (4, 12)


def test_mac_counter_matches_hand_counts():
    assert model_macs(build_model("dscnn", 12)) == 25 * 20 * 64 * 40 + 4 * (25 * 20 * 64 * 9 + 25 * 20 * 64 * 64) + 64 * 12
    # GRU: per step 3 gates x (input + recurrent) matmuls, 49 steps per layer: scan must be multiplied out
    gru = 49 * 3 * (40 * 64 + 64 * 64) + 49 * 3 * (64 * 64 + 64 * 64) + 64 * 12
    assert model_macs(build_model("gru", 12)) == gru


def test_mac_counter_scan_and_dense():
    w = jnp.ones((8, 16))
    assert count_macs(lambda x: x @ w, jnp.ones((3, 8))) == 3 * 8 * 16
    step = lambda c, _: (c @ jnp.ones((16, 16)), None)  # noqa: E731
    assert count_macs(lambda c: jax.lax.scan(step, c, None, length=7)[0], jnp.ones((2, 16))) == 7 * 2 * 16 * 16
