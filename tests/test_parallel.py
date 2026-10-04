"""Data parallelism must not change the math: N-device sharded training == single-device training."""

import jax
import numpy as np
import optax
import pytest
from flax import nnx

from earshot.models import build_model
from earshot.parallel import data_mesh, replicate, shard_batch
from earshot.train import evaluate, train_step


def _flat(model):
    return {p: np.asarray(v[...]) for p, v in nnx.to_flat_state(nnx.state(model, (nnx.Param, nnx.BatchStat)))}


@pytest.mark.parametrize("name", ["dscnn", "gru", "kwt"])
def test_sharded_training_matches_single_device(name):
    mesh = data_mesh(8)
    single, sharded = build_model(name, 12, seed=0), build_model(name, 12, seed=0)
    # SGD, not Adam: some gradients are mathematically ~0 (e.g. attention key biases, which softmax
    # ignores), so they are pure float noise from the reduction order. Adam's first step is roughly
    # lr * sign(g), which would turn that noise into full-size updates; SGD keeps updates linear in g.
    opt1 = nnx.Optimizer(single, optax.sgd(0.05), wrt=nnx.Param)
    opt8 = nnx.Optimizer(sharded, optax.sgd(0.05), wrt=nnx.Param)
    replicate(mesh, sharded, opt8)

    rng = np.random.default_rng(0)
    for step in range(3):
        x = rng.normal(size=(32, 49, 40, 1)).astype(np.float32)
        y = rng.integers(0, 12, 32).astype(np.int32)
        key = jax.random.fold_in(jax.random.key(0), step)
        loss1, _ = train_step(single, opt1, x, y, key)
        x8, y8 = shard_batch(mesh, x, y)
        assert len(x8.sharding.device_set) == 8 and not x8.sharding.is_fully_replicated
        loss8, _ = train_step(sharded, opt8, x8, y8, key)
        np.testing.assert_allclose(loss8, loss1, rtol=1e-5)

    a, b = _flat(single), _flat(sharded)
    assert a.keys() == b.keys()
    for k in a:  # params AND BatchNorm running stats (global-batch statistics)
        np.testing.assert_allclose(b[k], a[k], rtol=1e-4, atol=1e-5, err_msg=str(k))

    # params stay replicated on all 8 devices after updates
    kernel = next(v[...] for p, v in nnx.to_flat_state(nnx.state(sharded, nnx.Param)) if p[-1] == "kernel")
    assert kernel.sharding.is_fully_replicated and len(kernel.sharding.device_set) == 8


def test_sharded_eval_pads_ragged_batches():
    mesh = data_mesh(8)
    model = build_model("dscnn", 12)
    x = np.random.default_rng(0).normal(size=(37, 49, 40)).astype(np.float16)
    y = np.zeros(37, np.int32)
    mean, std = np.zeros(40, np.float32), np.ones(40, np.float32)
    _, ref = evaluate(model, x, y, mean, std, batch_size=16)
    replicate(mesh, model)
    _, got = evaluate(model, x, y, mean, std, batch_size=16, mesh=mesh)
    assert got.shape == (37, 12)
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-5)


def test_mesh_rejects_too_many_devices():
    with pytest.raises(ValueError):
        data_mesh(len(jax.devices()) + 1)
