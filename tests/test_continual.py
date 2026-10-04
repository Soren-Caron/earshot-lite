import jax
import numpy as np
import optax
from flax import nnx

from earshot.continual import CLASSES, TASK_OF_CLASS, TASKS, cl_step, evaluate, seen_mask
from earshot.models import build_model


def test_task_layout():
    assert len(CLASSES) == 10 and len(set(CLASSES)) == 10
    assert seen_mask(0).sum() == len(TASKS[0])
    assert seen_mask(len(TASKS) - 1).all()


def test_unseen_classes_get_no_gradient():
    model = build_model("dscnn", len(CLASSES))
    opt = nnx.Optimizer(model, optax.adam(1e-2), wrt=nnx.Param)  # no weight decay, so only gradients move weights
    before = np.asarray(model.head.kernel[...]).copy()
    rng = np.random.default_rng(0)
    x = rng.normal(size=(16, 49, 40, 1)).astype(np.float32)
    y = rng.integers(0, len(TASKS[0]), 16).astype(np.int32)
    for i in range(3):
        cl_step(model, opt, x, y, jax.random.key(i), seen_mask(0))
    after = np.asarray(model.head.kernel[...])
    unseen = ~np.asarray(seen_mask(0))
    np.testing.assert_array_equal(after[:, unseen], before[:, unseen])
    assert np.abs(after[:, ~unseen] - before[:, ~unseen]).max() > 0


def test_evaluate_ignores_future_tasks_and_never_predicts_unseen():
    model = build_model("dscnn", len(CLASSES))
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(len(CLASSES)), 4).astype(np.int32)
    x = rng.normal(size=(len(y), 49, 40)).astype(np.float16)
    mean, std = np.zeros(40, np.float32), np.ones(40, np.float32)
    r = evaluate(model, x, y, mean, std, stage=0)
    assert len(r["acc_per_task"]) == 1 and r["acc_old"] is None
    r1 = evaluate(model, x, y, mean, std, stage=1)
    assert len(r1["acc_per_task"]) == 2 and r1["acc_old"] is not None
    assert TASK_OF_CLASS.tolist() == [0] * 6 + [1] * 2 + [2] * 2
