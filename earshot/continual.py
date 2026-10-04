"""Continual (class-incremental) keyword learning with a replay buffer.

The model learns keywords in three stages and must keep recognizing the old
ones, with a single head over every class seen so far (no task id at test time):

    T0: yes no up down left right     T1: on off     T2: stop go

Methods, all starting from the same T0 model:
    naive        fine-tune on the new words only          (catastrophic forgetting)
    replay:K     new words + K stored clips per old class; each step trains on
                 half new data, half buffer (experience replay)
    joint        retrain on all data seen so far          (upper bound)

Normalization statistics come from T0 data only (no peeking at future tasks).
The 10-way head is allocated up front, with unseen classes masked out of the
softmax, which is equivalent to growing the head.

    python -m earshot.continual --methods naive replay:10 replay:50 replay:250 joint
    python -m earshot.continual --report        # merge results + chart
"""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from earshot import checkpoint, plots
from earshot.data.dataset import batches, make_task, norm_stats
from earshot.models import build_model
from earshot.train import augment

TASKS = (("yes", "no", "up", "down", "left", "right"), ("on", "off"), ("stop", "go"))
CLASSES = [w for t in TASKS for w in t]
TASK_OF_CLASS = np.array([i for i, t in enumerate(TASKS) for _ in t])
OUT = Path("runs/continual")
RESULTS = Path("results/m5")


@nnx.jit
def cl_step(model, optimizer, x, y, key, seen):
    """Train step with unseen classes masked out of the softmax. seen: (num_classes,) bool."""
    x = augment(x, key)

    def loss_fn(model):
        logits = jnp.where(seen, model(x), -1e9)
        return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


@nnx.jit
def masked_predict(model, x, seen):
    return jnp.where(seen, model(x), -1e9).argmax(-1)


def seen_mask(stage: int) -> jnp.ndarray:
    return jnp.asarray(TASK_OF_CLASS <= stage)


def evaluate(model, x, y, mean, std, stage: int) -> dict:
    """Per-task accuracy over the classes seen through `stage` (single head, no task id)."""
    model.eval()
    seen = seen_mask(stage)
    keep = TASK_OF_CLASS[y] <= stage
    x, y = x[keep], y[keep]
    pred = np.concatenate([np.asarray(masked_predict(model, xb, seen)) for xb, _ in batches(x, y, 1024, mean, std, shuffle=False)])
    correct = pred == y
    per_class = np.array([correct[y == c].mean() for c in range(len(CLASSES)) if TASK_OF_CLASS[c] <= stage])
    per_task = [float(np.mean([correct[y == c].mean() for c in np.flatnonzero(TASK_OF_CLASS == t)])) for t in range(stage + 1)]
    old = per_class[TASK_OF_CLASS[: len(per_class)] < stage]
    return {
        "stage": stage,
        "acc_seen": float(per_class.mean()),  # mean over seen classes
        "acc_per_task": per_task,
        "acc_old": float(old.mean()) if len(old) else None,
        "acc_new": per_task[stage],
    }


def train_stage(model, x, y, mean, std, stage, *, epochs, lr, rng, key, replay=None, batch_size=128, warmup_epochs=0.5):
    """Train on (x, y); if replay=(xr, yr), every step also draws an equal-size batch from the buffer."""
    steps = len(y) // batch_size
    schedule = optax.warmup_cosine_decay_schedule(0.0, lr, max(1, int(warmup_epochs * steps)), steps * epochs, lr * 0.01)
    optimizer = nnx.Optimizer(model, optax.adamw(schedule, weight_decay=1e-4), wrt=nnx.Param)
    seen = seen_mask(stage)
    step = 0
    for _ in range(epochs):
        model.train()
        for xb, yb in batches(x, y, batch_size, mean, std, shuffle=True, rng=rng, drop_last=True):
            if replay is not None:
                xr, yr = replay
                idx = rng.integers(0, len(yr), batch_size)
                xb = np.concatenate([xb, ((xr[idx].astype(np.float32) - mean) / std)[..., None]])
                yb = np.concatenate([yb, yr[idx]])
            cl_step(model, optimizer, xb, yb, jax.random.fold_in(key, step), seen)
            step += 1
    return model


def load_data():
    data = {s: make_task(s, CLASSES, silence=False, unknown=False) for s in ("train", "test")}
    assert data["train"].classes == CLASSES
    return data


def task_slice(task, stage: int, upto: bool = False):
    keep = (TASK_OF_CLASS[task.y] <= stage) if upto else (TASK_OF_CLASS[task.y] == stage)
    return task.x[keep], task.y[keep]


def task0_model(data, args):
    """Train (or load the cached) shared T0 model and the T0-only normalization stats."""
    ckpt = OUT / "task0"
    x0, y0 = task_slice(data["train"], 0)
    if (ckpt / "model.npz").exists():
        norm = np.load(ckpt / "norm.npz")
        model = checkpoint.load_weights(build_model("dscnn", len(CLASSES)), ckpt / "model.npz")
        return model, norm["mean"], norm["std"]
    mean, std = norm_stats(x0)
    model = build_model("dscnn", len(CLASSES), seed=args.seed)
    t0 = time.perf_counter()
    train_stage(model, x0, y0, mean, std, 0, epochs=args.t0_epochs, lr=3e-3, warmup_epochs=1.0,
                rng=np.random.default_rng(args.seed), key=jax.random.key(args.seed))
    print(f"T0 trained in {time.perf_counter() - t0:.0f}s", flush=True)
    checkpoint.save(model, ckpt, {"model": "dscnn", "classes": CLASSES})
    np.savez(ckpt / "norm.npz", mean=mean, std=std)
    return model, mean, std


def run_method(method: str, data, args) -> dict:
    model, mean, std = task0_model(data, args)
    train, test = data["train"], data["test"]
    rng, key = np.random.default_rng(args.seed + 1), jax.random.key(args.seed + 1)
    per_class = int(method.split(":")[1]) if method.startswith("replay") else 0

    history = [evaluate(model, test.x, test.y, mean, std, 0)]
    buffer_x, buffer_y = [], []
    t0 = time.perf_counter()
    for stage in range(1, len(TASKS)):
        if per_class:  # store exemplars of the classes just finished (random, class-balanced)
            x_prev, y_prev = task_slice(train, stage - 1)
            for c in np.unique(y_prev):
                idx = rng.choice(np.flatnonzero(y_prev == c), per_class, replace=False)
                buffer_x.append(x_prev[idx])
                buffer_y.append(y_prev[idx])
        if method == "joint":
            x, y = task_slice(train, stage, upto=True)
        else:
            x, y = task_slice(train, stage)
        replay = (np.concatenate(buffer_x), np.concatenate(buffer_y)) if per_class else None
        train_stage(model, x, y, mean, std, stage, epochs=args.epochs, lr=args.lr, rng=rng,
                    key=jax.random.fold_in(key, stage), replay=replay)
        history.append(evaluate(model, test.x, test.y, mean, std, stage))
        h = history[-1]
        print(f"{method:12s} after T{stage}: seen {h['acc_seen']:.4f}  old {h['acc_old']:.4f}  new {h['acc_new']:.4f}", flush=True)

    n_old_train = int((TASK_OF_CLASS[train.y] < len(TASKS) - 1).sum())
    return {
        "method": method,
        "buffer_per_class": per_class,
        "buffer_clips_final": per_class * int((TASK_OF_CLASS < len(TASKS) - 1).sum()),
        "buffer_frac_of_old_data": per_class * int((TASK_OF_CLASS < len(TASKS) - 1).sum()) / n_old_train,
        "history": history,
        "forgetting_t0": history[0]["acc_per_task"][0] - history[-1]["acc_per_task"][0],
        "train_s": time.perf_counter() - t0,
        "config": {k: v for k, v in vars(args).items() if k != "methods"},
    }


def report():
    rows = [json.loads(p.read_text()) for p in sorted(RESULTS.glob("*.json"))]
    order = {"naive": 0, "joint": 99}
    rows.sort(key=lambda r: order.get(r["method"], 1 + r["buffer_per_class"] / 1e4))
    Path("results").mkdir(exist_ok=True)
    Path("results/m5_continual.json").write_text(json.dumps(rows, indent=2))
    plots.continual(rows, Path("docs/m5_continual.png"), [", ".join(t) for t in TASKS])
    print("| Method | Buffer | T0 acc after T0 -> after T2 | Old classes (final) | New classes (final) | All seen (final) |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        h0, hf = r["history"][0], r["history"][-1]
        buf = f"{r['buffer_per_class']}/class ({r['buffer_frac_of_old_data']:.1%} of old data)" if r["buffer_per_class"] else "-"
        print(f"| {r['method']} | {buf} | {h0['acc_per_task'][0]:.1%} -> {hf['acc_per_task'][0]:.1%} | "
              f"{hf['acc_old']:.1%} | {hf['acc_new']:.1%} | {hf['acc_seen']:.1%} |")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="*", default=[])
    p.add_argument("--report", action="store_true")
    p.add_argument("--t0-epochs", type=int, default=20)
    p.add_argument("--epochs", type=int, default=10, help="epochs per incremental stage (over that stage's data)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    if args.methods:
        data = load_data()
        RESULTS.mkdir(parents=True, exist_ok=True)
        for m in args.methods:
            r = run_method(m, data, args)
            (RESULTS / f"{m.replace(':', '-')}.json").write_text(json.dumps(r, indent=2))
    if args.report:
        report()


if __name__ == "__main__":
    main()
