"""Train a keyword-spotting model on the 12-class Speech Commands task.

Usage:
    python -m earshot.train --model dscnn --epochs 30
    python -m earshot.train --epochs 1 --max-train 2048   # smoke test
    python -m earshot.train --devices 8                    # data-parallel over 8 devices (e.g. TPU)
    python -m earshot.train --devices 8 --sim-cpu-devices 8   # same, on 8 simulated CPU devices
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from earshot import checkpoint, plots
from earshot.parallel import data_mesh, replicate, shard_batch, simulate_cpu_devices
from earshot.data.dataset import batches, make_task, norm_stats
from earshot.models import TRAIN_DEFAULTS, build_model, count_params

SHIFT = 5  # max time shift, frames (100 ms)
TIME_MASK = 8  # max SpecAugment time-mask width, frames
FREQ_MASK = 6  # max SpecAugment freq-mask width, mel bins


def augment(x: jax.Array, key: jax.Array) -> jax.Array:
    """Random time shift (edge-padded) + one time mask + one freq mask. x: (B, T, F, 1) normalized."""
    b, t, f, _ = x.shape
    ks = jax.random.split(key, 5)
    padded = jnp.pad(x, ((0, 0), (SHIFT, SHIFT), (0, 0), (0, 0)), mode="edge")
    shifts = jax.random.randint(ks[0], (b,), 0, 2 * SHIFT + 1)
    x = jax.vmap(lambda a, s: jax.lax.dynamic_slice_in_dim(a, s, t, axis=0))(padded, shifts)

    def band(k_width, k_start, size, max_width):
        width = jax.random.randint(k_width, (b, 1), 0, max_width + 1)
        start = jax.random.randint(k_start, (b, 1), 0, size)
        pos = jnp.arange(size)[None, :]
        return (pos >= start) & (pos < start + width)

    tmask = band(ks[1], ks[2], t, TIME_MASK)[:, :, None, None]
    fmask = band(ks[3], ks[4], f, FREQ_MASK)[:, None, :, None]
    return jnp.where(tmask | fmask, 0.0, x)  # 0 = per-bin mean after normalization


@nnx.jit
def train_step(model, optimizer, x, y, key):
    x = augment(x, key)

    def loss_fn(model):
        logits = model(x)
        return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean(), logits

    (loss, logits), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)
    return loss, (logits.argmax(-1) == y).mean()


@nnx.jit
def predict(model, x):
    return model(x)


def evaluate(model, x, y, mean, std, batch_size: int = 1024, mesh=None) -> tuple[float, np.ndarray]:
    model.eval()
    out = []
    for xb, _ in batches(x, y, batch_size, mean, std, shuffle=False):
        n = len(xb)
        if mesh is not None:  # pad the last batch so it splits evenly across devices
            xb = np.concatenate([xb, np.zeros((-n % mesh.size, *xb.shape[1:]), xb.dtype)])
            xb = shard_batch(mesh, xb)
        out.append(np.asarray(predict(model, xb))[:n])
    logits = np.concatenate(out)
    return float((logits.argmax(-1) == y).mean()), logits


def batch1_latency_us(model, n: int = 300) -> float:
    """Median single-clip jitted inference latency on the default device."""
    model.eval()
    x = jnp.zeros((1, 49, 40, 1), jnp.float32)
    predict(model, x).block_until_ready()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        predict(model, x).block_until_ready()
        times.append(time.perf_counter() - t0)
    return float(np.median(times) * 1e6)


def confusion(y_true, y_pred, n):
    cm = np.zeros((n, n), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="dscnn")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=None, help="default: per-model, see models.TRAIN_DEFAULTS")
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--warmup-epochs", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-train", type=int, default=None, help="cap training examples (smoke tests)")
    p.add_argument("--cache", type=Path, default=Path("data/cache"))
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--devices", type=int, default=1, help="data-parallel over the first N devices")
    p.add_argument("--sim-cpu-devices", type=int, default=0, help="expose N simulated CPU devices (testing)")
    args = p.parse_args(argv)
    if args.sim_cpu_devices:
        simulate_cpu_devices(args.sim_cpu_devices)
    for k, v in TRAIN_DEFAULTS[args.model].items():
        if getattr(args, k) is None:
            setattr(args, k, v)

    run_dir = args.out or Path("runs") / f"{args.model}-{datetime.now():%Y%m%d-%H%M%S}"
    rng = np.random.default_rng(args.seed)

    train, val, test = (make_task(s, cache_dir=args.cache, seed=args.seed) for s in ("train", "val", "test"))
    if args.max_train:
        keep = np.sort(rng.choice(len(train.y), size=min(args.max_train, len(train.y)), replace=False))
        train.x, train.y = train.x[keep], train.y[keep]
    mean, std = norm_stats(train.x)
    print(f"train {len(train.y)}  val {len(val.y)}  test {len(test.y)}  classes {len(train.classes)}")

    model = build_model(args.model, len(train.classes), seed=args.seed)
    n_params = count_params(model)
    steps_per_epoch = len(train.y) // args.batch_size
    total = steps_per_epoch * args.epochs
    schedule = optax.warmup_cosine_decay_schedule(
        0.0, args.lr, int(args.warmup_epochs * steps_per_epoch), total, end_value=args.lr * 0.01
    )
    tx = optax.adamw(schedule, weight_decay=args.weight_decay)
    if args.grad_clip:
        tx = optax.chain(optax.clip_by_global_norm(args.grad_clip), tx)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    print(f"{args.model}: {n_params:,} params, {steps_per_epoch} steps/epoch")

    mesh = None
    if args.devices > 1:
        if args.batch_size % args.devices:
            raise ValueError(f"global batch {args.batch_size} not divisible by {args.devices} devices")
        mesh = data_mesh(args.devices)
        replicate(mesh, model, optimizer)
        print(f"data parallel: {args.batch_size // args.devices} per device on {list(mesh.devices)}")

    config = {
        "model": args.model,
        "model_kwargs": {},
        "classes": train.classes,
        "params": n_params,
        "platform": jax.devices()[0].platform,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez(run_dir / "norm.npz", mean=mean, std=std)

    history, best_val, base_key, step = [], -1.0, jax.random.key(args.seed), 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        losses, accs = [], []
        for xb, yb in batches(train.x, train.y, args.batch_size, mean, std, shuffle=True, rng=rng, drop_last=True):
            if mesh is not None:
                xb, yb = shard_batch(mesh, xb, yb)
            loss, acc = train_step(model, optimizer, xb, yb, jax.random.fold_in(base_key, step))
            losses.append(loss)
            accs.append(acc)
            step += 1
        train_loss, train_acc = float(jnp.mean(jnp.stack(losses))), float(jnp.mean(jnp.stack(accs)))
        epoch_s = time.perf_counter() - t0
        val_acc, _ = evaluate(model, val.x, val.y, mean, std, mesh=mesh)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "epoch_s": epoch_s,
                "samples_per_s": steps_per_epoch * args.batch_size / epoch_s,
            }
        )
        with open(run_dir / "history.jsonl", "a") as f:
            f.write(json.dumps(history[-1]) + "\n")
        marker = ""
        if val_acc > best_val:
            best_val = val_acc
            checkpoint.save(model, run_dir, config)
            marker = " *"
        print(
            f"epoch {epoch:3d}  loss {train_loss:.4f}  train {train_acc:.4f}  val {val_acc:.4f}  ({epoch_s:.1f}s){marker}",
            flush=True,
        )

    model, _ = checkpoint.load(run_dir)
    test_acc, logits = evaluate(model, test.x, test.y, mean, std)
    pred = logits.argmax(-1)
    cm = confusion(test.y, pred, len(test.classes))
    per_class = {c: float(cm[i, i] / max(cm[i].sum(), 1)) for i, c in enumerate(test.classes)}
    metrics = {
        "model": args.model,
        "params": n_params,
        "best_val_acc": best_val,
        "test_acc": test_acc,
        "per_class_test_acc": per_class,
        "confusion": cm.tolist(),
        "latency_us_batch1_jax": batch1_latency_us(model),
        "device": str(jax.devices()[0]),
        "history": history,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plots.training_curves(history, run_dir / "curves.png")
    plots.confusion_matrix(cm, test.classes, run_dir / "confusion.png", title=f"{args.model} test acc {test_acc:.2%}")
    print(f"test acc {test_acc:.4f}  (best val {best_val:.4f})  latency {metrics['latency_us_batch1_jax']:.0f} us")
    print(f"saved to {run_dir}")
    return metrics


if __name__ == "__main__":
    main()
