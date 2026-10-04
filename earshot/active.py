"""Pool-based active learning: the model picks which clips are worth labelling.

Simulated oracle: the 12-class training set is the unlabelled pool and its
existing labels are revealed only for clips a strategy selects. Every
strategy starts from the same random 2% seed set; each round retrains the
model from scratch on the labelled set, scores the remaining pool, and buys
labels up to the next budget.

    random   uniform sample                       (baseline)
    entropy  highest predictive entropy
    margin   smallest top-1 minus top-2 probability

The claim being tested, fixed before running: an uncertainty strategy reaches
within 1 point of the full-data model (same recipe, 100% labels) using a
fraction of the labels, and does so with fewer labels than random sampling.
The GRU is the learner because it trains ~5x faster per clip than the DS-CNN
on CPU; every round retrains from scratch.

    python -m earshot.active --full                        # full-data reference
    python -m earshot.active --strategies random entropy margin
    python -m earshot.active --report
"""

import argparse
import json
import time
from pathlib import Path

import jax
import numpy as np
import optax
from flax import nnx

from earshot import plots
from earshot.data.dataset import batches, make_task, norm_stats
from earshot.models import TRAIN_DEFAULTS, build_model
from earshot.train import evaluate, predict, train_step

FRACTIONS = (0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30)
RESULTS = Path("results/m6")


def train_on(model_name, x, y, mean, std, *, min_steps, epochs, seed, batch_size=128):
    """Train from scratch for max(min_steps, epochs over x) steps: small labelled sets get enough updates."""
    hp = TRAIN_DEFAULTS[model_name]
    model = build_model(model_name, 12, seed=seed)
    steps = max(min_steps, epochs * (len(y) // batch_size))
    schedule = optax.warmup_cosine_decay_schedule(0.0, hp["lr"], steps // 20, steps, hp["lr"] * 0.01)
    tx = optax.adamw(schedule, weight_decay=hp["weight_decay"])
    if hp["grad_clip"]:
        tx = optax.chain(optax.clip_by_global_norm(hp["grad_clip"]), tx)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    rng, key, step = np.random.default_rng(seed), jax.random.key(seed), 0
    model.train()
    while step < steps:
        for xb, yb in batches(x, y, batch_size, mean, std, shuffle=True, rng=rng, drop_last=True):
            train_step(model, optimizer, xb, yb, jax.random.fold_in(key, step))
            step += 1
            if step == steps:
                break
    return model, steps


def pool_probs(model, x, mean, std) -> np.ndarray:
    model.eval()
    dummy = np.zeros(len(x), np.int32)
    logits = np.concatenate([np.asarray(predict(model, xb)) for xb, _ in batches(x, dummy, 2048, mean, std, shuffle=False)])
    return np.asarray(jax.nn.softmax(logits, axis=-1))


def acquire(strategy: str, probs: np.ndarray, k: int, rng) -> np.ndarray:
    """Indices (into the pool) of the k clips to label next."""
    if strategy == "random":
        return rng.choice(len(probs), size=k, replace=False)
    if strategy == "entropy":
        score = -(probs * np.log(probs + 1e-12)).sum(-1)  # high = uncertain
    elif strategy == "margin":
        top2 = np.sort(probs, axis=-1)[:, -2:]
        score = -(top2[:, 1] - top2[:, 0])  # small margin = uncertain
    else:
        raise ValueError(strategy)
    return np.argsort(-score, kind="stable")[:k]


def run_strategy(strategy, train, test, mean, std, args) -> dict:
    n = len(train.y)
    rng = np.random.default_rng(args.seed)
    labelled = np.zeros(n, bool)
    labelled[rng.choice(n, size=int(FRACTIONS[0] * n), replace=False)] = True  # same seed set for every strategy
    rounds, selected = [], {}
    for i, frac in enumerate(FRACTIONS):
        t0 = time.perf_counter()
        idx = np.flatnonzero(labelled)
        selected[f"{frac:.2f}"] = idx
        model, steps = train_on(args.model, train.x[idx], train.y[idx], mean, std,
                                min_steps=args.min_steps, epochs=args.epochs, seed=args.seed)
        acc, _ = evaluate(model, test.x, test.y, mean, std)
        rounds.append({
            "frac": frac,
            "n_labelled": int(labelled.sum()),
            "test_acc": acc,
            "steps": steps,
            "class_counts": np.bincount(train.y[idx], minlength=len(train.classes)).tolist(),
            "round_s": time.perf_counter() - t0,
        })
        print(f"{strategy:8s} {frac:5.0%} labelled ({labelled.sum():5d})  test {acc:.4f}  ({rounds[-1]['round_s']:.0f}s)", flush=True)
        if i + 1 < len(FRACTIONS):
            pool = np.flatnonzero(~labelled)
            k = int(FRACTIONS[i + 1] * n) - int(labelled.sum())
            probs = pool_probs(model, train.x[pool], mean, std) if strategy != "random" else np.zeros((len(pool), 1))
            labelled[pool[acquire(strategy, probs, k, rng)]] = True
    np.savez(RESULTS / f"{strategy}_labelled.npz", **selected)  # labelled-set indices per round
    return {"strategy": strategy, "model": args.model, "rounds": rounds, "classes": train.classes,
            "config": {k: v for k, v in vars(args).items() if k != "strategies"}}


def report():
    full = json.loads((RESULTS / "full.json").read_text())
    rows = [json.loads((RESULTS / f"{s}.json").read_text()) for s in ("random", "entropy", "margin") if (RESULTS / f"{s}.json").exists()]
    target = full["test_acc"] - 0.01
    print(f"full data: {full['test_acc']:.4f} ({full['n_labelled']} labels)   target (full - 1pt): {target:.4f}\n")
    print("| Strategy | " + " | ".join(f"{f:.0%}" for f in FRACTIONS) + " | Labels to reach full - 1 pt |")
    print("|---|" + "---|" * (len(FRACTIONS) + 1))
    for r in rows:
        hit = next((x for x in r["rounds"] if x["test_acc"] >= target), None)
        r["reached_target_at"] = hit["frac"] if hit else None
        reach = f"{hit['frac']:.0%} ({hit['n_labelled']:,})" if hit else f"not within {FRACTIONS[-1]:.0%}"
        print(f"| {r['strategy']} | " + " | ".join(f"{x['test_acc']:.1%}" for x in r["rounds"]) + f" | {reach} |")
    Path("results/m6_active.json").write_text(json.dumps({"full": full, "target": target, "strategies": rows}, indent=2))
    plots.active_learning(rows, full, Path("docs/m6_active.png"))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--strategies", nargs="*", default=[])
    p.add_argument("--full", action="store_true", help="train the 100%%-labels reference with the same recipe")
    p.add_argument("--report", action="store_true")
    p.add_argument("--model", default="gru")
    p.add_argument("--min-steps", type=int, default=1500)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    if args.full or args.strategies:
        train, test = make_task("train"), make_task("test")
        mean, std = norm_stats(train.x)  # pool-based AL: unlabelled features are available
        RESULTS.mkdir(parents=True, exist_ok=True)
    if args.full:
        t0 = time.perf_counter()
        model, steps = train_on(args.model, train.x, train.y, mean, std, min_steps=args.min_steps, epochs=args.epochs, seed=args.seed)
        acc, _ = evaluate(model, test.x, test.y, mean, std)
        print(f"full     100% labelled ({len(train.y)})  test {acc:.4f}  ({time.perf_counter() - t0:.0f}s)", flush=True)
        (RESULTS / "full.json").write_text(json.dumps({"n_labelled": len(train.y), "test_acc": acc, "steps": steps, "model": args.model}))
    for s in args.strategies:
        (RESULTS / f"{s}.json").write_text(json.dumps(run_strategy(s, train, test, mean, std, args), indent=2))
    if args.report:
        report()


if __name__ == "__main__":
    main()
