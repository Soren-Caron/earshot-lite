"""Data-parallel scaling benchmark: training throughput on 1, 2, 4, 8 devices.

    python -m earshot.scaling                          # whatever devices exist (e.g. TPU v5e-8)
    python -m earshot.scaling --sim-cpu-devices 8      # 8 simulated CPU devices

Times the real jitted train step (AdamW + augmentation + BN) on synthetic
batches that are already on device, so it measures device compute and
communication, not the host input pipeline. Two regimes:
    weak   - fixed per-device batch, global batch grows with devices
    strong - fixed global batch, split across more devices
Writes results/m4_scaling_<platform>.json and docs/m4_scaling_<platform>.png.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np


def bench(model_name: str, n_devices: int, global_batch: int, steps: int, warmup: int = 3) -> float:
    """Training samples/second for one (model, devices, batch) config."""
    import jax
    import optax
    from flax import nnx

    from earshot.models import TRAIN_DEFAULTS, build_model
    from earshot.parallel import data_mesh, replicate, shard_batch
    from earshot.train import train_step

    hp = TRAIN_DEFAULTS[model_name]
    model = build_model(model_name, 12)
    optimizer = nnx.Optimizer(model, optax.adamw(hp["lr"], weight_decay=hp["weight_decay"]), wrt=nnx.Param)
    mesh = data_mesh(n_devices)
    replicate(mesh, model, optimizer)
    rng = np.random.default_rng(0)
    x, y = shard_batch(
        mesh,
        rng.normal(size=(global_batch, 49, 40, 1)).astype(np.float32),
        rng.integers(0, 12, global_batch).astype(np.int32),
    )
    key = jax.random.key(0)
    for i in range(warmup):
        train_step(model, optimizer, x, y, jax.random.fold_in(key, i))[0].block_until_ready()
    t0 = time.perf_counter()
    for i in range(steps):
        loss, _ = train_step(model, optimizer, x, y, jax.random.fold_in(key, warmup + i))
    loss.block_until_ready()
    return global_batch * steps / (time.perf_counter() - t0)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["dscnn", "gru", "kwt"])
    p.add_argument("--per-device-batch", type=int, default=128)
    p.add_argument("--global-batch", type=int, default=1024)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--sim-cpu-devices", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=Path("."))
    args = p.parse_args(argv)

    import jax

    if args.sim_cpu_devices:
        jax.config.update("jax_num_cpu_devices", args.sim_cpu_devices)
    from earshot import plots

    devices = jax.devices()
    counts = [n for n in (1, 2, 4, 8) if n <= len(devices)]
    platform = devices[0].platform + (f"-sim{len(devices)}" if args.sim_cpu_devices else "")
    print(f"{len(devices)} x {devices[0].device_kind} ({platform})")

    rows = []
    for name in args.models:
        for n in counts:
            for regime, batch in (("weak", args.per_device_batch * n), ("strong", args.global_batch)):
                sps = bench(name, n, batch, args.steps)
                rows.append({"model": name, "devices": n, "regime": regime, "global_batch": batch, "samples_per_s": sps})
                print(f"{name:6s} {regime:6s} devices {n}  batch {batch:5d}  {sps:10.0f} samples/s", flush=True)
    for r in rows:
        base = next(b for b in rows if b["model"] == r["model"] and b["regime"] == r["regime"] and b["devices"] == 1)
        r["speedup"] = r["samples_per_s"] / base["samples_per_s"]
        r["efficiency"] = r["speedup"] / r["devices"]

    out = {
        "platform": platform,
        "device_kind": devices[0].device_kind,
        "n_devices": len(devices),
        "jax": jax.__version__,
        "per_device_batch": args.per_device_batch,
        "global_batch": args.global_batch,
        "steps": args.steps,
        "rows": rows,
    }
    json_path = args.out_dir / f"results/m4_scaling_{platform}.json"
    png_path = args.out_dir / f"docs/m4_scaling_{platform}.png"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(out, indent=2))
    plots.scaling(out, png_path)
    print(f"wrote {json_path} and {png_path}")


if __name__ == "__main__":
    main()
