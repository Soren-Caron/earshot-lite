"""Compare trained models: accuracy vs size vs compute vs speed.

    python -m earshot.benchmark runs/dscnn-m1 runs/gru-m3 runs/kwt-m3

Re-measures every model's latency in one process so the numbers are comparable,
adds the int8 C engine row from <dscnn run>/quant_report.json if present, and
writes results/m3_benchmark.json + docs/m3_models.png.
"""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from earshot import checkpoint, plots
from earshot.flops import model_macs
from earshot.models import count_params
from earshot.train import predict

NAMES = {"dscnn": "DS-CNN", "gru": "GRU", "kwt": "KWT (transformer)"}


def _median_us(fn, n: int) -> float:
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times) * 1e6)


def latency(model, batch: int, n: int, rounds: int = 3) -> float:
    """Best-of-rounds median wall time per call (us). Best-of guards against efficiency-core scheduling."""
    model.eval()
    x = jnp.zeros((batch, 49, 40, 1), jnp.float32)
    call = lambda: predict(model, x).block_until_ready()  # noqa: E731
    call()
    return min(_median_us(call, n) for _ in range(rounds))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("runs", type=Path, nargs="+")
    p.add_argument("--out-json", type=Path, default=Path("results/m3_benchmark.json"))
    p.add_argument("--out-png", type=Path, default=Path("docs/m3_models.png"))
    args = p.parse_args(argv)

    rows = []
    for run in args.runs:
        model, cfg = checkpoint.load(run)
        metrics = json.loads((run / "metrics.json").read_text())
        params = count_params(model)
        rows.append(
            {
                "name": NAMES.get(cfg["model"], cfg["model"]),
                "model": cfg["model"],
                "run": run.as_posix(),
                "precision": "float32",
                "runtime": "JAX (CPU)",
                "test_acc": metrics["test_acc"],
                "params": params,
                "size_kb": params * 4 / 1024,
                "macs": model_macs(model),
                "latency_us_batch1": latency(model, 1, 300),
                "throughput_us_per_clip_batch512": latency(model, 512, 20) / 512,
            }
        )
        print(f"{rows[-1]['name']:<20} done")

        report = run / "quant_report.json"
        if report.exists():
            q = json.loads(report.read_text())
            rows.append(
                {
                    **rows[-1],
                    "name": rows[-1]["name"] + " int8",
                    "precision": "int8",
                    "runtime": "plain C (1 core)",
                    "test_acc": q["c_acc"],
                    "size_kb": q["bytes"]["total_params_bytes"] / 1024,
                    "latency_us_batch1": q["c_us_per_inference"],
                    "throughput_us_per_clip_batch512": None,
                }
            )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps({"device": str(jax.devices()[0]), "rows": rows}, indent=2)
    )
    plots.model_comparison(rows, args.out_png)

    print()
    print("| Model | Precision / runtime | Test acc | Params | Size | MACs / clip | Latency, batch 1 | Throughput (batch 512) |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        tput = f"{r['throughput_us_per_clip_batch512']:.0f} us/clip" if r["throughput_us_per_clip_batch512"] else "-"
        print(
            f"| {r['name']} | {r['precision']}, {r['runtime']} | {r['test_acc']:.2%} | {r['params']:,} | "
            f"{r['size_kb']:.1f} KB | {r['macs'] / 1e6:.1f}M | {r['latency_us_batch1'] / 1000:.2f} ms | {tput} |"
        )
    print(f"\nwrote {args.out_json} and {args.out_png}")


if __name__ == "__main__":
    main()
