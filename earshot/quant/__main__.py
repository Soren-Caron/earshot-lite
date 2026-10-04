"""Quantize a trained run, export the C model, and verify the whole chain on the test set.

    python -m earshot.quant runs/dscnn-m1

Checks, in order:
  1. folded float numpy graph == JAX model           (BN folding / padding)
  2. int8 numpy reference vs float                   (quantization cost)
  3. C engine == int8 numpy reference, bit-exact     (the C implementation)
Writes csrc/earshot_model.c and <run>/quant_report.json.
"""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from earshot import checkpoint
from earshot.data.dataset import make_task
from earshot.quant.cbuild import build_runner, run_c
from earshot.quant.export_c import write_model_c
from earshot.quant.graph import forward_float
from earshot.quant.pipeline import normalize, quantize_run
from earshot.quant.quantize import dequantize_logits, forward_int8, model_bytes, quantize_input
from earshot.train import evaluate


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--out", type=Path, default=Path("csrc/earshot_model.c"))
    p.add_argument("--n-calib", type=int, default=2000)
    p.add_argument("--bench-repeat", type=int, default=2)
    args = p.parse_args(argv)

    layers, qm, norm = quantize_run(args.run, n_calib=args.n_calib)
    test = make_task("test")
    xf = normalize(test.x, norm)

    model, _ = checkpoint.load(args.run)
    jax_acc, jax_logits = evaluate(model, test.x, test.y, norm["mean"], norm["std"])
    float_logits = forward_float(layers, xf)
    print(f"[1] folded float vs JAX: max |diff| {np.abs(float_logits - jax_logits).max():.2e}")

    q_logits = forward_int8(qm, quantize_input(qm, test.x))
    float_pred, q_pred = float_logits.argmax(-1), q_logits.argmax(-1)
    float_acc, q_acc = (float_pred == test.y).mean(), (q_pred == test.y).mean()
    agree = (float_pred == q_pred).mean()
    logit_err = np.abs(dequantize_logits(qm, q_logits) - float_logits).mean()
    print(f"[2] float acc {float_acc:.4f}  int8 acc {q_acc:.4f}  top-1 agreement {agree:.4f}  mean |logit err| {logit_err:.3f}")

    model_c = write_model_c(qm, args.out, source=args.run.as_posix())
    with tempfile.TemporaryDirectory() as d:
        exe = build_runner(model_c, Path(d) / "runner.exe")
        c_logits, _ = run_c(exe, test.x, len(qm.classes))
        # best of 5: on hybrid CPUs a trial can land on an efficiency core and run ~2x slower
        us = min(run_c(exe, test.x[:500], len(qm.classes), repeat=args.bench_repeat)[1] for _ in range(5))
    mismatched = int((c_logits != q_logits).any(axis=1).sum())
    c_acc = (c_logits.argmax(-1) == test.y).mean()
    print(f"[3] C vs int8 reference: {mismatched}/{len(test.y)} clips differ  C acc {c_acc:.4f}  {us:.0f} us/inference")

    report = {
        "run": args.run.as_posix(),
        "n_test": int(len(test.y)),
        "jax_acc": float(jax_acc),
        "float_folded_acc": float(float_acc),
        "int8_ref_acc": float(q_acc),
        "c_acc": float(c_acc),
        "int8_vs_float_top1_agreement": float(agree),
        "c_vs_ref_mismatched_clips": mismatched,
        "c_bit_exact": mismatched == 0,
        "c_us_per_inference": us,
        "bytes": model_bytes(qm),
        "input": {"scale": float(qm.input_scale), "zp": qm.input_zp},
        "layers": [
            {"name": L.name, "op": L.op, "out_shape": list(L.out_shape), "out_scale": L.out_scale, "out_zp": L.out_zp}
            for L in qm.layers
        ],
    }
    (args.run / "quant_report.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {model_c} and {args.run / 'quant_report.json'}")
    if mismatched:
        raise SystemExit("C engine is NOT bit-exact with the int8 reference")


if __name__ == "__main__":
    main()
