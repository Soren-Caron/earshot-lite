"""Inference graph as a flat list of layers, plus a numpy forward pass.

`fold_dscnn` turns a trained checkpoint into conv/dwconv/avgpool/fc layers with
BatchNorm folded into the conv weights and biases. The numpy forward here is
the bridge between JAX and the int8 reference: it must match JAX in float,
and the int8 forward in quantize.py reuses `conv_acc` on integer data.

Layouts (NHWC activations, H = time, W = mel):
    conv   w (kh, kw, cin, cout)
    dwconv w (kh, kw, c)
    fc     w (in, out)
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

BN_EPS = 1e-5  # nnx.BatchNorm default; DSCNN doesn't override it


@dataclass
class Layer:
    op: str  # "conv" | "dwconv" | "avgpool" | "fc"
    w: np.ndarray | None = None
    b: np.ndarray | None = None
    stride: tuple[int, int] = (1, 1)
    relu: bool = False
    name: str = field(default="")


def same_pad(n: int, k: int, s: int) -> tuple[int, int, int]:
    """XLA 'SAME' padding: returns (pad_before, pad_after, out_size). Extra padding goes after."""
    out = -(-n // s)
    total = max((out - 1) * s + k - n, 0)
    return total // 2, total - total // 2, out


def conv_acc(x: np.ndarray, w: np.ndarray, stride: tuple[int, int]) -> np.ndarray:
    """Zero-padded SAME conv (or depthwise conv if w.ndim == 3) as a sum over kernel taps.

    x: (N, H, W, C). Works in whatever dtype x/w are in; the int8 path passes
    zero-point-centered integers as float64, where every sum is exact (< 2**53).
    """
    kh, kw = w.shape[:2]
    sh, sw = stride
    n, h, wd, _ = x.shape
    pt, pb, oh = same_pad(h, kh, sh)
    pl, pr, ow = same_pad(wd, kw, sw)
    xp = np.pad(x, ((0, 0), (pt, pb), (pl, pr), (0, 0)))
    cout = w.shape[-1]
    acc = np.zeros((n, oh, ow, cout), dtype=np.result_type(x, w))
    for i in range(kh):
        for j in range(kw):
            patch = xp[:, i : i + sh * (oh - 1) + 1 : sh, j : j + sw * (ow - 1) + 1 : sw, :]
            acc += patch * w[i, j] if w.ndim == 3 else patch @ w[i, j]
    return acc


def _fold(kernel, bn_prefix, p, depthwise=False):
    gamma, beta = p[f"{bn_prefix}/scale"], p[f"{bn_prefix}/bias"]
    mean, var = p[f"{bn_prefix}/mean"], p[f"{bn_prefix}/var"]
    k = gamma / np.sqrt(var + BN_EPS)
    w = kernel[:, :, 0, :] * k if depthwise else kernel * k  # scale each output channel
    return w.astype(np.float32), (beta - mean * k).astype(np.float32)


def fold_dscnn(weights: dict | Path) -> list[Layer]:
    p = dict(np.load(weights)) if isinstance(weights, (str, Path)) else weights
    layers = [Layer("conv", *_fold(p["stem/conv/kernel"], "stem/bn", p), stride=(2, 2), relu=True, name="stem")]
    n_blocks = len({k.split("/")[1] for k in p if k.startswith("blocks/")})
    for i in range(n_blocks):
        pre = f"blocks/{i}/layers"
        dw = _fold(p[f"{pre}/0/conv/kernel"], f"{pre}/0/bn", p, depthwise=True)
        pw = _fold(p[f"{pre}/1/conv/kernel"], f"{pre}/1/bn", p)
        layers.append(Layer("dwconv", *dw, relu=True, name=f"block{i}.dw"))
        layers.append(Layer("conv", *pw, relu=True, name=f"block{i}.pw"))
    layers.append(Layer("avgpool", name="pool"))
    layers.append(Layer("fc", p["head/kernel"].astype(np.float32), p["head/bias"].astype(np.float32), name="head"))
    return layers


def forward_float(layers: list[Layer], x: np.ndarray, collect: bool = False):
    """x: (N, H, W, 1) normalized features. Returns logits, or (logits, [per-layer outputs])."""
    outs = []
    for L in layers:
        if L.op in ("conv", "dwconv"):
            x = conv_acc(x, L.w, L.stride) + L.b
        elif L.op == "avgpool":
            x = x.mean(axis=(1, 2))
        elif L.op == "fc":
            x = x @ L.w + L.b
        if L.relu:
            x = np.maximum(x, 0)
        outs.append(x)
    return (x, outs) if collect else x
