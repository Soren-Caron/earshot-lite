"""Post-training int8 quantization and the integer reference forward pass.

Scheme (TFLite-style):
    weights      int8, symmetric, per output channel     w ~ s_w[c] * q
    activations  int8, asymmetric, per tensor            x ~ s_x * (q - zp_x)
    bias         int32 at scale s_in * s_w[c]
    requant      acc * s_in * s_w / s_out  ~  (acc * M0 + 2**(shift-1)) >> shift
                 with M0 an int32 in [2**30, 2**31), evaluated in int64 (round half up)
    ReLU         fused: clamp the output to [zp_out, 127]

`forward_int8` is the specification the C engine must match bit-for-bit.
"""

from dataclasses import dataclass

import numpy as np

from earshot.quant.graph import Layer, conv_acc, forward_float, same_pad


@dataclass
class QLayer:
    op: str
    name: str
    in_shape: tuple[int, ...]  # (H, W, C) or (C,)
    out_shape: tuple[int, ...]
    stride: tuple[int, int]
    pad: tuple[int, int]  # (top, left)
    in_zp: int
    out_zp: int
    out_scale: float
    act_min: int
    mult: np.ndarray  # int32, one per output channel
    shift: np.ndarray  # int32, one per output channel
    w: np.ndarray | None = None  # int8, same layout as the float Layer
    b: np.ndarray | None = None  # int32


@dataclass
class QModel:
    layers: list[QLayer]
    input_scale: np.float32
    input_zp: int
    norm_mean: np.ndarray  # float32 (n_mels,)
    norm_std: np.ndarray
    classes: list[str]

    @property
    def output_scale(self) -> float:
        return self.layers[-1].out_scale

    @property
    def output_zp(self) -> int:
        return self.layers[-1].out_zp


def qparams(lo: float, hi: float) -> tuple[float, int]:
    """Asymmetric int8 scale/zero-point covering [lo, hi] (always including 0)."""
    lo, hi = min(float(lo), 0.0), max(float(hi), 0.0)
    scale = max(hi - lo, 1e-8) / 255.0
    zp = int(np.clip(np.round(-128 - lo / scale), -128, 127))
    return scale, zp


def quantize_multiplier(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Real multipliers m (0 < m < 2**31) -> (M0, shift) with m ~ M0 * 2**-shift, M0 in [2**30, 2**31)."""
    frac, exp = np.frexp(np.asarray(m, dtype=np.float64))  # m = frac * 2**exp, frac in [0.5, 1)
    m0 = np.round(frac * (1 << 31)).astype(np.int64)
    carry = m0 == (1 << 31)
    m0[carry] //= 2
    exp = exp + carry
    shift = 31 - exp
    assert (m0 > 0).all() and (shift >= 1).all() and (shift <= 62).all(), (m, shift)
    return m0.astype(np.int32), shift.astype(np.int32)


def requantize(acc: np.ndarray, mult: np.ndarray, shift: np.ndarray) -> np.ndarray:
    acc = acc.astype(np.int64)
    s = shift.astype(np.int64)
    return (acc * mult.astype(np.int64) + (np.int64(1) << (s - 1))) >> s


def calibrate(layers: list[Layer], x: np.ndarray) -> list[tuple[float, float]]:
    """Min/max of the input and of every layer output over calibration data x (N, H, W, 1)."""
    _, outs = forward_float(layers, x, collect=True)
    return [(float(x.min()), float(x.max()))] + [(float(o.min()), float(o.max())) for o in outs]


def _per_channel_weights(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reduce = tuple(range(w.ndim - 1))  # everything but the output-channel axis
    scale = np.maximum(np.abs(w).max(axis=reduce), 1e-12) / 127.0
    q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
    return q, scale


def quantize_model(
    layers: list[Layer],
    ranges: list[tuple[float, float]],
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    classes: list[str],
    input_hw: tuple[int, int] = (49, 40),
) -> QModel:
    input_scale, input_zp = qparams(*ranges[0])
    in_scale, in_zp = input_scale, input_zp
    shape = (*input_hw, 1)
    qlayers = []
    for L, rng in zip(layers, ranges[1:]):
        out_scale, out_zp = qparams(*rng)
        act_min = out_zp if L.relu else -128
        if L.op in ("conv", "dwconv"):
            (pt, _, oh), (pl, _, ow) = same_pad(shape[0], L.w.shape[0], L.stride[0]), same_pad(shape[1], L.w.shape[1], L.stride[1])
            out_shape, pad = (oh, ow, L.w.shape[-1]), (pt, pl)
        elif L.op == "avgpool":
            out_shape, pad = (shape[-1],), (0, 0)
        else:
            out_shape, pad = (L.w.shape[-1],), (0, 0)

        w_q = b_q = None
        if L.op == "avgpool":
            n_out = out_shape[0]
            real_mult = np.full(n_out, in_scale / (out_scale * shape[0] * shape[1]))
        else:
            w_q, w_scale = _per_channel_weights(L.w)
            acc_scale = in_scale * w_scale
            b_q = np.round(L.b / acc_scale).astype(np.int32)
            real_mult = acc_scale / out_scale
        mult, shift = quantize_multiplier(real_mult)
        qlayers.append(
            QLayer(L.op, L.name, shape, out_shape, L.stride, pad, in_zp, out_zp, out_scale, act_min, mult, shift, w_q, b_q)
        )
        shape, in_scale, in_zp = out_shape, out_scale, out_zp

    return QModel(
        qlayers,
        np.float32(input_scale),
        input_zp,
        np.asarray(norm_mean, np.float32),
        np.asarray(norm_std, np.float32),
        list(classes),
    )


def quantize_input(qm: QModel, logmel: np.ndarray) -> np.ndarray:
    """Raw log-mel (N, T, F) -> int8 (N, T, F, 1). All float32 math, round half to even (C: lrintf)."""
    v = (np.asarray(logmel, np.float32) - qm.norm_mean) / qm.norm_std
    q = np.rint(v / qm.input_scale) + np.float32(qm.input_zp)
    return np.clip(q, -128, 127).astype(np.int8)[..., None]


def forward_int8(qm: QModel, xq: np.ndarray) -> np.ndarray:
    """xq: int8 (N, H, W, 1) -> int8 logits (N, num_classes)."""
    x = xq
    for L in qm.layers:
        centered = x.astype(np.float64) - L.in_zp  # exact small integers
        if L.op in ("conv", "dwconv"):
            acc = np.rint(conv_acc(centered, L.w.astype(np.float64), L.stride)).astype(np.int64) + L.b
        elif L.op == "avgpool":
            acc = np.rint(centered.sum(axis=(1, 2))).astype(np.int64)
        else:
            acc = np.rint(centered @ L.w.astype(np.float64)).astype(np.int64) + L.b
        y = requantize(acc, L.mult, L.shift) + L.out_zp
        x = np.clip(y, L.act_min, 127).astype(np.int8)
    return x


def dequantize_logits(qm: QModel, q: np.ndarray) -> np.ndarray:
    return (q.astype(np.float32) - qm.output_zp) * qm.output_scale


def model_bytes(qm: QModel) -> dict:
    w = sum(L.w.size for L in qm.layers if L.w is not None)
    b = 4 * sum(L.b.size for L in qm.layers if L.b is not None)
    rq = 8 * sum(L.mult.size for L in qm.layers)
    act = max(int(np.prod(L.out_shape)) for L in qm.layers)
    return {"weights_int8": w, "bias_int32": b, "requant": rq, "total_params_bytes": w + b + rq, "max_activation": act}
