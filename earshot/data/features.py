"""Log-mel front-end.

Deliberately written as plain numpy (frame -> Hann -> rFFT -> power -> mel -> log)
so it can be ported line-for-line to C later. Every constant lives here.
"""

import numpy as np

SAMPLE_RATE = 16000
CLIP_SAMPLES = 16000  # 1 s
WIN = 640  # 40 ms
HOP = 320  # 20 ms
N_FFT = 1024
N_MELS = 40
FMIN = 20.0
FMAX = 7600.0
LOG_EPS = 1e-6
N_FRAMES = 1 + (CLIP_SAMPLES - WIN) // HOP  # 49


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank() -> np.ndarray:
    """HTK-style triangular filters, shape (N_MELS, N_FFT // 2 + 1)."""
    n_bins = N_FFT // 2 + 1
    bin_hz = np.arange(n_bins) * SAMPLE_RATE / N_FFT
    edges = mel_to_hz(np.linspace(hz_to_mel(FMIN), hz_to_mel(FMAX), N_MELS + 2))
    fb = np.zeros((N_MELS, n_bins), dtype=np.float64)
    for m in range(N_MELS):
        lo, center, hi = edges[m], edges[m + 1], edges[m + 2]
        rising = (bin_hz - lo) / (center - lo)
        falling = (hi - bin_hz) / (hi - center)
        fb[m] = np.maximum(0.0, np.minimum(rising, falling))
    return fb.astype(np.float32)


def hann_window() -> np.ndarray:
    """Periodic Hann window of length WIN."""
    n = np.arange(WIN)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * n / WIN)).astype(np.float32)


_FB = mel_filterbank()
_WINDOW = hann_window()
_FRAME_IDX = np.arange(N_FRAMES)[:, None] * HOP + np.arange(WIN)[None, :]


def fix_length(audio: np.ndarray) -> np.ndarray:
    """Zero-pad or truncate the last axis to CLIP_SAMPLES."""
    n = audio.shape[-1]
    if n >= CLIP_SAMPLES:
        return audio[..., :CLIP_SAMPLES]
    pad = [(0, 0)] * (audio.ndim - 1) + [(0, CLIP_SAMPLES - n)]
    return np.pad(audio, pad)


def log_mel(audio: np.ndarray) -> np.ndarray:
    """audio: (..., CLIP_SAMPLES) float32 in [-1, 1] -> (..., N_FRAMES, N_MELS) float32."""
    audio = np.asarray(audio, dtype=np.float32)
    frames = audio[..., _FRAME_IDX] * _WINDOW  # (..., 49, 640)
    spec = np.fft.rfft(frames, n=N_FFT, axis=-1)  # (..., 49, 513)
    power = (spec.real**2 + spec.imag**2).astype(np.float32)
    mel = power @ _FB.T  # (..., 49, 40)
    return np.log(mel + LOG_EPS).astype(np.float32)


def pcm16_to_float(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32) / 32768.0
