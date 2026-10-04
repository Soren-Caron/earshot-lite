"""Compile and drive the C engine from Python (parity tests, benchmarks).

Compiler: $CC if set, else gcc / clang / cc on PATH, else `python -m ziglang cc`
(pip install ziglang) so it works on a bare Windows box too.
"""

import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

CSRC = Path(__file__).resolve().parents[2] / "csrc"
CFLAGS = ["-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-ffp-contract=off"]


def find_cc() -> list[str]:
    if os.environ.get("CC"):
        return shlex.split(os.environ["CC"])
    for cc in ("gcc", "clang", "cc"):
        if shutil.which(cc):
            return [cc]
    try:
        import ziglang  # noqa: F401
    except ImportError as e:
        raise RuntimeError("no C compiler found: set $CC, install gcc/clang, or `pip install ziglang`") from e
    return [sys.executable, "-m", "ziglang", "cc"]


def build_runner(model_c: Path, out_exe: Path, extra_flags: list[str] = ()) -> Path:
    out_exe = Path(out_exe)
    srcs = [CSRC / "earshot.c", CSRC / "test_runner.c", Path(model_c)]
    cmd = [*find_cc(), *CFLAGS, *extra_flags, f"-I{CSRC}", *map(str, srcs), "-o", str(out_exe), "-lm"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"C build failed:\n{' '.join(cmd)}\n{res.stdout}\n{res.stderr}")
    return out_exe


def run_c(exe: Path, logmel: np.ndarray, num_classes: int, repeat: int = 1) -> tuple[np.ndarray, float]:
    """logmel (N, T, F) raw features -> (int8 logits (N, num_classes), microseconds per inference)."""
    x = np.ascontiguousarray(logmel, dtype="<f4")
    with tempfile.TemporaryDirectory() as d:
        fin, fout = Path(d) / "x.f32", Path(d) / "y.i8"
        x.tofile(fin)
        res = subprocess.run([str(exe), str(fin), str(len(x)), str(fout), str(repeat)], capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"runner failed ({res.returncode}): {res.stderr}")
        logits = np.fromfile(fout, dtype=np.int8).reshape(len(x), num_classes)
    return logits, float(res.stdout.split()[0])
