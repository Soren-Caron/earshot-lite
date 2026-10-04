"""Download Google Speech Commands v0.02.

Usage: python -m earshot.data.download
"""

import argparse
import urllib.request
from pathlib import Path

from tqdm import tqdm

URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
DEFAULT_OUT = Path("data/raw/speech_commands_v0.02.tar.gz")


def download(url: str = URL, out: Path = DEFAULT_OUT) -> Path:
    if out.exists():
        print(f"already downloaded: {out}")
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        with tqdm(total=total, unit="B", unit_scale=True, desc=out.name) as bar:
            while chunk := resp.read(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.rename(out)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    download(out=p.parse_args().out)
