#!/usr/bin/env python3
"""
Fetch MVTec-LOCO.

The dataset is not redistributed with this repository. It is published by
MVTec under CC BY-NC-SA 4.0 — free for research, not for commercial use,
and derivative works carry the same licence.

    python scripts/download_mvtec_loco.py            # → ./data
    python scripts/download_mvtec_loco.py --to /somewhere/else

Roughly 6 GB compressed. Point CASCADE_DATA at the result if you put it
somewhere other than ./data.
"""
from __future__ import annotations

import argparse
import sys
import tarfile
import urllib.request
from pathlib import Path

URL = ("https://www.mydrive.ch/shares/48237/1b9106ccdfbb09a0c414bd49fe44a14a/"
       "download/430647091-1646842701/mvtec_loco_anomaly_detection.tar.xz")

LICENCE = """\
MVTec LOCO AD is licensed CC BY-NC-SA 4.0 (Attribution-NonCommercial-
ShareAlike). Research use is permitted; commercial use is not.

  Bergmann et al., "Beyond Dents and Scratches: Logical Constraints in
  Unsupervised Anomaly Detection and Localization", IJCV 2022.
"""


def progress(done: int, block: int, total: int) -> None:
    if total > 0:
        pct = min(100, done * block * 100 // total)
        print(f"\r  {pct:3d}%  {done * block / 1e9:.1f} / {total / 1e9:.1f} GB",
              end="", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--to", type=Path, default=Path("data"))
    p.add_argument("--keep-archive", action="store_true")
    args = p.parse_args()

    print(LICENCE)
    if input("Proceed? [y/N] ").strip().lower() not in ("y", "yes"):
        sys.exit(0)

    args.to.mkdir(parents=True, exist_ok=True)
    archive = args.to / "mvtec_loco.tar.xz"

    if not archive.exists():
        print(f"\ndownloading to {archive}")
        urllib.request.urlretrieve(URL, archive, reporthook=progress)
        print()

    print(f"extracting into {args.to}")
    with tarfile.open(archive) as tar:
        tar.extractall(args.to, filter="data")

    if not args.keep_archive:
        archive.unlink()

    print(f"\ndone. If this is not ./data, set CASCADE_DATA={args.to.resolve()}")


if __name__ == "__main__":
    main()
