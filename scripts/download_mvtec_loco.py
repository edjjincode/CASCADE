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

    flatten(args.to)

    if not args.keep_archive:
        archive.unlink()

    found = sorted(d.name for d in args.to.iterdir()
                   if d.is_dir() and (d / "train").is_dir())
    print(f"\ndone. {len(found)} categories: {', '.join(found)}")
    print(f"images are at {args.to}/<category>/test/logical_anomalies/000.png")
    if args.to.resolve() != Path("data").resolve():
        print(f"set CASCADE_DATA={args.to.resolve()}")


def flatten(root: Path) -> None:
    """Lift the archive's single top-level directory out of the way.

    The tarball unpacks into one wrapper directory, which would make every
    path a level deeper than the mask pack's keys and than every example
    in the README. Moving the categories up keeps
    `<root>/<category>/test/...` true however the dataset was obtained.
    """
    wrappers = [d for d in root.iterdir()
                if d.is_dir() and not (d / "train").is_dir()
                and any((c / "train").is_dir() for c in d.iterdir() if c.is_dir())]
    if len(wrappers) != 1:
        return
    for category in wrappers[0].iterdir():
        category.rename(root / category.name)
    wrappers[0].rmdir()


if __name__ == "__main__":
    main()
