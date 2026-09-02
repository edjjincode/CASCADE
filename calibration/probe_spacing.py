#!/usr/bin/env python3
"""
The spacing probe — how fine can a grid get and still be read?

`w_min` is the spacing below which the alignment scaffold's own labels
stop being legible. It matters because the grid fit always wants the
finest spacing it can get — a finer grid is a more sensitive one — and
without a floor it would happily produce a grid too dense to read, whose
labels the model would then guess at.

The probe puts nothing in the picture but the scaffold itself: a square
grid at spacing w, every intersection labelled `(row, col)` exactly as
`scaffold/align.py` draws them, one dot marked red, and the question
"what are the red dot's coordinates?". `w_min` is the smallest spacing
read exactly right 95% of the time.

    python calibration/probe_spacing.py --report     # from saved responses
    python calibration/probe_spacing.py --run        # re-query a backend
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import cv2
import numpy as np

HERE      = Path(__file__).parent
RESPONSES = HERE / "responses"

WIDTH, HEIGHT = 800, 600
MARGIN        = 30

# Matched to cascade/scaffold/align.py, so the probe measures the grid
# the pipeline actually draws rather than a prettier stand-in.
DOT_RADIUS   = 3
PROBE_RADIUS = 4
LABEL_SCALE  = 0.4
LABEL_THICK  = 1
LABEL_OFFSET = (5, -5)

SPACINGS        = [10, 15, 20, 25, 30, 40, 50, 60, 70, 100, 150]
TRIALS_PER_CELL = 30
SEED            = 42

TARGET_ACCURACY = 0.95

QUESTION = (
    "The image shows a grid of dots. Each dot is labeled with its "
    "(row, col) coordinate.\n\n"
    "One dot is red. What is the (row, col) coordinate of the red dot?\n\n"
    'Answer with JSON only: {"row": <int>, "col": <int>}'
)


# ═════════════════════════════════════════════════════════════════════
def draw(spacing: int) -> tuple[np.ndarray, dict]:
    """A labelled grid at `spacing`, and where each coordinate landed."""
    image = np.full((HEIGHT, WIDTH, 3), 255, np.uint8)

    n_cols = max(3, (WIDTH  - 2 * MARGIN) // spacing + 1)
    n_rows = max(3, (HEIGHT - 2 * MARGIN) // spacing + 1)
    x0 = (WIDTH  - (n_cols - 1) * spacing) // 2
    y0 = (HEIGHT - (n_rows - 1) * spacing) // 2

    where = {}
    for row in range(1, n_rows + 1):
        for col in range(1, n_cols + 1):
            x, y = x0 + (col - 1) * spacing, y0 + (row - 1) * spacing
            where[(row, col)] = (x, y)
            cv2.circle(image, (x, y), DOT_RADIUS, (0, 0, 0), -1)
            cv2.putText(image, f"({row},{col})",
                        (x + LABEL_OFFSET[0], y + LABEL_OFFSET[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, LABEL_SCALE,
                        (0, 0, 0), LABEL_THICK, cv2.LINE_AA)
    return image, where


def verdict(true: tuple, read: tuple | None) -> str:
    """Exact, one cell out, or further — the three ways a reading can land."""
    if read is None:
        return "unreadable"
    if tuple(read) == tuple(true):
        return "exact"
    if max(abs(read[0] - true[0]), abs(read[1] - true[1])) == 1:
        return "off_by_1"
    return "off_by_n"


# ═════════════════════════════════════════════════════════════════════
def report(path: Path) -> int:
    """Exact-match accuracy by spacing, and the floor it implies."""
    data  = json.loads(path.read_text())
    tally: dict[int, dict[str, int]] = {}
    for trial in data["trials"]:
        counts = tally.setdefault(trial["spacing"], {})
        counts[trial["verdict"]] = counts.get(trial["verdict"], 0) + 1
        counts["total"] = counts.get("total", 0) + 1

    print(f"{data['model']}   {len(data['trials'])} trials\n")
    print(f"{'w (px)':>7}  {'exact':>10}  {'off by 1':>9}  {'further':>8}")
    floor = None
    for spacing in sorted(tally):
        counts   = tally[spacing]
        total    = counts["total"]
        exact    = counts.get("exact", 0)
        accuracy = exact / total
        if floor is None and accuracy >= TARGET_ACCURACY:
            floor = spacing
        mark = "  <- w_min" if floor == spacing else ""
        print(f"{spacing:>7}  {exact:>3}/{total:<3} {accuracy:>6.1%}  "
              f"{counts.get('off_by_1', 0):>9}  "
              f"{counts.get('off_by_n', 0) + counts.get('unreadable', 0):>8}{mark}")

    print(f"\nw_min = {floor} px "
          f"(finest spacing read exactly ≥{TARGET_ACCURACY:.0%} of the time)")
    return floor


def run(out: Path, model: str | None) -> None:
    """Re-query a backend. Costs one call per trial."""
    import sys
    sys.path.insert(0, str(HERE.parent))
    from cascade.client import Client

    client  = Client(model=model)
    rng     = random.Random(SEED)
    scratch = out.parent / "images"
    scratch.mkdir(parents=True, exist_ok=True)

    scored = []
    for spacing in SPACINGS:
        image, where = draw(spacing)
        for k in range(TRIALS_PER_CELL):
            true = rng.choice(list(where))
            marked = image.copy()
            cv2.circle(marked, where[true], PROBE_RADIUS, (0, 0, 255), -1)
            path = scratch / f"w{spacing:03d}_t{k:02d}.png"
            cv2.imwrite(str(path), marked)

            reply = client.ask(QUESTION, images=[path])
            found = re.search(r'"row"\s*:\s*(\d+).*?"col"\s*:\s*(\d+)', reply, re.S)
            read  = (int(found[1]), int(found[2])) if found else None
            scored.append({"spacing": spacing, "true": list(true),
                           "read": list(read) if read else None,
                           "verdict": verdict(true, read)})
        print(f"  w={spacing}")

    out.write_text(json.dumps({"model": client.model, "probe": "spacing",
                               "trials": scored}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--report", action="store_true", help="score saved responses")
    p.add_argument("--run", action="store_true", help="re-query a backend")
    p.add_argument("--model", help="backend to probe (default: config.MODEL)")
    p.add_argument("--responses", type=Path,
                   default=RESPONSES / "spacing_gpt-5-image-mini.json")
    args = p.parse_args()

    if args.run:
        run(args.responses, args.model)
    report(args.responses)
