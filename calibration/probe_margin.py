#!/usr/bin/env python3
"""
The margin probe — how close is too close to call?

`m` is the distance within which a mark cannot be reliably assigned to
one side of a line. The alignment scaffold needs it because a reference
sitting within `m` of a cell boundary is a reference the model may read
into the wrong cell, and the grid fit has to keep clear of that.

The probe measures it directly and without any product in the picture: a
horizontal line, two dots at different distances on opposite sides, and
the question "which dot is closer?". Accuracy as a function of the
difference between the two distances says how fine a distinction the
backend can actually make. `m` is the smallest difference it gets right
95% of the time.

    python calibration/probe_margin.py --report     # from saved responses
    python calibration/probe_margin.py --run        # re-query a backend
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

HERE      = Path(__file__).parent
RESPONSES = HERE / "responses"

WIDTH, HEIGHT = 800, 400
LINE_Y        = HEIGHT // 2
DOT_RADIUS    = 3
X_MARGIN      = 80
X_SEPARATION  = 200

DISTANCES       = [5, 10, 15, 20, 30, 50]
TRIALS_PER_CELL = 30
SEED            = 42

TARGET_ACCURACY = 0.95

QUESTION = (
    "The image shows a horizontal black line and two dots labeled A and B.\n\n"
    "Which dot is closer to the line?\n\n"
    "Answer with a single letter: A or B."
)


# ═════════════════════════════════════════════════════════════════════
def draw(a_xy: tuple, b_xy: tuple) -> np.ndarray:
    image = np.full((HEIGHT, WIDTH, 3), 255, np.uint8)
    cv2.line(image, (10, LINE_Y), (WIDTH - 10, LINE_Y), (0, 0, 0), 2)
    for (x, y), letter in ((a_xy, "A"), (b_xy, "B")):
        cv2.circle(image, (x, y), DOT_RADIUS, (0, 0, 0), -1)
        # Put the label on the far side of the dot from the line, so it
        # never occludes the distance being judged.
        offset = -14 if y < LINE_Y else 22
        cv2.putText(image, letter, (x - 7, y + offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    return image


def trials(seed: int = SEED, per_cell: int = TRIALS_PER_CELL) -> list[dict]:
    """Every (d_A, d_B) pair, with A above the line and B below it.

    The dots are placed on opposite sides so neither can be judged by
    comparing it against the other's position — only against the line.
    Their horizontal positions are random, and which side of the image
    each falls on is randomised too, so left/right cannot stand in for
    the answer.
    """
    rng = random.Random(seed)
    out = []
    for d_a in DISTANCES:
        for d_b in DISTANCES:
            for k in range(per_cell):
                x_a = rng.randint(X_MARGIN, WIDTH // 2 - X_SEPARATION // 2)
                x_b = rng.randint(WIDTH // 2 + X_SEPARATION // 2, WIDTH - X_MARGIN)
                if rng.random() < 0.5:
                    x_a, x_b = x_b, x_a
                out.append({
                    "id": f"dA{d_a:03d}_dB{d_b:03d}_t{k:02d}",
                    "d_a": d_a, "d_b": d_b, "delta_d": abs(d_a - d_b),
                    "a_xy": (x_a, LINE_Y - d_a), "b_xy": (x_b, LINE_Y + d_b),
                    # Equal distances have no correct answer; they are
                    # generated for symmetry and dropped when scoring.
                    "closer": "A" if d_a < d_b else "B" if d_b < d_a else None,
                })
    return out


# ═════════════════════════════════════════════════════════════════════
def report(path: Path) -> int:
    """Accuracy by distance difference, and the threshold it implies."""
    data  = json.loads(path.read_text())
    scored = [t for t in data["trials"] if t["closer"] is not None]

    tally: dict[int, list[int]] = {}
    for trial in scored:
        hit, total = tally.setdefault(trial["delta_d"], [0, 0])
        tally[trial["delta_d"]] = [hit + bool(trial["correct"]), total + 1]

    print(f"{data['model']}   {len(scored)} scored trials\n")
    print(f"{'Δd (px)':>8}  {'correct':>10}  {'accuracy':>9}")
    threshold = None
    for delta_d in sorted(tally):
        hit, total = tally[delta_d]
        accuracy = hit / total
        if threshold is None and accuracy >= TARGET_ACCURACY:
            threshold = delta_d
        mark = "  <- m" if threshold == delta_d else ""
        print(f"{delta_d:>8}  {hit:>4}/{total:<5}  {accuracy:>8.1%}{mark}")

    print(f"\nm = {threshold} px "
          f"(smallest difference read correctly ≥{TARGET_ACCURACY:.0%} of the time)")
    return threshold


def run(out: Path, model: str | None) -> None:
    """Re-query a backend. Costs one call per trial."""
    import sys
    sys.path.insert(0, str(HERE.parent))
    from cascade.client import Client

    client  = Client(model=model)
    scratch = out.parent / "images"
    scratch.mkdir(parents=True, exist_ok=True)

    scored = []
    for n, trial in enumerate(trials(), 1):
        path = scratch / f"{trial['id']}.png"
        cv2.imwrite(str(path), draw(trial["a_xy"], trial["b_xy"]))
        reply  = client.ask(QUESTION, images=[path]).strip().upper()
        answer = "A" if reply.startswith("A") else "B" if reply.startswith("B") else "?"
        scored.append({"delta_d": trial["delta_d"], "answer": answer,
                       "closer": trial["closer"],
                       "correct": answer == trial["closer"]})
        if n % 50 == 0:
            print(f"  {n} trials")

    out.write_text(json.dumps({"model": client.model, "probe": "margin",
                               "trials": scored}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--report", action="store_true", help="score saved responses")
    p.add_argument("--run", action="store_true", help="re-query a backend")
    p.add_argument("--model", help="backend to probe (default: config.MODEL)")
    p.add_argument("--responses", type=Path,
                   default=RESPONSES / "margin_gpt-5-image-mini.json")
    args = p.parse_args()

    if args.run:
        run(args.responses, args.model)
    report(args.responses)
