"""
The alignment scaffold.

A range-based rule asks where something is or how big it is. Asked over a
raw photograph, a model has to estimate a continuous quantity by eye,
against a reference it can only remember — and its estimates drift with
scale, crop and lighting.

The scaffold puts the ruler in the picture. A grid, fitted on the normal
references and anchored to the object that defines the frame, is drawn
with every intersection labelled `(row, col)`. Now the model does not
estimate anything: it reads a label off the image, and the question asks
whether that label lies in an interval computed in code.

Two shapes of ruler, for the two kinds of quantity:

    position   a 2D grid over the frame — where the edge falls
    extent     ticks along the object's own long axis — how far it reaches

The second exists because an object free to rotate has no meaningful
image-aligned size. Ticks laid along its own axis, counted outward from
its centre, measure the same thing at any angle.
"""
from __future__ import annotations

import cv2
import numpy as np

from .. import config
from ..types import Grid, Instance

LABEL_SCALE     = 0.4
LABEL_THICKNESS = 1
LABEL_OFFSET    = (5, -5)
FRAME_COLOR     = (255, 0, 0)     # BGR: blue
FRAME_THICKNESS = 2

# Ticks along an object's own axis only mean something if it has one.
MIN_ASPECT = 1.2


def render(image: np.ndarray, grid: Grid, edge_type: str,
           frame: tuple | None = None,
           targets: list[Instance] | None = None) -> np.ndarray:
    """Draw the scaffold for a range-based rule and return the result."""
    out = image.copy()
    if edge_type == "contour":
        for instance in targets or []:
            draw_extent_ticks(out, instance.mask, grid.spacing)
    else:
        height, width = out.shape[:2]
        draw_grid(out, grid, frame or (0, 0, width, height))
    return out


def draw_grid(image: np.ndarray, grid: Grid, frame: tuple) -> None:
    """Label every intersection of the grid inside `frame`, in place.

    Rows and columns count from 1 at the frame's top-left, so a
    coordinate is something a viewer can read without being told a
    convention.
    """
    x0, y0, x1, y1 = frame
    origin_x, origin_y = grid.origin

    row, y = 1, y0 + origin_y
    while y < y1:
        col, x = 1, x0 + origin_x
        while x < x1:
            _tick(image, x, y, f"({row},{col})")
            col, x = col + 1, x + grid.spacing
        row, y = row + 1, y + grid.spacing

    cv2.rectangle(image, (int(x0), int(y0)), (int(x1), int(y1)),
                  FRAME_COLOR, FRAME_THICKNESS)


def draw_extent_ticks(image: np.ndarray, mask: np.ndarray,
                      spacing: float) -> int:
    """Mark ticks along the object's long axis. Returns how many were drawn.

    That count is the reading: the number of ticks a viewer counts along
    the object is what the question asks about, so the same count is what
    `calibrate.tick_span` computes for the references.
    """
    points = np.argwhere(np.asarray(mask) > 0)
    if len(points) < 4 or spacing <= 0:
        return 0

    (cx, cy), (width, height), angle = cv2.minAreaRect(
        points[:, ::-1].astype(np.float32))
    long_side, short_side = max(width, height), max(min(width, height), 1e-6)
    if long_side / short_side < MIN_ASPECT:
        return 0                       # too round to have a long axis

    if height > width:
        angle += 90.0                  # the long side is the other one
    theta = np.deg2rad(angle)
    step_x, step_y = np.cos(theta), np.sin(theta)

    reach = int(np.floor(long_side / 2 / spacing))
    for n, k in enumerate(range(-reach, reach + 1), start=1):
        _tick(image, cx + k * spacing * step_x, cy + k * spacing * step_y,
              f"({n}, 1)")
    return 2 * reach + 1


def _tick(image: np.ndarray, x: float, y: float, label: str) -> None:
    x, y = int(x), int(y)
    color = _readable_against(image, x, y)
    cv2.circle(image, (x, y), config.GRID_DOT_RADIUS, color, -1)
    cv2.putText(image, label, (x + LABEL_OFFSET[0], y + LABEL_OFFSET[1]),
                cv2.FONT_HERSHEY_SIMPLEX, LABEL_SCALE, color,
                LABEL_THICKNESS, cv2.LINE_AA)


def _readable_against(image: np.ndarray, x: int, y: int) -> tuple:
    """Black on light ground, white on dark — so no tick is ever lost.

    A grid the model cannot read is worse than no grid, because it still
    occupies the image.
    """
    height, width = image.shape[:2]
    patch = image[max(0, y - 3):min(height, y + 4),
                  max(0, x - 3):min(width, x + 4)]
    if patch.size == 0:
        return (255, 255, 255)
    luma = float(cv2.cvtColor(patch, cv2.COLOR_BGR2YUV)[:, :, 0].mean())
    return (0, 0, 0) if luma > 128 else (255, 255, 255)
