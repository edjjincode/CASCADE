"""
Calibration — turning K normal images into a grid and a normal range.

A range-based rule constrains a continuous quantity: how high the liquid
comes, how long the cable is. A model cannot read a continuous quantity
off a photograph reliably, and asking it to compare against a remembered
reference is worse. So the quantity is discretised: a grid is laid over
the image, the references are measured in code, and the question becomes
membership in an integer interval — which a model *can* answer.

The grid is not arbitrary. It has to be coarse enough that its labels are
legible and that ordinary manufacturing variation stays inside one cell,
and fine enough that a real deviation crosses out of it. Those two
demands are what the fit below balances:

    every reference of an edge falls in the same cell, with margin   (3)
    w* = the smallest spacing for which that holds                   (4)
    a* = the offset that centres the references in their cells       (5)

`m` and `w_min` come from `config`; they are calibrated once per backend
on synthetic probes (see calibration/) and never per category.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from . import config
from .types import Grid, Instance

# Edge pixels within this many pixels of a mask's corner are geometry, not
# the edge itself, and would inflate its spread.
CORNER_MARGIN = 20

# A boundary pixel counts as horizontal when its gradient leans that way
# by at least this ratio. Below 1.0 so gently concave stretches of a
# horizontal edge are still read as horizontal.
HORIZONTAL_RATIO = 0.85


# ═════════════════════════════════════════════════════════════════════
# Measuring an edge
# ═════════════════════════════════════════════════════════════════════
def classify_boundary(mask: np.ndarray) -> dict:
    """Assign every boundary pixel to exactly one of the four edges.

    Projecting a mask onto its axes double-counts corners: the same pixel
    is the topmost of its column and the leftmost of its row. Instead,
    each boundary pixel is assigned by the direction its gradient points,
    so a pixel belongs to the edge it actually lies on.
    """
    binary = np.asarray(mask, dtype=np.uint8)
    empty  = {"top": {}, "bottom": {}, "left": {}, "right": {}}
    if binary.max() == 0:
        return empty

    kernel   = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    boundary = binary - cv2.erode(binary, kernel, iterations=1)

    surface = binary.astype(np.float32)
    grad_y  = cv2.Sobel(surface, cv2.CV_32F, 0, 1, ksize=3)
    grad_x  = cv2.Sobel(surface, cv2.CV_32F, 1, 0, ksize=3)

    rows, cols = np.nonzero(binary)
    middle     = (rows.min() + rows.max()) / 2
    left, right = float(cols.min()), float(cols.max())
    inset       = (right - left) * 0.25       # corners are not the edge

    out = {"top": {}, "bottom": {}, "left": {}, "right": {}}
    for row, col in zip(*np.nonzero(boundary)):
        dy, dx = float(grad_y[row, col]), float(grad_x[row, col])
        if dy == 0.0 and dx == 0.0:
            continue

        if abs(dy) >= HORIZONTAL_RATIO * abs(dx):
            side   = "top" if dy > 0 else "bottom"
            in_half = row <= middle if side == "top" else row >= middle
            if in_half and left + inset <= col <= right - inset:
                best = out[side].get(col)
                if best is None or (row < best if side == "top" else row > best):
                    out[side][col] = row
        else:
            side = "left" if dx > 0 else "right"
            best = out[side].get(row)
            if best is None or (col < best if side == "left" else col > best):
                out[side][row] = col

    return out


def flatten_edge(samples: dict) -> dict:
    """Drop the parts of an edge that are not the edge.

    A segmented region's boundary picks up excursions — a bottle's
    shoulder read as part of its fill line, a ragged patch of a mask.
    Those sit far from the edge's typical position, so two passes of
    median-absolute-deviation strip them and leave the flat stretch that
    the rule is actually about. This is a statement about the *shape* of
    an edge, not about any particular product.
    """
    if len(samples) < 3:
        return samples

    keys   = sorted(samples)
    values = np.array([samples[k] for k in keys], dtype=float)

    for scale, floor in ((3.0, 5.0), (1.5, 3.0)):
        median = np.median(values)
        spread = np.median(np.abs(values - median))
        keep   = np.abs(values - median) <= max(spread * scale, floor)
        if not keep.any():
            return samples
        keys   = [k for k, ok in zip(keys, keep) if ok]
        values = values[keep]

    return {k: samples[k] for k in keys}


def measure_edges(masks: list[np.ndarray], edge_type: str,
                  origins: list | tuple = (0, 0)) -> tuple[list, list]:
    """Measure each constrained edge across the K reference masks.

    Returns one group per edge and axis, each holding the extreme values
    that edge took across the references. `origins` is the corner each
    mask is measured from — one per mask, or a single pair shared by all
    — so a grid fitted here transfers to any image whose frame is found
    the same way.
    """
    masks = [np.squeeze(m) for m in masks if m is not None and m.size]
    if not masks:
        return [], []

    if origins and isinstance(origins[0], (int, float)):
        origins = [tuple(origins)] * len(masks)
    origins = list(origins) + [(0, 0)] * (len(masks) - len(origins))

    if edge_type == "contour":
        # An intrinsic dimension: the rule is about the object's own
        # extent, which no image-aligned box captures once it can rotate.
        # Its minimum-area rectangle gives an extent that does not care
        # how the object was laid down.
        major, minor = [], []
        for mask in masks:
            points = np.argwhere(mask > 0)
            if len(points) < 4:
                continue
            (_, _), (width, height), _ = cv2.minAreaRect(
                points[:, ::-1].astype(np.float32))
            major.append(float(max(width, height)))
            minor.append(float(min(width, height)))
        if not major:
            return [], []
        return ([{"edge": "major", "values": major}],
                [{"edge": "minor", "values": minor}])

    boundaries = [classify_boundary(mask) for mask in masks]
    for boundary in boundaries:
        for side in ("top", "bottom", "left", "right"):
            boundary[side] = flatten_edge(boundary[side])

    wanted_y = ["top", "bottom"] if edge_type == "rectangular" else \
               ["top"] if edge_type == "horizontal" else []
    wanted_x = ["left", "right"] if edge_type in ("rectangular", "vertical") else []

    groups_y = _collect(boundaries, wanted_y, [o[1] for o in origins])
    groups_x = _collect(boundaries, wanted_x, [o[0] for o in origins])
    return groups_y, groups_x


def _collect(boundaries: list[dict], edges: list[str], origins: list) -> list:
    """Gather the range each edge spans, per reference, in its own frame."""
    groups = []
    for edge in edges:
        values = []
        for boundary, origin in zip(boundaries, origins):
            samples = boundary.get(edge, {})
            if samples:
                values.append(float(min(samples.values())) - origin)
                values.append(float(max(samples.values())) - origin)
        if values:
            groups.append({"edge": edge, "values": values})
    return groups


# ═════════════════════════════════════════════════════════════════════
# Fitting the grid
# ═════════════════════════════════════════════════════════════════════
def drift(groups: list, axis: str) -> float:
    """How far the object as a whole shifts between normal references.

    When an object translates without changing size, both of its opposite
    edges move together, and the midpoint between them moves with it.
    That midpoint's range is the drift — the part of an edge's variation
    that says nothing about the property the rule constrains.
    """
    first, second = ("top", "bottom") if axis == "y" else ("left", "right")
    a = next((g for g in groups if g["edge"] == first), None)
    b = next((g for g in groups if g["edge"] == second), None)
    if not a or not b or not a["values"] or not b["values"]:
        return 0.0
    low  = (min(a["values"]) + min(b["values"])) / 2
    high = (max(a["values"]) + max(b["values"])) / 2
    return max(0.0, high - low)


def margins(edge: str, outer: float, inner: float) -> tuple[float, float]:
    """The clearance an edge needs on each side of its cell.

    Drift pushes an object's leading edge outward and its trailing edge
    inward by the same amount, so the two sides of a cell do not need
    equal clearance. The side facing the direction of travel absorbs the
    drift; the other side only needs the reading margin m. Giving both
    sides the larger clearance would coarsen the grid for no reason and
    blunt the very deviations the rule is meant to catch.
    """
    if edge in ("top", "left"):
        return outer, inner
    if edge in ("bottom", "right"):
        return inner, outer
    return inner, inner          # an intrinsic dimension does not drift


def feasible_offsets(values: list, spacing: float,
                     low: float, high: float) -> list:
    """Where the grid may start so this edge stays inside one cell.

    An offset works when every reference value sits at least `low` past
    the cell's near boundary and `high` short of its far one. Cells
    repeat, so the workable offsets form an interval on a circle of
    circumference `spacing`, which may wrap.
    """
    first, last = min(values), max(values)
    if last - first > spacing - low - high:
        return []                                  # too wide for any cell
    start = (last - (spacing - high)) % spacing
    end   = (first - low) % spacing
    return [(start, end)] if start <= end else [(start, spacing), (0, end)]


def common_offset(intervals: list, spacing: float) -> float | None:
    """An offset every edge can live with, or None if there is none."""
    for offset in range(int(spacing)):
        if all(any(lo - 1e-9 <= offset <= hi + 1e-9 for lo, hi in segments)
               for segments in intervals):
            return float(offset)
    return None


def fit_grid(groups_y: list, groups_x: list,
             m: float | None = None, w_min: float | None = None,
             search: int = 500) -> Grid | None:
    """Find the finest grid that keeps every reference edge in one cell.

    Both axes share one spacing, so a cell is square and a coordinate
    means the same thing whichever way it is read.

    Spacing grows from the smallest value that could possibly work — no
    cell can be narrower than the widest edge's spread plus its two
    margins, nor narrower than the legibility floor — and the first
    spacing that admits an offset on both axes wins. Finest means most
    sensitive: a coarser grid would hide smaller deviations.
    """
    m     = config.M if m is None else m
    w_min = config.DELTA_MIN if w_min is None else w_min
    if not groups_y and not groups_x:
        return None

    outer_y = drift(groups_y, "y") + m
    outer_x = drift(groups_x, "x") + m

    lower = float(w_min)
    for groups, outer in ((groups_y, outer_y), (groups_x, outer_x)):
        for group in groups:
            low, high = margins(group["edge"], outer, m)
            spread    = max(group["values"]) - min(group["values"])
            lower     = max(lower, spread + low + high)
    lower = int(math.ceil(lower))

    for spacing in range(lower, lower + search + 1):
        offsets = []
        for groups, outer in ((groups_y, outer_y), (groups_x, outer_x)):
            intervals = []
            for group in groups:
                low, high = margins(group["edge"], outer, m)
                segments  = feasible_offsets(group["values"], spacing, low, high)
                if not segments:
                    break
                intervals.append(segments)
            else:
                offsets.append(common_offset(intervals, spacing) if intervals else 0.0)
                continue
            break
        if len(offsets) < 2 or any(o is None for o in offsets):
            continue

        offset_y, offset_x = offsets
        return Grid(
            origin  = (offset_x, offset_y),
            spacing = float(spacing),
            normal_cells = _normal_cells(groups_y, offset_y, groups_x,
                                         offset_x, float(spacing)),
        )
    return None


def _normal_cells(groups_y: list, offset_y: float,
                  groups_x: list, offset_x: float, spacing: float) -> dict:
    """The closed cell interval each edge occupies in the references.

    This is what the verification question quotes, so it has to be read
    the same way the scaffold is drawn — a positional edge by which cell
    it falls in, an intrinsic dimension by how many ticks it spans.
    """
    cells = {}
    for groups, offset in ((groups_y, offset_y), (groups_x, offset_x)):
        for group in groups:
            values = group["values"]
            if not values:
                continue
            if group["edge"] in ("major", "minor"):
                cells[group["edge"]] = tick_span(values, spacing)
            else:
                cells[group["edge"]] = [cell_of(min(values), offset, spacing),
                                        cell_of(max(values), offset, spacing)]
    return cells


def cell_of(value: float, offset: float, spacing: float) -> int:
    """The row or column number a coordinate falls in, as the grid is drawn.

    `align.draw_grid` prints its first tick at the offset and numbers it
    1, so the tick immediately above (or left of) a coordinate carries
    the number below. Deriving the reading from the same rule that draws
    the labels is what keeps the question and the picture in agreement:
    the interval the question quotes is a range of labels the model can
    actually see.
    """
    return int(math.floor((value - offset) / spacing)) + 1


def tick_span(values: list, spacing: float) -> list:
    """How many ticks an intrinsic dimension covers, at its extremes.

    A contour scaffold marks ticks outward from the object's centre, so
    what is countable is the number of ticks the object spans — an odd
    number, the centre tick plus as many each way as fit.
    """
    counts = [2 * int(math.floor(v / 2 / spacing)) + 1 for v in values]
    return [min(counts), max(counts)]


# ═════════════════════════════════════════════════════════════════════
# From references to a grid
# ═════════════════════════════════════════════════════════════════════
def calibrate(reference_roles: list[dict[str, list[Instance]]], edge_type: str,
              anchor_frames: list | None = None) -> Grid | None:
    """Fit the grid for a range-based rule from its K reference images.

    The first target is the one measured: it is the object whose extent
    the rule constrains. An anchor, where the rule has one, supplies the
    frame those measurements are expressed in — each reference is
    measured from its *own* anchor's corner, so a bottle photographed a
    little to the left contributes the same fill height as one centred.
    Without that, ordinary framing jitter would enter the measurement as
    spread and force a coarser grid than the property deserves.

    The frame used here is the frame `scaffold.align.draw_grid` draws in,
    which is what keeps the cell numbers the question quotes the same
    ones the model can read off the picture.
    """
    primary = next((role for roles in reference_roles for role in roles), None)
    if primary is None:
        return None

    frames = list(anchor_frames or [])
    masks, origins = [], []
    for n, roles in enumerate(reference_roles):
        if not roles.get(primary):
            continue
        masks.append(roles[primary][0].mask)
        frame = frames[n] if n < len(frames) else None
        origins.append((int(frame[0]), int(frame[1])) if frame else (0, 0))
    if not masks:
        return None

    groups_y, groups_x = measure_edges(masks, edge_type, origins)
    return fit_grid(groups_y, groups_x)
