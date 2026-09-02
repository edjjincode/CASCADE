"""
The disambiguation scaffold.

A composition-based rule is about which instances there are and how they
correspond. Asked over a raw photograph, a model has to hold every
instance apart in its head while it answers, and that is where it fails:
it conflates two instances of the same type, or loses track of which one
it already counted.

The scaffold moves that bookkeeping out of the model's head and onto the
image. Each instance is given a colour and a number, chosen by the engine
from the correspondence the matching program found:

    same colour, same number      these two go together
    different colour or number    these are distinct instances
    a colour with no partner      this instance matched nothing

so a missing, extra or mispaired instance is something the model can
*see* rather than something it has to deduce.
"""
from __future__ import annotations

import cv2
import numpy as np

from .. import config
from ..types import Instance, MaskingSpec

Roles = dict[str, list[Instance]]


def render(image: np.ndarray, roles: Roles, assignment: dict,
           spec: MaskingSpec) -> np.ndarray:
    """Draw the scaffold over `image` and return the result.

    Two roles are drawn differently on purpose. A role listed in
    `visual_property_roles` is one whose own colour or texture is what
    the rule asks about, so filling it would paint over the evidence; it
    is outlined instead. A role in `skip_label_roles` is one where a
    number adds clutter and no information — a poured, countless
    substance — so it gets a colour but no ID.
    """
    out = image.copy()

    for role, instances in roles.items():
        outline_only = role in spec.visual_property_roles
        for index, instance in enumerate(instances):
            color = assignment["colors"][(role, index)]
            mask  = _fit(instance.mask, out.shape)
            out = _outline(out, mask, color) if outline_only \
                  else _fill(out, mask, color)

    for role, instances in roles.items():
        if role in spec.skip_label_roles:
            continue
        for index, instance in enumerate(instances):
            out = _label(out, _fit(instance.mask, out.shape),
                         assignment["labels"][(role, index)])
    return out


def _fit(mask: np.ndarray, shape) -> np.ndarray:
    """Match a mask to the image it is drawn on."""
    mask = np.squeeze(mask)
    if mask.shape[:2] != shape[:2]:
        mask = cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    return mask > 0


def _fill(image: np.ndarray, mask: np.ndarray, color) -> np.ndarray:
    """Tint the instance, keeping enough of the photograph to see it."""
    overlay = image.copy()
    overlay[mask] = color
    return cv2.addWeighted(image, 1 - config.MASK_ALPHA,
                           overlay, config.MASK_ALPHA, 0)


def _outline(image: np.ndarray, mask: np.ndarray, color,
             thickness: int = 3) -> np.ndarray:
    """Mark the instance without covering a single one of its pixels."""
    out = image.copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, tuple(int(c) for c in color), thickness)
    return out


def _label(image: np.ndarray, mask: np.ndarray, text: str,
           scale: float = 0.8, thickness: int = 2) -> np.ndarray:
    """Print the instance's ID at its centre, on a plate so it stays legible."""
    if not text:
        return image
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return image
    cy, cx = int(rows.mean()), int(cols.mean())

    font = cv2.FONT_HERSHEY_SIMPLEX
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad = 5

    out = image.copy()
    cv2.rectangle(out,
                  (cx - width // 2 - pad, cy - height // 2 - pad),
                  (cx + width // 2 + pad, cy + height // 2 + pad + baseline),
                  (255, 255, 255), -1)
    cv2.putText(out, text, (cx - width // 2, cy + height // 2),
                font, scale, (0, 0, 0), thickness)
    return out
