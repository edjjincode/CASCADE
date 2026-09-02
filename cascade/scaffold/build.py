"""
Choosing a scaffold.

Which scaffold a constraint gets is decided by what kind of rule it is,
and nothing else. A rule about composition needs its instances told
apart; a rule about a range needs a ruler. That is the whole dispatch,
and it is the reason the pipeline never branches on the product.
"""
from __future__ import annotations

import numpy as np

from ..types import Decomposition, Grid, Instance
from . import align, disambiguate, engine

Roles = dict[str, list[Instance]]


def build(image: np.ndarray, decomposition: Decomposition, roles: Roles,
          grid: Grid | None = None,
          frame: tuple | None = None) -> np.ndarray:
    """Draw the scaffold this constraint calls for.

    `frame` is the anchor's bounding box, where the rule has an anchor
    and it was found. The grid is laid out from its corner, which is the
    frame calibration measured in — cell (1,1) means the same place on
    every image, however the object happens to sit in the photograph.
    """
    if decomposition.comprehend.is_range:
        if grid is None:
            return image.copy()          # nothing was measurable to draw
        primary = next(iter(roles), None)
        return align.render(image, grid, decomposition.edge_type or "rectangular",
                            frame=frame,
                            targets=roles.get(primary, []) if primary else [])

    spec = decomposition.comprehend.masking_spec
    return disambiguate.render(image, roles, engine.execute(spec, roles), spec)
