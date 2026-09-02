"""
The pipeline.

Two phases, and the split between them is where the cost lives.

**Calibration** runs once per constraint, on K normal images. It grounds
the rule (three model calls), and for a range-based rule measures the
normal interval from the references (no model calls at all). What it
produces — a `Decomposition` and a `Grid` — is then fixed.

**Verification** runs per image. For each constraint it draws one
scaffold and asks one question: a single model call, seeing a single
image, answering about a single rule. Nothing is compared against a
remembered reference, because everything the references had to say was
already turned into text at calibration time.

An image is anomalous when any of its constraints is violated.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import calibrate as calibration
from . import parse, question
from .client import Client
from .detect import MaskPack, MaskSource, anchor_frame, assign_roles
from .scaffold.build import build as build_scaffold
from .types import Constraint, Decomposition, Grid, ImageResult, Judgment


class Calibrated:
    """A constraint that is ready to be checked against any query image."""

    def __init__(self, decomposition: Decomposition, grid: Grid | None = None):
        self.decomposition = decomposition
        self.grid          = grid

    @property
    def constraint_id(self) -> str:
        return self.decomposition.constraint_id

    def to_dict(self) -> dict:
        return {"decomposition": self.decomposition.to_dict(),
                "grid": self.grid.to_dict() if self.grid else None}

    @classmethod
    def from_dict(cls, d: dict) -> "Calibrated":
        return cls(Decomposition.from_dict(d["decomposition"]),
                   Grid.from_dict(d["grid"]) if d.get("grid") else None)


class Pipeline:
    """Calibrate constraints on reference images, then check query images."""

    def __init__(self, client: Client | None = None,
                 masks: MaskSource | None = None):
        self._client = client
        self.masks   = masks or MaskPack()

    @property
    def client(self) -> Client:
        """The model client, created on first use so cached work stays keyless."""
        if self._client is None:
            self._client = Client()
        return self._client

    # ── calibration, once per constraint ─────────────────────────────
    def calibrate(self, constraint: Constraint,
                  references: list) -> Calibrated:
        """Ground the rule and, if it is a range, measure its normal interval."""
        decomposition = parse.decompose(constraint, references, self.client)
        return Calibrated(decomposition, self.fit(decomposition, references))

    def fit(self, decomposition: Decomposition, references: list) -> Grid | None:
        """Measure the normal range from the references. No model involved."""
        if not decomposition.comprehend.is_range:
            return None
        anchor = decomposition.locate.anchor if decomposition.locate else None
        return calibration.calibrate(
            [assign_roles(self.masks, image, decomposition.targets)
             for image in references],
            decomposition.edge_type or "rectangular",
            [anchor_frame(self.masks, image, anchor) for image in references],
        )

    # ── verification, once per image and constraint ──────────────────
    def scaffold(self, calibrated: Calibrated, image) -> np.ndarray:
        """Draw the image the model will be shown."""
        picture = cv2.imread(str(image))
        if picture is None:
            raise FileNotFoundError(f"cannot read {image}")
        decomposition = calibrated.decomposition
        roles = assign_roles(self.masks, image, decomposition.targets)
        frame = anchor_frame(self.masks, image,
                             decomposition.locate.anchor
                             if decomposition.locate else None)
        return build_scaffold(picture, decomposition, roles,
                              calibrated.grid, frame)

    def verify(self, calibrated: Calibrated, image,
               out_dir: Path | None = None) -> Judgment:
        """Check one constraint against one image. Exactly one model call."""
        picture = self.scaffold(calibrated, image)
        prompt  = question.build(calibrated.decomposition, calibrated.grid)

        path = Path(out_dir or ".") / f"{calibrated.constraint_id}_{Path(image).stem}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), picture)

        reply  = self.client.ask(prompt, images=[path])
        answer = question.read_answer(reply)
        return Judgment(
            constraint_id = calibrated.constraint_id,
            answer        = answer,
            violated      = question.is_violation(answer),
            question      = prompt,
            scaffold_path = str(path),
            raw_response  = reply,
        )

    def check(self, image, calibrated: list[Calibrated],
              out_dir: Path | None = None) -> ImageResult:
        """Check every constraint of a category against one image."""
        judgments = [self.verify(c, image, out_dir) for c in calibrated]
        return ImageResult(
            image_path = str(image),
            category   = calibrated[0].decomposition.constraint.category
                         if calibrated else "",
            judgments  = judgments,
        )
