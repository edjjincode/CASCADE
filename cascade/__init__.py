"""
CASCADE — Constraint-Aware Scaffolded Decomposition for logical anomaly
detection.

A logical anomaly is not a defect in the pixels; it is a violated rule
about what should be present and where. CASCADE turns each rule into a
question a vision-language model can answer, and gives it a picture drawn
so the answer is legible:

    constraints/  a rule, in plain language
      parse       -> what kind of rule it is, and what it talks about
      detect      -> instance masks for those objects
      calibrate   -> the normal range, measured from K references
      scaffold    -> the image, overlaid so the rule is readable off it
      question    -> the rule, stated affirmatively
      verify      -> one model call; the answer, inverted

An image is anomalous when any of its constraints is violated.
"""
from .types import (
    Constraint, Comprehend, Locate, Compare, Decomposition,
    MaskingSpec, Primitive, Target, Anchor, FreeEdge,
    Instance, Grid, Judgment, ImageResult,
)

__all__ = [
    "Constraint", "Comprehend", "Locate", "Compare", "Decomposition",
    "MaskingSpec", "Primitive", "Target", "Anchor", "FreeEdge",
    "Instance", "Grid", "Judgment", "ImageResult",
]
