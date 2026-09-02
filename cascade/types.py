"""
Types — the contract between the model and the code.

CASCADE splits the work in three layers:

    1.  the model chooses a *strategy*, and only from the closed
        vocabularies defined at the top of this file;
    2.  `cascade.scaffold.plan` compiles that strategy into a matching
        program, deterministically;
    3.  `cascade.scaffold.engine` executes the program with pure
        geometry — no model in the loop.

Everything below is layer 1's output format. If a stage returns a value
outside these vocabularies the parser rejects it, which is why no product
category is ever named in the code.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

# ═════════════════════════════════════════════════════════════════════
# Closed vocabularies
# ═════════════════════════════════════════════════════════════════════

# What kind of rule this is. A range-based rule constrains a continuous
# extent (a fill level, a length) and needs the alignment scaffold; a
# composition-based rule constrains what is present and how the parts
# correspond, and needs the disambiguation scaffold.
TAXONOMY_CLASSES = ("range-based", "composition-based")

# How the instances of a composition-based rule bind. `property` binds an
# instance to an attribute of itself; `relational` binds instances of two
# roles to each other.
BINDING_TYPES = ("property", "relational")

# How instances of the two roles are put into correspondence.
MATCHING_CRITERIA = ("none", "spatial_rank", "2d_grid", "mirror", "nn_match")

# How the scaffold assigns overlay colour.
COLOR_SCHEMES = ("unique_per_type", "shared_per_match",
                 "gradient_by_rank", "per_instance")

# How the scaffold assigns the printed numeric ID.
ID_SCHEMES = ("instance_index", "match_index",
              "natural_position", "role_position")

# The axis a spatial ordering runs along, when one applies.
PRIMARY_AXES = ("horizontal", "vertical", "none")

# The measurable edge of a range-based rule.
EDGES = ("top", "bottom", "left", "right",
         "major", "minor", "length", "diameter")

# The eight primitives the engine can execute. The model may reference
# them by name in a matching program but cannot define new ones.
PRIMITIVES = ("sort", "sort_lex", "reflect", "rotate", "group_by",
              "zip_align", "nn_match", "explicit_pair")

# Shape of the region a range-based rule measures, inferred from which
# edges are constrained.
EDGE_TYPES = ("horizontal", "vertical", "rectangular", "contour")


def _check(value, vocabulary, field_name):
    """Reject anything outside a closed vocabulary."""
    if value not in vocabulary:
        raise ValueError(
            f"{field_name}={value!r} is outside its vocabulary {vocabulary}"
        )
    return value


# ═════════════════════════════════════════════════════════════════════
# Input
# ═════════════════════════════════════════════════════════════════════
@dataclass
class Constraint:
    """One atomic unit of a LogicQA bullet — the unit CASCADE verifies."""
    constraint_id: str          # "PP-L1"
    category:      str          # "pushpins"
    text:          str          # natural-language rule

    def to_dict(self) -> dict:
        return asdict(self)


# ═════════════════════════════════════════════════════════════════════
# Stage 1 — Comprehend  (text only, no image)
# ═════════════════════════════════════════════════════════════════════
@dataclass
class Primitive:
    """One step of a matching program."""
    op:   str
    args: dict = field(default_factory=dict)

    def __post_init__(self):
        _check(self.op, PRIMITIVES, "op")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MaskingSpec:
    """How to disambiguate instances for a composition-based rule.

    The model fills in the *strategy*: which roles correspond
    (`role_pairs`), by what criterion (`matching_criterion`), and how the
    result should be drawn (`color_scheme`, `id_scheme`). It does not
    write `matching_program` — that is compiled from the criterion by
    `cascade.scaffold.plan`, deterministically, and executed by the
    engine. Same strategy in, same program out, every run.
    """
    binding_type:       str
    matching_criterion: str = "none"
    role_pairs:         list = field(default_factory=list)
    matching_program:   list[Primitive] = field(default_factory=list)
    color_scheme:     str = "unique_per_type"
    id_scheme:        str = "instance_index"
    primary_axis:     Optional[str] = None
    rationale:        str = ""

    # Roles drawn with a colour but no printed ID — used when the ID
    # would clutter without adding information.
    skip_label_roles: list = field(default_factory=list)
    # Roles whose own colour or texture *is* what the rule is about.
    # These are outlined rather than filled, so the overlay does not
    # cover the very property being checked.
    visual_property_roles: list = field(default_factory=list)

    def __post_init__(self):
        _check(self.binding_type, BINDING_TYPES, "binding_type")
        _check(self.matching_criterion, MATCHING_CRITERIA, "matching_criterion")
        _check(self.color_scheme, COLOR_SCHEMES, "color_scheme")
        _check(self.id_scheme, ID_SCHEMES, "id_scheme")
        if self.primary_axis is not None:
            _check(self.primary_axis, PRIMARY_AXES, "primary_axis")
        self.matching_program = [
            p if isinstance(p, Primitive) else Primitive(**p)
            for p in self.matching_program
        ]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["matching_program"] = [p.to_dict() for p in self.matching_program]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "MaskingSpec":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Comprehend:
    """Stage 1 output. Pure linguistic analysis — no objects, no image."""
    taxonomy_class: str
    binding_type:   Optional[str] = None       # None for range-based
    masking_spec:   Optional[MaskingSpec] = None
    reasoning:      str = ""

    def __post_init__(self):
        _check(self.taxonomy_class, TAXONOMY_CLASSES, "taxonomy_class")
        if self.binding_type is not None:
            _check(self.binding_type, BINDING_TYPES, "binding_type")

    @property
    def is_range(self) -> bool:
        return self.taxonomy_class == "range-based"

    @property
    def is_composition(self) -> bool:
        return self.taxonomy_class == "composition-based"

    def to_dict(self) -> dict:
        return {
            "taxonomy_class": self.taxonomy_class,
            "binding_type":   self.binding_type,
            "masking_spec":   self.masking_spec.to_dict() if self.masking_spec else None,
            "reasoning":      self.reasoning,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Comprehend":
        spec = d.get("masking_spec")
        return cls(
            taxonomy_class = d["taxonomy_class"],
            binding_type   = d.get("binding_type"),
            masking_spec   = MaskingSpec.from_dict(spec) if spec else None,
            reasoning      = d.get("reasoning", ""),
        )


# ═════════════════════════════════════════════════════════════════════
# Stage 2 — Locate  (one reference image)
# ═════════════════════════════════════════════════════════════════════
@dataclass
class Target:
    """An object role the rule talks about."""
    name:           str
    prompts:        list = field(default_factory=list)  # segmentation phrases
    expected_count: Optional[int] = None
    generic_prompt: Optional[str] = None   # shared phrase for sibling roles

    def __post_init__(self):
        if not self.prompts:
            self.prompts = [self.name]
        if self.generic_prompt is None:
            self.generic_prompt = min(self.prompts, key=len)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Anchor:
    """The object the measurement is expressed relative to.

    None means the rule is self-contained: it needs no frame of
    reference beyond the objects it already names.
    """
    name:    str
    prompts: list = field(default_factory=list)

    def __post_init__(self):
        if not self.prompts:
            self.prompts = [self.name]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Locate:
    """Stage 2 output. Which objects, and what to segment them by."""
    targets:   list[Target] = field(default_factory=list)
    anchor:    Optional[Anchor] = None
    reasoning: str = ""

    def to_dict(self) -> dict:
        return {
            "targets":   [t.to_dict() for t in self.targets],
            "anchor":    self.anchor.to_dict() if self.anchor else None,
            "reasoning": self.reasoning,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Locate":
        anchor = d.get("anchor")
        return cls(
            targets   = [Target(**t) for t in d.get("targets", [])],
            anchor    = Anchor(**anchor) if anchor else None,
            reasoning = d.get("reasoning", ""),
        )


# ═════════════════════════════════════════════════════════════════════
# Stage 3 — Compare  (K reference images; range-based only)
# ═════════════════════════════════════════════════════════════════════
@dataclass
class FreeEdge:
    """An edge that is free to move, and so is worth measuring.

    Stage 3 looks at the K references together and reports which of a
    target's edges shift between them. A rigid edge — one held in place
    by the object's own geometry — carries no information; a free edge is
    where the constrained quantity actually lives, so it is exactly the
    edge the grid has to resolve.
    """
    target: str
    edge:   str
    reason: str = ""

    def __post_init__(self):
        _check(self.edge, EDGES, "edge")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Compare:
    """Stage 3 output.

    An empty list means nothing was observed to move, and the
    conservative reading is to measure the whole bounding box.
    """
    free_edges: list[FreeEdge] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"free_edges": [e.to_dict() for e in self.free_edges]}

    @classmethod
    def from_dict(cls, d: dict) -> "Compare":
        return cls(free_edges=[FreeEdge(**e) for e in d.get("free_edges", [])])


# ═════════════════════════════════════════════════════════════════════
# Combined decomposition
# ═════════════════════════════════════════════════════════════════════
@dataclass
class Decomposition:
    """A constraint after all three grounding stages.

    Computed once per constraint, cached, and reused for every query
    image of that category.
    """
    constraint: Constraint
    comprehend: Comprehend
    locate:     Optional[Locate] = None
    compare:    Optional[Compare] = None
    edge_type:  Optional[str] = None       # range-based only

    def __post_init__(self):
        if self.edge_type is not None:
            _check(self.edge_type, EDGE_TYPES, "edge_type")

    @property
    def constraint_id(self) -> str:
        return self.constraint.constraint_id

    @property
    def targets(self) -> list[Target]:
        return self.locate.targets if self.locate else []

    def to_dict(self) -> dict:
        return {
            "constraint": self.constraint.to_dict(),
            "comprehend": self.comprehend.to_dict(),
            "locate":     self.locate.to_dict() if self.locate else None,
            "compare":    self.compare.to_dict() if self.compare else None,
            "edge_type":  self.edge_type,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Decomposition":
        return cls(
            constraint = Constraint(**d["constraint"]),
            comprehend = Comprehend.from_dict(d["comprehend"]),
            locate     = Locate.from_dict(d["locate"]) if d.get("locate") else None,
            compare    = Compare.from_dict(d["compare"]) if d.get("compare") else None,
            edge_type  = d.get("edge_type"),
        )


# ═════════════════════════════════════════════════════════════════════
# Detection
# ═════════════════════════════════════════════════════════════════════
@dataclass(eq=False)          # identity, not pixel-wise array comparison
class Instance:
    """One detected object instance.

    `mask` is a boolean HxW array; `bbox` is (x0, y0, x1, y1) in pixels.
    """
    role: str
    mask: object                     # numpy.ndarray[bool], HxW
    bbox: tuple

    @property
    def centroid(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bbox
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


# ═════════════════════════════════════════════════════════════════════
# Calibration  (range-based only)
# ═════════════════════════════════════════════════════════════════════
@dataclass
class Grid:
    """The alignment scaffold's grid, fitted on the K references.

    `origin` is a*, `spacing` is w*, both in pixels and both relative to
    the anchor's bounding box. `normal_cells` is the closed integer
    interval the references occupy, per constrained edge — this is what
    the verification question quotes.
    """
    origin:       tuple[float, float]           # (a_x, a_y)
    spacing:      float                         # w*, shared by both axes
    normal_cells: dict = field(default_factory=dict)   # edge -> [lo, hi]

    def to_dict(self) -> dict:
        return {
            "origin":       list(self.origin),
            "spacing":      self.spacing,
            "normal_cells": self.normal_cells,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Grid":
        return cls(
            origin       = tuple(d["origin"]),
            spacing      = d["spacing"],
            normal_cells = d.get("normal_cells", {}),
        )


# ═════════════════════════════════════════════════════════════════════
# Verification
# ═════════════════════════════════════════════════════════════════════
@dataclass
class Judgment:
    """The verdict for one constraint on one image.

    The question is always phrased affirmatively, so the model answering
    "Yes" means the rule holds. `violated` is that answer inverted; a
    leading question never reaches the model.
    """
    constraint_id: str
    answer:        str                # the model's literal Yes / No
    violated:      bool               # True == anomaly
    question:      str = ""
    scaffold_path: Optional[str] = None
    raw_response:  str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ImageResult:
    """The verdict for one image: anomalous if any constraint is violated."""
    image_path: str
    category:   str
    judgments:  list[Judgment] = field(default_factory=list)

    @property
    def is_anomaly(self) -> bool:
        return any(j.violated for j in self.judgments)

    @property
    def violations(self) -> list[str]:
        return [j.constraint_id for j in self.judgments if j.violated]

    def to_dict(self) -> dict:
        return {
            "image_path": self.image_path,
            "category":   self.category,
            "prediction": "anomaly" if self.is_anomaly else "normal",
            "violations": self.violations,
            "judgments":  [j.to_dict() for j in self.judgments],
        }
