"""
Constraint grounding — turning one sentence into something executable.

Three stages, each answering a question the next one needs, and each
looking at exactly as much as it needs to:

    1. Comprehend   What kind of rule is this?          text only
    2. Locate       Which objects does it talk about?   one reference image
    3. Compare      Which edges may vary?               K reference images

Stage 3 runs only for range-based rules; a composition-based rule has no
extent to measure. The result is a `Decomposition`, computed once per
constraint and reused for every query image of that category.

The model's answers are validated against the closed vocabularies in
`cascade.types`. A stage that returns something outside its vocabulary
falls back to the safe default for its case rather than inventing a new
one, and an unparseable reply is an error, not a guess.
"""
from __future__ import annotations

from .client import Client, load_prompt, parse_json
from .scaffold.plan import build_matching_program, resolve_roles
from .types import (
    Anchor, Compare, Comprehend, Constraint, Decomposition,
    FreeEdge, Locate, MaskingSpec, Target,
    BINDING_TYPES, COLOR_SCHEMES, EDGES, ID_SCHEMES,
    MATCHING_CRITERIA, PRIMARY_AXES, TAXONOMY_CLASSES,
)


# ═════════════════════════════════════════════════════════════════════
# Stage 1 — Comprehend
# ═════════════════════════════════════════════════════════════════════
def comprehend(constraint: Constraint, client: Client) -> Comprehend:
    """Classify the rule, and for a composition-based rule, plan its scaffold.

    Text only. This stage never sees an image, which is the point: the
    kind of rule a sentence expresses is a property of the sentence.
    """
    reply = client.ask_json(
        load_prompt("stage1_comprehend", "SYSTEM") + "\n\n" +
        load_prompt("stage1_comprehend", "USER").format(
            category        = constraint.category,
            constraint_text = constraint.text,
        )
    )
    return _read_comprehend(reply)


def _read_comprehend(reply: dict) -> Comprehend:
    taxonomy = reply.get("taxonomy_class")
    if taxonomy not in TAXONOMY_CLASSES:
        raise ValueError(f"taxonomy_class={taxonomy!r} is not one of {TAXONOMY_CLASSES}")

    binding = reply.get("binding_type")
    if binding in ("null", "None", "none", ""):
        binding = None
    if binding is not None and binding not in BINDING_TYPES:
        raise ValueError(f"binding_type={binding!r} is not one of {BINDING_TYPES} or null")

    if taxonomy == "range-based":
        # A range rule constrains an extent, not a correspondence; it has
        # no binding and needs no masking spec.
        return Comprehend("range-based", None, None, reply.get("reasoning", "").strip())

    binding = binding or "property"
    return Comprehend(
        taxonomy_class = "composition-based",
        binding_type   = binding,
        masking_spec   = _read_masking_spec(reply.get("masking_spec"), binding),
        reasoning      = reply.get("reasoning", "").strip(),
    )


def _read_masking_spec(spec: dict | None, binding: str) -> MaskingSpec:
    """Validate the scaffold strategy, then compile it into a program."""
    spec = spec if isinstance(spec, dict) else {}
    relational = binding == "relational"

    def pick(key, vocabulary, default):
        value = spec.get(key)
        return value if value in vocabulary else default

    # No default axis: an axis is a claim about how the roles are laid
    # out, and the criteria that need one (spatial_rank, mirror) read it
    # from the model. Inventing one would put a statement in the visual
    # cues that the scaffold does not support.
    axis = pick("primary_axis", PRIMARY_AXES, "none")

    out = MaskingSpec(
        binding_type       = binding,
        matching_criterion = pick("matching_criterion", MATCHING_CRITERIA,
                                  "spatial_rank" if relational else "none"),
        # A property rule binds an instance to itself, so it pairs nothing.
        role_pairs         = (spec.get("role_pairs") or []) if relational else [],
        color_scheme       = pick("color_scheme", COLOR_SCHEMES,
                                  "shared_per_match" if relational else "unique_per_type"),
        id_scheme          = pick("id_scheme", ID_SCHEMES,
                                  "match_index" if relational else "instance_index"),
        primary_axis       = axis if axis != "none" else None,
        rationale          = spec.get("rationale", ""),
        skip_label_roles      = _strings(spec.get("skip_label_roles")),
        visual_property_roles = _strings(spec.get("visual_property_roles")),
    )
    out.matching_program = build_matching_program(
        out.matching_criterion, out.role_pairs, out.primary_axis
    )
    return out


def _strings(value) -> list:
    return [str(v) for v in value if isinstance(v, str)] if isinstance(value, list) else []


# ═════════════════════════════════════════════════════════════════════
# Stage 2 — Locate
# ═════════════════════════════════════════════════════════════════════
def locate(constraint: Constraint, grounding: Comprehend,
           reference_image, client: Client) -> Locate:
    """Map the entities the sentence names onto objects visible in an image.

    A constraint names things in a person's words, not in words a
    segmentation model accepts. This stage looks at one normal reference
    image and returns, for each entity, the short noun phrases that will
    find it — plus the anchor, if the rule needs a frame of reference.
    """
    reply = client.ask_json(
        load_prompt("stage2_locate").format(
            category        = constraint.category,
            constraint_text = constraint.text,
            taxonomy_class  = grounding.taxonomy_class,
            binding_type    = grounding.binding_type,
        ),
        images=[reference_image],
    )
    return _read_locate(reply)


def _read_locate(reply: dict) -> Locate:
    raw_targets = reply.get("targets")
    if not isinstance(raw_targets, list) or not raw_targets:
        raise ValueError(f"targets must be a non-empty list, got {raw_targets!r}")

    targets = []
    for entry in raw_targets:
        name = str(entry.get("name", "")).strip()
        if not name:
            raise ValueError(f"target has no name: {entry!r}")
        count = entry.get("expected_count")
        targets.append(Target(
            name           = name,
            prompts        = _phrases(entry.get("sam3_prompts"), name),
            expected_count = int(count) if count is not None else None,
            generic_prompt = str(entry.get("generic_prompt", "")).strip() or None,
        ))

    anchor = None
    raw_anchor = reply.get("anchor")
    if isinstance(raw_anchor, dict):
        name = str(raw_anchor.get("name", "")).strip()
        if name:
            anchor = Anchor(name, _phrases(raw_anchor.get("sam3_prompts"), name))

    return Locate(targets, anchor, reply.get("reasoning", ""))


def _phrases(value, fallback: str) -> list:
    """Normalise a segmentation-phrase field to a non-empty list of strings."""
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return [fallback]
    cleaned = [str(v).strip() for v in value if str(v).strip()]
    return cleaned or [fallback]


# ═════════════════════════════════════════════════════════════════════
# Stage 3 — Compare
# ═════════════════════════════════════════════════════════════════════
def compare(constraint: Constraint, grounding: Comprehend, located: Locate,
            reference_images: list, client: Client) -> Compare | None:
    """Decide which edges the rule leaves free, by looking across references.

    An edge that varies freely between normal images is not part of the
    rule, and measuring it would manufacture anomalies. Returns None for
    a composition-based rule, which has nothing to measure.
    """
    if not grounding.is_range:
        return None

    targets = ", ".join(
        t.name + (f"({t.expected_count})" if t.expected_count else "")
        for t in located.targets
    )
    reply = client.ask_json(
        load_prompt("stage3_compare").format(
            k               = len(reference_images),
            category        = constraint.category,
            constraint_text = constraint.text,
            targets_str     = targets,
            anchor_str      = located.anchor.name if located.anchor
                              else "null (self-contained)",
        ),
        images=list(reference_images),
    )
    return _read_compare(reply)


def _read_compare(reply: dict) -> Compare:
    raw = reply.get("free_edges", [])
    if not isinstance(raw, list):
        raise ValueError(f"free_edges must be a list, got {raw!r}")

    free = []
    for entry in raw:
        target = str(entry.get("target", "")).strip()
        reason = str(entry.get("reason", "")).strip()
        # The model sometimes packs several edges into one string
        # ("top, bottom" or "left/right"); split and keep known edges.
        tokens = str(entry.get("edge", "")).replace("/", ",").split(",")
        for token in (t.strip().lower() for t in tokens):
            if target and token in EDGES:
                free.append(FreeEdge(target, token, reason))
    return Compare(free)


# ═════════════════════════════════════════════════════════════════════
# Edge type
# ═════════════════════════════════════════════════════════════════════
def infer_edge_type(free_edges: list[FreeEdge]) -> str:
    """What shape of region the free edges imply, hence how to measure it.

    A free edge is one that varies across normal references, so it is
    exactly the edge whose range is worth measuring. An intrinsic
    dimension makes the rule about the object's own extent (contour);
    otherwise the free sides decide whether one coordinate suffices or
    the whole box is needed.

    With no free edge reported, nothing was observed to vary, and the
    conservative reading is to measure the whole box.
    """
    edges = {e.edge for e in free_edges}

    if edges & {"major", "minor", "length", "diameter"}:
        return "contour"

    sides = edges & {"top", "bottom", "left", "right"}
    if sides == {"top"}:
        return "horizontal"
    if sides and sides <= {"left", "right"}:
        return "vertical"
    return "rectangular"


# ═════════════════════════════════════════════════════════════════════
# All three stages
# ═════════════════════════════════════════════════════════════════════
def decompose(constraint: Constraint, reference_images: list,
              client: Client) -> Decomposition:
    """Run the three stages and package the result.

    Costs three model calls per constraint, once. Every query image of
    the category then reuses this.
    """
    grounded = comprehend(constraint, client)
    located  = locate(constraint, grounded, reference_images[0], client)
    compared = compare(constraint, grounded, located, reference_images, client)

    bind_roles(grounded, located)

    return Decomposition(
        constraint = constraint,
        comprehend = grounded,
        locate     = located,
        compare    = compared,
        edge_type  = infer_edge_type(compared.free_edges) if compared else None,
    )


def bind_roles(grounded: Comprehend, located: Locate) -> None:
    """Point Stage 1's roles at the targets Stage 2 actually found.

    Stage 1 plans the scaffold before anything has been looked at, so it
    names roles from the sentence. This re-reads those names against the
    objects Stage 2 named, and recompiles the matching program over the
    resolved names, so every step of the program refers to a role that
    exists.
    """
    spec = grounded.masking_spec
    if spec is None or not spec.role_pairs:
        return
    spec.role_pairs = resolve_roles(spec.role_pairs,
                                    [t.name for t in located.targets])
    spec.matching_program = build_matching_program(
        spec.matching_criterion, spec.role_pairs, spec.primary_axis)
