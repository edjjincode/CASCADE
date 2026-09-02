"""
Compiling a strategy into a program.

The model chose a `matching_criterion` — how instances of two roles
should be put into correspondence. This module turns that one word into
the sequence of primitives that carries it out. Nothing here calls a
model: the same criterion always compiles to the same program, which is
why a scaffold's structure is reproducible even though the strategy
behind it came from a language model.

The engine understands eight primitives. Only these appear below, and
only these can appear at all.
"""
from __future__ import annotations

from ..types import Primitive


def resolve_roles(role_pairs: list, target_names: list[str]) -> list:
    """Reconcile the names Stage 1 used with the names Stage 2 gave.

    The two stages name the same thing at different moments: Stage 1
    reads roles out of the sentence ("pushpin"), Stage 2 names what it
    found in the image ("yellow_pushpin"). They agree often but not
    always, and a pair whose names do not line up is a pair that never
    gets matched — the scaffold quietly falls back to colouring by type,
    and the correspondence the rule is about is never drawn.

    So a role resolves to the target whose name it equals, or failing
    that to the one target whose name contains it as a word. A role that
    matches nothing, or matches ambiguously, is dropped. This compares
    words with words; it knows nothing about what they denote.
    """
    def words(name: str) -> set:
        return set(str(name).lower().replace("-", "_").split("_"))

    lookup = {name.lower(): name for name in target_names}
    resolved = []
    for pair in role_pairs:
        if len(pair) != 2:
            continue
        mapped = []
        for role in pair:
            exact = lookup.get(str(role).lower())
            if exact:
                mapped.append(exact)
                continue
            contained = [name for name in target_names if words(role) <= words(name)]
            if len(contained) == 1:
                mapped.append(contained[0])
        if len(mapped) == 2:
            resolved.append(mapped)
    return resolved


def build_matching_program(criterion: str, role_pairs: list,
                           primary_axis: str | None = None) -> list[Primitive]:
    """Compile `criterion` into an executable matching program.

    With no pair of roles to match there is nothing to compile: a
    property rule binds each instance to an attribute of itself, and the
    engine colours it per type without any correspondence step.
    """
    if criterion == "none" or not role_pairs:
        return []

    program: list[Primitive] = []
    for role_a, role_b in role_pairs:
        program += _pair_program(criterion, role_a, role_b, primary_axis)
    return program


def _pair_program(criterion: str, a: str, b: str,
                  axis: str | None) -> list[Primitive]:
    if criterion == "spatial_rank":
        # The roles are separated along one axis, so their instances are
        # ordered along the other; rank i of A pairs with rank i of B.
        across = _across(axis)
        return [
            Primitive("sort", {"target": a, "key": across}),
            Primitive("sort", {"target": b, "key": across}),
            Primitive("zip_align", {"a": a, "b": b}),
        ]

    if criterion == "2d_grid":
        # Reading order — top to bottom, then left to right.
        return [
            Primitive("sort_lex", {"target": a, "keys": ["y", "x"]}),
            Primitive("sort_lex", {"target": b, "keys": ["y", "x"]}),
            Primitive("zip_align", {"a": a, "b": b}),
        ]

    if criterion == "mirror":
        # The roles face each other. Reflecting one across the axis of
        # symmetry puts corresponding instances on top of one another,
        # and then proximity is the correspondence.
        return [
            Primitive("reflect", {"target": b, "axis": _mirror(axis)}),
            Primitive("nn_match", {"a": a, "b": b, "metric": "spatial"}),
        ]

    if criterion == "nn_match":
        # No global order to exploit — each instance pairs with the
        # nearest instance of the other role.
        return [Primitive("nn_match", {"a": a, "b": b, "metric": "spatial"})]

    return []


def _across(primary_axis: str | None) -> str:
    """The axis instances are ordered along, given the axis dividing roles.

    Roles split left/right are each ordered top to bottom, and the other
    way round. With no axis stated, order vertically.
    """
    return "x" if primary_axis == "vertical" else "y"


def _mirror(primary_axis: str | None) -> str:
    """The axis to reflect across for a mirror-symmetric pairing."""
    return "y" if primary_axis == "horizontal" else "x"
