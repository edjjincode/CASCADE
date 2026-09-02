"""
The engine — pure geometry, no model.

`plan.py` compiled the model's strategy into a matching program. This
module runs it. Every primitive below is a statement about geometry:
where instances are, how they are ordered, which is nearest to which.
None of them knows what the objects are, and none of them calls a model,
so the same masks and the same program always give the same scaffold.

Eight primitives, each grounded in one structure:

    sort            a total order
    sort_lex        a product order, for reading order across a grid
    zip_align       a rank-wise bijection
    reflect         a group action, for mirror symmetry
    rotate          a group action, for rotational symmetry
    nn_match        nearest neighbour in a metric space
    group_by        equivalence classes
    explicit_pair   a correspondence given outright

Their output is a correspondence between instances. Two schemes then turn
that correspondence into what the viewer actually sees: a colour and a
printed ID per instance. Same colour means "these go together"; different
colour means "these are distinct" — which is exactly the distinction a
model conflates when it is left to bind instances on its own.
"""
from __future__ import annotations

import math

from ..types import Instance, MaskingSpec, Primitive

Roles = dict[str, list[Instance]]
Key   = tuple[str, int]        # (role, index within role)


# ═════════════════════════════════════════════════════════════════════
# Primitives
# ═════════════════════════════════════════════════════════════════════
def _key(instance: Instance, axis: str) -> float:
    if axis == "x":
        return instance.centroid[0]
    if axis == "y":
        return instance.centroid[1]
    if axis == "area":
        return float(instance.mask.sum())
    raise ValueError(f"unknown sort key {axis!r}")


def op_sort(instances: list[Instance], key: str = "y",
            reverse: bool = False) -> list[Instance]:
    """Order instances along one axis, or by size."""
    return sorted(instances, key=lambda i: _key(i, key), reverse=reverse)


def op_sort_lex(instances: list[Instance], keys: list[str]) -> list[Instance]:
    """Order by several axes at once — ["y", "x"] is reading order."""
    return sorted(instances, key=lambda i: tuple(_key(i, k) for k in keys))


def op_zip_align(a: list[Instance], b: list[Instance]) -> list[tuple]:
    """Pair by rank. A surplus on either side is left unpaired, which is
    what makes a missing or extra instance visible in the scaffold."""
    return list(zip(a, b))


def op_reflect(instances: list[Instance], axis: str) -> list[Instance]:
    """Mirror instances across the image's centre line.

    Used before matching two roles that face each other: reflecting one
    of them lands corresponding instances on top of one another, so
    proximity becomes the correspondence.
    """
    out = []
    for i in instances:
        height, width = i.mask.shape[:2]
        x0, y0, x1, y1 = i.bbox
        if axis == "y":                       # mirror left-right
            mask, bbox = i.mask[:, ::-1], (width - x1, y0, width - x0, y1)
        elif axis == "x":                     # mirror top-bottom
            mask, bbox = i.mask[::-1, :], (x0, height - y1, x1, height - y0)
        else:
            raise ValueError(f"unknown reflect axis {axis!r}")
        out.append(Instance(i.role, mask, bbox))
    return out


def op_rotate(instances: list[Instance], angle: float) -> list[Instance]:
    """Turn instances about the image centre, for rotational symmetry."""
    import numpy as np
    out = []
    for i in instances:
        turns = int(round(angle / 90)) % 4
        mask  = np.rot90(i.mask, -turns)
        height, width = i.mask.shape[:2]
        x0, y0, x1, y1 = i.bbox
        for _ in range(turns):                # each quarter turn, clockwise
            x0, y0, x1, y1 = height - y1, x0, height - y0, x1
            height, width = width, height
        out.append(Instance(i.role, mask, (x0, y0, x1, y1)))
    return out


def op_nn_match(a: list[Instance], b: list[Instance],
                metric: str = "spatial") -> list[tuple]:
    """Pair each instance of A with its nearest free instance of B.

    Greedy and one-to-one: once an instance of B is taken it cannot be
    taken again, so a compartment holding two objects leaves one of them
    unmatched instead of quietly double-counting.
    """
    if metric != "spatial":
        raise ValueError(f"unknown nn_match metric {metric!r}")
    free, pairs = list(b), []
    for one in a:
        if not free:
            break
        nearest = min(free, key=lambda other: math.dist(one.centroid, other.centroid))
        free.remove(nearest)
        pairs.append((one, nearest))
    return pairs


def op_group_by(instances: list[Instance], key: str,
                n_bins: int = 2) -> dict[int, list[Instance]]:
    """Split instances into equal bands along an axis."""
    if not instances:
        return {}
    coords = [_key(i, key) for i in instances]
    low, high = min(coords), max(coords)
    if high - low < 1e-6:
        return {0: list(instances)}
    width  = (high - low) / n_bins
    groups: dict[int, list[Instance]] = {}
    for instance, coord in zip(instances, coords):
        band = min(int((coord - low) / width), n_bins - 1)
        groups.setdefault(band, []).append(instance)
    return groups


def op_explicit_pair(a: list[Instance], b: list[Instance],
                     pairs: list) -> list[tuple]:
    """Pair by index, when the correspondence is simply given."""
    return [(a[i], b[j]) for i, j in pairs
            if 0 <= i < len(a) and 0 <= j < len(b)]


PRIMITIVES = {
    "sort": op_sort, "sort_lex": op_sort_lex, "zip_align": op_zip_align,
    "reflect": op_reflect, "rotate": op_rotate, "nn_match": op_nn_match,
    "group_by": op_group_by, "explicit_pair": op_explicit_pair,
}

_REORDER = ("sort", "sort_lex", "reflect", "rotate", "group_by")
_PAIR    = ("zip_align", "nn_match", "explicit_pair")


# ═════════════════════════════════════════════════════════════════════
# Running a program
# ═════════════════════════════════════════════════════════════════════
def run_program(program: list[Primitive], roles: Roles) -> list[tuple[Key, Key]]:
    """Execute the program and return the correspondence it found.

    Instances are carried through as `(role, index)` keys, so an
    instance stays identifiable no matter how many times the program
    sorts, reflects or rotates it.
    """
    tagged: dict[str, list[tuple[Key, Instance]]] = {
        role: [((role, index), instance) for index, instance in enumerate(items)]
        for role, items in roles.items()
    }

    pairs: list[tuple[Key, Key]] = []
    for step in program:
        args = dict(step.args)

        if step.op in _REORDER:
            role = args.pop("target")
            keys, instances = zip(*tagged[role]) if tagged[role] else ((), ())
            result = PRIMITIVES[step.op](list(instances), **args)
            if isinstance(result, dict):                 # group_by
                result = [i for band in sorted(result) for i in result[band]]
            lookup = {id(instance): key for key, instance in tagged[role]}
            # A primitive may return new objects (reflect, rotate); those
            # carry no identity, so fall back to position.
            tagged[role] = [
                (lookup.get(id(instance), keys[index]), instance)
                for index, instance in enumerate(result)
            ]

        elif step.op in _PAIR:
            a_role, b_role = args.pop("a"), args.pop("b")
            a_keys, a_items = _split(tagged.get(a_role, []))
            b_keys, b_items = _split(tagged.get(b_role, []))
            a_index = {id(i): k for k, i in zip(a_keys, a_items)}
            b_index = {id(i): k for k, i in zip(b_keys, b_items)}
            for one, other in PRIMITIVES[step.op](a_items, b_items, **args):
                pairs.append((a_index[id(one)], b_index[id(other)]))

        else:
            raise ValueError(f"unknown primitive {step.op!r}")

    return pairs


def _split(tagged: list) -> tuple[list, list]:
    return ([k for k, _ in tagged], [i for _, i in tagged])


# ═════════════════════════════════════════════════════════════════════
# Colour and ID
# ═════════════════════════════════════════════════════════════════════
# Distinct at a glance, and distinguishable in greyscale.
PALETTE = [
    (0, 165, 255),    # orange
    (255, 50, 50),    # blue-red
    (255, 255, 0),    # cyan
    (0, 255, 0),      # green
    (255, 0, 255),    # magenta
    (0, 215, 255),    # gold
    (128, 128, 128),  # grey
    (255, 0, 0),      # blue
    (200, 200, 0),    # teal
    (100, 0, 200),    # purple
]

# When a rule names a colour, the overlay should not contradict it: a
# constraint about the red one is easier to check if the red one is red.
# These are colour words, not object names — the scaffold never gets a
# per-product palette.
COLOR_WORDS = {
    "red": (0, 0, 255), "green": (0, 255, 0), "blue": (255, 0, 0),
    "yellow": (0, 255, 255), "orange": (0, 165, 255), "purple": (128, 0, 128),
    "pink": (203, 192, 255), "cyan": (255, 255, 0), "magenta": (255, 0, 255),
    "brown": (42, 42, 165), "gold": (0, 215, 255), "silver": (192, 192, 192),
    "gray": (128, 128, 128), "grey": (128, 128, 128),
    "white": (220, 220, 220), "black": (50, 50, 50),
}


def distinct_colors(n: int) -> list[tuple]:
    """`n` colours, no two of which read as the same colour.

    A fixed palette cannot do this: fifteen compartments against ten
    colours wraps, and then match 1 and match 11 are the same green while
    the cue says a shared colour means a pair. Spacing hues by the golden
    angle keeps consecutive ranks far apart and never repeats within a
    run, and alternating lightness separates hues that are close anyway.
    """
    import colorsys

    out = []
    for i in range(max(n, 0)):
        hue       = (i * 0.6180339887) % 1.0
        lightness = (0.45, 0.62, 0.54)[i % 3]
        r, g, b   = colorsys.hls_to_rgb(hue, lightness, 0.95)
        out.append((int(b * 255), int(g * 255), int(r * 255)))    # BGR
    return out


def role_color(role: str) -> tuple | None:
    """The colour a role names, if it names one."""
    for word in role.lower().replace("-", "_").split("_"):
        if word in COLOR_WORDS:
            return COLOR_WORDS[word]
        if len(word) > 3 and word.endswith("s") and word[:-1] in COLOR_WORDS:
            return COLOR_WORDS[word[:-1]]
    return None


def _shade(color: tuple, rank: int, total: int) -> tuple:
    if total <= 1:
        return color
    factor = 0.4 + 0.6 * rank / (total - 1)
    return tuple(int(c * factor) for c in color)


def assign_colors(scheme: str, roles: Roles,
                  pairs: list[tuple[Key, Key]]) -> dict[Key, tuple]:
    """Turn the correspondence into a colour per instance."""
    colors: dict[Key, tuple] = {}

    if scheme == "shared_per_match":
        # Matched instances share a colour, so the pairing is visible. The
        # colour never comes from a role's name: this scheme exists to
        # tell matches apart, and a role called "yellow_pushpin" would
        # otherwise paint every match the same yellow and erase the
        # distinction it was chosen to draw.
        #
        # Anything the matching left over continues the same sequence, so
        # it ends up with a colour nothing else has — which is exactly
        # what the visual cue tells the model to look for.
        matched  = {key for pair in pairs for key in pair}
        leftover = [key for key in _all_keys(roles) if key not in matched]
        wheel    = distinct_colors(len(pairs) + len(leftover))
        for rank, pair in enumerate(pairs):
            for key in pair:
                colors[key] = wheel[rank]
        for rank, key in enumerate(leftover):
            colors[key] = wheel[len(pairs) + rank]
        return colors

    elif scheme == "gradient_by_rank":
        # Order itself is the information, so shade along it.
        ordered = [key for pair in pairs for key in pair]
        base    = PALETTE[0]
        for rank, key in enumerate(ordered):
            colors[key] = _shade(base, rank, len(ordered))

    elif scheme == "per_instance":
        # Every instance distinct: identity matters, type does not.
        keys  = _all_keys(roles)
        wheel = distinct_colors(len(keys))
        for n, key in enumerate(keys):
            colors[key] = wheel[n]

    elif scheme != "unique_per_type":
        raise ValueError(f"unknown color_scheme {scheme!r}")

    # Instances the scheme left out — every instance under
    # unique_per_type — take a colour standing for their role. Where the
    # role's own name says a colour, use it: a rule about the red one is
    # easier to check when the red one is red.
    for n, role in enumerate(roles):
        color = role_color(role) or PALETTE[(len(pairs) + n) % len(PALETTE)]
        for index in range(len(roles[role])):
            colors.setdefault((role, index), color)
    return colors


def assign_labels(scheme: str, roles: Roles,
                  pairs: list[tuple[Key, Key]]) -> dict[Key, str]:
    """Turn the correspondence into a printed ID per instance."""
    labels: dict[Key, str] = {}

    if scheme == "match_index":
        # Matched instances share a number, so a pair reads as one thing.
        for rank, pair in enumerate(pairs):
            for key in pair:
                labels[key] = f"#{rank + 1}"

        # Whatever the matching left over continues the same sequence.
        # Numbering it within its role instead would collide: the sixteenth
        # pushpin, unmatched, would print #7 alongside the pushpin that
        # really is match 7, and the surplus — the anomaly itself — would
        # read as a duplicate of something else.
        nxt = len(pairs) + 1
        for key in _all_keys(roles):
            if key not in labels:
                labels[key] = f"#{nxt}"
                nxt += 1
        return labels

    elif scheme == "role_position":
        for role, items in roles.items():
            prefix = role[0].upper() if len(role) > 3 else role.upper()
            for index in range(len(items)):
                labels[(role, index)] = f"{prefix}.{index + 1}"

    elif scheme not in ("instance_index", "natural_position"):
        raise ValueError(f"unknown id_scheme {scheme!r}")

    # instance_index and natural_position both number within the role;
    # they differ in whether the caller sorted the role first. The same
    # numbering also fills in for anything left unmatched above.
    for role, items in roles.items():
        for index in range(len(items)):
            labels.setdefault((role, index), f"#{index + 1}")
    return labels


def _all_keys(roles: Roles) -> list[Key]:
    return [(role, index) for role, items in roles.items()
            for index in range(len(items))]


# ═════════════════════════════════════════════════════════════════════
# The whole execution
# ═════════════════════════════════════════════════════════════════════
def execute(spec: MaskingSpec, roles: Roles) -> dict:
    """Run the matching program, then colour and number the result."""
    pairs = run_program(spec.matching_program, roles) \
            if spec.matching_program else []
    return {
        "pairs":  pairs,
        "colors": assign_colors(spec.color_scheme, roles, pairs),
        "labels": assign_labels(spec.id_scheme, roles, pairs),
    }
