"""
Writing the question.

Two rules govern every question CASCADE asks, and both exist to keep the
answer about the image rather than about the phrasing.

**The question is always affirmative.** It states what a normal product
looks like and asks whether that holds — never "is anything wrong here?".
A question that suggests a defect gets defects found, and the verdict is
recovered afterwards by inverting the answer in code, so the model is
never told which reply counts as the alarm.

**The reference is a number, not a memory.** For a range-based rule the
normal interval was measured in code from the K references and is written
into the question as integers. The model reads one coordinate off the
scaffold and checks membership; it is never asked to recall, estimate or
compare against an image it saw earlier.

The templates themselves live in `prompts/`, so what is sent can be read
without running anything.
"""
from __future__ import annotations

import re

from .client import load_prompt
from .types import Decomposition, Grid


# ═════════════════════════════════════════════════════════════════════
# The rule, as a question
# ═════════════════════════════════════════════════════════════════════
_VERBS = {"contains": "contain", "carries": "carry", "has": "have",
          "displays": "display", "includes": "include"}

_PATTERNS = [
    (r"^[Ee]ach (.+?) (contains?|has|have|carries|carry|displays?|includes?) (.+)$",
     lambda m: f"Does each {m[1]} {_VERBS.get(m[2], m[2])} {m[3]}?"),
    (r"^(?:The|A|An) (.+?) (contains?|has|have|carries|carry|displays?|includes?) (.+)$",
     lambda m: f"Does the {m[1]} {_VERBS.get(m[2], m[2])} {m[3]}?"),
    (r"^(?:The|A) (.+?) connects? (.+)$",  lambda m: f"Does the {m[1]} connect {m[2]}?"),
    (r"^(?:The|A) (.+?) (?:is|are) (.+)$", lambda m: f"Is the {m[1]} {m[2]}?"),
    (r"^(?:The|A) (.+?) reads? (.+)$",     lambda m: f"Does the {m[1]} read {m[2]}?"),
]


def as_question(text: str) -> str:
    """Restate a rule as the question that asks whether it holds."""
    text = text.strip().rstrip(".")
    for pattern, rewrite in _PATTERNS:
        match = re.match(pattern, text)
        if match:
            return rewrite(match)
    return f"Is it true that {text[0].lower()}{text[1:]}?"


def as_range_question(decomposition: Decomposition, grid: Grid) -> str:
    """State the measured normal interval and ask whether the image is in it.

    The wording follows what the scaffold actually shows: ticks along an
    object are counted, a fill line is read off a row, a placed object is
    read off all four of its edges.
    """
    name  = decomposition.targets[0].name.replace("_", " ") \
            if decomposition.targets else "target"
    cells = grid.normal_cells

    if decomposition.edge_type == "contour":
        span = cells.get("major") or cells.get("minor")
        if span:
            return f"Does the {name} occupy {_interval(span)} tick marks?"

    elif decomposition.edge_type == "horizontal":
        if "top" in cells:
            return f"Is the {name} level positioned at row {_interval(cells['top'])}?"

    else:
        stated = [f"{label}={_interval(cells[edge])}"
                  for edge, label in (("top", "row_start"), ("bottom", "row_end"),
                                      ("left", "col_start"), ("right", "col_end"))
                  if edge in cells]
        if stated:
            return f"Does the {name} occupy " + ", ".join(stated) + "?"

    # Nothing was measurable, so there is no interval to quote; fall back
    # to the rule itself rather than quoting a number that does not exist.
    return as_question(decomposition.constraint.text)


def _interval(bounds) -> str:
    low, high = bounds
    return str(low) if low == high else f"{low}~{high}"


# ═════════════════════════════════════════════════════════════════════
# What the scaffold shows
# ═════════════════════════════════════════════════════════════════════
def visual_cues(decomposition: Decomposition) -> str:
    """Tell the model how to read the scaffold it is about to be shown.

    Every line here describes something the renderer actually drew. A cue
    for a mark that is not on the image is worse than no cue: the model
    will look for it, and answer as though it found it.
    """
    spec  = decomposition.comprehend.masking_spec
    lines = []

    if decomposition.comprehend.is_range:
        if decomposition.edge_type == "contour":
            lines.append("Numbered tick marks (1), (2), (3), … are drawn along "
                         "each target's long axis.")
        else:
            lines.append("A coordinate grid is overlaid on the image. Each "
                         "intersection is labelled (row, col).")
        for target in decomposition.targets:
            lines.append(f"  - target: {target.name} ({target.prompts[0]})")
        if decomposition.locate and decomposition.locate.anchor:
            lines.append(f"The grid is positioned relative to "
                         f"'{decomposition.locate.anchor.name}'.")
        lines.append("Do not read any background pattern as the grid — only "
                     "the numbered marks that are explicitly drawn.")

    elif spec and spec.binding_type == "relational":
        lines.append("Instances that were matched to each other share the same "
                     "colour and the same ID.")
        lines.append("An instance with a colour no other instance shares was "
                     "matched to nothing.")
        for target in decomposition.targets:
            lines.append(f"  - {target.name}: {target.prompts[0]}")
        if spec.primary_axis and spec.matching_criterion in ("spatial_rank", "mirror"):
            lines.append(f"Instances were matched along the "
                         f"{spec.primary_axis} axis.")
        lines.append("Check whether the matched instances satisfy the rule.")

    else:
        lines.append("Each type of object is shown in a distinct colour, and "
                     "its instances are numbered within that type.")
        for target in decomposition.targets:
            count = f"×{target.expected_count}" if target.expected_count \
                    else "(count unspecified)"
            lines.append(f"  - {target.name} {count}: {target.prompts[0]}")
        lines.append("Verify each instance independently, by how it looks.")

    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════
# The whole prompt
# ═════════════════════════════════════════════════════════════════════
def build(decomposition: Decomposition, grid: Grid | None = None) -> str:
    """The verification prompt for one constraint on one image.

    The range template comes in two sections, one per ruler, because a
    grid is read and ticks are counted. Picking the section from the same
    `edge_type` that decided what to draw is what keeps the instructions
    describing the image the model is actually looking at.
    """
    if decomposition.comprehend.is_range:
        section = "TICKS" if decomposition.edge_type == "contour" else "GRID"
        text    = (as_range_question(decomposition, grid) if grid is not None
                   else as_question(decomposition.constraint.text))
        return load_prompt("question_range", section).format(
            visual_cues=visual_cues(decomposition), question=text)

    return load_prompt("question_composition").format(
        visual_cues=visual_cues(decomposition),
        question=as_question(decomposition.constraint.text))


# ═════════════════════════════════════════════════════════════════════
# Reading the answer back
# ═════════════════════════════════════════════════════════════════════
_ANSWERS = (
    re.compile(r"\[\[\s*FINAL\s*ANSWER\s*:\s*(Yes|No)\s*\]\]", re.IGNORECASE),
    re.compile(r"Status\s*:\s*\[?\s*(YES|NO)\s*\]?", re.IGNORECASE),
)


def read_answer(reply: str) -> str:
    """Pull "Yes", "No" — or "Unknown" — out of the model's reply."""
    for pattern in _ANSWERS:
        match = pattern.search(reply or "")
        if match:
            return match.group(1).capitalize()

    last = (reply or "").strip().split("\n")[-1].strip().strip(".")
    return last.capitalize() if last.lower() in ("yes", "no") else "Unknown"


def is_violation(answer: str) -> bool:
    """Invert the answer into a verdict.

    The question asked whether the rule holds, so "No" is the anomaly.
    An unreadable reply is not evidence of a defect, and is not treated
    as one.
    """
    return answer.lower() == "no"
