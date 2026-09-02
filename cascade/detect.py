"""
Instance masks — an input to CASCADE, not a part of it.

CASCADE's claim is about scaffolding, not segmentation: given instance
masks for the objects a constraint names, it draws the image so the rule
can be read off it. Where those masks come from is somebody else's
problem, so this module treats segmentation as an interface with one
method and ships a default implementation that reads precomputed masks.

    MaskSource.masks(image, phrases) -> {phrase: [Instance, ...]}

The repository ships a mask pack covering the MVTec-LOCO images, produced
once with SAM3 from the segmentation phrases Stage 2 returned. Reading it
needs no GPU and no model, so a clone can run the whole pipeline with
nothing but an API key. To segment with something else, implement the
protocol and pass it to the pipeline.

Roles are resolved from phrases afterwards, by `assign_roles`, using only
the structure of the target names — never the product they describe.
"""
from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from . import config
from .types import Instance, Target

# Two detections of the same object overlap heavily. Above this, treat
# them as one and keep the larger.
DUPLICATE_IOU = 0.5

# A handful of stray pixels is not an instance.
MIN_AREA = 50


class MaskSource(Protocol):
    """Anything that can produce instance masks for an image."""

    def masks(self, image: str | Path,
              phrases: list[str]) -> dict[str, list[Instance]]:
        """Return the instances each phrase finds in `image`.

        `Instance.role` is set to the phrase; `assign_roles` maps
        phrases onto the constraint's target names afterwards.
        """
        ...


# ═════════════════════════════════════════════════════════════════════
# The shipped default: precomputed masks
# ═════════════════════════════════════════════════════════════════════
@dataclass
class MaskPack(MaskSource):
    """Instance masks read from a pack on disk, one file per category.

    A pack is a gzipped JSON file mapping image name -> phrase ->
    run-length-encoded masks. It is plain text under the compression, so
    what the pipeline is given can be inspected without running anything.
    """
    root: Path = config.MASKS

    def __post_init__(self):
        self.root = Path(self.root)
        self._cache: dict[str, dict] = {}
        self._index_cache: dict[str, dict] = {}

    def _index(self, category: str) -> dict:
        """Every key this pack answers to, exact and relaxed.

        A dataset is not always laid out the way it was published. Ours
        groups two categories' test splits into `orange/`, `blue/` and so
        on, and someone else may sort theirs differently again. The pack
        is keyed by MVTec-LOCO's own layout, and looking up the relaxed
        form as well — split plus file name, with any extra grouping
        dropped — lets the same pack serve either.
        """
        if category not in self._index_cache:
            index = {}
            for key in self._pack(category)["images"]:
                index.setdefault(relax(key), key)
                index[key] = key
            self._index_cache[category] = index
        return self._index_cache[category]

    def _pack(self, category: str) -> dict:
        if category not in self._cache:
            path = self.root / f"{category}.json.gz"
            if not path.exists():
                raise FileNotFoundError(
                    f"No mask pack for {category!r} at {path}. The packs "
                    f"ship in masks/; set CASCADE_MASKS if they live "
                    f"elsewhere, or pass a MaskSource of your own to the "
                    f"pipeline (see masks/README.md)."
                )
            with gzip.open(path, "rt") as f:
                self._cache[category] = json.load(f)
        return self._cache[category]

    def masks(self, image: str | Path,
              phrases: list[str]) -> dict[str, list[Instance]]:
        image = Path(image)
        pack  = self._pack(category_of(image))
        key   = pack_key(image)
        index = self._index(category_of(image))
        entry = pack["images"].get(index.get(key) or index.get(relax(key), ""))
        if entry is None:
            raise KeyError(
                f"{key!r} is not in the mask pack for {pack['category']}. "
                f"The packs cover the three train/good references and a "
                f"sample of the test split — see masks/README.md, and "
                f"MaskPack.covers({pack['category']!r}) for the list. "
                f"(CASCADE_DATA is {config.DATA_ROOT}.)"
            )

        height, width = entry["size"]
        out: dict[str, list[Instance]] = {}
        for phrase in phrases:
            found = entry["objects"].get(phrase, [])
            out[phrase] = [
                Instance(role=phrase,
                         mask=decode_rle(item["rle"], height, width),
                         bbox=tuple(item["bbox"]))
                for item in found
            ]
        return out

    def covers(self, category: str) -> set[str]:
        """Image stems this pack has masks for."""
        return set(self._pack(category)["images"])


def category_of(image: Path) -> str:
    """Which category an image belongs to, from its path."""
    for part in Path(image).parts[::-1]:
        if part in config.CATEGORIES:
            return part
    raise ValueError(f"cannot tell which category {image} belongs to")


def pack_key(image: str | Path, root: Path | None = None) -> str:
    """How an image is named inside a mask pack.

    A file name alone is not unique — MVTec-LOCO reuses `000.png` in
    every split — so the key is the path from the dataset root down,
    without its extension. That makes the key stable wherever the dataset
    is unpacked, and readable when the pack is opened by hand.
    """
    image = Path(image).resolve()
    root  = Path(root or config.DATA_ROOT).resolve()
    try:
        relative = image.relative_to(root)
    except ValueError:
        # Outside the dataset root: fall back to the deepest path that
        # still starts at the category.
        parts    = image.parts
        category = category_of(image)
        relative = Path(*parts[len(parts) - 1 - parts[::-1].index(category):])
    return relative.with_suffix("").as_posix()


def relax(key: str) -> str:
    """A pack key with any extra grouping levels dropped.

    `juice_bottle/test/logical_anomalies/orange/134` and
    `juice_bottle/test/logical_anomalies/134` name the same photograph;
    the `orange/` is one dataset copy's own filing, not part of the
    image's identity. Keeping category, split and file name discards
    exactly that.
    """
    parts = key.split("/")
    return "/".join(parts[:3] + parts[-1:]) if len(parts) > 4 else key


# ═════════════════════════════════════════════════════════════════════
# Run-length encoding
# ═════════════════════════════════════════════════════════════════════
def encode_rle(mask: np.ndarray) -> str:
    """Column-major run lengths, starting from a run of zeros.

    Masks are mostly large flat regions, so runs compress them to a
    fraction of a bitmap while staying human-readable in the pack.
    """
    flat = np.asarray(mask, dtype=bool).T.ravel()
    if flat.size == 0:
        return ""
    # Boundaries between runs, plus the two ends.
    edges  = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    starts = np.concatenate(([0], edges))
    runs   = np.diff(np.concatenate((starts, [flat.size])))
    if flat[0]:                       # always begin with a zero run
        runs = np.concatenate(([0], runs))
    return " ".join(map(str, runs.tolist()))


def decode_rle(rle: str, height: int, width: int) -> np.ndarray:
    """Inverse of `encode_rle`."""
    flat  = np.zeros(height * width, dtype=bool)
    value, at = False, 0
    for run in (int(r) for r in rle.split()):
        if value:
            flat[at:at + run] = True
        at += run
        value = not value
    return flat.reshape(width, height).T


# ═════════════════════════════════════════════════════════════════════
# Phrases -> roles
# ═════════════════════════════════════════════════════════════════════
def assign_roles(source: MaskSource, image: str | Path,
                 targets: list[Target]) -> dict[str, list[Instance]]:
    """Group the detected instances under the role names the rule uses.

    Several targets often share one segmentation phrase, because a
    detector cannot see the difference the rule cares about: `left_pin`
    and `right_pin` are both just "pin". The phrase finds every instance,
    and the target name says which of them this role means — a leading
    `left_`, `right_`, `top_` or `bottom_` takes the extreme along that
    axis, and `expected_count` says how many.

    This reads the target's name, never the product it names.
    """
    phrases = []
    for target in targets:
        for phrase in target.prompts + [target.generic_prompt]:
            if phrase and phrase not in phrases:
                phrases.append(phrase)

    found = source.masks(image, phrases)

    roles: dict[str, list[Instance]] = {}
    claimed: list[Instance] = []
    for target in targets:
        pool = deduplicate([
            instance
            for phrase in target.prompts + [target.generic_prompt]
            for instance in found.get(phrase, [])
        ])
        pool = [i for i in pool if not _overlaps_any(i, claimed)]
        chosen = _select(pool, target)
        roles[target.name] = [
            Instance(target.name, i.mask, i.bbox) for i in chosen
        ]
        claimed.extend(chosen)
    return roles


def anchor_frame(source: MaskSource, image: str | Path,
                 anchor) -> tuple[int, int, int, int] | None:
    """The bounding box of the object a rule measures against.

    A range-based rule constrains where something sits *within* something
    else — how high the juice comes up the bottle, not how high it comes
    up the photograph. Measuring in the anchor's frame is what lets a
    grid fitted on the references still mean the same thing on an image
    where the whole object sits a centimetre to the left.

    Returns None when the rule has no anchor, or when the anchor is not
    found in this image; the caller then falls back to the image itself.
    """
    if anchor is None:
        return None
    found = source.masks(image, list(anchor.prompts))
    boxes = [i.bbox for instances in found.values() for i in instances]
    if not boxes:
        return None
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def _select(pool: list[Instance], target: Target) -> list[Instance]:
    """Which of the phrase's instances this target's name refers to.

    Only a positional name selects: `left_pin` means the leftmost of the
    pins, and `expected_count` says how many of them count as "the left
    ones". Everything the phrase found is otherwise kept.

    In particular a plain role never truncates to `expected_count`. That
    number is what a *normal* image contains, and a query image with one
    object too many is precisely the anomaly being looked for — trimming
    the surplus would delete the evidence before the model ever sees it.
    """
    name  = target.name.lower()
    count = target.expected_count or 1

    for prefix, axis, take_first in (("left_", 0, True), ("right_", 0, False),
                                     ("top_", 1, True),  ("bottom_", 1, False)):
        if name.startswith(prefix):
            pool = sorted(pool, key=lambda i: i.centroid[axis])
            return pool[:count] if take_first else pool[-count:]

    return pool


def deduplicate(instances: list[Instance]) -> list[Instance]:
    """Collapse detections of the same object, keeping the larger."""
    kept: list[Instance] = []
    for instance in sorted(instances, key=lambda i: -int(i.mask.sum())):
        if not _overlaps_any(instance, kept):
            kept.append(instance)
    return kept


def _overlaps_any(instance: Instance, others: list[Instance]) -> bool:
    return any(iou(instance.mask, other.mask) > DUPLICATE_IOU for other in others)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0
