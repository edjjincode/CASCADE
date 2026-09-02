"""
Scoring — turning per-image verdicts into the numbers a table reports.

CASCADE decides one rule at a time, and an image is anomalous when any
of its rules is violated. That gives two things worth scoring, and they
answer different questions:

  **Per image.** Did we call this image anomalous, and was it? This is
  the detection number, and it is what `AUROC` and `balanced accuracy`
  below report. Note that the decision is a disjunction over constraints,
  so a category with more constraints has more chances to raise a false
  alarm — precision and recall move in opposite directions as constraints
  are added, which is why both are reported and not just accuracy.

  **Per constraint.** Of the images that actually violate this rule, how
  many did we catch, and how often did we cry wolf? This is what says
  whether a particular scaffold works, and it is the number to look at
  when a rule is behind. It needs per-constraint ground truth, which not
  every dataset has; without it only the image-level block is printed.

A note on AUROC. The pipeline emits a decision, not a probability, so
there is no continuous score to sweep a threshold over. The score used
here is the *number* of violated constraints — an image failing three
rules ranks above one failing a single rule. That is an ordinal score
with as many levels as there are constraints, so its ROC curve is a short
staircase rather than a smooth arc, and its AUROC is correspondingly
coarse. It is comparable across methods scored the same way; it is not
comparable to an AUROC computed from a continuous anomaly map.

Nothing here calls a model. It reads what `run.py test` already wrote.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


# ═════════════════════════════════════════════════════════════════════
# Ground truth
# ═════════════════════════════════════════════════════════════════════
# Directory names that mark an image as anomalous, in MVTec-LOCO's own
# layout and in the flattened one. Anything else is treated as normal.
ANOMALY_DIRS = {"anomaly", "logical_anomalies", "structural_anomalies"}
NORMAL_DIRS  = {"normal", "good", "train", "validation"}


def label_of(image_path: str | Path) -> bool | None:
    """True if the image is anomalous, from where it sits on disk.

    MVTec-LOCO stores the label in the path, so no annotation file is
    needed. An image under neither kind of directory returns None and is
    left out of the scoring rather than guessed at.
    """
    parts = {p.lower() for p in Path(image_path).parts}
    if parts & ANOMALY_DIRS:
        return True
    if parts & NORMAL_DIRS:
        return False
    return None


# ═════════════════════════════════════════════════════════════════════
# Counts
# ═════════════════════════════════════════════════════════════════════
@dataclass
class Counts:
    """A confusion matrix, and everything derived from one."""
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def add(self, predicted: bool, actual: bool) -> None:
        if   predicted and actual:         self.tp += 1
        elif predicted and not actual:     self.fp += 1
        elif not predicted and actual:     self.fn += 1
        else:                              self.tn += 1

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def precision(self) -> float:
        return _ratio(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> float:
        """Also the true positive rate."""
        return _ratio(self.tp, self.tp + self.fn)

    @property
    def specificity(self) -> float:
        return _ratio(self.tn, self.tn + self.fp)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return _ratio(2 * p * r, p + r)

    @property
    def accuracy(self) -> float:
        return _ratio(self.tp + self.tn, self.n)

    @property
    def balanced_accuracy(self) -> float:
        """The mean of recall and specificity.

        Preferred over plain accuracy because the splits are not balanced
        — a detector that answers "normal" to everything still scores
        well on accuracy alone.
        """
        return (self.recall + self.specificity) / 2

    def to_dict(self) -> dict:
        return {
            "n": self.n, "tp": self.tp, "fp": self.fp,
            "fn": self.fn, "tn": self.tn,
            "precision": self.precision, "recall": self.recall,
            "specificity": self.specificity, "f1": self.f1,
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
        }


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


# ═════════════════════════════════════════════════════════════════════
# AUROC
# ═════════════════════════════════════════════════════════════════════
def auroc(scores: list[float], labels: list[bool]) -> float:
    """Area under the ROC curve, via the rank identity.

    AUROC is the probability that a random anomaly outranks a random
    normal image, so it can be computed from ranks alone — no threshold
    sweep, and ties handled exactly by averaging their ranks (which is
    what the coarse score here produces a lot of).
    """
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return float("nan")

    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        shared = (i + j) / 2 + 1          # 1-based, averaged over the tie
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1

    rank_sum = sum(r for r, label in zip(ranks, labels) if label)
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


# ═════════════════════════════════════════════════════════════════════
# The report
# ═════════════════════════════════════════════════════════════════════
@dataclass
class Report:
    """What one run of `run.py test` scored."""
    category:        str = ""
    image_level:     Counts = field(default_factory=Counts)
    auroc:           float = float("nan")
    per_constraint:  dict[str, Counts] = field(default_factory=dict)
    skipped:         int = 0          # images whose label could not be read

    def to_dict(self) -> dict:
        return {
            "category":       self.category,
            "image_level":    self.image_level.to_dict(),
            "auroc":          self.auroc,
            "per_constraint": {c: k.to_dict()
                               for c, k in sorted(self.per_constraint.items())},
            "skipped":        self.skipped,
        }

    def __str__(self) -> str:
        counts = self.image_level
        lines = [
            f"{self.category or 'all'}   {counts.n} images "
            f"({counts.tp + counts.fn} anomalous, {counts.fp + counts.tn} normal)",
            "",
            f"  image level    F1 {counts.f1:.3f}   "
            f"balanced acc {counts.balanced_accuracy:.3f}   "
            f"AUROC {self.auroc:.3f}",
            f"                 precision {counts.precision:.3f}   "
            f"recall {counts.recall:.3f}   "
            f"specificity {counts.specificity:.3f}",
            f"                 tp {counts.tp}  fp {counts.fp}  "
            f"fn {counts.fn}  tn {counts.tn}",
        ]
        if self.per_constraint:
            lines += ["", "  per constraint        F1   prec    rec"
                          "     tp   fp   fn   tn"]
            for constraint_id, k in sorted(self.per_constraint.items()):
                lines.append(
                    f"    {constraint_id:<12s} {k.f1:6.3f} {k.precision:6.3f} "
                    f"{k.recall:6.3f}  {k.tp:4d} {k.fp:4d} {k.fn:4d} {k.tn:4d}")
        if self.skipped:
            lines += ["", f"  {self.skipped} images skipped — "
                          f"no normal/anomaly directory in their path"]
        return "\n".join(lines)


def score(results: list[dict], truth: dict[str, set[str]] | None = None) -> Report:
    """Score what `run.py test` wrote.

    `results` is the list of `ImageResult` dicts. `truth`, when given,
    maps an image path to the set of constraint ids that image actually
    violates, and unlocks the per-constraint block; the image-level block
    needs no annotation beyond the directory layout.
    """
    report = Report(category=results[0]["category"] if results else "")
    scores, labels = [], []

    for result in results:
        actual = label_of(result["image_path"])
        if actual is None:
            report.skipped += 1
            continue

        violations = set(result["violations"])
        report.image_level.add(bool(violations), actual)
        scores.append(len(violations))
        labels.append(actual)

        if truth is None:
            continue
        violated_here = truth.get(result["image_path"], set())
        for judgment in result["judgments"]:
            constraint_id = judgment["constraint_id"]
            report.per_constraint.setdefault(constraint_id, Counts()).add(
                bool(judgment["violated"]), constraint_id in violated_here)

    report.auroc = auroc(scores, labels)
    return report


def load(path: str | Path) -> list[dict]:
    """Read a results file written by `run.py test`."""
    return json.loads(Path(path).read_text())


def load_truth(path: str | Path) -> dict[str, set[str]]:
    """Read per-constraint ground truth: {image path: [constraint id, ...]}."""
    return {image: set(ids)
            for image, ids in json.loads(Path(path).read_text()).items()}
