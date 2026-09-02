#!/usr/bin/env python3
"""
CASCADE — command line.

    python run.py constraints                 the rules, and how they are classified
    python run.py show     PP-L1              one rule's decomposition and question
    python run.py scaffold PP-L1 IMAGE        draw the scaffold  (no API key)
    python run.py check    PP-L1 IMAGE        draw it, ask, and report the verdict
    python run.py calibrate juice_bottle      ground and calibrate a category
    python run.py fit       juice_bottle      re-fit the grids only (no API key)
    python run.py test     juice_bottle DIR   every constraint over every image
    python run.py evaluate RESULTS.json       score a test run

`constraints`, `show`, `scaffold`, `fit` and `evaluate` read what is
already on disk and need no API key. `check`, `calibrate` and `test` call
a model, so they need one — see .env.example.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from cascade import config
from cascade.pipeline import Calibrated, Pipeline
from cascade.types import Constraint

CALIBRATION = config.EXAMPLES / "calibration.json"


# ═════════════════════════════════════════════════════════════════════
def load_units() -> list[dict]:
    data = json.loads((config.CONSTRAINTS / "atomic_units.json").read_text())
    return data["units"] if isinstance(data, dict) and "units" in data else data


def unit(constraint_id: str) -> dict:
    for u in load_units():
        if u["id"] == constraint_id:
            return u
    sys.exit(f"no constraint {constraint_id!r}. Try: python run.py constraints")


def load_calibration() -> dict[str, Calibrated]:
    if not CALIBRATION.exists():
        return {}
    return {cid: Calibrated.from_dict(d)
            for cid, d in json.loads(CALIBRATION.read_text()).items()}


def save_calibration(entries: dict[str, Calibrated]) -> None:
    CALIBRATION.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION.write_text(json.dumps(
        {cid: c.to_dict() for cid, c in entries.items()}, indent=2))


# ═════════════════════════════════════════════════════════════════════
def cmd_constraints(args) -> None:
    """List the rules, grouped by category."""
    calibrated = load_calibration()
    category   = None
    for u in load_units():
        if u["category"] != category:
            category = u["category"]
            print(f"\n{category}")
        marks = u.get("reference_labels", {})
        kind  = marks.get("taxonomy", "?")
        bind  = marks.get("binding") or "-"
        print(f"  {u['id']:8s} {kind:18s} {bind:11s} "
              f"{_readiness(u, calibrated):10s} {u['text']}")

    ready = sum(1 for u in load_units()
                if _readiness(u, calibrated) == "ready")
    print(f"\n{len(load_units())} constraints, {len(calibrated)} grounded, "
          f"{ready} ready to check. Cached in {CALIBRATION.name}.")
    print("A range rule is ready once its grid is fitted; run "
          "`calibrate <category>` if one is missing.")


def _readiness(unit: dict, calibrated: dict) -> str:
    """How far this rule has got: not grounded, grounded, or ready to check.

    A composition rule needs nothing beyond grounding. A range rule also
    needs a grid measured from the references, and without one it can
    only fall back to asking the rule in words — which is the thing the
    alignment scaffold exists to avoid.
    """
    entry = calibrated.get(unit["id"])
    if entry is None:
        return "-"
    if not entry.decomposition.comprehend.is_range:
        return "ready"
    return "ready" if entry.grid else "no grid"


def cmd_show(args) -> None:
    """Print one rule's decomposition, its scaffold plan, and its question."""
    from cascade import question

    u = unit(args.constraint_id)
    print(f"{u['id']}  ({u['category']})\n  {u['text']}\n")
    slots = re.findall(r"\{(\w+)\}", u["text"])
    if slots:
        print(f"  note     this rule names a variant it does not fix "
              f"({', '.join('{' + s + '}' for s in slots)}). The slot is sent "
              f"as written, and the\n           segmentation phrases below "
              f"were chosen for the reference image's variant.\n")

    entry = load_calibration().get(u["id"])
    if entry is None:
        print("Not calibrated yet — run:")
        print(f"  python run.py calibrate {u['category']}")
        return

    d = entry.decomposition
    print(f"  kind     {d.comprehend.taxonomy_class}"
          f"{' / ' + d.comprehend.binding_type if d.comprehend.binding_type else ''}")
    print(f"  targets  {', '.join(t.name for t in d.targets)}")
    if d.locate and d.locate.anchor:
        print(f"  anchor   {d.locate.anchor.name}")
    if d.comprehend.masking_spec:
        spec = d.comprehend.masking_spec
        print(f"  matching {spec.matching_criterion} -> "
              f"{' '.join(p.op for p in spec.matching_program) or '(none)'}")
        if spec.binding_type == "relational":
            pairs = " ".join("+".join(pair) for pair in spec.role_pairs)
            print(f"  pairs    {pairs or '(none resolved — the roles Stage 1 named '
                  f'did not line up with what Stage 2 found, so the scaffold '
                  f'colours by type instead)'}")
        print(f"  drawn as {spec.color_scheme} + {spec.id_scheme}")
    if entry.grid:
        print(f"  grid     spacing {entry.grid.spacing:.0f}px, "
              f"normal cells {entry.grid.normal_cells}")

    print("\n" + "-" * 68)
    print(question.build(d, entry.grid))


def cmd_scaffold(args) -> None:
    """Draw the scaffold for one rule on one image. No model call."""
    import cv2

    entry = load_calibration().get(args.constraint_id)
    if entry is None:
        sys.exit(f"{args.constraint_id} is not calibrated. "
                 f"Run: python run.py calibrate <category>")

    pipeline = Pipeline()
    picture  = pipeline.scaffold(entry, args.image)
    out = Path(args.out or config.OUT_ROOT /
               f"{args.constraint_id}_{Path(args.image).stem}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), picture)
    print(f"wrote {out}")
    for line in role_report(pipeline, entry, args.image):
        print(f"  {line}")


def role_report(pipeline, entry, image) -> list[str]:
    """What the scaffold was drawn from, and what it could not draw.

    A scaffold is only as good as the masks under it. When a role a rule
    pairs on is not found, the drawing still succeeds — it just stops
    showing the correspondence the rule is about, and nothing in the
    picture says so. Counting the roles out loud is what keeps a scaffold
    from looking more informative than it is.
    """
    from cascade.detect import assign_roles

    decomposition = entry.decomposition
    roles = assign_roles(pipeline.masks, image, decomposition.targets)
    lines = ["found  " + ", ".join(f"{role}×{len(items)}"
                                   for role, items in roles.items())]

    spec = decomposition.comprehend.masking_spec
    if spec:
        missing = sorted({role for pair in spec.role_pairs for role in pair
                          if not roles.get(role)})
        if missing:
            lines.append(f"NOT DRAWN: {', '.join(missing)} — no instance was "
                         f"found, so the correspondence this rule is about is "
                         f"not in the picture")
    return lines


def cmd_check(args) -> None:
    """Draw the scaffold, ask the question, report the verdict."""
    entry = load_calibration().get(args.constraint_id)
    if entry is None:
        sys.exit(f"{args.constraint_id} is not calibrated. "
                 f"Run: python run.py calibrate <category>")

    judgment = Pipeline().verify(entry, args.image, config.OUT_ROOT)
    print(f"scaffold  {judgment.scaffold_path}")
    print(f"answer    {judgment.answer}")
    print(f"verdict   {'VIOLATED' if judgment.violated else 'satisfied'}")
    if args.verbose:
        print("\n" + judgment.raw_response)


def cmd_calibrate(args) -> None:
    """Ground and calibrate every constraint of a category."""
    references = references_for(args.category, args.references)
    pipeline   = Pipeline()
    entries  = load_calibration()
    for u in load_units():
        if u["category"] != args.category:
            continue
        print(f"  {u['id']} ...", end=" ", flush=True)
        entry = pipeline.calibrate(
            Constraint(u["id"], u["category"], u["text"]), references)
        entries[u["id"]] = entry
        print(f"{entry.decomposition.comprehend.taxonomy_class}"
              f"{', grid ' + str(int(entry.grid.spacing)) + 'px' if entry.grid else ''}")
    save_calibration(entries)
    print(f"wrote {CALIBRATION}")


def cmd_fit(args) -> None:
    """Re-measure the normal range from the references. No model call.

    Grounding is the expensive half and it does not depend on the
    references, so a grid can be re-fitted on its own — after changing
    `m` or `w_min`, or against reference images of your own. Only
    range-based rules have a grid; the rest are unaffected.
    """
    pipeline  = Pipeline()
    entries   = load_calibration()
    reference = references_for(args.category, args.references)

    fitted = 0
    for constraint_id, entry in entries.items():
        if entry.decomposition.constraint.category != args.category:
            continue
        if not entry.decomposition.comprehend.is_range:
            continue
        entry.grid = pipeline.fit(entry.decomposition, reference)
        fitted += entry.grid is not None
        if entry.grid:
            print(f"  {constraint_id:8s} spacing {entry.grid.spacing:5.0f}px  "
                  f"normal cells {entry.grid.normal_cells}")
        else:
            print(f"  {constraint_id:8s} no grid — no spacing keeps every "
                  f"reference in one readable cell. The references disagree "
                  f"about this target by more than a grid can hold.")
        for note in ambiguous_targets(pipeline, entry, reference):
            print(f"           note: {note}")
    save_calibration(entries)
    print(f"\nfitted {fitted}, wrote {CALIBRATION}")


def ambiguous_targets(pipeline, entry, references: list) -> list[str]:
    """Warn where the rule did not determine what gets measured.

    A range is read off one region. Where the references hold several
    instances of the measured role, the largest is taken — a tie-break,
    not a decision the rule made — so it is worth saying out loud.
    """
    from cascade.detect import assign_roles

    decomposition = entry.decomposition
    role = decomposition.targets[0].name if decomposition.targets else None
    if role is None:
        return []
    counts = [len(assign_roles(pipeline.masks, image, decomposition.targets)
                  .get(role, [])) for image in references]
    if max(counts, default=0) <= 1:
        return []
    return [f"'{role}' matched {counts} instances across the references; "
            f"the largest is measured"]


def references_for(category: str, given: list | None) -> list:
    """The K normal images a category is calibrated on.

    MVTec-LOCO's own `train/good` is the default, so the same three
    images are used wherever the dataset was unpacked.
    """
    if given:
        return [Path(p) for p in given]
    folder = config.DATA_ROOT / category / "train" / "good"
    chosen = sorted(folder.glob("*.png"))[:config.K_REFERENCE]
    if len(chosen) < config.K_REFERENCE:
        sys.exit(f"need {config.K_REFERENCE} reference images in {folder} "
                 f"(found {len(chosen)}). See scripts/download_mvtec_loco.py, "
                 f"or pass them explicitly.")
    return chosen


def cmd_test(args) -> None:
    """Check every calibrated constraint of a category over a directory."""
    entries = [c for c in load_calibration().values()
               if c.decomposition.constraint.category == args.category]
    if not entries:
        sys.exit(f"nothing calibrated for {args.category}")

    images   = sorted(Path(args.images).glob("*.png"))
    pipeline = Pipeline()
    results  = []
    for image in images:
        result = pipeline.check(image, entries, config.OUT_ROOT / args.category)
        results.append(result.to_dict())
        print(f"  {image.name:32s} {result.to_dict()['prediction']:8s} "
              f"{','.join(result.violations)}")

    out = config.OUT_ROOT / f"{args.category}_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")

    from cascade import evaluate
    print("\n" + str(evaluate.score(results)))


def cmd_evaluate(args) -> None:
    """Score a results file. Reads only what is already on disk."""
    from cascade import evaluate

    results = evaluate.load(args.results)
    truth   = evaluate.load_truth(args.truth) if args.truth else None
    report  = evaluate.score(results, truth)
    print(report)
    if not args.truth:
        print("\n  (pass --truth to also score each constraint separately)")

    if args.json:
        Path(args.json).write_text(json.dumps(report.to_dict(), indent=2))
        print(f"\nwrote {args.json}")


# ═════════════════════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("constraints", help="list the rules").set_defaults(fn=cmd_constraints)

    p = sub.add_parser("show", help="one rule's decomposition and question")
    p.add_argument("constraint_id"); p.set_defaults(fn=cmd_show)

    p = sub.add_parser("scaffold", help="draw the scaffold (no API key)")
    p.add_argument("constraint_id"); p.add_argument("image")
    p.add_argument("--out"); p.set_defaults(fn=cmd_scaffold)

    p = sub.add_parser("check", help="draw, ask, and report the verdict")
    p.add_argument("constraint_id"); p.add_argument("image")
    p.add_argument("-v", "--verbose", action="store_true"); p.set_defaults(fn=cmd_check)

    p = sub.add_parser("calibrate", help="ground and calibrate a category")
    p.add_argument("category", choices=config.CATEGORIES)
    p.add_argument("references", nargs="*", help="K normal images "
                   "(default: <category>/train/good)")
    p.set_defaults(fn=cmd_calibrate)

    p = sub.add_parser("fit", help="re-fit the grids only (no API key)")
    p.add_argument("category", choices=config.CATEGORIES)
    p.add_argument("references", nargs="*", help="K normal images "
                   "(default: <category>/train/good)")
    p.set_defaults(fn=cmd_fit)

    p = sub.add_parser("test", help="every constraint over every image")
    p.add_argument("category", choices=config.CATEGORIES)
    p.add_argument("images"); p.set_defaults(fn=cmd_test)

    p = sub.add_parser("evaluate", help="score a test run (no API key)")
    p.add_argument("results", help="a results file written by `test`")
    p.add_argument("--truth", help="{image: [constraint id, ...]} for "
                                   "per-constraint scores")
    p.add_argument("--json", help="also write the report as JSON")
    p.set_defaults(fn=cmd_evaluate)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
