"""
The tests.

    python -m pytest tests -q

They need no API key, no GPU and no dataset. What they check is the part
of CASCADE that is deterministic — which is, by design, almost all of it:
a model chooses a strategy from a closed vocabulary, and everything after
that is compiled and executed in code. So the strategies are exercised on
synthetic inputs whose right answer is known by construction.

The two properties worth defending, and the reason each has a test:

  **The question and the picture agree.** The scaffold is drawn by one
  piece of code and quoted by another. If the cell a reference falls in
  is numbered 4 on the image and quoted as 3 in the text, every answer is
  wrong for a reason no amount of prompting fixes.

  **A strategy means what it says.** `mirror` has to pair across the axis,
  `nn_match` has to leave a surplus instance unpaired. These are what the
  scaffold *shows*; if they are wrong the picture is a confident lie.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from cascade import calibrate, detect, evaluate, parse, question
from cascade.scaffold import align, engine, plan
from cascade.types import (Anchor, Comprehend, Compare, Constraint,
                           Decomposition, FreeEdge, Grid, Instance, Locate,
                           MaskingSpec, Target)


# ═════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════
def box(x0, y0, x1, y1, shape=(400, 400)) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def instance(role, x0, y0, x1, y1, shape=(400, 400)) -> Instance:
    return Instance(role, box(x0, y0, x1, y1, shape), (x0, y0, x1, y1))


# ═════════════════════════════════════════════════════════════════════
# The closed vocabularies
# ═════════════════════════════════════════════════════════════════════
def test_a_value_outside_the_vocabulary_is_refused():
    """A strategy the engine cannot execute must not survive parsing."""
    with pytest.raises(ValueError):
        MaskingSpec(binding_type="property", matching_criterion="vibes")
    with pytest.raises(ValueError):
        Comprehend(taxonomy_class="mood-based")
    with pytest.raises(ValueError):
        FreeEdge("liquid", "northwest")


def test_the_model_never_supplies_the_program():
    """Whatever the model says, the program is compiled from the strategy."""
    spec = parse._read_masking_spec({
        "matching_criterion": "mirror",
        "role_pairs":         [["left_pin", "right_pin"]],
        "primary_axis":       "horizontal",
        # a program the model made up, which must be ignored
        "matching_program":   [{"op": "drop_everything", "args": {}}],
    }, "relational")
    assert [p.op for p in spec.matching_program] == ["reflect", "nn_match"]


# ═════════════════════════════════════════════════════════════════════
# Roles
# ═════════════════════════════════════════════════════════════════════
def test_role_names_resolve_onto_what_was_actually_found():
    """Stage 1 names roles from the sentence; Stage 2 names what it sees."""
    assert plan.resolve_roles([["compartment", "pushpin"]],
                              ["compartment", "yellow_pushpin"]) \
        == [["compartment", "yellow_pushpin"]]


def test_an_ambiguous_role_name_is_dropped_not_guessed():
    """`icon` fits two targets equally. Pairing the wrong one is worse
    than pairing nothing, because the scaffold would still look right."""
    assert plan.resolve_roles([["icon", "label"]],
                              ["fruit_icon", "fruit_icon_label"]) == []


def test_a_role_is_never_paired_with_itself():
    """`fruit_icon` has no target of its own once the icon was found as
    `banana_icon`, and it would otherwise resolve to the label its
    partner already is."""
    assert plan.resolve_roles([["fruit_icon", "fruit_icon_label"]],
                              ["banana_icon", "fruit_icon_label"]) == []


def test_a_surplus_object_survives_role_assignment():
    """The extra pushpin *is* the anomaly. Trimming the pool to the count
    a normal image has would delete the evidence before it is drawn."""
    target = Target(name="pushpin", prompts=["pushpin"], expected_count=15)
    pool   = [instance("pushpin", i * 10, 0, i * 10 + 8, 8) for i in range(16)]
    assert len(detect._select(pool, target)) == 16


def test_a_positional_name_does_select():
    """`left_connector` means the leftmost one, and there is one of it."""
    target = Target(name="left_connector", prompts=["connector"],
                    expected_count=1)
    pool   = [instance("connector", 300, 0, 340, 40),
              instance("connector", 10, 0, 50, 40)]
    chosen = detect._select(pool, target)
    assert [i.bbox[0] for i in chosen] == [10]


# ═════════════════════════════════════════════════════════════════════
# The matching strategies
# ═════════════════════════════════════════════════════════════════════
def test_nn_match_leaves_the_surplus_instance_unpaired():
    """16 pushpins into 15 compartments: one has to end up alone, and
    being alone is what the picture has to show."""
    roles = {
        "compartment": [instance("compartment", i * 60, 0, i * 60 + 50, 50)
                        for i in range(15)],
        "pushpin":     [instance("pushpin", i * 60 + 10, 10, i * 60 + 30, 30)
                        for i in range(15)]
                       + [instance("pushpin", 100, 15, 120, 35)],
    }
    spec = MaskingSpec(binding_type="relational", matching_criterion="nn_match",
                       role_pairs=[["compartment", "pushpin"]])
    spec.matching_program = plan.build_matching_program(
        spec.matching_criterion, spec.role_pairs, None)

    pairs  = engine.execute(spec, roles)["pairs"]
    paired = {key for pair in pairs for key in pair}
    assert len(pairs) == 15, "every compartment pairs with one pushpin"
    unpaired = [k for k in (("pushpin", i) for i in range(16)) if k not in paired]
    assert len(unpaired) == 1, "the surplus pushpin is left over, and shown so"


def test_the_surplus_instance_gets_a_number_of_its_own():
    """Under match_index a number names a pair, so the leftover must not
    reuse one: the sixteenth pushpin printing #7 next to the real match 7
    turns the anomaly into a duplicate of something else."""
    roles = {
        "compartment": [instance("compartment", i * 60, 0, i * 60 + 50, 50)
                        for i in range(15)],
        "pushpin":     [instance("pushpin", i * 60 + 10, 10, i * 60 + 30, 30)
                        for i in range(15)]
                       + [instance("pushpin", 100, 15, 120, 35)],
    }
    spec = MaskingSpec(binding_type="relational", matching_criterion="nn_match",
                       role_pairs=[["compartment", "pushpin"]],
                       color_scheme="shared_per_match", id_scheme="match_index")
    spec.matching_program = plan.build_matching_program(
        spec.matching_criterion, spec.role_pairs, None)

    result = engine.execute(spec, roles)
    labels = result["labels"]
    assert len(set(labels.values())) == 16, "15 pairs plus one leftover"
    assert "#16" in labels.values()

    # and its colour is shared with nothing, as the visual cue promises
    colors = result["colors"]
    leftover = next(k for k, v in labels.items() if v == "#16")
    assert sum(1 for c in colors.values() if c == colors[leftover]) == 1


def test_mirror_pairs_across_the_axis_not_along_it():
    """Reflection is the point: leftmost pairs with rightmost."""
    roles = {
        "left_slot":  [instance("left_slot", 10, y, 40, y + 20)
                       for y in (0, 100, 200)],
        "right_slot": [instance("right_slot", 300, y, 330, y + 20)
                       for y in (0, 100, 200)],
    }
    spec = MaskingSpec(binding_type="relational", matching_criterion="mirror",
                       role_pairs=[["left_slot", "right_slot"]],
                       primary_axis="vertical")
    spec.matching_program = plan.build_matching_program(
        spec.matching_criterion, spec.role_pairs, spec.primary_axis)

    pairs = engine.execute(spec, roles)["pairs"]
    assert {(dict(p)["left_slot"], dict(p)["right_slot"]) for p in pairs} \
        == {(0, 2), (1, 1), (2, 0)}


def test_matched_instances_share_a_colour_and_an_id():
    """The whole disambiguation scaffold rests on this, and it broke once:
    colours taken from role names made every pair the same colour."""
    roles = {"a": [instance("a", i * 60, 0, i * 60 + 50, 50) for i in range(3)],
             "b": [instance("b", i * 60, 100, i * 60 + 50, 150) for i in range(3)]}
    spec = MaskingSpec(binding_type="relational",
                       matching_criterion="spatial_rank",
                       role_pairs=[["a", "b"]], color_scheme="shared_per_match",
                       id_scheme="match_index", primary_axis="horizontal")
    spec.matching_program = plan.build_matching_program(
        spec.matching_criterion, spec.role_pairs, spec.primary_axis)

    result = engine.execute(spec, roles)
    colors = result["colors"]
    for pair in result["pairs"]:
        assert len({colors[key] for key in pair}) == 1, "a pair shares one colour"
    assert len({colors[("a", i)] for i in range(3)}) == 3, "pairs must differ"


# ═════════════════════════════════════════════════════════════════════
# The grid: what is quoted is what is drawn
# ═════════════════════════════════════════════════════════════════════
def test_the_cell_quoted_is_the_cell_drawn():
    """`calibrate.cell_of` reads the grid; `align.draw_grid` draws it.
    They have to number the same tick the same way."""
    grid = Grid(origin=(7.0, 11.0), spacing=60.0)
    for value in (11.0, 40.0, 71.0, 200.0):
        row = calibrate.cell_of(value, grid.origin[1], grid.spacing)
        # the tick this cell is named after, as draw_grid would place it
        tick = grid.origin[1] + (row - 1) * grid.spacing
        assert tick <= value < tick + grid.spacing


def test_a_grid_is_fine_enough_to_separate_normal_from_deviant():
    """Three references agreeing to within 12 px must land in one cell,
    and a bottle filled 200 px lower must not."""
    groups_y = [{"edge": "top", "values": [300.0, 312.0]}]
    grid = calibrate.fit_grid(groups_y, [])
    assert grid is not None
    normal  = calibrate.cell_of(306.0, grid.origin[1], grid.spacing)
    deviant = calibrate.cell_of(506.0, grid.origin[1], grid.spacing)
    assert normal != deviant


def test_no_reference_lands_on_a_cell_the_grid_never_draws():
    """The first tick is numbered 1, so a value above it falls in cell 0 —
    a row the model cannot read and the question must never quote.

    These are BB-L2's real measurements, where the granola pile's bottom
    edge disagrees by 458 px across the three references. No grid can
    both hold that and stay readable, and the honest answer is no grid
    rather than a cell that is not drawn.
    """
    groups_y = [{"edge": "top", "values": [87.0, 101.0]},
                {"edge": "bottom", "values": [678.0, 1136.0]}]
    groups_x = [{"edge": "left", "values": [613.0, 805.0]},
                {"edge": "right", "values": [1315.0, 1410.0]}]
    assert calibrate.fit_grid(groups_y, groups_x) is None


def test_every_fitted_cell_is_one_the_grid_draws():
    """The complement: where a grid *is* fitted, no cell may be 0."""
    grid = calibrate.fit_grid(
        [{"edge": "top", "values": [587.0, 602.0]},
         {"edge": "bottom", "values": [843.0, 865.0]}],
        [{"edge": "left", "values": [47.0, 64.0]},
         {"edge": "right", "values": [316.0, 340.0]}])
    assert grid is not None
    assert min(min(bounds) for bounds in grid.normal_cells.values()) >= 1, \
        grid.normal_cells


def test_translation_is_measured_out_by_the_anchor():
    """The same object photographed three times, shifted bodily each
    time, must not read as a varying quantity."""
    masks   = [box(100 + d, 100 + d, 300 + d, 260 + d) for d in (0, 25, 50)]
    origins = [(d, d) for d in (0, 25, 50)]
    groups_y, _ = calibrate.measure_edges(masks, "horizontal", origins)
    spread = max(groups_y[0]["values"]) - min(groups_y[0]["values"])
    assert spread == 0.0, "an anchored measurement sees no shift"


def test_an_unanchored_measurement_does_see_the_shift():
    """The complement of the test above — otherwise it proves nothing."""
    masks = [box(100 + d, 100 + d, 300 + d, 260 + d) for d in (0, 25, 50)]
    groups_y, _ = calibrate.measure_edges(masks, "horizontal", (0, 0))
    assert max(groups_y[0]["values"]) - min(groups_y[0]["values"]) == 50.0


def test_contour_measures_extent_at_any_angle():
    """A rotated bar is the same bar. An image-aligned box is not."""
    canvas = np.zeros((400, 400), dtype=np.uint8)
    import cv2
    cv2.line(canvas, (100, 100), (300, 300), 1, 20)
    groups_y, groups_x = calibrate.measure_edges([canvas.astype(bool)], "contour")
    major = groups_y[0]["values"][0]
    minor = groups_x[0]["values"][0]
    assert major > minor * 3


# ═════════════════════════════════════════════════════════════════════
# The question
# ═════════════════════════════════════════════════════════════════════
def range_decomposition() -> Decomposition:
    return Decomposition(
        constraint = Constraint("JB-L5", "juice_bottle",
                                "The juice fills at least 90% of the bottle."),
        comprehend = Comprehend(taxonomy_class="range-based"),
        locate     = Locate(targets=[Target("liquid", ["pale liquid"])],
                            anchor=Anchor("bottle", ["glass bottle"])),
        compare    = Compare([FreeEdge("liquid", "top")]),
        edge_type  = "horizontal",
    )


def test_the_question_is_never_leading():
    """The model is asked whether the rule holds, phrased affirmatively.
    The inversion to `anomaly` happens in code, where it is auditable."""
    grid = Grid(origin=(0, 10), spacing=60, normal_cells={"top": [4, 4]})
    text = question.build(range_decomposition(), grid)
    for leading in ("anomal", "defect", "wrong", "violat", "abnormal"):
        assert leading not in text.lower(), f"{leading!r} leads the model"
    assert question.is_violation("No") is True
    assert question.is_violation("Yes") is False


def test_the_instructions_describe_the_ruler_that_was_drawn():
    """A grid is read; ticks are counted. Telling the model to count the
    marks on a grid describes an image it was not given."""
    grid = Grid(origin=(0, 10), spacing=60, normal_cells={"top": [4, 4]})
    on_a_grid = question.build(range_decomposition(), grid)
    assert "(row, col)" in on_a_grid and "Count the numbered tick" not in on_a_grid

    contour = range_decomposition()
    contour.edge_type = "contour"
    contour.compare   = Compare([FreeEdge("liquid", "major")])
    on_ticks = question.build(
        contour, Grid(origin=(0, 0), spacing=60, normal_cells={"major": [3, 3]}))
    assert "Count the numbered tick" in on_ticks and "(row, col)" not in on_ticks


def test_the_question_quotes_the_measured_interval():
    grid = Grid(origin=(0, 10), spacing=60, normal_cells={"top": [4, 6]})
    text = question.build(range_decomposition(), grid)
    assert "4" in text and "6" in text


# ═════════════════════════════════════════════════════════════════════
# Masks and scoring
# ═════════════════════════════════════════════════════════════════════
def test_run_length_encoding_round_trips():
    rng  = np.random.default_rng(0)
    mask = rng.random((37, 53)) > 0.7
    assert np.array_equal(
        detect.decode_rle(detect.encode_rle(mask), 37, 53), mask)


def test_an_all_true_mask_round_trips():
    """The encoding always begins with a zero run, so this is the edge case."""
    mask = np.ones((8, 5), dtype=bool)
    assert np.array_equal(detect.decode_rle(detect.encode_rle(mask), 8, 5), mask)


def test_auroc_is_the_probability_an_anomaly_outranks_a_normal():
    assert evaluate.auroc([0, 0, 1, 2], [False, False, True, True]) == 1.0
    assert evaluate.auroc([2, 2, 0, 0], [False, False, True, True]) == 0.0
    assert evaluate.auroc([1, 1, 1, 1], [False, False, True, True]) == 0.5


def test_an_unlabelled_image_is_skipped_not_guessed():
    results = [{"image_path": "somewhere/000.png", "category": "pushpins",
                "violations": [], "judgments": []}]
    assert evaluate.score(results).skipped == 1
