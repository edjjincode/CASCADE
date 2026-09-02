"""
Configuration — every tunable value in CASCADE, in one place.

Nothing here is category-specific. The same constants apply to every
product category; CASCADE never branches on what the object is.

Paths are resolved from environment variables so that a clone works
without editing source. See `.env.example`.
"""
from __future__ import annotations

import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parent.parent

CONSTRAINTS = REPO_ROOT / "constraints"       # LogicQA bullets + atomic units
PROMPTS     = REPO_ROOT / "prompts"           # verbatim prompt texts
MASKS       = Path(os.getenv("CASCADE_MASKS", REPO_ROOT / "masks"))
EXAMPLES    = REPO_ROOT / "examples"

# MVTec-LOCO images. Not redistributed — see scripts/download_mvtec_loco.py.
DATA_ROOT   = Path(os.getenv("CASCADE_DATA", REPO_ROOT / "data"))

OUT_ROOT    = Path(os.getenv("CASCADE_OUT", REPO_ROOT / "out"))

CATEGORIES = [
    "breakfast_box",
    "juice_bottle",
    "pushpins",
    "screw_bag",
    "splicing_connectors",
]

# ── Model ────────────────────────────────────────────────────────────
# Any OpenAI-compatible endpoint works. The paper's main results use
# gpt-5-image-mini through OpenRouter; Table VIII reports three others.
MODEL      = os.getenv("CASCADE_MODEL", "openai/gpt-5-image-mini")
BASE_URL   = os.getenv("CASCADE_BASE_URL", "https://openrouter.ai/api/v1")
API_KEY    = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY") or ""

MAX_TOKENS = int(os.getenv("CASCADE_MAX_TOKENS", 16384))
# Reasoning models only; ignored by backends that do not accept it.
REASONING  = os.getenv("CASCADE_REASONING", "high")

# ── Alignment scaffold geometry ──────────────────────────────────────
# Both constants are calibrated once per backend on synthetic probes and
# then held fixed across every constraint and category. The probes, their
# raw results, and the figures are in calibration/ — no API calls needed
# to reproduce them, the responses are saved.
#
#   M            reading margin m, in pixels. A dot closer than m to a
#                grid line is not reliably assigned to a side.
#                Probe: two dots at ±d from a line, "which is closer?"
#                Accuracy crosses 95% at d = 25 px.
#
#   DELTA_MIN    grid spacing floor w_min, in pixels. Below it, cell
#                labels are too dense to read off correctly.
#                Probe: read the (row, col) of a marked dot.
#                Exact-match accuracy crosses 95% at spacing 60 px.
#
# Paper: eq. (4)  w* = max(w_min, R + 2m)
#        eq. (5)  a* = min_k r_k - m
M         = int(os.getenv("CASCADE_M", 25))
DELTA_MIN = int(os.getenv("CASCADE_DELTA_MIN", 60))

K_REFERENCE = int(os.getenv("CASCADE_K", 3))   # normal reference images

# ── Scaffold rendering ───────────────────────────────────────────────
MASK_ALPHA      = 0.45
GRID_DOT_RADIUS = 3
GRID_FONT_SCALE = 0.35
