# Calibration

Two constants govern how fine the alignment scaffold's grid may be:

| | | |
|---|---|---|
| `m` | 25 px | the distance within which a mark cannot be reliably assigned to one side of a line |
| `w_min` | 60 px | the spacing below which grid labels stop being read correctly |

They enter the fit in `cascade/calibrate.py` as the margin each reference
needs inside its cell, and as the floor on spacing:

```
w* = max(w_min, R + margins)        the finest legible, unambiguous grid
a* = the offset that keeps every reference clear of a cell boundary
```

Both are properties of the **backend**, not of any product. They are
measured once, on synthetic stimuli with no product in them at all, and
then held fixed across every constraint and every category. That is the
whole reason they can be constants: nothing about a tangerine or a cable
enters their derivation.

## Reproducing them

The responses are saved, so both reports run offline and instantly:

```bash
python calibration/probe_margin.py  --report
python calibration/probe_spacing.py --report
```

```
Δd (px)     correct   accuracy          w (px)       exact   off by 1   further
      5   127/180       70.6%               10    0/30    0.0%          0       29
     10   154/180       85.6%               25    2/30    6.7%         14       14
     15   111/120       92.5%               40   16/30   53.3%         12        2
     20   109/120       90.8%               50   26/30   86.7%          3        1
     25    60/60       100.0%  <- m         60   60/60  100.0%          0        0  <- w_min
```

Both thresholds are step transitions rather than gradual slopes, which is
why a single number is a fair summary: 90.8% → 100.0% between 20 and 25
px, and 86.7% → 100.0% between 50 and 60 px. The 95% criterion in the
paper is the conservative reading of both.

## Calibrating a different backend

```bash
python calibration/probe_margin.py  --run --model google/gemini-3.1-flash-lite
python calibration/probe_spacing.py --run --model google/gemini-3.1-flash-lite
```

Each `--run` re-queries the backend and overwrites the response file,
then prints the report. Cost is one call per trial — 900 for the margin
probe, 330 for the spacing probe — so it is worth doing once per backend
and no more. Put the resulting numbers in `CASCADE_M` and
`CASCADE_DELTA_MIN` (see `.env.example`).

## What the probes look like

**Margin.** A horizontal line and two dots, `d_A` above and `d_B` below,
positioned so neither can be judged against the other — only against the
line. *Which dot is closer?* Accuracy as a function of `|d_A − d_B|` is
how fine a distinction the backend can make.

**Spacing.** A square grid at spacing `w`, every intersection labelled
`(row, col)` with the same dot radius, font scale and label offset that
`cascade/scaffold/align.py` uses. One dot is red. *What are its
coordinates?* Exact-match accuracy as a function of `w` is how dense a
grid the backend can still read.

The second probe deliberately draws the pipeline's own grid rather than a
cleaner one, so the number it produces applies to the scaffold that is
actually shipped.

## Files

| | |
|---|---|
| `probe_margin.py` | stimulus, question, scoring, and the `m` report |
| `probe_spacing.py` | the same, for `w_min` |
| `responses/*.json` | every trial's answer, per backend |

Stimulus images are regenerated on demand and not stored; the generators
are seeded, so `--run` reproduces the same stimuli every time.
