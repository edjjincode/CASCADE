# Examples

`calibration.json` holds the cached grounding for all 19 constraints —
what the three stages made of each rule, and, for a range rule, the grid
fitted on its references. It is what `show` prints and what `check` and
`test` reuse, so neither has to re-ground a rule that was grounded once.

```bash
python run.py constraints           # all 19, and how far each has got
python run.py show PP-L1            # the decomposition and the exact question
```

Grounding is cached because it costs three model calls per rule and its
answer does not depend on the query image. Re-running `calibrate` for a
category overwrites that category's entries in place.

`show` needs nothing at all. To redraw a scaffold you also need the
images, which are MVTec-LOCO's and are not redistributed here:

```bash
python scripts/download_mvtec_loco.py
python run.py scaffold PP-L1 data/pushpins/test/logical_anomalies/000.png
```

`scaffold.png` below is what that produces, included so the result can be
seen without downloading anything.

## `pushpins/` — PP-L1

> Each compartment of the box of pushpins contains exactly one pushpin.

A **composition** rule with a **relational** binding: it is not about how
many pushpins there are in total, but about the correspondence between
compartments and pushpins. Stage 1 chose `nn_match` — with no global
order to exploit, each compartment takes the nearest free pushpin — and
`shared_per_match` + `match_index`, so a matched pair shares a colour and
a number.

The query is an anomalous image: fifteen compartments, sixteen pushpins.
In `scaffold.png` every compartment is tinted and numbered, and its
pushpin carries the same colour and number — except one:

```
compartment #14   its own colour
pushpin     #14   the same colour, the same number      matched
pushpin     #16   a colour nothing else has             matched nothing
```

`#16` is the second pin sitting in `#14`'s cell. Sixteen colours are
generated for sixteen instances, so no two share one by accident — which
is what lets "a colour nothing else has" mean "matched nothing".

The greedy one-to-one matching is what makes this visible. A compartment
takes one pushpin and no more, so the surplus is left over rather than
quietly absorbed, and the model sees an unpaired object instead of having
to count to sixteen.

`scaffold.png` is what the model is actually shown. The question it is
asked, verbatim, is what `run.py show PP-L1` prints.

The three normal images it was calibrated on are
`pushpins/train/good/{000,001,002}.png`, and the query is
`pushpins/test/logical_anomalies/000.png` — MVTec-LOCO's own paths, which
are also the keys the shipped mask pack uses.

Figures derived from MVTec LOCO AD, CC BY-NC-SA 4.0 (Bergmann et al.,
IJCV 2022).
