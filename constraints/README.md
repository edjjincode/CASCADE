# Constraints

This directory holds the natural-language constraints CASCADE verifies, and
nothing else — no code, no derived artifacts. Two files:

| file | what | count |
|---|---|---|
| [`logicqa_bullets.json`](logicqa_bullets.json) | LogicQA's published rules, verbatim | 14 |
| [`atomic_units.json`](atomic_units.json) | the units actually evaluated | 19 |

## Where they come from

> LogicQA (Kwon et al., ACL 2025 Industry), Appendix C.2 'MVTec LOCO AD Dataset - Normality Definition for each class'

- paper: https://aclanthology.org/2025.acl-industry.29.pdf
- arXiv: https://arxiv.org/abs/2503.20252
- upstream dataset: MVTec LOCO AD (Bergmann et al., IJCV 2022, 'Beyond Dents and Scratches')

Verbatim copy of LogicQA's bulleted per-class rules. Template slots {fruit}, {color} are kept as-is; they are specialized per-image at runtime.

We do not write our own constraints. Using LogicQA's wording verbatim keeps
every vision-language method on the same constraint input.

## Splitting compound bullets

A LogicQA bullet can carry several claims at once, and a single Yes/No verdict
cannot cover claims that need different kinds of evidence. Such bullets are split
into atomic units, one verdict each:

```
JB-L1  "The juice bottle is filled with {fruit} juice and carries exactly two labels."
   -> JB-L1a  The juice bottle is filled with {fruit} juice.        (a colour property)
   -> JB-L1b  The juice bottle carries exactly two labels.          (a count)
```

**Policy.** Multi-aspect bullets split where clause-level taxonomy differs. 5 bullets split: BB-L1→kept, SC-L1→a/b, JB-L1→a/b, JB-L2→a/b/c, JB-L3→a/b. BB-L3 added from existing GT.

The split is part of the released input: it is fixed in `atomic_units.json` rather
than re-derived per run, so the constraint set is identical across runs and across
methods. Every unit carries `from_bullet`, pointing back to the bullet it came from.

JB-L4 (`The fill level is the same for each bottle.`) has no atomic unit of its
own; JB-L5 states the same fill level as an explicit range.

## Two kinds of constraint

| kind | in the paper | verified by |
|---|---|---|
| `composition-based` | **binding** | property binding — attribute bound to object independently, type-based masking |
| `range-based` | **calibration** | continuous measurable quantity verified against a range/threshold |

`binding` splits further into `property` (each instance checked on its own) and
`relational` (instances must be matched across roles first). That choice decides
how the disambiguation scaffold assigns colours and IDs — see `cascade/scaffold/`.

## The 19 units

### Breakfast Box (BB)

| id | from | kind | constraint |
|---|---|---|---|
| `BB-L1` | BB-L1 | binding / property | The breakfast box contains exactly two tangerines and one nectarine. |
| `BB-L2` | BB-L2 | **calibration** | The ratio and relative position of the cereals and the mix of banana chips and almonds on the right-hand side are fixed. |
| `BB-L3` | BB-L3 | binding / property | The right compartment contains cereal and a mix of banana chips and almonds. |

### Juice Bottle (JB)

| id | from | kind | constraint |
|---|---|---|---|
| `JB-L1a` | JB-L1 | binding / property | The juice bottle is filled with {fruit} juice. |
| `JB-L1b` | JB-L1 | binding / property | The juice bottle carries exactly two labels. |
| `JB-L2a` | JB-L2 | **calibration** | The fruit icon label is attached to the center of the bottle. |
| `JB-L2b` | JB-L2 | binding / **relational** | The {fruit} icon is positioned exactly at the center of the fruit icon label. |
| `JB-L2c` | JB-L2 | binding / property | The {fruit} icon clearly indicates the type of {fruit} juice. |
| `JB-L3a` | JB-L3 | **calibration** | The text label is attached to the lower part of the bottle. |
| `JB-L3b` | JB-L3 | binding / property | The text label reads "100% Juice". |
| `JB-L5` | JB-L5 | **calibration** | The juice fills at least 90% of the bottle, but not 100%. |

### Pushpins (PP)

| id | from | kind | constraint |
|---|---|---|---|
| `PP-L1` | PP-L1 | binding / **relational** | Each compartment of the box of pushpins contains exactly one pushpin. |

### Screw Bag (SB)

| id | from | kind | constraint |
|---|---|---|---|
| `SB-L1` | SB-L1 | binding / property | A screw bag contains exactly two washers, two nuts, one long screw, and one short screw. |
| `SB-L2` | SB-L2 | **calibration** | The long screw and the short screw are each longer than 3 times the diameter of the washer. |

### Splicing Connectors (SC)

| id | from | kind | constraint |
|---|---|---|---|
| `SC-L1a` | SC-L1 | binding / **relational** | Exactly two splicing connectors are linked by exactly one cable. |
| `SC-L1b` | SC-L1 | binding / **relational** | The two splicing connectors have the same number of cable clamps. |
| `SC-L2` | SC-L2 | binding / property | The number of clamps has a one-to-one correspondence to the {color} of the cable. |
| `SC-L3` | SC-L3 | binding / **relational** | The cable connects to the same slot index on both the left connector and the right connector. |
| `SC-L4` | SC-L4 | **calibration** | The cable length is roughly longer than the length of the splicing connector terminal block. |

**19 units — 13 binding (5 of them relational) and 6 calibration.**

The `reference_labels` in `atomic_units.json` are annotations for reading the
table above. The pipeline does not consume them: it re-derives the kind from the
constraint text at run time (`cascade/parse.py`, Stage 1), and the derived value
can differ from the annotation.

## Listing them from the CLI

```bash
python run.py constraints        # all 19, grouped by category, with the kind
                                 # Stage 1 derived and whether each is ready
python run.py show BB-L1         # one unit: its decomposition, and the exact
                                 # prompt the model is sent
```

Both read the repository and need no API key.
