# Prompts

Every call CASCADE makes to a language model is in this directory, verbatim.
There are five, and no others: three read the constraint once per category,
and one of the last two asks the question once per constraint per image.

Each file opens with a header comment recording what goes in, which model
sees it, what comes out, and the section of the paper it corresponds to. The
prompt itself follows, exactly as the model receives it — fields in braces
are substituted before the call.

| File | Model | Runs | Output |
|---|---|---|---|
| `stage1_comprehend.txt` | LLM, text only | once per constraint | `taxonomy_class`, `binding_type`, `masking_spec` |
| `stage2_locate.txt` | VLM, 1 reference image | once per constraint | `targets[]`, `anchor` |
| `stage3_compare.txt` | VLM, K reference images | range-based only | `free_edges[]` |
| `question_composition.txt` | VLM, 1 scaffold | per constraint, per query image | Yes / No |
| `question_range.txt` | VLM, 1 scaffold | per constraint, per query image | YES / NO |

## Where the cost goes

Stages 1–3 run once per constraint and their output is cached to disk, so a
query image costs exactly one call per constraint — the verification call.
Adding a reference image does not add a verification call; it only sharpens
the calibration the question is written against.

## What the model is allowed to decide

Stages 1–3 choose from closed vocabularies. The parser rejects anything else,
so a stage cannot invent a category, an operator, or an axis:

```
taxonomy_class   range-based | composition-based
binding_type     property | relational | (none)
matching_criterion   none | spatial_rank | 2d_grid | mirror | nn_match
color_scheme     unique_per_type | shared_per_match | gradient_by_rank | per_instance
id_scheme        instance_index | match_index | natural_position | role_position
primary_axis     horizontal | vertical | none
edge             top | bottom | left | right | major | minor | length | diameter
```

The model picks a strategy from these; it never composes the program that
carries the strategy out. Turning a `masking_spec` into an executable
matching program is deterministic and happens in `cascade/scaffold/`, with no
model in the loop. This is why the same constraint yields the same scaffold
structure on every run, and why no product category is named anywhere in the
code.

## What the model is not asked

It is not asked to measure the references. For a range-based constraint the
normal range is computed in code from the K references at calibration time
and written into the question as an integer range, so the verification call
reads one coordinate off the query image and checks membership. It is also
not asked whether anything looks wrong: every question states the rule in the
affirmative, and the verdict is inverted afterwards, so that a leading
question cannot bias the answer.
