# Mask pack

Precomputed instance masks, so the pipeline runs without a GPU or a
segmentation model.

CASCADE's contribution is scaffolding, not segmentation. Given instance
masks for the objects a rule names, it draws the image so the rule can be
read off it; where those masks came from is a separate question. Treating
segmentation as an **input** rather than a stage keeps that boundary
where it belongs — and means a clone can run the whole pipeline from a
plain `pip install`.

## Format

One gzipped JSON file per category:

```json
{
  "version": 1,
  "category": "pushpins",
  "segmenter": "SAM3 promptable concept segmentation",
  "images": {
    "normal/pushpins/pushpins_crop_000": {
      "size": [1000, 1650],
      "objects": {
        "compartment": [{"bbox": [x0, y0, x1, y1], "rle": "0 132 18 ..."}, ...],
        "pushpin":     [...]
      }
    }
  }
}
```

- **Keys** are the image's path relative to the dataset root, without its
  extension. A bare file name would not be unique — MVTec-LOCO reuses
  `000.png` in every split.
- **`objects`** is keyed by *segmentation phrase*, not by role. The
  phrases are what Stage 2 returned for a constraint's targets; several
  targets often share one phrase, because a detector cannot see the
  distinction the rule cares about (`left_pin` and `right_pin` are both
  just "pin"). `cascade/detect.py::assign_roles` resolves phrases onto
  roles afterwards, from the target names alone.
- **`rle`** is column-major run-length encoding, starting from a run of
  zeros. Masks are large flat regions, so this is a fraction of a bitmap
  and still readable when the file is opened by hand.

## Reading it

```python
from cascade.detect import MaskPack
pack = MaskPack()
found = pack.masks("data/pushpins/test/logical_anomalies/000.png", ["pushpin"])
```

## Using a different segmenter

`MaskSource` is a protocol with one method. Implement it and pass it in:

```python
class MySegmenter:
    def masks(self, image, phrases) -> dict[str, list[Instance]]:
        ...

Pipeline(masks=MySegmenter())
```

Nothing downstream knows or cares where the masks came from.

## Licence

These masks are derived from MVTec LOCO AD and carry that dataset's
licence, **CC BY-NC-SA 4.0** — research use, with attribution, and
share-alike on derivative works. The images themselves are not
redistributed here; see `scripts/download_mvtec_loco.py`.

> Bergmann et al., "Beyond Dents and Scratches: Logical Constraints in
> Unsupervised Anomaly Detection and Localization", IJCV 2022.
