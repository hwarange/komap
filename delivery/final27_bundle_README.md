# Selected competition code and completed five-fold review

The active inference entrypoint is
`delivery/final27_repro/run_submission.py`, with its frozen `manifest.json` and
`environment.yml`. The manifest identifies the highest observed 90-image OOF
mIoU among the four fixed review27 recipes and records its exact ten weights,
image hashes, source hashes, inference policy, and software versions.

Use the recorded Linux/Python environment and restore the ten test images and
selected checkpoints at the paths listed in the manifest. From the project root:

```bash
python delivery/final27_repro/run_submission.py --check
python delivery/final27_repro/run_submission.py --run --gpus 0 1 2 3
```

The runner requires a new output directory and idle GPUs. By default it writes
`submissions/final27_selected/masks`, `final27_masks.zip`, and verification
receipts. Ten original-size RGB masks use the original four class colors. The
ten checkpoints contribute five fold probabilities per family; the two family
means are combined by 50:50 soft voting before unchanged postprocessing.

MiT refers to U-Net with the MiT-b2 encoder. Where selected, the cap6 inference
policy changes only MiT inputs whose original width is exactly675 pixels; other
inputs retain their original scale. Where selected, the UNet++ weights were
trained for120 epochs and then continued in a separate six-epoch stage with a
new optimizer and schedule. That stage is not an uninterrupted126-epoch run.
Fold0 continuation is reused from review26; folds1–4 are from review27.

`runs/review27_finetune_full/results.json` contains the four-recipe ranking,
class and fold results, remaining72-image results, and uncertainty estimates.
`per_image_rows.json` provides the underlying90-image metrics. The mean is
computed per image over its valid classes, then averaged across images. These
are development OOF results, not the unavailable hidden test score. The same90
images informed earlier model development, and the full90 comparison does not
isolate interface sampling from additional training and optimizer reset.

The prior final-pair and final-strengthening code and evidence remain in the
archive for provenance. Their earlier selections are historical; the active
release is the `final27_repro` manifest. The previous root README is preserved
under `historical/final_strengthened_README.md`. Original submissions and code
ZIPs remain unchanged outside this archive.

This source bundle includes Python programs, environment YAML, training and
evaluation settings, per-image results, and completed-run receipts. It excludes
training/test image binaries, GT focus maps, checkpoint binaries, probability
arrays, mask binaries, and other ZIP files. The ten selected masks are supplied
separately as `final27_masks.zip`; checkpoint and dataset references/hashes are
recorded so the required files can be restored.

The environment snapshot has been checked on the reference server. A fresh
environment installation has not been tested. The source bundle is validated
against completed server receipts; excluded large checkpoints and probability
arrays are not reread during local archive construction. No official competition
submission is performed by these programs.

The included archive builders are provenance utilities and additionally require
the prior code archives and separate mask ZIPs at their recorded paths. They
are not needed to run the active inference entrypoint.
