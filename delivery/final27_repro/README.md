# Review27 selected-model reproduction

This versioned package preserves the previous `delivery/final_pair_repro` and
`submissions/final_pair_mit4_unet3` artifacts. It becomes active only when the
completed four-recipe, 90-image OOF ranking selects a changed recipe.

The selection rule is the highest **observed 90-image OOF mIoU** among the four
fixed combinations of original/interface-continued UNet++ and original/width675
cap6 MiT inference. These images have been used during model development; this
is not an independent external test score. Confidence intervals are reported
with the comparison, rather than used as an automatic adoption gate.

`prepare_manifest.py --prepare` is CPU-only and refuses to overwrite a manifest.
It checks the completed ranking, pins ten selected weights, source files, the
ten test inputs, and reference packages. The ranking has four eligible recipe
scores and two additional standalone-model diagnostic scores. Only the four
recipes are ranked. Completion, plan, row, evaluator-source, and checkpoint
receipt hashes bind the selected weights to the actual evaluated run. If the original recipe wins, it refuses
unnecessary regeneration. Do not prepare before review27 completes.

From the project root in the recorded Linux/Python environment:

```bash
.venv/bin/python delivery/final27_repro/prepare_manifest.py --prepare
.venv/bin/python delivery/final27_repro/run_submission.py --check
.venv/bin/python delivery/final27_repro/run_submission.py --run --gpus 0 1 2 3
```

The default output is the new directory `submissions/final27_selected`; an
existing directory is rejected. `--check` is read-only and does not initialize
CUDA. `--run` requires distinct idle GPUs, creates fresh float32 probabilities
for all ten test images and ten selected weights, checks every array, and saves
ten original-size RGB masks plus `final27_masks.zip`. It also records per-array,
per-mask, ZIP, source, and checkpoint checksums. Existing jobs are not stopped;
failure cleanup only affects this invocation's own prediction processes.

Inference uses the unchanged shared `src/data.py`, `src/infer.py`, and
`src/train.py`. MiT means U-Net with MiT-b2 encoder. Both model families use eight
dihedral TTA transforms. Five fold probabilities are averaged within each
family, followed by 50:50 probability soft voting and the original
postprocessing. The optional cap6 policy applies only to MiT inputs with
original width exactly675 pixels (`extra_scale=0.75`); all other inputs keep
their original scale. No test labels are loaded or used.

If selected, UNet++ continuation checkpoints represent **120 epochs followed by
a separate six-epoch stage**, with a new optimizer and schedule. They are not a
single uninterrupted126-epoch run. Fold0 comes from review26; folds1–4 come from
review27. All selected checkpoints are `last.pt`.

The runner explicitly enables cuDNN benchmarking and TF32 as in the OOF
validation code, and records these flags. Small numerical differences may occur
on other hardware. The included environment snapshot was inspected on the
reference server; a fresh installation has not been tested. Weights and test
images are referenced by their hashes and are not embedded in a source bundle.

This package generates reviewable local artifacts. It does not upload or submit
an entry to the competition.
