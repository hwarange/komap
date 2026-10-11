# Final35: train70 MiT2 + UNet++3

This is the final approved two-model submission. Both selected checkpoints were trained from ImageNet initialization on the official 70 training images only. The official 20 validation images were used for model selection and evaluation, never optimizer updates. Historical 90-image checkpoints, the 126-epoch continuation, and the cap6 inference policy are excluded.

The official validation20 image-macro mIoU is **80.62075310543737%**. A separate train70 internal alloy-group 5-fold study gives **80.77829756479382%** for the same recipe; those 56-image fold checkpoints are not used in this submission. Neither number is an undisclosed test10 score.

Selected checkpoints (exact SHA256 values and all training arguments are in `manifest.json`):

- `runs/review32_mit2_train70/models/mit2/base/last.pt`: U-Net decoder / MiT-B2 encoder, fixed epoch60, target width2560, scale multiplier1.
- `runs/submission31_train70/models/unet3/base/last.pt`: UNet++ / ResNet34, fixed epoch120, target width3840, scale multiplier1.

Each model uses grayscale percentile normalization, tile1536/overlap128, original resolution-adaptive integer scale, BF16 mixed precision and dihedral8 TTA. Its probabilities are restored to original size and stored as float16, matching the validation protocol. The two arrays are converted to float32 and averaged 50:50, then passed through the unchanged Primary-Si component filter. Output PNG masks use the original dimensions and four exact RGB colors: Al(153,127,76), Al3Ni(25,76,153), Primary Si(76,178,76), Eutectic Si(204,204,204).

## Reproduction

Use Linux, an NVIDIA CUDA GPU environment, and Python3.12.3. `environment.yml` is the exact working-environment snapshot; a clean installation has not been independently tested. Two idle GPUs with at least30GB free each are checked before inference. The observed reference hardware is two L40S48GB GPUs. No training or pretrained download is performed by the inference entry point.

Preserve project-relative paths for the two checkpoints, the ten `data/test/images/*_image.png` inputs, and the small provenance files named in `manifest.json`. The `source/` directory contains exact frozen copies of the original training model factory, data utilities, and inference functions; the entry point imports those copies.

From the project root:

```bash
python delivery/final35_repro/run_submission.py --out submissions/final35_reproduced
python delivery/final35_repro/run_submission.py --execute --gpus 0 1 --out submissions/final35_reproduced
```

The first command performs a read-only CPU preflight; the second runs inference into a new directory. Existing output directories are never overwritten. The runner verifies source/checkpoint/input/provenance hashes, strict checkpoint shape loading and finite tensors; it records worker completions, output hashes, RGB palette/dimension checks and ZIP CRC checks in `run_receipt.json` and `all.done`.

`masks/` contains10 PNGs and `final35_masks.zip` contains those masks alone. `prediction_area_fractions.json` contains predicted two-dimensional pixel fractions, not chemical composition or measured accuracy. `parts/` stores the20 intermediate probability arrays for auditing and is not needed in the mask ZIP.

The `prepare_manifest.py` script documents the one-time freeze; do not rerun it over a completed bundle. Use the already-frozen manifest for reproduction. Hash pinning guarantees identity of inputs, weights and code; exact floating-point output reproducibility across different CUDA hardware/software is not promised.
