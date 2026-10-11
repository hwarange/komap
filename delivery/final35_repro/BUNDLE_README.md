# Final35 code and reproducibility evidence

Final approved recipe: **MiT2 original + UNet++3 base120, 50:50 soft voting**, trained on the official70 training images only. Official validation20 image-macro mIoU: **80.62075310543737%**. Train70 internal alloy-group5fold OOF, a separate supporting evaluation: **80.77829756479382%**. No test10 accuracy score is available.

This code ZIP includes the executable Python entry point, frozen source, environment YAML, complete model settings, and small verification/provenance files. **Original images, trained weights and probability arrays are excluded.** Submit the ten generated RGB masks separately in `final35_masks.zip`.

## Required external files

Extract this code ZIP into a project directory. Place the user-provided ten original test images at:

```text
data/test/images/*_image.png
```

Place the two preserved trained checkpoints at these exact paths:

```text
runs/review32_mit2_train70/models/mit2/base/last.pt
runs/submission31_train70/models/unet3/base/last.pt
```

Their SHA256 values are respectively:

```text
2739b79a9e052808d318b621fb2d1cb9ae31b43473f4266e73fb275cd199aee1
1668b0ef0e25e9be093ed531d60a2425e45f9cda694f421e023d3d09e9e253a8
```

The selected weights are backed up separately; the inference entry point never downloads them or retrains a model. Training/validation images are unnecessary for inference. The supplied audit JSON files document the training70/validation20 split and fresh ImageNet initialization.

## Run from the extracted project root

Create the recorded Linux/Python3.12.3 environment using `delivery/final35_repro/environment.yml`. This is a working-environment snapshot; a fresh installation has not been independently tested. Two idle CUDA GPUs with at least30GB free each are required by the guarded reference runner (tested on two L40S48GB GPUs).

```bash
# Read-only CPU input/source/provenance checks; creates no output directory.
python delivery/final35_repro/run_submission.py --out submissions/final35_reproduced

# Execute once. The output path must be new.
python delivery/final35_repro/run_submission.py --execute --gpus 0 1 --out submissions/final35_reproduced
```

Read `delivery/final35_repro/README.md` and `manifest.json` for the exact normalization, integer scale policy, TTA8, float16 storage/float32 averaging, unchanged postprocessing and RGB palette. Output includes `masks/`, `final35_masks.zip`, per-image predicted pixel-area fractions, and hash/format verification receipts. Predicted area fractions are not chemical composition or measured accuracy.

The archive preserves original project-relative metadata paths so the guarded preflight can check them. Historical paths appearing inside those small metadata files are provenance references, not selected models or additional runtime weight requirements. No old90-image model, cap6 policy or UNet126 continuation is used.

Final35 was completed without further training. The recorded ten masks passed original-dimension/RGB/palette checks, ZIP CRC/SHA checks, independent re-combination of all20 stored model probabilities, and local copy validation. Changing CUDA hardware/software can affect floating-point results; the recorded checks guarantee the exact reference files and recipe, not bitwise equivalence on every environment.
