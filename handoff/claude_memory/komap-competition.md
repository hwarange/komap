---
name: komap-competition
description: "KoMaP competition goal, metric, judging weights, data quirks, and local GPU constraint for the Komap project"
metadata:
  node_type: memory
  type: project
  originSessionId: df8d7375-2335-419b-861a-6a4d16ad2a94
  modified: 2026-10-05T21:15:17.156Z
---

KoMaP contest: 4-phase segmentation of Al-alloy optical micrographs (Al=brown 153,127,76; Al3Ni=blue 25,76,153; Primary Si=green 76,178,76; Eutectic Si="white" 204,204,204). Data in data/{train(70),valid(20),test(10, no masks)}. Metric: per-image 4-class mIoU averaged over the 10 test images; reference ~75%.

Schedule (2026 KoMaP AI 경진대회, from a linkareer listing; confirm with the official notice):
- Submission deadline 2026-10-15; preliminary results 2026-10-22; final presentations 2026-10-29 at KINTEX.
- Prizes total 13M won: grand prize 5M, plus two 2M awards (KIMS, KICET).
- This appears to be the first edition, so there are no past scores to calibrate a rank cutoff.

Rules as the user understands them (2026-10-05):
- A class absent from both GT and prediction is either excluded (3-class mean) or scored 100%. Our nan-ignore metric matches.
- External pretrained weights are allowed.
- Valid data may be used for training.
- Do NOT use the test images for pseudo-labeling or semi-supervised training.
- Submission: 4-color RGB PNG masks, same format as the provided masks.

Judging (100): task understanding 5, data-processing soundness 15, model mIoU 50, insight/use-case 10, originality/new algorithms 20.

EDA findings (2026-10-04): images grayscale saved as RGBA, 5 resolutions (675–1440 wide) with the same field of view; strong per-image brightness variation; Eutectic Si is thin and only ~7% of pixels (hardest class); Primary Si is absent in 2 images (false positives zero that class's IoU).

Deeper EDA (2026-10-04), figures in eda/figures/:
- Primary Si is absent in A57_step1_sample3 (train) and A71_step2_sample1 (valid). Both labels look correct.
- When Primary Si is present it covers at least 1.53% of the image, and components under 0.01% of image area hold ~0% of its pixels.
- Resolution ceiling: a perfect mask downsampled 1/2 and upsampled back scores only 73.2% mIoU. Work at native resolution.
- Intensity-only k-means baseline scores 39.5% mIoU.
- Primary Si and Eutectic Si almost never touch (462 of 12.6M boundary pixels).
- Within the same alloy, later steps (3–4) have more Al3Ni and less Eutectic Si (9 of 10 pairs). Within-step variance is still larger than the step effect.

Third EDA round (2026-10-04):
- Test is skewed toward low-contrast images: 6/10 have p1 > 60, versus 23/70 train and 5/20 valid. Per-image p1–p99 stretch aligns class gray levels across the two styles, so make it the default preprocessing.
- In 675-wide images, Al3Ni and Eutectic Si are about 1.9 px thick. Plan: rescale all images to a common scale, then map predictions back to the original size.
- The best per-image intensity LUT scores 65.1% mIoU (Eutectic Si 29.8%).
- Corners are about 10 gray levels darker than the center (worst about 28).
- The largest Primary Si objects are about 520 px, so use tiles of at least 512 px.
- No near-duplicate images. A51 and A74 look odd but are correct.
- The user chose a driver upgrade (LG 15G880 laptop); Colab only if heavy models are needed. Check that the chosen PyTorch build supports sm_61 (Pascal).

Baseline (2026-10-04): runs/baseline, U-Net + ResNet34, 60 epochs (~45 min on GTX 1060), FP32 only because Pascal FP16 is slow.
- Valid mIoU: 78.33 plain, 78.77 with TTA + postprocessing. Best epoch 40; plateau after that.
- Eutectic Si 58.5 (weakest), Al3Ni 74.3.
- 79% of error pixels lie within 1 px of a GT boundary.
- Main confusions: Eutectic Si predicted as Al3Ni (12%) or Al (9%).
- 675px images average 77.5. Worst image A71 (Al3Ni/Eutectic Si object confusion).
- Scale rule: 675px images ×2, all others native.

Round 1 (2026-10-05), full write-up in reports/round1_results.md:
- 17 experiments × 5-fold CV on 90 images, grouped by alloy (splits/folds5.json). Ran via src/launch.py on 4×L40S in 2h41m.
- Reference: mean of seeds 0/1 = 78.31 CV (TTA + postprocess); seed gap 0.35. Seed 0 was unlucky on A181_step4_sample4 (missed a big Primary Si block).
- Best single changes: x2_c1024 +0.75 (3.8× cost), convnext_t +0.73, unetpp +0.61 (most consistent; best Eutectic Si +1.4).
- Smaller gains: boundary +0.41, lovasz +0.38, c768 +0.48.
- No effect: scse, classw, x2 with crop 512. Clearly worse: segformer −1.52.
- 675px ×2 rule validated (+1.6 on 675px images).
- UnetPlusPlus + tu-convnext fails to build in smp 0.5.
- Revised expectation: 80–81% realistic.
- Round 2 proposed, not started: ensemble measurement, combinations, LR tuning, SiP–SiE adjacency postprocess.

Round 6 is RUNNING on the server (started 2026-10-06 06:13 via scripts/run_round6.sh; expected done ~12–13h).
- Check runs/round6_driver.log and runs/round6_done.txt.
- When done, run scripts/review_round6.sh, then build sub3. Steps are in the "★" section of handoff/HANDOFF_latest.md.

Round 6 plan (agreed 2026-10-06, details in reports/plan_round6.md):
- A1 learning curve (54/36 train imgs) decides whether to do full-90 training. A2 tests ×1.25 scale TTA.
- B adds x2 + UNet++ (B1), MicroNet r34 / se_resnext50 UNet++ with an ImageNet control (B2), and EffNetV2-S / HRNet for diversity (B3).
- C: full-90 training with 3 seeds, only if A1 shows ≥ +0.3 going from 54 to 72 images.
- D builds sub3 and replaces sub2 only if it beats it by ≥ +0.1.
- Accuracy work stops 10/7; the report is the focus from 10/8.

Rounds 4–5 and sub2 (2026-10-06), write-up in reports/round4_5_results.md and reports/idea_journey.md:
- Rot90-only aug helps UNet++ / MiT / x2 but not ConvNeXt at 120 epochs.
- Noise (−0.68 when removed) and gamma (−0.23) are the key intensity augs.
- Si component classifier (src/si_classifier.py, applied only at ≥0.8 confidence) gives +0.10.
- sub2 = 16 exps × 5 folds + Si classifier: CV 79.96, current best. Pipeline in src/submit.py.
- Failed: hires, DINOv3, skeleton loss, phase-anchored HM, lowres, copypaste, EMA, bw 5×.
- Handoff zip made for moving to another laptop. The SSH key is not included; register a new key via the Jupyter terminal.

Rounds 2–3 (2026-10-05), write-up in reports/round2_3_results.md:
- Best singles: cnx_bw_e120 79.33, unetpp_bw_e120 79.32, cnx_bw 79.19 (fast).
- Ensemble plateaus ~79.7 CV (all-12 mean 79.73, est test 80.51; nested greedy 79.69). Picks diverse models: unetpp_bw, unet_mit, x2_c1024, cnx_bw.
- Boundary weighting stacks. Nothing helped: bs16, ConvNeXt-S/B, MiT-B4, crop 768 on ConvNeXt, Lovász, lr 1e-4. Going 60→120 epochs gives only +0.1.
- Conclusion: the wall is data and boundary ambiguity, not compute.
- Handoff convention: keep a single file, handoff/HANDOFF_latest.md, with no timestamped copies (user request).

Status at end of 2026-10-04 session. Proposed experiments, none approved or started yet:
1. Boundary loss (Lovász + boundary-weighted CE).
2. Upscale all inputs ×2.
3. Stronger encoder.
4. Final 5-fold ensemble on train+valid.
The user's understanding: Eutectic Si is the main bottleneck.

Company server (set up 2026-10-05):
- Access: `ssh komap` (alias in ~/.ssh/config), user j-j15a106 at 70.12.130.104, key auth with the laptop's id_ed25519. The user's password did not work for SSH.
- Host jupyter04 (TLJH JupyterHub): 4× L40S 46GB, 128 CPU, 1TB RAM, Ubuntu 24.04, internet OK.
- A GPU monitor agent runs, and the home directory has other people's work. Only touch ~/komap.
- Project at ~/komap with a .venv built by uv: torch 2.14.1+cu126, smp 0.5.0.
- Sync code with `tar czf - src | ssh komap 'cd ~/komap && tar xzf -'` (no rsync on the laptop).
- train.py has `--amp` (BF16). Baseline epoch: 6s on L40S vs 42s on the GTX 1060.

Environment: project venv at .venv (Python 3.13, EDA packages; upgrading pip inside it broke it once on Windows, so skip the pip upgrade); no torch installed yet; GTX 1060 6GB with driver 419.72 (CUDA 10.1), too old for current PyTorch. The driver must be updated before training locally; Colab is the alternative.

**Why:** Non-model criteria are worth 50 points, so the plan and report matter as much as mIoU.
**How to apply:** Optimize per-image mIoU, especially Eutectic Si, and keep EDA and design rationale for the report. See [[user-prefers-plain-korean]].
