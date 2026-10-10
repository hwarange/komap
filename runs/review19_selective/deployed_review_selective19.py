"""CPU-only evaluation of one fixed uncertainty-gated blend of saved OOF maps.

This is a proxy for selective refinement, not new ROI inference or training.
The gate never reads labels: use the x4 prediction only where sub6's top-two
class probability margin is strictly below 0.15, with fixed one-third weight.
"""
import os

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from data import load_split
from infer import image_iou, postprocess
from review_grains15 import metrics

MARGIN = 0.15
ZOOM_WEIGHT = 1 / 3
BASELINE_90 = 80.7884256007465
UNIFORM_90 = 80.90140560679338
BASE = "x2_mit_bw_r90"
UNET = "x3_unetpp_bw_e120_r90"
CONTROL = "x4_mit_c1024_acc2_bw_r90"
CANDIDATE = "x4_mit_c1536_bw_r90"
OLD = ROOT / "runs/review17_tiles"
CROP = ROOT / "runs/review18_crop"


def choose(base, zoom):
    """Return (selected probabilities, boolean gate), without consulting labels.

    Both inputs are [4, H, W] probability maps on the original image grid.
    For base=(pb+pu)/2, (2*base+zoom)/3 equals (pb+pu+zoom)/3.
    Values outside the gate are copied unchanged from base.
    """
    if base.ndim != 3 or base.shape[0] != 4 or base.shape != zoom.shape:
        raise ValueError("Expected matching [4, H, W] probability arrays")
    top_two = np.partition(base, 2, axis=0)[2:]
    gate = (top_two[1] - top_two[0]) < MARGIN
    selected = base.copy()
    selected[:, gate] = (2 * base[:, gate] + zoom[:, gate]) / 3
    return selected, gate


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def receipt(path):
    path = Path(path)
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path),
            "bytes": path.stat().st_size}


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    return value


def write_new_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(clean(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def probability(path, item, expected_hash=None):
    source = receipt(path)
    if expected_hash is not None:
        assert source["sha256"] == expected_hash, f"OOF changed after crop review: {path}"
    p = np.load(path, allow_pickle=False).astype(np.float32)
    assert p.shape == (4, *item["gray"].shape), str(path)
    assert np.isfinite(p).all() and p.min() >= 0 and p.max() <= 1, str(path)
    assert np.max(np.abs(p.sum(0) - 1)) < 0.005, str(path)
    return p, source


def old_completion(fold_of):
    done_path = OLD / "all.done"
    done = load_json(done_path)
    assert done["training"] is False and done["submission_changed"] is False
    records = [receipt(done_path)]
    seen = set()
    for fold in range(5):
        path = OLD / f"fold{fold}_result.json"
        result = load_json(path)
        expected = {name for name, value in fold_of.items() if value == fold}
        assert result["fold"] == fold and result["tile"] == 1536 and result["batch"] == 2
        assert len(expected) == 18 and set(result["rows"]) == expected
        assert not seen.intersection(expected)
        seen.update(expected)
        records.append(receipt(path))
    assert seen == set(fold_of)
    return records


def crop_completion(items, fold_of, crop_review):
    done_path = CROP / "all.done"
    done = load_json(done_path)
    assert done["outcome"] == "four_job_crop_pilot_complete"
    assert done["pilot_folds"] == [0, 1]
    assert done["further_training_scheduled"] is False and done["submission_changed"] is False
    results_path = CROP / "results.json"
    assert sha256(results_path) == done["results_sha256"], "Crop completion/result hash mismatch"
    result = load_json(results_path)
    assert result["folds"] == [0, 1] and result["images"] == 36
    assert result["training_started_by_review"] is False and result["submission_changed"] is False
    records = [receipt(done_path), receipt(results_path)]
    runs = []
    previous_runs = {(r["experiment"], r["fold"]): r for r in result["run_receipts"]}
    for fold in (0, 1):
        names = {item["name"] for item in items if fold_of[item["name"]] == fold}
        cfg_a, run_a = crop_review.validate_run(CONTROL, 1024, fold, names)
        cfg_b, run_b = crop_review.validate_run(CANDIDATE, 1536, fold, names)
        differences = {key for key in set(cfg_a) | set(cfg_b) if cfg_a.get(key) != cfg_b.get(key)}
        assert "crop" in differences and differences.issubset(crop_review.ALLOWED_CONFIG_DIFFERENCES)
        for run in (run_a, run_b):
            prior = previous_runs[(run["experiment"], fold)]
            for key in ("args_sha256", "log_sha256", "result_sha256"):
                assert run[key] == prior[key], f"Crop run changed: {run['experiment']} fold{fold} {key}"
            runs.append(run)
    source_hashes = {(row["name"], row["source"]): row["sha256"] for row in result["source_files"]}
    for item in items:
        for source in ("x2", "unet", "control", "candidate"):
            assert (item["name"], source) in source_hashes
    return records, runs, source_hashes, result["sub6_reference"]["cv_mIoU"]


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("existing", "crop"))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    output = args.out_dir.resolve()
    expected_output = (ROOT / "runs/review19_selective" / args.stage).resolve()
    if output != expected_output:
        parser.error(f"--out-dir must be runs/review19_selective/{args.stage}")
    artifacts = ("results.json", "per_image_rows.json", "summary.txt")
    for filename in artifacts:
        if (output / filename).exists():
            raise FileExistsError(f"Refuse to overwrite an existing review artifact: {output / filename}")
    # review_crop18 sets interop threads on import; do not set them twice.
    crop_review = None
    if args.stage == "crop":
        import review_crop18 as crop_review
    else:
        cv2.setNumThreads(1)
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)

    fold_path = ROOT / "splits/folds5.json"
    fold_of = load_json(fold_path)
    all_items = load_split("data", "train") + load_split("data", "valid")
    assert len(all_items) == len(fold_of) == 90
    assert {item["name"] for item in all_items} == set(fold_of)
    assert set(fold_of.values()) == set(range(5))
    # Every photo of an alloy must stay in one OOF fold.
    alloy_folds = {}
    for item in all_items:
        alloy_folds.setdefault(item["alloy"], set()).add(fold_of[item["name"]])
        assert item["lab"] is not None and item["lab"].shape == item["gray"].shape
        assert np.isin(item["lab"], np.arange(4)).all()
    assert all(len(folds) == 1 for folds in alloy_folds.values())
    selected_folds = list(range(5)) if args.stage == "existing" else [0, 1]
    items = [item for item in all_items if fold_of[item["name"]] in selected_folds]
    assert len(items) == (90 if args.stage == "existing" else 36)
    metadata = old_completion(fold_of)
    run_receipts, source_hashes = [], {}
    baseline_expected = BASELINE_90
    if args.stage == "crop":
        extra, run_receipts, source_hashes, baseline_expected = crop_completion(items, fold_of, crop_review)
        metadata.extend(extra)

    arms = {"existing_x4": OLD / "oof"} if args.stage == "existing" else {
        "matched_crop1024": ROOT / "runs/oof" / CONTROL,
        "crop1536": ROOT / "runs/oof" / CANDIDATE,
    }
    source_keys = {"existing_x4": "old", "matched_crop1024": "control", "crop1536": "candidate"}
    labels = ["sub6", *[f"{arm}_{method}" for arm in arms for method in ("uniform", "selected")]]
    rows = {label: {} for label in labels}
    sources, masks, timings, gate_rows = [], [], {}, {}
    pixels_total = gated_total = 0
    for index, item in enumerate(items, 1):
        item_started = time.monotonic()
        name = item["name"]
        mask_path = ROOT / "data" / item["split"] / "masks" / f"{name}_mask.png"
        masks.append({"name": name, "fold": fold_of[name], **receipt(mask_path)})
        probs = []
        for key, exp in (("x2", BASE), ("unet", UNET)):
            path = ROOT / "runs/oof" / exp / f"{name}.npy"
            p, record = probability(path, item, source_hashes.get((name, key)))
            probs.append(p)
            sources.append({"name": name, "source": key, "fold": fold_of[name], **record})
        pb, pu = probs
        base = (pb + pu) / 2

        def score(p):
            return {"tta_pp": image_iou(item["lab"], postprocess(p)).tolist(),
                    "low": bool(item["low_contrast"])}

        rows["sub6"][name] = score(base)
        gate_reference = None
        for arm, directory in arms.items():
            source = source_keys[arm]
            zoom, record = probability(directory / f"{name}.npy", item, source_hashes.get((name, source)))
            sources.append({"name": name, "source": source, "fold": fold_of[name], **record})
            selected, gate = choose(base, zoom)
            if gate_reference is None:
                gate_reference = gate
            else:
                assert np.array_equal(gate_reference, gate), "Gate must not depend on x4 arm"
            rows[f"{arm}_uniform"][name] = score((pb + pu + zoom) / 3)
            rows[f"{arm}_selected"][name] = score(selected)
        count = int(gate_reference.sum())
        area = int(gate_reference.size)
        pixels_total += area
        gated_total += count
        gate_rows[name] = {"fold": fold_of[name], "gated_pixels": count, "total_pixels": area,
                           "area_fraction": count / area, "low_contrast": bool(item["low_contrast"])}
        timings[name] = time.monotonic() - item_started
        if index % 6 == 0 or index == len(items):
            print(f"scored {index}/{len(items)} images; CPU only; elapsed={time.monotonic() - started:.1f}s", flush=True)

    scoring_finished = time.monotonic()
    baseline = metrics(rows["sub6"], rows["sub6"], fold_of, repeats=10000)
    assert abs(baseline["cv_mIoU"] - baseline_expected) <= 1e-5, (
        f"Baseline mismatch: {baseline['cv_mIoU']} versus {baseline_expected}")
    comparisons = {}
    for arm in arms:
        for method in ("uniform", "selected"):
            key = f"{arm}_{method}_vs_sub6"
            comparisons[key] = metrics(rows[f"{arm}_{method}"], rows["sub6"], fold_of, repeats=10000)
        comparisons[f"{arm}_selected_vs_uniform"] = metrics(
            rows[f"{arm}_selected"], rows[f"{arm}_uniform"], fold_of, repeats=10000)
    if args.stage == "crop":
        comparisons["crop1536_selected_vs_matched_crop1024_selected"] = metrics(
            rows["crop1536_selected"], rows["matched_crop1024_selected"], fold_of, repeats=10000)
    else:
        uniform = comparisons["existing_x4_uniform_vs_sub6"]["cv_mIoU"]
        assert abs(uniform - UNIFORM_90) <= 1e-5, f"Uniform reference mismatch: {uniform} versus {UNIFORM_90}"

    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": args.stage,
        "script_sha256": sha256(Path(__file__)),
        "folds": selected_folds,
        "images": len(items),
        "fixed_rule": {"base": "(x2 MiT + x3 UNet++) / 2",
                       "gate": "sub6 top1 probability minus top2 probability < 0.15",
                       "margin_threshold": MARGIN, "strict_less_than": True,
                       "inside_gate": "(x2 MiT + x3 UNet++ + selected x4 MiT) / 3",
                       "outside_gate": "sub6 unchanged", "x4_weight_inside_gate": ZOOM_WEIGHT,
                       "label_based_gate": False, "threshold_or_weight_search": False,
                       "postprocessing": "existing postprocess applied once after final blend"},
        "baseline": baseline,
        "baseline_expected_cv_mIoU": baseline_expected,
        "baseline_drift_pp": baseline["cv_mIoU"] - baseline_expected,
        "uniform_expected_cv_mIoU": UNIFORM_90 if args.stage == "existing" else None,
        "comparisons": comparisons,
        "gate_area": {"gated_pixels": gated_total, "total_pixels": pixels_total,
                      "pixel_weighted_fraction": gated_total / pixels_total,
                      "mean_image_fraction": float(np.mean([r["area_fraction"] for r in gate_rows.values()])),
                      "fraction_per_fold": {
                          str(fold): sum(r["gated_pixels"] for r in gate_rows.values() if r["fold"] == fold)
                          / sum(r["total_pixels"] for r in gate_rows.values() if r["fold"] == fold)
                          for fold in selected_folds}},
        "timing_seconds": {"elapsed_before_writing": time.monotonic() - started,
                           "input_validation_and_scoring": scoring_finished - started,
                           "metrics_and_bootstrap": time.monotonic() - scoring_finished,
                           "per_image": timings},
        "provenance": {"folds": receipt(fold_path), "masks": masks, "source_files": sources,
                       "completion_and_results": metadata, "crop_run_receipts": run_receipts},
        "caveats": [
            "Selective blending of stored full-image OOF predictions is a proxy; no region is cropped, enlarged or inferred again",
            "This evaluation cannot establish runtime or memory savings from actual selective ROI inference",
            "All 90 images have been reviewed repeatedly; this is exploratory, not an independent confirmatory test",
            "The gate uses only sub6 probabilities and is identical for both crop arms; confidently wrong pixels may be missed",
            "The gate restricts probability replacement; global Primary Si postprocessing can also change final labels outside the gate",
            "Threshold 0.15 and x4 weight one-third are fixed before scoring; no threshold or weight search is performed",
            "mIoU is the mean of per-image means across present-in-GT-or-prediction classes, preserving existing scoring and postprocessing",
            "Confidence intervals resample alloy groups within folds 10000 times with seed0; training-seed and model-selection uncertainty are not included",
            "No training, additional GPU inference, submission replacement, or expansion to new folds is performed",
        ],
        "training_started": False, "gpu_inference_started": False, "submission_changed": False,
        "cpu_threads": 2, "opencv_threads": 1,
    }
    lines = [f"Created UTC: {result['created_utc']}", f"Stage: {args.stage}; images={len(items)}; folds={selected_folds}",
             "Fixed rule: sub6 top1-top2 margin < 0.15; inside use equal-three, outside preserve sub6.",
             f"Baseline mIoU={baseline['cv_mIoU']:.6f}; expected={baseline_expected:.6f}",
             f"Selected pixel fraction={gated_total / pixels_total:.6f}; mean image fraction={result['gate_area']['mean_image_fraction']:.6f}"]
    for key, value in comparisons.items():
        lines.append(f"{key}: mIoU={value['cv_mIoU']:.6f}; delta={value['delta']:+.6f} pp; CI95={value['group_bootstrap_ci95']}")
        lines.append(f"  folds={value['delta_per_fold']}; classes={value['delta_per_class']}; low={value['delta_low_contrast']}; improved={value['improved_images']}/{value['images']}")
    lines.append(f"Elapsed before writing: {result['timing_seconds']['elapsed_before_writing']:.3f} seconds")
    lines.extend(result["caveats"])
    output.mkdir(parents=True, exist_ok=True)
    write_new_json(output / "per_image_rows.json", {
        "stage": args.stage, "folds": selected_folds,
        "fold_of": {item["name"]: fold_of[item["name"]] for item in items}, "rows": rows, "gate": gate_rows})
    with (output / "summary.txt").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    # Written last: its presence indicates all requested outputs were produced.
    write_new_json(output / "results.json", result)
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
