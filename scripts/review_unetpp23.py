"""CPU-only fixed-recipe review of the 4x UNet++ two-fold pilot.

Read completed last-epoch runs and saved OOF maps. Never launch training,
extend folds, tune weights, or replace a submission.
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
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np
import torch

from data import CLASSES, load_split
from infer import image_iou, postprocess
from review_grains15 import metrics

OUT = ROOT / "runs/review23_unetpp4"
EXP = "x4_unetpp_c1536_bw_r90"
EXISTING = {
    "x2_mit": "x2_mit_bw_r90",
    "x3_unet60": "x3_unetpp_bw_r90",
    "x3_unet120": "x3_unetpp_bw_e120_r90",
    "x4_mit": "x4_mit_c1536_bw_r90",
}
FOLDS = (0, 1)
EXPECTED = {"arch": "UnetPlusPlus", "encoder": "resnet34", "width": 2560,
            "scale_multiplier": 2.0, "crop": 1536, "tile": 1536, "epochs": 60,
            "bs": 4, "accum_steps": 1, "seed": 0}
ARTIFACTS = ("per_image_rows.json", "summary.txt", "results.json")
RECIPES = {
    "x3_unet60_single": {"x3_unet60": 1.0},
    "x3_unet120_single": {"x3_unet120": 1.0},
    "x4_unet60_single": {"x4_unet60": 1.0},
    "sub6": {"x2_mit": 0.5, "x3_unet120": 0.5},
    "x4_mit_x3_unet120_pair": {"x4_mit": 0.5, "x3_unet120": 0.5},
    "x2_mit_x4_unet60_pair": {"x2_mit": 0.5, "x4_unet60": 0.5},
    "x4_mit_x4_unet60_pair": {"x4_mit": 0.5, "x4_unet60": 0.5},
}
COMPARISONS = {
    "x4_vs_x3_unet60": ("x4_unet60_single", "x3_unet60_single"),
    "x4_unet60_vs_x3_unet120": ("x4_unet60_single", "x3_unet120_single"),
    "x4_mit_x3_unet120_vs_sub6": ("x4_mit_x3_unet120_pair", "sub6"),
    "replace_unet_vs_sub6": ("x2_mit_x4_unet60_pair", "sub6"),
    "both4x_vs_sub6": ("x4_mit_x4_unet60_pair", "sub6"),
    "both4x_vs_x4_mit_x3_unet120": ("x4_mit_x4_unet60_pair", "x4_mit_x3_unet120_pair"),
}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def under_root(name):
    path = (ROOT / name).resolve()
    assert path.is_relative_to(ROOT), f"Path escapes project: {name}"
    return path


def receipt(path):
    path = Path(path)
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path),
            "bytes": path.stat().st_size}


def verify_hashes(records):
    assert records
    for name, digest in records.items():
        assert sha256(under_root(name)) == digest, f"Pinned input changed: {name}"


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    return value


def write_new_atomic(path, text):
    """Publish a fully flushed sibling temporary file without replacing a target."""
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=".review23_", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)  # Atomic and fails if destination already exists.
    finally:
        temporary.unlink()


def write_new_json(path, value):
    write_new_atomic(path, json.dumps(clean(value), ensure_ascii=False, indent=2,
                                      allow_nan=False) + "\n")


def immutable_plan():
    path = OUT / "plan.json"
    plan = load_json(path)
    assert plan["experiment"] == EXP and plan["folds"] == list(FOLDS)
    assert all(plan["config"].get(key) == value for key, value in EXPECTED.items())
    assert len(plan["jobs"]) == 2
    assert {job["fold"] for job in plan["jobs"]} == set(FOLDS)
    assert {job["id"] for job in plan["jobs"]} == {f"unetpp_fold{fold}" for fold in FOLDS}
    assert all(job["arm"] == "unetpp" for job in plan["jobs"])
    assert sha256(ROOT / "splits/folds5.json") == plan["split_sha256"]
    assert load_json(OUT / "started.json")["plan_sha256"] == sha256(path)
    required = {"src/train.py", "src/data.py", "src/infer.py", "src/review_grains15.py",
                "scripts/review_unetpp23.py"}
    assert required.issubset(plan["source_hashes"])
    assert len(plan["data_hashes"]) == 180
    assert len(plan["baseline_oof_hashes"]) == 144
    expected_run_paths = {f"runs/cv/{experiment}/fold{fold}/{filename}"
                          for experiment in EXISTING.values() for fold in FOLDS
                          for filename in ("args.json", "result.json")}
    assert set(plan["baseline_run_hashes"]) == expected_run_paths
    for key in ("source_hashes", "data_hashes", "baseline_oof_hashes", "baseline_run_hashes"):
        verify_hashes(plan[key])
    return plan, {"plan": receipt(path), "started": receipt(OUT / "started.json"),
                  "split": receipt(ROOT / "splits/folds5.json"),
                  "frozen_sources": [receipt(under_root(name)) for name in plan["source_hashes"]],
                  "pinned_data_files": len(plan["data_hashes"]),
                  "pinned_existing_oof_files": len(plan["baseline_oof_hashes"]),
                  "baseline_runs": [receipt(under_root(name)) for name in plan["baseline_run_hashes"]]}


def validate_run(fold, names, plan):
    run = ROOT / "runs/cv" / EXP / f"fold{fold}"
    cfg = load_json(run / "args.json")
    job = next(job for job in plan["jobs"] if job["fold"] == fold)
    assert cfg == job["args"], "Actual trainer arguments differ from launch snapshot"
    assert len(job["validation_names"]) == 18 and set(job["validation_names"]) == names
    assert cfg["fold"] == fold
    assert all(cfg.get(key) == value for key, value in plan["config"].items()), "Config differs from frozen plan"
    assert not cfg.get("init") and not cfg.get("full") and not cfg.get("max_iters")
    assert cfg["spi"] == 8
    assert Path(cfg["save_oof"]).resolve() == (ROOT / "runs/oof" / EXP).resolve()
    assert Path(cfg["out"]).resolve() == run.resolve()
    assert Path(cfg["folds_file"]).resolve() == (ROOT / "splits/folds5.json").resolve()
    logs = [json.loads(line) for line in (run / "log.jsonl").read_text().splitlines() if line.strip()]
    assert [row["epoch"] for row in logs] == list(range(1, 61)), "Incomplete 60-epoch run"
    assert all(math.isfinite(row["loss"]) and math.isfinite(row["train_sec"])
               and row["microbatches"] == 144 and row["optimizer_updates"] == 144
               and row["accum_steps"] == 1 and row["effective_batch"] == 4 for row in logs)
    result = load_json(run / "result.json")
    assert result["args"] == cfg
    assert len(result["rows"]) == len(names) == 18
    assert {row["name"] for row in result["rows"]} == names
    checkpoint = torch.load(run / "last.pt", map_location="cpu", weights_only=True)
    assert checkpoint["epoch"] == 60 and checkpoint["args"] == cfg
    assert checkpoint["model"]
    del checkpoint
    return {"experiment": EXP, "fold": fold, "epochs": 60,
            "training_seconds": sum(row["train_sec"] for row in logs),
            "files": [receipt(run / name) for name in ("args.json", "log.jsonl", "result.json", "last.pt")]}


def probability(path, item, expected_hash=None):
    record = receipt(path)
    if expected_hash is not None:
        assert record["sha256"] == expected_hash, f"Historical OOF changed: {path}"
    p = np.load(path, allow_pickle=False).astype(np.float32)
    assert p.shape == (4, *item["gray"].shape)
    assert np.isfinite(p).all() and p.min() >= 0 and p.max() <= 1
    assert np.max(np.abs(p.sum(0) - 1)) < 0.005
    return p, record


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    os.chdir(ROOT)
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    for filename in ARTIFACTS:
        if (OUT / filename).exists():
            raise FileExistsError(f"Refuse to overwrite completed review: {OUT / filename}")
    plan, provenance = immutable_plan()
    fold_of = load_json(ROOT / "splits/folds5.json")
    all_items = load_split("data", "train") + load_split("data", "valid")
    assert len(all_items) == len(fold_of) == 90
    assert {item["name"] for item in all_items} == set(fold_of)
    alloy_folds = {}
    data_paths = set()
    for item in all_items:
        alloy_folds.setdefault(item["alloy"], set()).add(fold_of[item["name"]])
        for suffix, directory in (("image", "images"), ("mask", "masks")):
            data_paths.add(f"data/{item['split']}/{directory}/{item['name']}_{suffix}.png")
    assert all(len(folds) == 1 for folds in alloy_folds.values()), "Alloy leakage across folds"
    assert data_paths == set(plan["data_hashes"]), "Data hashes do not cover the exact 90-image dataset"
    items = [item for item in all_items if fold_of[item["name"]] in FOLDS]
    assert len(items) == 36
    expected_oof = {f"runs/oof/{experiment}/{item['name']}.npy"
                    for experiment in EXISTING.values() for item in items}
    assert expected_oof == set(plan["baseline_oof_hashes"]), "Baseline OOF snapshot does not cover all four sources"
    run_receipts = [validate_run(fold, {item["name"] for item in items
                                      if fold_of[item["name"]] == fold}, plan) for fold in FOLDS]
    rows = {label: {} for label in RECIPES}
    sources = []
    for index, item in enumerate(items, 1):
        name = item["name"]
        probabilities = {}
        for source, experiment in {**EXISTING, "x4_unet60": EXP}.items():
            relative = f"runs/oof/{experiment}/{name}.npy"
            p, record = probability(ROOT / relative, item,
                                    plan["baseline_oof_hashes"][relative] if source in EXISTING else None)
            probabilities[source] = p
            sources.append({"name": name, "source": source, **record})
        for label, recipe in RECIPES.items():
            assert sum(recipe.values()) == 1.0
            p = sum(probabilities[source] * weight for source, weight in recipe.items())
            rows[label][name] = {"tta_pp": image_iou(item["lab"], postprocess(p)).tolist(),
                                 "low": bool(item["low_contrast"])}
        for suffix, directory in (("image", "images"), ("mask", "masks")):
            record = receipt(ROOT / "data" / item["split"] / directory / f"{name}_{suffix}.png")
            assert record["sha256"] == plan["data_hashes"][record["path"]]
            sources.append({"name": name, "source": suffix, **record})
        if index % 6 == 0:
            print(f"scored {index}/36 images; CPU only", flush=True)
    comparisons = {label: metrics(rows[candidate], rows[reference], fold_of, repeats=10000)
                   for label, (candidate, reference) in COMPARISONS.items()}
    arrays = {label: np.asarray([row["tta_pp"] for row in values.values()], dtype=float)
              for label, values in rows.items()}
    scores = {label: float(np.nanmean(values, axis=1).mean() * 100) for label, values in arrays.items()}
    class_scores = {label: dict(zip(CLASSES, (np.nanmean(values, axis=0) * 100).tolist()))
                    for label, values in arrays.items()}
    class_counts = {label: dict(zip(CLASSES, np.isfinite(values).sum(axis=0).tolist()))
                    for label, values in arrays.items()}
    # Check the frozen inputs again before publishing, including data read earlier.
    for key in ("source_hashes", "data_hashes", "baseline_oof_hashes", "baseline_run_hashes"):
        verify_hashes(plan[key])
    assert sha256(ROOT / "splits/folds5.json") == plan["split_sha256"]
    assert sha256(OUT / "plan.json") == provenance["plan"]["sha256"]
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "folds": list(FOLDS),
              "images": len(items), "dataset_pool_images": 90, "training_seed": 0,
              "scores": scores, "class_scores": class_scores, "class_image_counts": class_counts,
              "comparisons": comparisons, "primary_comparison": "x4_vs_x3_unet60",
              "primary_ensemble_comparison": "replace_unet_vs_sub6", "class_order": CLASSES,
              "fixed_recipes": RECIPES, "source_experiments": {**EXISTING, "x4_unet60": EXP},
              "threshold_or_weight_search": False,
              "run_receipts": run_receipts, "provenance": {**provenance, "source_files": sources},
              "caveats": [
                  "Exploratory pilot36 from folds0/1 of a repeatedly examined 90-image dataset; not an independent test or full90 result",
                  "Each image is evaluated out of fold and all images from an alloy remain together; only training seed0 is evaluated",
                  "Primary resolution comparison uses x3 and x4 UNet++ at 60 epochs with crop1536; the adopted x3 UNet++120 comparison also differs in training duration",
                  "Existing predictions are reused; x2 MiT60, x4 MiT crop1536 at60, x3 UNet++60 and x3 UNet++120 are separate historical runs",
                  "New x4 UNet++60 uses its last checkpoint, ImageNet initialization and unchanged crop1536; higher image scale reduces the original-image field of view per crop",
                  "All pair recipes are pixel-probability soft voting at 50:50; no weight search, probability calibration or automatic adoption",
                  "Grouped bootstrap resamples alloy groups within each fold10000 times with seed0; training-seed and repeated-selection uncertainty are not included",
                  "mIoU averages per-image class means; classes absent from both prediction and truth are excluded within an image",
                  "Probabilities are saved TTA OOF arrays and existing postprocessing is applied after each fixed blend; no real ROI reinference",
                  "CPU review elapsed time excludes GPU training/inference and is not deployment latency; no automatic remaining-fold extension or submission replacement",
              ],
              "elapsed_seconds": time.monotonic() - started, "cpu_threads": 2, "opencv_threads": 1,
              "training_started_by_review": False, "gpu_inference_started": False,
              "actual_roi_inference": False, "further_training_scheduled": False,
              "automatic_model_adoption": False, "submission_changed": False}
    lines = [f"Created UTC: {result['created_utc']}", "Exploratory 4x UNet++ pilot: folds0/1, 36 images, seed0, 60 epochs."]
    for label, score in scores.items():
        lines.append(f"{label}: mIoU={score:.6f}; classes={class_scores[label]}")
    for label, value in comparisons.items():
        lines.append(f"{label}: delta={value['delta']:+.6f} pp; CI95={value['group_bootstrap_ci95']}")
        lines.append(f"  folds={value['delta_per_fold']}; classes={value['delta_per_class']}; improved={value['improved_images']}/36")
    lines.extend(result["caveats"])
    write_new_json(OUT / "per_image_rows.json", {"folds": list(FOLDS),
                   "fold_of": {item["name"]: fold_of[item["name"]] for item in items}, "rows": rows})
    write_new_atomic(OUT / "summary.txt", "\n".join(lines) + "\n")
    write_new_json(OUT / "results.json", result)  # Publication marker, written last.
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
