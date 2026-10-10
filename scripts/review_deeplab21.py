"""CPU-only fixed-weight DeepLabV3+ ensemble pilot, folds 0/1 only.

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

OUT = ROOT / "runs/review21_deeplab"
EXP = "x2_deeplabv3plus_r34_bw_r90"
BASE = "x2_mit_bw_r90"
UNET = "x3_unetpp_bw_e120_r90"
FOLDS = (0, 1)
EXPECTED = {"arch": "DeepLabV3Plus", "encoder": "resnet34", "width": 2560,
            "scale_multiplier": 1.0, "crop": 1024, "tile": 1024, "epochs": 60,
            "bs": 4, "accum_steps": 1, "seed": 0}
ARTIFACTS = ("per_image_rows.json", "summary.txt", "results.json")


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


def write_new_atomic(path, text):
    """Publish a fully flushed sibling temporary file without replacing a target."""
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=".review21_", delete=False) as stream:
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
    assert sha256(ROOT / "splits/folds5.json") == plan["split_sha256"]
    assert load_json(OUT / "started.json")["plan_sha256"] == sha256(path)
    required = {"src/train.py", "src/data.py", "src/infer.py", "src/review_grains15.py",
                "scripts/review_deeplab21.py"}
    assert required.issubset(plan["source_hashes"])
    sources = []
    for name, digest in plan["source_hashes"].items():
        source = (ROOT / name).resolve()
        assert source.is_relative_to(ROOT)
        assert sha256(source) == digest, f"Source changed after launch: {name}"
        sources.append(receipt(source))
    return plan, {"plan": receipt(path), "started": receipt(OUT / "started.json"),
                  "split": receipt(ROOT / "splits/folds5.json"), "frozen_sources": sources}


def validate_run(fold, names, plan):
    run = ROOT / "runs/cv" / EXP / f"fold{fold}"
    cfg = load_json(run / "args.json")
    job = next(job for job in plan["jobs"] if job["fold"] == fold)
    assert cfg == job["args"], "Actual trainer arguments differ from launch snapshot"
    assert len(job["validation_names"]) == 18 and set(job["validation_names"]) == names
    assert cfg["fold"] == fold
    assert all(cfg.get(key) == value for key, value in plan["config"].items()), "Config differs from frozen plan"
    assert not cfg.get("init") and not cfg.get("full")
    assert Path(cfg["save_oof"]).resolve() == (ROOT / "runs/oof" / EXP).resolve()
    assert Path(cfg["out"]).resolve() == run.resolve()
    assert Path(cfg["folds_file"]).resolve() == (ROOT / "splits/folds5.json").resolve()
    logs = [json.loads(line) for line in (run / "log.jsonl").read_text().splitlines() if line.strip()]
    assert [row["epoch"] for row in logs] == list(range(1, 61)), "Incomplete 60-epoch run"
    assert all(math.isfinite(row["loss"]) for row in logs)
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
    for item in all_items:
        alloy_folds.setdefault(item["alloy"], set()).add(fold_of[item["name"]])
    assert all(len(folds) == 1 for folds in alloy_folds.values()), "Alloy leakage across folds"
    items = [item for item in all_items if fold_of[item["name"]] in FOLDS]
    assert len(items) == 36
    run_receipts = [validate_run(fold, {item["name"] for item in items
                                      if fold_of[item["name"]] == fold}, plan) for fold in FOLDS]
    prior_path = ROOT / "runs/review18_crop/results.json"
    prior = load_json(prior_path)
    assert prior["folds"] == list(FOLDS) and prior["images"] == 36
    done = load_json(prior_path.parent / "all.done")
    assert done["results_sha256"] == sha256(prior_path)
    historical = {(row["name"], row["source"]): row["sha256"] for row in prior["source_files"]}
    provenance["prior_reference"] = [receipt(prior_path), receipt(prior_path.parent / "all.done")]
    rows = {label: {} for label in ("sub6", "deeplab_single", "deeplab_equal_three")}
    sources = []
    for index, item in enumerate(items, 1):
        name = item["name"]
        probabilities = []
        for source, experiment in (("x2", BASE), ("unet", UNET), ("deeplab", EXP)):
            p, record = probability(ROOT / "runs/oof" / experiment / f"{name}.npy", item,
                                    historical[(name, source)] if source != "deeplab" else None)
            probabilities.append(p)
            sources.append({"name": name, "source": source, **record})
        pb, pu, pn = probabilities
        for label, p in (("sub6", (pb + pu) / 2), ("deeplab_single", pn),
                         ("deeplab_equal_three", (pb + pu + pn) / 3)):
            rows[label][name] = {"tta_pp": image_iou(item["lab"], postprocess(p)).tolist(),
                                 "low": bool(item["low_contrast"])}
        for suffix, directory in (("image", "images"), ("mask", "masks")):
            sources.append({"name": name, "source": suffix,
                            **receipt(ROOT / "data" / item["split"] / directory / f"{name}_{suffix}.png")})
        if index % 6 == 0:
            print(f"scored {index}/36 images; CPU only", flush=True)
    comparisons = {"equal_three_vs_sub6": metrics(rows["deeplab_equal_three"], rows["sub6"], fold_of, repeats=10000),
                   "deeplab_single_vs_sub6": metrics(rows["deeplab_single"], rows["sub6"], fold_of, repeats=10000)}
    scores = {label: float(np.nanmean(np.asarray([row["tta_pp"] for row in values.values()]), axis=1).mean() * 100)
              for label, values in rows.items()}
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "folds": list(FOLDS),
              "images": len(items), "dataset_pool_images": 90, "training_seed": 0,
              "scores": scores, "comparisons": comparisons,
              "primary_comparison": "equal_three_vs_sub6", "class_order": CLASSES,
              "fixed_recipe": {"sub6": "(x2 MiT + x3 UNet++) / 2",
                               "candidate": "(x2 MiT + x3 UNet++ + x2 DeepLabV3+) / 3",
                               "threshold_or_weight_search": False},
              "run_receipts": run_receipts, "provenance": {**provenance, "source_files": sources},
              "caveats": [
                  "Exploratory pilot36 from folds0/1 of a repeatedly examined 90-image dataset; not an independent test or full90 result",
                  "Each image is evaluated out of fold and all images from an alloy remain together; only training seed0 is evaluated",
                  "Historical sub6 references are not retrained: MiT60 epochs and UNet++120 epochs; DeepLabV3+60 epochs uses its last checkpoint",
                  "Primary objective is ensemble complementarity, not the highest DeepLab single-model score; no architecture-only causal claim",
                  "Grouped bootstrap resamples alloy groups within each fold10000 times with seed0; training-seed and repeated-selection uncertainty are not included",
                  "mIoU averages per-image class means; classes absent from both prediction and truth are excluded within an image",
                  "Probabilities are saved TTA OOF arrays and existing postprocessing is applied after each fixed blend; no real ROI reinference",
                  "CPU review elapsed time excludes GPU training/inference and is not deployment latency; no automatic remaining-fold extension or submission replacement",
              ],
              "elapsed_seconds": time.monotonic() - started, "cpu_threads": 2, "opencv_threads": 1,
              "training_started_by_review": False, "gpu_inference_started": False,
              "actual_roi_inference": False, "further_training_scheduled": False,
              "automatic_model_adoption": False, "submission_changed": False}
    lines = [f"Created UTC: {result['created_utc']}", "Exploratory DeepLabV3+ pilot: folds0/1, 36 images, seed0, 60 epochs."]
    for label, score in scores.items():
        lines.append(f"{label}: mIoU={score:.6f}")
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
