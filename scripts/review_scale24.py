"""CPU-only comparison after two single-holdout scale runs and review23 finish.

Reuse one existing train72/validation18 split, without repeated-fold training.
Read completed last checkpoints and saved TTA probabilities; never train,
tune weights, replace a submission, or pool the 18-image scores with pilot36.
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

OUT = ROOT / "runs/review24_single_scale"
PRIOR = ROOT / "runs/review23_unetpp4"
EXISTING = {
    "mit2": "x2_mit_bw_r90",
    "mit4": "x4_mit_c1536_bw_r90",
    "unet3": "x3_unetpp_bw_e120_r90",
    "deeplab2": "x2_deeplabv3plus_r34_bw_r90",
}
NEW = {"mit5": "x5_mit_c1536_bw_r90", "deeplab4": "x4_deeplabv3plus_r34_bw_r90"}
UNET4 = "x4_unetpp_c1536_bw_r90"
EXPECTED = {
    "mit5": dict(arch="Unet", encoder="mit_b2", width=2560, scale_multiplier=2.5,
                 crop=1536, tile=1536, epochs=60, bs=2, accum_steps=2, seed=0),
    "deeplab4": dict(arch="DeepLabV3Plus", encoder="resnet34", width=2560, scale_multiplier=2.0,
                     crop=1024, tile=1024, epochs=60, bs=4, accum_steps=1, seed=0),
    "unet4": dict(arch="UnetPlusPlus", encoder="resnet34", width=2560, scale_multiplier=2.0,
                  crop=1536, tile=1536, epochs=60, bs=4, accum_steps=1, seed=0),
}
RECIPES = {
    "sub6": {"mit2": 0.5, "unet3": 0.5},
    "mit4_unet3_pair": {"mit4": 0.5, "unet3": 0.5},
    "mit5_single": {"mit5": 1.0},
    "deeplab4_single": {"deeplab4": 1.0},
    "mit4_single": {"mit4": 1.0},
    "deeplab2_single": {"deeplab2": 1.0},
    "unet4_single": {"unet4": 1.0},
    "mit5_unet3_pair": {"mit5": 0.5, "unet3": 0.5},
    "mit4_unet4_pair": {"mit4": 0.5, "unet4": 0.5},
    "mit5_unet4_pair": {"mit5": 0.5, "unet4": 0.5},
    "mit4_unet3_deeplab4_triple": {"mit4": 1 / 3, "unet3": 1 / 3, "deeplab4": 1 / 3},
}
COMPARISONS = {
    "mit5_vs_mit4": ("mit5_single", "mit4_single"),
    "deeplab4_vs_deeplab2": ("deeplab4_single", "deeplab2_single"),
    "mit5_unet3_vs_mit4_unet3": ("mit5_unet3_pair", "mit4_unet3_pair"),
    "mit5_unet3_vs_sub6": ("mit5_unet3_pair", "sub6"),
    "mit4_unet4_vs_mit4_unet3": ("mit4_unet4_pair", "mit4_unet3_pair"),
    "mit4_unet4_vs_sub6": ("mit4_unet4_pair", "sub6"),
    "mit5_unet4_vs_mit4_unet4": ("mit5_unet4_pair", "mit4_unet4_pair"),
    "mit5_unet4_vs_mit5_unet3": ("mit5_unet4_pair", "mit5_unet3_pair"),
    "mit5_unet4_vs_sub6": ("mit5_unet4_pair", "sub6"),
    "deeplab4_triple_vs_mit4_unet3": ("mit4_unet3_deeplab4_triple", "mit4_unet3_pair"),
    "deeplab4_triple_vs_sub6": ("mit4_unet3_deeplab4_triple", "sub6"),
}
ARTIFACTS = ("per_image_rows.json", "summary.txt", "results.json")


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
    path = Path(path).resolve()
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
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=".review24_", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def write_new_json(path, value):
    write_new_atomic(path, json.dumps(clean(value), ensure_ascii=False, indent=2,
                                      allow_nan=False) + "\n")


def immutable_plan():
    path = OUT / "plan.json"
    plan = load_json(path)
    assert plan["fixed_holdout_fold"] == 0 and plan["repeated_cross_validation"] is False
    assert len(plan["jobs"]) == 2 and {job["id"] for job in plan["jobs"]} == set(NEW)
    for job in plan["jobs"]:
        assert job["arm"] == job["id"] and job["fold"] == 0
        assert job["expected_microbatches"] == (288 if job["arm"] == "mit5" else 144)
        assert job["expected_updates"] == 144
        assert all(job["args"].get(k) == v for k, v in EXPECTED[job["arm"]].items())
    assert sha256(ROOT / "splits/folds5.json") == plan["split_sha256"]
    assert load_json(OUT / "started.json")["plan_sha256"] == sha256(path)
    required = {"src/train.py", "src/data.py", "src/infer.py", "src/review_grains15.py",
                "scripts/review_scale24.py"}
    assert required.issubset(plan["source_hashes"])
    assert len(plan["data_hashes"]) == 180 and len(plan["baseline_oof_hashes"]) == 72
    run_paths = {f"runs/cv/{experiment}/fold0/{filename}" for experiment in EXISTING.values()
                 for filename in ("args.json", "result.json")}
    assert set(plan["baseline_run_hashes"]) == run_paths
    assert set(plan["prerequisite_hashes"]) == {"runs/review23_unetpp4/plan.json",
                                                "runs/review23_unetpp4/started.json"}
    for key in ("source_hashes", "data_hashes", "baseline_oof_hashes", "baseline_run_hashes",
                "prerequisite_hashes"):
        verify_hashes(plan[key])
    return plan, {"plan": receipt(path), "started": receipt(OUT / "started.json"),
                  "split": receipt(ROOT / "splits/folds5.json"),
                  "frozen_sources": [receipt(under_root(name)) for name in plan["source_hashes"]],
                  "pinned_data_files": len(plan["data_hashes"]),
                  "pinned_existing_oof_files": len(plan["baseline_oof_hashes"]),
                  "baseline_runs": [receipt(under_root(name)) for name in plan["baseline_run_hashes"]]}


def validate_run(job, experiment, expected, names):
    fold = job["fold"]
    run = ROOT / "runs/cv" / experiment / f"fold{fold}"
    cfg = load_json(run / "args.json")
    assert cfg == job["args"], "Actual trainer arguments differ from launch snapshot"
    assert len(job["validation_names"]) == 18 and set(job["validation_names"]) == names
    assert cfg["fold"] == fold and all(cfg.get(k) == v for k, v in expected.items())
    assert not cfg.get("init") and not cfg.get("full") and not cfg.get("max_iters")
    assert cfg["spi"] == 8
    assert under_root(cfg["save_oof"]) == (ROOT / "runs/oof" / experiment).resolve()
    assert under_root(cfg["out"]) == run.resolve()
    assert under_root(cfg["folds_file"]) == (ROOT / "splits/folds5.json").resolve()
    logs = [json.loads(line) for line in (run / "log.jsonl").read_text().splitlines() if line.strip()]
    assert [row["epoch"] for row in logs] == list(range(1, 61)), "Incomplete 60-epoch run"
    microbatches = 576 // cfg["bs"]
    updates = microbatches // cfg["accum_steps"]
    assert microbatches == job.get("expected_microbatches", microbatches)
    assert updates == job.get("expected_updates", updates) == 144
    assert all(math.isfinite(row["loss"]) and math.isfinite(row["train_sec"])
               and row["microbatches"] == microbatches and row["optimizer_updates"] == updates
               and row["accum_steps"] == cfg["accum_steps"] and row["effective_batch"] == 4
               for row in logs)
    result = load_json(run / "result.json")
    assert result["args"] == cfg
    assert len(result["rows"]) == len(names) == 18
    assert {row["name"] for row in result["rows"]} == names
    assert all("tta" in row and "tta_pp" in row for row in result["rows"])
    checkpoint = torch.load(run / "last.pt", map_location="cpu", weights_only=True)
    assert checkpoint["epoch"] == 60 and checkpoint["args"] == cfg and checkpoint["model"]
    del checkpoint
    return {"experiment": experiment, "fold": fold, "epochs": 60,
            "training_seconds": sum(row["train_sec"] for row in logs),
            "files": [receipt(run / name) for name in ("args.json", "log.jsonl", "result.json", "last.pt")]}


def completed_unet_dependency(plan, fold_of):
    """Require both earlier GPU jobs and their CPU review to finish successfully."""
    verify_hashes(plan["prerequisite_hashes"])
    prior_plan = load_json(PRIOR / "plan.json")
    prior_done = load_json(PRIOR / "all.done")
    prior_results = load_json(PRIOR / "results.json")
    assert prior_plan["experiment"] == UNET4 and prior_plan["folds"] == [0, 1]
    assert len(prior_plan["jobs"]) == 2 and {j["fold"] for j in prior_plan["jobs"]} == {0, 1}
    assert prior_plan["split_sha256"] == plan["split_sha256"]
    assert prior_plan["data_hashes"] == plan["data_hashes"]
    assert prior_done["plan_sha256"] == sha256(PRIOR / "plan.json")
    assert load_json(PRIOR / "started.json")["plan_sha256"] == prior_done["plan_sha256"]
    assert prior_done["results_sha256"] == sha256(PRIOR / "results.json")
    assert prior_done["outcome"] == "unetpp4_two_fold_pilot_complete"
    assert prior_done["folds"] == prior_results["folds"] == [0, 1]
    assert prior_done["images"] == prior_results["images"] == 36
    assert prior_done["epochs"] == 60 and prior_done["new_training_jobs"] == 2
    assert prior_results["source_experiments"]["x4_unet60"] == UNET4
    prior_sources = prior_results["provenance"]["source_files"]
    oof_hashes = {r["path"]: r["sha256"] for r in prior_sources if r["source"] == "x4_unet60"}
    expected_paths = {f"runs/oof/{UNET4}/{name}.npy" for name, fold in fold_of.items() if fold in (0, 1)}
    assert set(oof_hashes) == expected_paths and len(oof_hashes) == 36
    verify_hashes(oof_hashes)
    runs = [validate_run(job, UNET4, EXPECTED["unet4"],
                         {n for n, f in fold_of.items() if f == job["fold"]})
            for job in prior_plan["jobs"]]
    historical_receipts = {r["path"]: r["sha256"]
                           for run in prior_results["run_receipts"] for r in run["files"]}
    for run in runs:
        assert all(historical_receipts[r["path"]] == r["sha256"] for r in run["files"])
    return oof_hashes, {"review23": [receipt(PRIOR / name)
                                   for name in ("plan.json", "started.json", "all.done", "results.json")],
                        "completed_runs": runs, "images": 36, "folds": [0, 1],
                        "scores_36_images": prior_results["scores"],
                        "note": "Separate completed pilot36 reference; never pooled with this holdout18"}


def probability(path, item, expected_hash=None):
    record = receipt(path)
    if expected_hash is not None:
        assert record["sha256"] == expected_hash, f"Historical OOF changed: {path}"
    p = np.load(path, allow_pickle=False).astype(np.float32)
    assert p.shape == (4, *item["gray"].shape)
    assert np.isfinite(p).all() and p.min() >= 0 and p.max() <= 1
    assert np.max(np.abs(p.sum(0) - 1)) < 0.005
    return p, record


def holdout_metrics(candidate, reference, fold_of):
    result = metrics(candidate, reference, fold_of, repeats=10000)
    # The shared helper uses CV field names; these new runs are one fixed split.
    result["holdout_mIoU"] = result.pop("cv_mIoU")
    result["base_holdout_mIoU"] = result.pop("base_cv_mIoU")
    return result


def main():
    started = time.monotonic()
    argparse.ArgumentParser(description=__doc__).parse_args()
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
    alloy_folds, data_paths = {}, set()
    for item in all_items:
        alloy_folds.setdefault(item["alloy"], set()).add(fold_of[item["name"]])
        for suffix, directory in (("image", "images"), ("mask", "masks")):
            data_paths.add(f"data/{item['split']}/{directory}/{item['name']}_{suffix}.png")
    assert all(len(folds) == 1 for folds in alloy_folds.values()), "Alloy leakage across splits"
    assert data_paths == set(plan["data_hashes"]), "Dataset snapshot differs"
    items = [item for item in all_items if fold_of[item["name"]] == 0]
    assert len(items) == 18 and len(all_items) - len(items) == 72
    names = {item["name"] for item in items}
    expected_oof = {f"runs/oof/{experiment}/{name}.npy" for experiment in EXISTING.values() for name in names}
    assert expected_oof == set(plan["baseline_oof_hashes"])
    for experiment in EXISTING.values():
        run = ROOT / "runs/cv" / experiment / "fold0"
        cfg, previous = load_json(run / "args.json"), load_json(run / "result.json")
        assert cfg["fold"] == 0 and cfg["epochs"] == (120 if experiment == EXISTING["unet3"] else 60)
        assert cfg == previous["args"] and not cfg.get("full")
        assert under_root(cfg["folds_file"]) == (ROOT / "splits/folds5.json").resolve()
        assert len(previous["rows"]) == 18 and {r["name"] for r in previous["rows"]} == names
    run_receipts = [validate_run(job, NEW[job["arm"]], EXPECTED[job["arm"]], names)
                    for job in plan["jobs"]]
    unet_hashes, dependency = completed_unet_dependency(plan, fold_of)
    rows = {label: {} for label in RECIPES}
    source_experiments = {**EXISTING, **NEW, "unet4": UNET4}
    sources = []
    for index, item in enumerate(items, 1):
        name, probabilities = item["name"], {}
        for source, experiment in source_experiments.items():
            relative = f"runs/oof/{experiment}/{name}.npy"
            expected = (plan["baseline_oof_hashes"][relative] if source in EXISTING
                        else unet_hashes[relative] if source == "unet4" else None)
            p, record = probability(ROOT / relative, item, expected)
            probabilities[source] = p
            sources.append({"name": name, "source": source, **record})
        for label, recipe in RECIPES.items():
            assert math.isclose(sum(recipe.values()), 1.0)
            # Match historical equal-vote arithmetic, including equal thirds.
            components = list(recipe)
            assert len(set(recipe.values())) == 1
            p = sum(probabilities[source] for source in components) / len(components)
            rows[label][name] = {"tta_pp": image_iou(item["lab"], postprocess(p)).tolist(),
                                 "low": bool(item["low_contrast"])}
        for suffix, directory in (("image", "images"), ("mask", "masks")):
            record = receipt(ROOT / "data" / item["split"] / directory / f"{name}_{suffix}.png")
            assert record["sha256"] == plan["data_hashes"][record["path"]]
            sources.append({"name": name, "source": suffix, **record})
        if index % 6 == 0:
            print(f"scored {index}/18 fixed-holdout images; CPU only", flush=True)
    comparisons = {label: holdout_metrics(rows[candidate], rows[reference], fold_of)
                   for label, (candidate, reference) in COMPARISONS.items()}
    arrays = {label: np.asarray([row["tta_pp"] for row in values.values()], dtype=float)
              for label, values in rows.items()}
    scores = {label: float(np.nanmean(values, axis=1).mean() * 100) for label, values in arrays.items()}
    class_scores = {label: dict(zip(CLASSES, (np.nanmean(values, axis=0) * 100).tolist()))
                    for label, values in arrays.items()}
    class_counts = {label: dict(zip(CLASSES, np.isfinite(values).sum(axis=0).tolist()))
                    for label, values in arrays.items()}
    # Confirm prior recipe reproduces the same 18-image subset of review23.
    prior_rows_path = PRIOR / "per_image_rows.json"
    prior_rows = load_json(prior_rows_path)["rows"]
    for current, prior in (("sub6", "sub6"), ("mit4_unet3_pair", "x4_mit_x3_unet120_pair"),
                           ("unet4_single", "x4_unet60_single"),
                           ("mit4_unet4_pair", "x4_mit_x4_unet60_pair")):
        for name in names:
            assert np.allclose(np.asarray(rows[current][name]["tta_pp"], dtype=float),
                               np.asarray(prior_rows[prior][name]["tta_pp"], dtype=float),
                               rtol=0, atol=1e-12, equal_nan=True), "Reference recipe failed to reproduce"
    dependency["per_image_rows"] = receipt(prior_rows_path)
    for key in ("source_hashes", "data_hashes", "baseline_oof_hashes", "baseline_run_hashes",
                "prerequisite_hashes"):
        verify_hashes(plan[key])
    verify_hashes(unet_hashes)
    for record in dependency["review23"]:
        assert sha256(under_root(record["path"])) == record["sha256"]
    assert sha256(ROOT / "splits/folds5.json") == plan["split_sha256"]
    assert sha256(OUT / "plan.json") == provenance["plan"]["sha256"]
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "fixed_holdout_fold": 0,
              "repeated_cross_validation": False, "images": 18, "training_images": 72,
              "dataset_pool_images": 90, "training_seed": 0, "epochs": 60,
              "scores": scores, "class_scores": class_scores, "class_image_counts": class_counts,
              "comparisons": comparisons, "primary_comparisons": ["mit5_vs_mit4", "deeplab4_vs_deeplab2"],
              "class_order": CLASSES, "fixed_recipes": RECIPES, "source_experiments": source_experiments,
              "threshold_or_weight_search": False, "run_receipts": run_receipts,
              "prerequisite_review23": dependency,
              "provenance": {**provenance, "source_files": sources},
              "caveats": [
                  "One fixed train72/validation18 split reuses existing fold0; no repeated-fold cross-validation or new independent test",
                  "These 18 images are repeatedly examined; scores are not directly comparable to 36-image or 90-image means",
                  "Each new model excludes these 18 images from training; all images of each alloy remain together across the split",
                  "Only seed0 is trained, and new models use their last checkpoint at epoch60; no best-epoch selection",
                  "MiT5 uses crop1536 and accumulation2 to retain effective batch4; the matched MiT4 reference also uses crop1536 at60 epochs",
                  "DeepLab4 keeps crop1024 and60 epochs like DeepLab2; this tests higher image scale with less original-image context per crop",
                  "The historical UNet++3 reference uses120 epochs while UNet++4 uses60, so their comparison also changes training duration",
                  "Pairs use50:50 and the triple uses equal thirds of per-pixel TTA probabilities, followed by unchanged postprocessing",
                  "Grouped bootstrap resamples alloy groups within this one holdout10000 times with seed0; it does not measure split, seed, or repeated-selection uncertainty",
                  "mIoU averages per-image class means; classes absent from both prediction and truth are excluded within an image",
                  "Review23 is separately linked as a completed36-image pilot; this review never pools its two folds with the18-image comparison",
                  "CPU review time excludes GPU training/inference and is not deployment latency; no ROI reinference, extra training or automatic model adoption",
              ],
              "elapsed_seconds": time.monotonic() - started, "cpu_threads": 2, "opencv_threads": 1,
              "training_started_by_review": False, "gpu_inference_started": False,
              "actual_roi_inference": False, "further_training_scheduled": False,
              "automatic_model_adoption": False, "submission_changed": False}
    lines = [f"Created UTC: {result['created_utc']}",
             "Single fixed holdout: train72/validation18, existing fold0, seed0,60 epochs; no repeated CV."]
    for label, score in scores.items():
        lines.append(f"{label}: holdout mIoU={score:.6f}; classes={class_scores[label]}")
    for label, value in comparisons.items():
        lines.append(f"{label}: delta={value['delta']:+.6f} pp; CI95={value['group_bootstrap_ci95']}")
        lines.append(f"  classes={value['delta_per_class']}; improved={value['improved_images']}/18")
    lines.append("Separate completed review23 pilot36: " + json.dumps(dependency["scores_36_images"]))
    lines.extend(result["caveats"])
    write_new_json(OUT / "per_image_rows.json", {"fixed_holdout_fold": 0, "repeated_cross_validation": False,
                   "fold_of": {name: 0 for name in sorted(names)}, "rows": rows})
    write_new_atomic(OUT / "summary.txt", "\n".join(lines) + "\n")
    write_new_json(OUT / "results.json", result)
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
