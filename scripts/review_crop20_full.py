"""CPU-only, fixed-recipe five-fold review of matched 60-epoch crop training.

Read ten completed runs; preserve the completed folds0/1 pilot and separately
report the remaining54 images. This evaluator does not train, infer with a GPU,
search thresholds or ensemble weights, or replace a competition submission.
"""
import os

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# This import alone sets torch's intra/inter-op threads and OpenCV threads.
import review_crop18 as crop_review
import review_selective19 as selective
import numpy as np

from data import CLASSES, load_split
from infer import image_iou, postprocess
from review_grains15 import metrics

OUT = ROOT / "runs/review20_crop_full"
PILOT = ROOT / "runs/review18_crop"
EXISTING = ROOT / "runs/review19_selective/existing"
FOLDS = tuple(range(5))
ARMS = {
    "control": (crop_review.CONTROL, 1024),
    "candidate": (crop_review.CANDIDATE, 1536),
}
LABELS = (
    "sub6", "old_tile1536_single", "old_tile1536_uniform",
    "matched_crop1024_single", "matched_crop1024_uniform", "matched_crop1024_selected",
    "crop1536_single", "crop1536_uniform", "crop1536_selected",
)
PAIRS = {
    "candidate_vs_control_single": ("crop1536_single", "matched_crop1024_single"),
    "candidate_uniform_vs_control_uniform": ("crop1536_uniform", "matched_crop1024_uniform"),
    "candidate_uniform_vs_sub6": ("crop1536_uniform", "sub6"),
    "candidate_uniform_vs_old_uniform": ("crop1536_uniform", "old_tile1536_uniform"),
    "candidate_selected_vs_uniform": ("crop1536_selected", "crop1536_uniform"),
    "candidate_selected_vs_sub6": ("crop1536_selected", "sub6"),
    "candidate_selected_vs_control_selected": ("crop1536_selected", "matched_crop1024_selected"),
    "control_uniform_vs_sub6": ("matched_crop1024_uniform", "sub6"),
    "control_selected_vs_sub6": ("matched_crop1024_selected", "sub6"),
    "control_selected_vs_uniform": ("matched_crop1024_selected", "matched_crop1024_uniform"),
    "candidate_single_vs_old_single": ("crop1536_single", "old_tile1536_single"),
    "control_single_vs_old_single": ("matched_crop1024_single", "old_tile1536_single"),
    "old_uniform_vs_sub6": ("old_tile1536_uniform", "sub6"),
}
PRIMARY = ("candidate_vs_control_single", "candidate_uniform_vs_control_uniform")
ARTIFACTS = ("per_image_rows.json", "summary.txt", "results.json")

load_json = crop_review.load_json
sha256 = crop_review.sha256
receipt = selective.receipt


def under_root(relative):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError(f"Source path outside project: {relative}")
    return path


def verify_record(record):
    path = under_root(record["path"])
    assert sha256(path) == record["sha256"], f"Changed recorded source: {path}"
    if "bytes" in record:
        assert path.stat().st_size == record["bytes"], f"Changed source size: {path}"
    return receipt(path)


def immutable_plan():
    """Validate the start-time snapshot without requiring the caller's all.done."""
    plan_path = OUT / "plan.json"
    plan = load_json(plan_path)
    started_path = OUT / "started.json"
    assert sha256(plan_path) == load_json(started_path)["plan_sha256"], "Run plan changed after launch"
    expected_jobs = {(arm, fold) for arm in ARMS for fold in (2, 3, 4)}
    jobs = {(job["arm"], job["fold"]): job for job in plan["jobs"]}
    assert len(plan["jobs"]) == len(jobs) == 6 and set(jobs) == expected_jobs
    required_sources = {
        "src/train.py", "src/data.py", "src/infer.py", "src/review_grains15.py",
        "scripts/run_crop18.py", "scripts/review_crop18.py",
        "scripts/review_selective19.py", "scripts/review_crop20_full.py",
    }
    assert required_sources.issubset(plan["source_sha256"])
    sources = [verify_record({"path": name, "sha256": digest})
               for name, digest in plan["source_sha256"].items()]
    assert plan["pilot_receipts"], "Missing immutable pilot receipt snapshot"
    original_pilot = [verify_record({"path": name, "sha256": digest})
                      for name, digest in plan["pilot_receipts"].items()]
    for filename, key in (("plan.json", "pilot_plan_sha256"),
                          ("results.json", "pilot_results_sha256"),
                          ("all.done", "pilot_done_sha256")):
        assert sha256(PILOT / filename) == plan[key], f"Pilot changed: {filename}"
    return plan, jobs, {
        "plan": receipt(plan_path), "started": receipt(started_path), "frozen_code": sources,
        "unchanged_pilot_start_receipts": original_pilot,
    }


def prior_reviews(fold_of):
    done_path, result_path = PILOT / "all.done", PILOT / "results.json"
    done, pilot = load_json(done_path), load_json(result_path)
    assert done["outcome"] == "four_job_crop_pilot_complete"
    assert done["pilot_folds"] == [0, 1] and done["submission_changed"] is False
    assert sha256(result_path) == done["results_sha256"]
    assert pilot["folds"] == [0, 1] and pilot["images"] == 36
    assert pilot["submission_changed"] is False
    assert pilot["training_started_by_review"] is False
    old_records = selective.old_completion(fold_of)

    existing_done_path, existing_path = EXISTING / "all.done", EXISTING / "results.json"
    existing_done, existing = load_json(existing_done_path), load_json(existing_path)
    assert existing_done["stage"] == existing["stage"] == "existing"
    assert existing_done["training_started"] is False
    assert existing_done["gpu_inference_started"] is False
    assert sha256(existing_path) == existing_done["results_sha256"]
    assert existing["folds"] == list(FOLDS) and existing["images"] == 90
    assert existing["submission_changed"] is False
    assert existing["fixed_rule"]["margin_threshold"] == selective.MARGIN == 0.15
    assert existing["fixed_rule"]["x4_weight_inside_gate"] == selective.ZOOM_WEIGHT == 1 / 3
    assert existing["fixed_rule"]["threshold_or_weight_search"] is False
    prior_source_hashes = {(r["name"], r["source"]): r["sha256"]
                           for r in existing["provenance"]["source_files"]}
    assert len(prior_source_hashes) == len(existing["provenance"]["source_files"]) == 270
    assert set(prior_source_hashes) == {(name, source) for name in fold_of
                                      for source in ("x2", "unet", "old")}
    pilot_hashes = {(r["name"], r["source"]): r["sha256"] for r in pilot["source_files"]}
    pilot_names = {name for name, fold in fold_of.items() if fold in (0, 1)}
    assert len(pilot_hashes) == len(pilot["source_files"]) == 180
    assert set(pilot_hashes) == {(name, source) for name in pilot_names
                                for source in ("x2", "unet", "old", "control", "candidate")}
    for key in set(prior_source_hashes).intersection(pilot_hashes):
        assert prior_source_hashes[key] == pilot_hashes[key], f"Reference receipts differ: {key}"
    source_hashes = {**prior_source_hashes, **pilot_hashes}
    provenance = existing["provenance"]
    records = [receipt(p) for p in (done_path, result_path, existing_done_path, existing_path)]
    records.extend(old_records)
    records.extend(verify_record(r) for r in provenance["completion_and_results"])
    fold_record = verify_record(provenance["folds"])
    assert fold_record["path"] == "splits/folds5.json"
    assert len(provenance["masks"]) == 90
    assert {r["name"] for r in provenance["masks"]} == set(fold_of)
    masks = []
    for record in provenance["masks"]:
        assert record["fold"] == fold_of[record["name"]]
        masks.append({"name": record["name"], "fold": record["fold"], **verify_record(record)})
    return pilot, source_hashes, {
        "completion_and_results": records, "folds": fold_record, "masks": masks,
    }


def completed_runs(items, fold_of, pilot, jobs):
    """Require equal configurations and unchanged completed pilot artifacts."""
    previous = {(r["experiment"], r["fold"]): r for r in pilot["run_receipts"]}
    assert len(previous) == len(pilot["run_receipts"]) == 4
    pilot_cfg, records = {}, []
    for fold in FOLDS:
        names = {item["name"] for item in items if fold_of[item["name"]] == fold}
        configs = {}
        for arm, (exp, crop) in ARMS.items():
            cfg, record = crop_review.validate_run(exp, crop, fold, names)
            assert Path(cfg["out"]).resolve() == (ROOT / "runs/cv" / exp / f"fold{fold}").resolve()
            assert cfg["folds_file"] == "splits/folds5.json" and cfg["data"] == "data"
            configs[arm] = cfg
            if fold in (0, 1):
                prior = previous[(exp, fold)]
                for key in ("args_sha256", "log_sha256", "result_sha256", "last_checkpoint_bytes"):
                    assert record[key] == prior[key], f"Pilot run changed: {exp} fold{fold} {key}"
                record["unchanged_from_pilot"] = True
                if fold == 0:
                    pilot_cfg[arm] = cfg
            else:
                assert cfg == jobs[(arm, fold)]["args"], f"Run differs from approved plan: {arm} fold{fold}"
                record["unchanged_from_pilot"] = None
                record["matches_full_review_start_plan"] = True
            reference = pilot_cfg[arm]
            differences = {key for key in set(reference) | set(cfg) if reference.get(key) != cfg.get(key)}
            assert differences.issubset({"fold", "out"}), f"Cross-fold configuration changed: {arm} {differences}"
            records.append(record)
        differences = {key for key in set(configs["control"]) | set(configs["candidate"])
                       if configs["control"].get(key) != configs["candidate"].get(key)}
        assert "crop" in differences and differences.issubset(crop_review.ALLOWED_CONFIG_DIFFERENCES)
    return records


def review_scope(rows, fold_of, folds):
    names = {name for name, fold in fold_of.items() if fold in folds}
    subset = {label: {name: values[name] for name in sorted(names)} for label, values in rows.items()}
    baseline = metrics(subset["sub6"], subset["sub6"], fold_of, repeats=10000)
    comparisons = {key: metrics(subset[left], subset[right], fold_of, repeats=10000)
                   for key, (left, right) in PAIRS.items()}
    arm_metrics = {}
    for label, values in subset.items():
        ordered = sorted(values)
        class_iou = np.asarray([values[name]["tta_pp"] for name in ordered], dtype=float) * 100
        image_miou = np.nanmean(class_iou, axis=1)
        low = np.asarray([values[name]["low"] for name in ordered], dtype=bool)
        groups = np.asarray([fold_of[name] for name in ordered])
        arm_metrics[label] = {
            "cv_mIoU": float(image_miou.mean()),
            "mIoU_per_fold": {str(fold): float(image_miou[groups == fold].mean()) for fold in folds},
            "IoU_per_class": dict(zip(CLASSES, np.nanmean(class_iou, axis=0).tolist())),
            "low_contrast_mIoU": float(image_miou[low].mean()) if low.any() else None,
            "low_contrast_images": int(low.sum()),
        }
    return {"folds": list(folds), "images": len(names), "baseline": baseline,
            "scores": {label: value["cv_mIoU"] for label, value in arm_metrics.items()},
            "arm_metrics": arm_metrics,
            "primary_comparisons": list(PRIMARY), "comparisons": comparisons}


def assert_pilot_repeat(pilot, scope):
    pairs = {
        "candidate_vs_control_single": "crop1536_vs_matched_crop1024_single",
        "candidate_uniform_vs_control_uniform": "crop1536_addition_vs_matched_crop1024_addition",
        "candidate_uniform_vs_sub6": "crop1536_addition_vs_sub6",
        "candidate_uniform_vs_old_uniform": "crop1536_addition_vs_original_tile1536_addition",
    }
    for current, old in pairs.items():
        for key in ("cv_mIoU", "base_cv_mIoU", "delta"):
            assert abs(scope["comparisons"][current][key] - pilot["comparisons"][old][key]) < 1e-5, (
                f"Pilot score drift: {current} {key}")
    assert abs(scope["baseline"]["cv_mIoU"] - pilot["sub6_reference"]["cv_mIoU"]) < 1e-5


def write_new_atomic(path, text):
    """Publish a fully-written file without ever replacing an existing name."""
    path = Path(path)
    assert path.parent.resolve() == OUT.resolve() and path.name in ARTIFACTS
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=OUT,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        # On the same filesystem, link publishes atomically and fails if target exists.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_new_json(path, value):
    write_new_atomic(path, json.dumps(crop_review.clean(value), ensure_ascii=False,
                                     indent=2, allow_nan=False) + "\n")


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    started = time.monotonic()
    os.chdir(ROOT)
    for name in ARTIFACTS:
        if (OUT / name).exists():
            raise FileExistsError(f"Refuse to overwrite completed review artifact: {OUT / name}")
    plan, jobs, frozen = immutable_plan()
    fold_of = load_json(ROOT / "splits/folds5.json")
    items = load_split("data", "train") + load_split("data", "valid")
    assert len(items) == len(fold_of) == 90 and set(fold_of.values()) == set(FOLDS)
    assert {item["name"] for item in items} == set(fold_of)
    alloy_folds = {}
    for item in items:
        assert item["lab"] is not None and item["lab"].shape == item["gray"].shape
        assert np.isin(item["lab"], np.arange(4)).all()
        alloy_folds.setdefault(item["alloy"], set()).add(fold_of[item["name"]])
    assert all(len(value) == 1 for value in alloy_folds.values())
    pilot, source_hashes, provenance = prior_reviews(fold_of)
    runs = completed_runs(items, fold_of, pilot, jobs)
    rows, sources, gates, timings = {label: {} for label in LABELS}, [], {}, {}
    for index, item in enumerate(items, 1):
        item_started, name = time.monotonic(), item["name"]
        paths = {
            "x2": ROOT / "runs/oof" / crop_review.BASE / f"{name}.npy",
            "unet": ROOT / "runs/oof" / crop_review.UNET / f"{name}.npy",
            "old": crop_review.OLD_TILE_ROOT / "oof" / f"{name}.npy",
            "control": ROOT / "runs/oof" / crop_review.CONTROL / f"{name}.npy",
            "candidate": ROOT / "runs/oof" / crop_review.CANDIDATE / f"{name}.npy",
        }
        probs = {}
        for source, path in paths.items():
            expected_hash = source_hashes.get((name, source))
            if fold_of[name] in (0, 1) or source in ("x2", "unet", "old"):
                assert expected_hash is not None
            probs[source], record = selective.probability(path, item, expected_hash)
            sources.append({"name": name, "fold": fold_of[name], "source": source,
                            "matched_prior_receipt": expected_hash is not None, **record})
        pb, pu, po, pc, pn = (probs[source] for source in ("x2", "unet", "old", "control", "candidate"))
        base = (pb + pu) / 2
        selected_control, gate = selective.choose(base, pc)
        selected_candidate, gate_candidate = selective.choose(base, pn)
        assert np.array_equal(gate, gate_candidate), "Gate must be identical for both crop arms"
        predictions = {
            "sub6": base, "old_tile1536_single": po, "old_tile1536_uniform": (pb + pu + po) / 3,
            "matched_crop1024_single": pc, "matched_crop1024_uniform": (pb + pu + pc) / 3,
            "matched_crop1024_selected": selected_control, "crop1536_single": pn,
            "crop1536_uniform": (pb + pu + pn) / 3, "crop1536_selected": selected_candidate,
        }
        for label, p in predictions.items():
            rows[label][name] = {"tta_pp": image_iou(item["lab"], postprocess(p)).tolist(),
                                 "low": bool(item["low_contrast"])}
        gates[name] = {"fold": fold_of[name], "gated_pixels": int(gate.sum()),
                       "total_pixels": int(gate.size), "area_fraction": float(gate.mean()),
                       "low_contrast": bool(item["low_contrast"])}
        timings[name] = time.monotonic() - item_started
        if index % 6 == 0:
            print(f"scored {index}/90 images; CPU only; elapsed={time.monotonic() - started:.1f}s", flush=True)
    scoring_finished = time.monotonic()
    scopes = {}
    for scope, folds in (("full90", FOLDS), ("remaining54", (2, 3, 4)), ("pilot36", (0, 1))):
        scopes[scope] = review_scope(rows, fold_of, folds)
        print(f"completed grouped-bootstrap metrics: {scope}", flush=True)
    assert abs(scopes["full90"]["baseline"]["cv_mIoU"] - selective.BASELINE_90) < 1e-5
    assert abs(scopes["full90"]["scores"]["old_tile1536_uniform"] - selective.UNIFORM_90) < 1e-5
    assert_pilot_repeat(pilot, scopes["pilot36"])
    # Protect against source/code changes during scoring as well as before it.
    for record in frozen["frozen_code"]:
        verify_record(record)
    assert sha256(OUT / "plan.json") == frozen["plan"]["sha256"]
    verify_record(frozen["started"])
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256(Path(__file__)), "stage": "full_crop_validation",
        "folds": list(FOLDS), "images": 90, "primary_comparisons": list(PRIMARY),
        "scopes": scopes,
        "comparisons": scopes["full90"]["comparisons"],
        "baseline": scopes["full90"]["baseline"],
        "baseline_expected_cv_mIoU": selective.BASELINE_90,
        "old_uniform_expected_cv_mIoU": selective.UNIFORM_90,
        "matched_design": {
            "control": crop_review.CONTROL, "candidate": crop_review.CANDIDATE,
            "epochs_both": 60, "seed_both": 0, "physical_batch_both": 2,
            "accumulation_steps_both": 2, "effective_batch_both": 4,
            "control_crop": 1024, "candidate_crop": 1536, "inference_tile_both": 1536,
            "checkpoint": "last.pt at epoch60, without best-epoch selection",
            "initialization": "fresh locked ImageNet; no fold checkpoint warm-start",
            "preserved_pilot_folds": [0, 1], "newly_trained_folds": [2, 3, 4],
            "uniform_recipe": "x2 MiT + x3 UNet++ + selected x4 MiT, each one-third",
            "existing_unet_reference_epochs": 120, "weight_search": False,
        },
        "fixed_selective_rule": {
            "base": "(x2 MiT + x3 UNet++) / 2", "margin_threshold": selective.MARGIN,
            "gate": "sub6 top1 probability minus top2 probability < 0.15",
            "inside_gate": "equal three-way blend", "outside_gate": "unchanged sub6 probabilities",
            "x4_weight_inside_gate": selective.ZOOM_WEIGHT, "label_based_gate": False,
            "threshold_or_weight_search": False, "actual_roi_inference": False,
        },
        "gate_area": {
            "pixel_weighted_fraction": sum(row["gated_pixels"] for row in gates.values())
                                       / sum(row["total_pixels"] for row in gates.values()),
            "mean_image_fraction": float(np.mean([row["area_fraction"] for row in gates.values()])),
            "fraction_per_fold": {str(fold): sum(row["gated_pixels"] for row in gates.values() if row["fold"] == fold)
                                  / sum(row["total_pixels"] for row in gates.values() if row["fold"] == fold)
                                  for fold in FOLDS},
        },
        "provenance": {**provenance, **frozen, "source_files": sources, "crop_run_receipts": runs},
        "timing_seconds": {"elapsed_before_writing": time.monotonic() - started,
                           "validation_and_scoring": scoring_finished - started,
                           "metrics_and_bootstrap": time.monotonic() - scoring_finished,
                           "per_image": timings},
        "caveats": [
            "Five-fold OOF validation covers 90 images; every photo from an alloy remains in one fold",
            "The 36-image pilot was used to decide whether to run the remaining folds; pilot36 and remaining54 are reported separately",
            "The remaining54 crop comparisons were unobserved at extension time, but these images were examined in earlier unrelated model experiments; this is not an independent confirmatory test",
            "All results use one training seed; this experiment cannot establish training-seed stability",
            "The crop arms use identical 60-epoch matched settings; crop changes also alter context, pixel coverage and normalization/loss spatial statistics",
            "The historical x4 reference had a different physical batch; matched newly trained crop1024 is the primary control",
            "Historical sub6 contains x2 MiT trained60 epochs and x3 UNet++ trained120 epochs; neither reference is retrained or shortened here",
            "Grouped bootstrap resamples alloy groups within folds10000 times with seed0; repeated model selection and training-seed uncertainty are not represented",
            "mIoU is the mean of per-image class means using existing postprocessing; absent-in-both classes are excluded per image",
            "Selective mixing reads saved full-image OOF maps; it neither crops uncertain regions nor measures real ROI inference speed",
            "The label-free gate is identical across crop arms; confidently wrong pixels may be missed and global Primary Si postprocessing can alter labels outside the gate",
            "The margin0.15 and one-third x4 weight are fixed; no threshold/weight search or automatic model adoption is performed",
        ],
        "training_started_by_review": False, "gpu_inference_started": False,
        "automatic_model_adoption": False, "threshold_or_weight_search": False,
        "submission_changed": False, "further_training_scheduled": False,
        "cpu_threads": 2, "opencv_threads": 1,
    }
    lines = [f"Created UTC: {result['created_utc']}",
             "Complete five-fold matched crop validation: 90 images, seed0, 60 epochs both crop arms.",
             "Report full90, remaining54, pilot36 separately; no automatic adoption or further training."]
    for scope, reviewed in scopes.items():
        lines.append(f"[{scope}] images={reviewed['images']}; folds={reviewed['folds']}")
        for label, score in reviewed["scores"].items():
            lines.append(f"  {label}: mIoU={score:.6f}")
        for key, value in reviewed["comparisons"].items():
            lines.append(f"  {key}: mIoU={value['cv_mIoU']:.6f}; base={value['base_cv_mIoU']:.6f}; delta={value['delta']:+.6f} pp; CI95={value['group_bootstrap_ci95']}")
            lines.append(f"    folds={value['delta_per_fold']}; classes={value['delta_per_class']}; low={value['delta_low_contrast']}; improved={value['improved_images']}/{value['images']}")
    lines.extend(result["caveats"])
    write_new_json(OUT / "per_image_rows.json", {"folds": list(FOLDS), "fold_of": fold_of,
                                                "rows": rows, "gate": gates})
    write_new_atomic(OUT / "summary.txt", "\n".join(lines) + "\n")
    # Publish last: this file means the requested review artifacts exist.
    write_new_json(OUT / "results.json", result)
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
