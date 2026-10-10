"""CPU-only fixed 50:50 x4 MiT crop1536 + x3 UNet++ OOF comparison."""
import os

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

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

OUT = ROOT / "runs/review22_pair"
PRIOR = ROOT / "runs/review20_crop_full"
EXPERIMENTS = {"x2": "x2_mit_bw_r90", "unet": "x3_unetpp_bw_e120_r90",
               "candidate": "x4_mit_c1536_bw_r90"}
ARTIFACTS = ("plan.json", "per_image_rows.json", "summary.txt", "results.json", "all.done")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def receipt(path):
    path = Path(path)
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path),
            "bytes": path.stat().st_size}


def verify(record):
    path = (ROOT / record["path"]).resolve()
    assert path.is_relative_to(ROOT.resolve())
    assert sha256(path) == record["sha256"], f"Changed source: {path}"
    if "bytes" in record:
        assert path.stat().st_size == record["bytes"]
    return receipt(path)


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(item) for item in value]
    return value


def write_new(path, value, raw=False):
    assert path.parent == OUT and path.name in ARTIFACTS
    text = value if raw else json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=OUT, prefix=".pair22_", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def scope(rows, fold_of, folds):
    selected = {label: {name: row for name, row in values.items() if fold_of[name] in folds}
                for label, values in rows.items()}
    arm_metrics = {}
    for label, values in selected.items():
        names = sorted(values)
        iou = np.asarray([values[name]["tta_pp"] for name in names], float) * 100
        miou = np.nanmean(iou, axis=1)
        low = np.asarray([values[name]["low"] for name in names], bool)
        groups = np.asarray([fold_of[name] for name in names])
        arm_metrics[label] = {
            "cv_mIoU": float(miou.mean()),
            "mIoU_per_fold": {str(f): float(miou[groups == f].mean()) for f in folds},
            "IoU_per_class": dict(zip(CLASSES, np.nanmean(iou, axis=0).tolist())),
            "class_valid_images": dict(zip(CLASSES, np.isfinite(iou).sum(0).tolist())),
            "low_contrast_mIoU": float(miou[low].mean()) if low.any() else None,
            "low_contrast_images": int(low.sum()),
        }
    compared = metrics(selected["x4_mit_x3_unetpp_pair"], selected["sub6"], fold_of, repeats=10000)
    return {"folds": list(folds), "images": len(selected["sub6"]),
            "scores": {label: value["cv_mIoU"] for label, value in arm_metrics.items()},
            "arm_metrics": arm_metrics, "pair_vs_sub6": compared}


def main():
    started = time.monotonic()
    os.chdir(ROOT)
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    assert not any((OUT / name).exists() for name in ARTIFACTS), "Refuse to overwrite review"
    old = load_json(PRIOR / "results.json")
    done = load_json(PRIOR / "all.done")
    prior_plan = load_json(PRIOR / "plan.json")
    assert done["results_sha256"] == sha256(PRIOR / "results.json")
    assert done["plan_sha256"] == sha256(PRIOR / "plan.json")
    assert done["outcome"] == "matched_crop_five_fold_validation_complete"
    assert old["folds"] == list(range(5)) and old["images"] == 90
    assert old["matched_design"]["candidate"] == EXPERIMENTS["candidate"]
    assert old["submission_changed"] is False and done["submission_changed"] is False
    frozen = [verify(record) for record in old["provenance"]["frozen_code"]]
    frozen.append(receipt(Path(__file__)))
    split_record = verify(old["provenance"]["folds"])
    assert split_record["path"] == "splits/folds5.json"
    fold_of = load_json(ROOT / split_record["path"])
    data_records = [verify({"path": path, "sha256": digest})
                    for path, digest in prior_plan["data_receipts"].items()]
    assert len(data_records) == 181
    assert sum(record["path"].startswith("data/") for record in data_records) == 180
    assert sum(record["path"] == "splits/folds5.json" for record in data_records) == 1
    historical = {(r["name"], r["source"]): r for r in old["provenance"]["source_files"]}
    assert len(historical) == 450
    items = load_split("data", "train") + load_split("data", "valid")
    assert len(items) == len(fold_of) == 90
    assert {item["name"] for item in items} == set(fold_of)
    alloy_folds = {}
    for item in items:
        assert item["lab"] is not None and item["gray"].shape == item["lab"].shape
        assert np.isin(item["lab"], np.arange(4)).all()
        alloy_folds.setdefault(item["alloy"], set()).add(fold_of[item["name"]])
    assert all(len(folds) == 1 for folds in alloy_folds.values())
    plan = {"created_utc": datetime.now(timezone.utc).isoformat(), "experiments": EXPERIMENTS,
            "fixed_recipes": {"sub6": "(x2 MiT60 + x3 UNet++120) / 2",
                              "x4_mit_x3_unetpp_pair": "(x4 MiT crop1536 60 + x3 UNet++120) / 2"},
            "weights": [0.5, 0.5], "soft_voting": True,
            "scope_folds": {"full90": [0, 1, 2, 3, 4], "remaining54": [2, 3, 4], "pilot36": [0, 1]},
            "data_receipts": data_records, "source_receipts": frozen, "split": split_record,
            "prior_receipts": [receipt(PRIOR / name) for name in ("plan.json", "results.json", "all.done")],
            "threshold_or_weight_search": False, "training_started_by_review": False,
            "gpu_inference_started": False, "submission_changed": False}
    write_new(OUT / "plan.json", plan)
    rows = {label: {} for label in plan["fixed_recipes"]}
    source_records = []
    for index, item in enumerate(items, 1):
        name = item["name"]
        probs = {}
        for source, exp in EXPERIMENTS.items():
            path = ROOT / "runs/oof" / exp / f"{name}.npy"
            old_record = historical[(name, source)]
            assert old_record["path"] == path.relative_to(ROOT).as_posix()
            record = verify(old_record)
            p = np.load(path, allow_pickle=False).astype(np.float32)
            assert p.shape == (4, *item["gray"].shape)
            assert np.isfinite(p).all() and p.min() >= 0 and p.max() <= 1
            assert np.max(np.abs(p.sum(0) - 1)) < 0.005
            probs[source] = p
            source_records.append({"name": name, "fold": fold_of[name], "source": source, **record})
        for label, p in (("sub6", (probs["x2"] + probs["unet"]) / 2),
                         ("x4_mit_x3_unetpp_pair", (probs["candidate"] + probs["unet"]) / 2)):
            rows[label][name] = {"tta_pp": image_iou(item["lab"], postprocess(p)).tolist(),
                                 "low": bool(item["low_contrast"])}
        if index % 10 == 0:
            print(f"scored {index}/90 CPU-only; {time.monotonic() - started:.1f}s", flush=True)
    scopes = {name: scope(rows, fold_of, folds) for name, folds in plan["scope_folds"].items()}
    for name, values in scopes.items():
        assert abs(values["scores"]["sub6"] - old["scopes"][name]["scores"]["sub6"]) < 1e-6
    for record in frozen + data_records + [split_record] + plan["prior_receipts"]:
        verify(record)
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "stage": "fixed_two_model_pair_review",
              "images": 90, "folds": list(range(5)), "scopes": scopes, "class_order": CLASSES,
              "plan": receipt(OUT / "plan.json"), "source_files": source_records,
              "elapsed_seconds": time.monotonic() - started, "cpu_threads": 2, "opencv_threads": 1,
              "caveats": [
                  "Reused saved TTA out-of-fold probabilities; fixed 50:50 soft voting, then unchanged postprocessing",
                  "x4 MiT crop1536 trained60 epochs, x3 UNet++ trained120 epochs, baseline x2 MiT trained60 epochs; one training seed",
                  "90 images from a repeatedly examined dataset; pilot36 and remaining54 are descriptive subsets, not independent confirmatory tests",
                  "Alloy groups remain in one fold; paired bootstrap resamples alloy groups within folds10000 times, RNG seed0",
                  "Bootstrap excludes training-seed and repeated-model-selection uncertainty",
                  "mIoU averages per-image class means; classes absent in both truth and prediction are excluded within an image",
                  "Stored OOF arrays were checked against completed review20 receipts; all180 data files and split were rehashed",
                  "No GPU inference, additional training, weight search, ROI reinference, automatic adoption, or submission replacement",
              ], "training_started_by_review": False, "gpu_inference_started": False,
              "threshold_or_weight_search": False, "automatic_model_adoption": False, "submission_changed": False}
    lines = ["Fixed equal two-model ensemble: x4 MiT crop1536 + x3 UNet++120"]
    for name, values in scopes.items():
        m = values["pair_vs_sub6"]
        lines.append(f"{name}: sub6={values['scores']['sub6']:.9f}; pair={values['scores']['x4_mit_x3_unetpp_pair']:.9f}; delta={m['delta']:+.9f}; CI95={m['group_bootstrap_ci95']}")
        lines.append(f"  folds={m['delta_per_fold']}; classes={m['delta_per_class']}; improved={m['improved_images']}/{m['images']}")
    lines.extend(result["caveats"])
    write_new(OUT / "per_image_rows.json", {"fold_of": fold_of, "rows": rows})
    write_new(OUT / "summary.txt", "\n".join(lines) + "\n", raw=True)
    write_new(OUT / "results.json", result)
    write_new(OUT / "all.done", {"created_utc": datetime.now(timezone.utc).isoformat(),
              "outcome": "fixed_two_model_pair_complete", "images": 90,
              "results_sha256": sha256(OUT / "results.json"), "plan_sha256": sha256(OUT / "plan.json"),
              "submission_changed": False, "training_started": False, "gpu_inference_started": False})
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
