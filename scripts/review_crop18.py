"""CPU-only review of matched crop1024/crop1536 training on folds 0 and 1.

This script only reads completed runs and OOF probabilities. It never launches
training, extends the fold set, or changes any submission/model/data files.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import cv2
import numpy as np
import torch

cv2.setNumThreads(1)
torch.set_num_threads(2)
torch.set_num_interop_threads(1)

from data import load_split
from infer import image_iou, postprocess
from review_grains15 import metrics

FOLDS = (0, 1)
CONTROL = "x4_mit_c1024_acc2_bw_r90"
CANDIDATE = "x4_mit_c1536_bw_r90"
BASE = "x2_mit_bw_r90"
UNET = "x3_unetpp_bw_e120_r90"
OLD_TILE_ROOT = ROOT / "runs/review17_tiles"
ALLOWED_CONFIG_DIFFERENCES = {"crop", "out", "save_oof"}


def read_json(path):
    return Path(path).read_text(encoding="utf-8-sig")


def load_json(path):
    return json.loads(read_json(path))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    return value


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def accumulation_steps(cfg):
    assert cfg["accum_steps"] == 2, "Expected recorded gradient accumulation of 2"
    return 2


def validate_run(exp, crop, fold, expected_names):
    run = ROOT / "runs/cv" / exp / f"fold{fold}"
    cfg = load_json(run / "args.json")
    assert cfg["seed"] == 0 and cfg["fold"] == fold and cfg["epochs"] == 60
    assert cfg["crop"] == crop and cfg["tile"] == 1536
    assert cfg["bs"] == 2 and accumulation_steps(cfg) == 2
    assert cfg["width"] == 2560 and cfg["scale_multiplier"] == 2.0
    assert cfg["arch"] == "Unet" and cfg["encoder"] == "mit_b2"
    assert not cfg.get("init") and not cfg.get("full")
    assert Path(cfg["save_oof"]).resolve() == (ROOT / "runs/oof" / exp).resolve()
    logs = [json.loads(line) for line in read_json(run / "log.jsonl").splitlines() if line.strip()]
    assert [row["epoch"] for row in logs] == list(range(1, 61)), f"Incomplete epochs: {exp} fold{fold}"
    assert all(math.isfinite(row["loss"]) for row in logs)
    final = load_json(run / "result.json")
    assert final["args"] == cfg
    assert len(final["rows"]) == len(expected_names) == 18
    assert {row["name"] for row in final["rows"]} == expected_names
    assert (run / "last.pt").is_file() and (run / "last.pt").stat().st_size > 0
    receipt = {
        "experiment": exp,
        "fold": fold,
        "crop": crop,
        "epochs": len(logs),
        "physical_batch": cfg["bs"],
        "accumulation_steps": 2,
        "effective_batch": 4,
        "inference_tile": cfg["tile"],
        "args_sha256": sha256(run / "args.json"),
        "log_sha256": sha256(run / "log.jsonl"),
        "result_sha256": sha256(run / "result.json"),
        "last_checkpoint_bytes": (run / "last.pt").stat().st_size,
        "training_seconds": sum(row["train_sec"] for row in logs),
    }
    return cfg, receipt


def probability(path, item):
    p = np.load(path, allow_pickle=False).astype(np.float32)
    assert p.shape == (4, *item["gray"].shape) and np.isfinite(p).all(), str(path)
    assert p.min() >= 0 and p.max() <= 1, str(path)
    assert np.max(np.abs(p.sum(0) - 1)) < 0.005, str(path)
    return p


def compare(rows_a, rows_b, fold_of):
    return metrics(rows_a, rows_b, fold_of, repeats=10000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, nargs="+", default=list(FOLDS))
    parser.add_argument("--out", type=Path, default=Path("runs/review18_crop/results.json"))
    args = parser.parse_args()
    if args.folds != list(FOLDS):
        parser.error("Only --folds 0 1 is allowed; this review cannot extend to other folds")
    output = args.out.resolve()
    assert output.is_relative_to(ROOT / "runs/review18_crop"), "Output must stay within runs/review18_crop"
    assert output.suffix == ".json"
    if output.exists():
        raise FileExistsError(f"Refuse to overwrite a completed review: {output}")
    old_done = load_json(OLD_TILE_ROOT / "all.done")
    assert not old_done["training"] and not old_done["submission_changed"]
    fold_of = load_json(ROOT / "splits/folds5.json")
    all_items = load_split("data", "train") + load_split("data", "valid")
    assert len(all_items) == len(fold_of) == 90
    assert {item["name"] for item in all_items} == set(fold_of)
    items = [item for item in all_items if fold_of[item["name"]] in FOLDS]
    assert len(items) == 36
    run_receipts = []
    for fold in FOLDS:
        expected = {item["name"] for item in items if fold_of[item["name"]] == fold}
        cfg_control, receipt_control = validate_run(CONTROL, 1024, fold, expected)
        cfg_candidate, receipt_candidate = validate_run(CANDIDATE, 1536, fold, expected)
        differences = {key for key in set(cfg_control) | set(cfg_candidate)
                       if cfg_control.get(key) != cfg_candidate.get(key)}
        assert differences.issubset(ALLOWED_CONFIG_DIFFERENCES), f"Unmatched configuration: {sorted(differences)}"
        assert "crop" in differences
        run_receipts.extend([receipt_control, receipt_candidate])
        old_fold = load_json(OLD_TILE_ROOT / f"fold{fold}_result.json")
        assert old_fold["fold"] == fold and old_fold["tile"] == 1536
        assert set(old_fold["rows"]) == expected

    labels = ("sub6", "old_tile1536_single", "matched_crop1024_single", "crop1536_single",
              "old_tile1536_addition", "matched_crop1024_addition", "crop1536_addition")
    rows = {label: {} for label in labels}
    source_files = []
    for index, item in enumerate(items, 1):
        name = item["name"]
        sources = {
            "x2": ROOT / "runs/oof" / BASE / f"{name}.npy",
            "unet": ROOT / "runs/oof" / UNET / f"{name}.npy",
            "old": OLD_TILE_ROOT / "oof" / f"{name}.npy",
            "control": ROOT / "runs/oof" / CONTROL / f"{name}.npy",
            "candidate": ROOT / "runs/oof" / CANDIDATE / f"{name}.npy",
        }
        probabilities = {key: probability(path, item) for key, path in sources.items()}
        for key, path in sources.items():
            source_files.append({"name": name, "source": key, "path": str(path.relative_to(ROOT)), "sha256": sha256(path)})
        pb, pu, po, pc, pn = (probabilities[key] for key in ("x2", "unet", "old", "control", "candidate"))
        for label in labels:
            if label == "sub6":
                p = (pb + pu) / 2
            elif label == "old_tile1536_single":
                p = po
            elif label == "matched_crop1024_single":
                p = pc
            elif label == "crop1536_single":
                p = pn
            else:
                extra = po if label == "old_tile1536_addition" else pc if label == "matched_crop1024_addition" else pn
                p = (pb + pu + extra) / 3
            rows[label][name] = {"tta_pp": image_iou(item["lab"], postprocess(p)).tolist(), "low": bool(item["low_contrast"])}
        if index % 6 == 0:
            print(f"scored {index}/36 images; CPU only", flush=True)

    pairs = {
        "crop1536_vs_matched_crop1024_single": ("crop1536_single", "matched_crop1024_single"),
        "crop1536_addition_vs_matched_crop1024_addition": ("crop1536_addition", "matched_crop1024_addition"),
        "crop1536_addition_vs_sub6": ("crop1536_addition", "sub6"),
        "matched_crop1024_addition_vs_sub6": ("matched_crop1024_addition", "sub6"),
        "crop1536_vs_original_tile1536_single": ("crop1536_single", "old_tile1536_single"),
        "matched_crop1024_vs_original_tile1536_single": ("matched_crop1024_single", "old_tile1536_single"),
        "crop1536_addition_vs_original_tile1536_addition": ("crop1536_addition", "old_tile1536_addition"),
        "original_tile1536_addition_vs_sub6": ("old_tile1536_addition", "sub6"),
    }
    comparisons = {key: compare(rows[left], rows[right], fold_of) for key, (left, right) in pairs.items()}
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256(Path(__file__)),
        "folds": list(FOLDS),
        "images": len(items),
        "dataset_pool_images": 90,
        "primary_comparisons": ["crop1536_vs_matched_crop1024_single", "crop1536_addition_vs_matched_crop1024_addition"],
        "comparisons": comparisons,
        "sub6_reference": compare(rows["sub6"], rows["sub6"], fold_of),
        "matched_design": {
            "control": CONTROL,
            "candidate": CANDIDATE,
            "seed": 0,
            "epochs": 60,
            "physical_batch_both": 2,
            "gradient_accumulation_both": 2,
            "effective_batch_both": 4,
            "control_crop": 1024,
            "candidate_crop": 1536,
            "inference_tile_both": 1536,
            "additional_ensemble": "x2 MiT 1/3 + x3 UNet++ 1/3 + selected x4 MiT 1/3",
            "weight_search": False,
        },
        "run_receipts": run_receipts,
        "source_files": source_files,
        "caveats": [
            "Exploratory screening on folds0/1 (36 images) from a repeatedly examined 90-image dataset; not an independent test or full five-fold validation",
            "Primary comparisons match batch2, accumulation2, epochs60, seed0, inference tile1536 and other recorded options; only training crop changes",
            "Matching effective batch alone does not reproduce historical batch4 training because BatchNorm and Dice reductions can differ; the newly trained matched control is primary",
            "Original crop1024 tile1536 and sub6 are secondary references, not substitutes for the matched-control comparison",
            "Changing training crop also changes visible context, pixel coverage and the spatial distribution used by normalization and the loss; no single causal mechanism is established",
            "Confidence intervals resample alloy groups within folds 10000 times with seed0; training-seed and repeated model-selection uncertainty are not included",
            "This review produces evidence only; no automatic submission replacement or extension to folds2/3/4 is performed",
        ],
        "training_started_by_review": False,
        "auto_extend_folds": False,
        "submission_changed": False,
        "cpu_threads": 2,
        "opencv_threads": 1,
    }
    save_json(output.parent / "per_image_rows.json", {"folds": list(FOLDS), "fold_of": {item["name"]: fold_of[item["name"]] for item in items}, "rows": rows})
    lines = [f"Created UTC: {result['created_utc']}",
             "Exploratory matched-control pilot: folds0/1, 36 previously examined images.",
             "Primary: crop1536 versus new crop1024; both batch2/accumulation2/60epoch/tile1536."]
    for key, m in comparisons.items():
        lines.append(f"{key}: mIoU={m['cv_mIoU']:.6f}; base={m['base_cv_mIoU']:.6f}; delta={m['delta']:+.6f} pp; CI95={m['group_bootstrap_ci95']}")
        lines.append(f"  folds={m['delta_per_fold']}; classes={m['delta_per_class']}; low={m['delta_low_contrast']}; improved={m['improved_images']}/{m['images']}")
    lines.extend(result["caveats"])
    (output.parent / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    save_json(output, result)
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
