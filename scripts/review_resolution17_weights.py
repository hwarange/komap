"""CPU-only comparison of two fixed MiT weights; no training or weight search."""
import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from datetime import datetime, timezone
import hashlib
import json
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

OUT = ROOT / "runs/review17_weights"
RETURN = ROOT / "runs/runpod_return_20261010"
EXPS = ("x2_mit_bw_r90", "x4_mit_bw_r90", "x3_unetpp_bw_e120_r90")
RECIPES = {
    "baseline_sub6": (0.5, 0.0, 0.5),
    "candidate_A_x4_0_1": (0.4, 0.1, 0.5),
    "candidate_B_x4_0_2": (0.3, 0.2, 0.5),
    "reference_equal_three": (1 / 3, 1 / 3, 1 / 3),
}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    if (OUT / "results.json").exists():
        raise RuntimeError("Output already exists; refuse to overwrite a completed review")
    state = read_json(RETURN / "state.json")
    assert state["stage"] == "complete" and state["all_sha256_verified"]
    verified = read_json(RETURN / "verified.json")
    manifest_path = RETURN / "manifest.json"
    assert sha256(manifest_path) == state["manifest_sha256"]
    manifest = {r["path"]: r for r in read_json(manifest_path)["files"]}
    mapping = {r["path"]: r["stored"] for r in verified["files"]}
    fold_of = read_json(ROOT / "splits/folds5.json")
    items = load_split("data", "train") + load_split("data", "valid")
    assert len(items) == len(fold_of) == 90 and {it["name"] for it in items} == set(fold_of)
    x4_keys = {f"runs/oof/{EXPS[1]}/{name}.npy" for name in fold_of}
    assert x4_keys.issubset(mapping) and x4_keys.issubset(manifest)
    assert all((ROOT / mapping[key]).resolve().is_relative_to(ROOT) for key in x4_keys)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = {name: {} for name in RECIPES}
    sources = []
    for index, it in enumerate(items, 1):
        name = it["name"]
        probs = []
        for exp in EXPS:
            key = f"runs/oof/{exp}/{name}.npy"
            relative = mapping[key] if exp == EXPS[1] else key
            path = ROOT / relative
            digest = sha256(path)
            if exp == EXPS[1]:
                assert path.stat().st_size == manifest[key]["size"]
                assert digest == manifest[key]["sha256"], f"Recovered OOF hash mismatch: {name}"
            p = np.load(path, allow_pickle=False).astype(np.float32)
            assert p.shape == (4, *it["gray"].shape) and np.isfinite(p).all()
            assert p.min() >= 0 and p.max() <= 1 and np.max(np.abs(p.sum(0) - 1)) < 0.005
            probs.append(p)
            sources.append({"logical_path": key, "actual_path": relative, "sha256": digest})
        pb, px, pu = probs
        for label, weights in RECIPES.items():
            if label == "baseline_sub6":
                p = (pb + pu) / 2
            elif label == "reference_equal_three":
                p = (pb + pu + px) / 3
            else:
                p = pb * weights[0] + px * weights[1] + pu * weights[2]
            rows[label][name] = {
                "tta_pp": image_iou(it["lab"], postprocess(p)).tolist(),
                "low": bool(it["low_contrast"]),
            }
        if index % 10 == 0:
            print(f"scored {index}/90 images", flush=True)
    baseline = rows["baseline_sub6"]
    comparisons = {}
    for label, values in rows.items():
        comparisons[label] = {
            "weights": dict(zip(EXPS, RECIPES[label])),
            "full": metrics(values, baseline, fold_of),
            "nonpilot": metrics({n: r for n, r in values.items() if fold_of[n] in (2, 3, 4)}, baseline, fold_of),
            "score_per_fold": {
                str(f): float(np.mean([np.nanmean(r["tta_pp"]) * 100 for n, r in values.items() if fold_of[n] == f]))
                for f in range(5)
            },
            "per_image_delta": {
                n: float((np.nanmean(r["tta_pp"]) - np.nanmean(baseline[n]["tta_pp"])) * 100)
                for n, r in sorted(values.items())
            },
        }
        print(f"summarized {label}", flush=True)
    previous_key = "runs/review16_additional/seed0_full_results.json"
    previous_path = ROOT / mapping.get(previous_key, previous_key)
    previous = read_json(previous_path)
    full_baseline = comparisons["baseline_sub6"]["full"]["cv_mIoU"]
    equal_three = comparisons["reference_equal_three"]["full"]["cv_mIoU"]
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Two fixed reduced x4 weights, preserving UNet++ weight at 0.5; no grid search or training",
        "script_sha256": sha256(Path(__file__)),
        "comparisons": comparisons,
        "provenance": {
            "return_manifest_sha256": sha256(manifest_path),
            "return_verification_sha256": sha256(RETURN / "verified.json"),
            "x4_recovered_oof": 90,
            "x4_conflict_paths_used": sum(mapping[k] != k for k in x4_keys),
            "baseline_oof": "Original-server x2 MiT and x3 UNet++ OOF, same model weights; not RunPod-regenerated OOF",
            "previous_runpod_baseline": previous["baseline"]["cv_mIoU"],
            "recomputed_baseline": full_baseline,
            "baseline_drift_pp": full_baseline - previous["baseline"]["cv_mIoU"],
            "previous_runpod_equal_three": previous["add_seed0_equal_three"]["full"]["cv_mIoU"],
            "recomputed_equal_three": equal_three,
            "equal_three_drift_pp": equal_three - previous["add_seed0_equal_three"]["full"]["cv_mIoU"],
            "source_files": sources,
        },
        "caveats": [
            "Exploratory fixed comparisons after inspecting full round16 results; no independent confirmatory test",
            "Nonpilot means folds2/3/4 of the earlier pilot; these outcomes are now already observed",
            "Bootstrap resamples alloy groups within folds; does not include training-seed or model-selection uncertainty",
            "All candidates evaluated using identical current-source probabilities, TTA, labels and postprocessing",
        ],
        "cpu_threads": 2,
        "opencv_threads": 1,
        "training_started": False,
        "submission_changed": False,
    }
    (OUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    lines = [f"Created UTC: {result['created_utc']}",
             "Fixed CPU-only comparison; all values percentages / percentage points.",
             f"Baseline drift from RunPod: {result['provenance']['baseline_drift_pp']:+.6f} pp",
             f"Equal-three drift from RunPod: {result['provenance']['equal_three_drift_pp']:+.6f} pp"]
    for label, comparison in comparisons.items():
        full, held = comparison["full"], comparison["nonpilot"]
        lines.append(f"{label}: full={full['cv_mIoU']:.6f}; delta={full['delta']:+.6f}; CI={full['group_bootstrap_ci95']}; nonpilot_delta={held['delta']:+.6f}; nonpilot_CI={held['group_bootstrap_ci95']}")
        lines.append(f"  fold_delta={full['delta_per_fold']}; class_delta={full['delta_per_class']}; low_delta={full['delta_low_contrast']}")
    lines.extend(result["caveats"])
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
