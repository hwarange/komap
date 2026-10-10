"""Build the selected final-pair code ZIP after verified test inference completes."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "delivery/final_pair_code_bundle.zip"
RECEIPT = ROOT / "delivery/final_pair_code_bundle_receipt.json"
EXECUTION = ROOT / "submissions/final_pair_mit4_unet3"
EXPS = ("x4_mit_c1536_bw_r90", "x3_unetpp_bw_e120_r90")
PACKAGE = (
    "README.md", "environment.yml", "run_submission.py", "manifest.json",
    "reference_server_receipts.json", "preflight_reference_server.json", "local_validation.json",
)
STATIC = (
    "src/submit.py", "src/train.py", "src/data.py", "src/infer.py",
    "requirements.txt", "scripts/summarize_phase_masks.py",
    "runs/review22_pair/plan.json", "runs/review22_pair/results.json",
    "runs/review22_pair/per_image_rows.json", "runs/review22_pair/summary.txt",
    "runs/review22_pair/all.done", "runs/review22_pair/completion_verification.json",
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def prepare() -> tuple[dict[str, bytes], dict]:
    # Use an explicit allowlist: never sweep weights, inputs, credentials or probabilities.
    payload = {name: (ROOT / name).read_bytes() for name in STATIC}
    # The Windows checkout has equivalent JSON with LF; preserve the experiment's exact CRLF bytes.
    split_bytes = (ROOT / "delivery/final_pair_fold_split_reference.json").read_bytes()
    split_record = read_json(ROOT / "runs/review22_pair/plan.json")["split"]
    assert digest(split_bytes) == split_record["sha256"] and len(split_bytes) == split_record["bytes"]
    assert json.loads(split_bytes) == read_json(ROOT / "splits/folds5.json")
    payload["splits/folds5.json"] = split_bytes
    payload["README.md"] = (ROOT / "delivery/final_pair_bundle_README.md").read_bytes()
    for name in PACKAGE:
        relative = f"delivery/final_pair_repro/{name}"
        payload[relative] = (ROOT / relative).read_bytes()
    manifest_bytes = payload["delivery/final_pair_repro/manifest.json"]
    manifest = json.loads(manifest_bytes)
    assert manifest["experiments"] == list(EXPS)
    assert manifest["recipe"] == dict.fromkeys(EXPS, 1)
    for item in manifest["source_files"]:
        assert digest(payload[item["path"]]) == item["sha256"], item["path"]
    for name, data in payload.items():
        if name.endswith(".py"):
            ast.parse(data.decode("utf-8"), filename=name)
        if name.endswith(".json"):
            json.loads(data)

    run_bytes = (EXECUTION / "run_receipt.json").read_bytes()
    summary_bytes = (EXECUTION / "summary_test.json").read_bytes()
    run, summary = json.loads(run_bytes), json.loads(summary_bytes)
    assert run["status"] == "complete", "Test inference must finish before packaging"
    assert run["manifest_sha256"] == digest(manifest_bytes), "Frozen manifest changed"
    assert run["entrypoint_sha256"] == digest(payload["delivery/final_pair_repro/run_submission.py"])
    assert run["prediction_reuse"] is False and run["training_performed"] is False
    assert run["prediction_exit_codes"] and not any(run["prediction_exit_codes"])
    assert run["preflight"]["verified_files"] == {"source_files": 4, "checkpoints": 10, "test_images": 10}
    assert summary["n_images"] == 10 and summary["n_models"] == 10
    assert summary["models_per_image"] == [10]
    assert summary["models_per_exp"] == dict.fromkeys(EXPS, 5)
    assert summary["recipe"] == dict.fromkeys(EXPS, 1.0)
    assert summary["full_mix"] == 0 and summary["si_classifier"] is None and not summary["problems"]
    assert len(run["masks"]) == 10
    mask_zip = (EXECUTION / "final_pair_masks.zip").read_bytes()
    assert digest(mask_zip) == run["zip_sha256"], "Local final mask ZIP differs from run receipt"
    with zipfile.ZipFile(EXECUTION / "final_pair_masks.zip") as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == {item["name"] for item in run["masks"]}
        assert len(archive.namelist()) == 10
        for item in run["masks"]:
            data = archive.read(item["name"])
            assert len(data) == item["size_bytes"] and digest(data) == item["sha256"]
    payload["submissions/final_pair_mit4_unet3/run_receipt.json"] = run_bytes
    payload["submissions/final_pair_mit4_unet3/summary_test.json"] = summary_bytes
    # Freeze immutable review22 evidence only; live final_model_selection.json is excluded.
    evidence = read_json(ROOT / "runs/review22_pair/completion_verification.json")
    for path, sha in evidence["files"].items():
        if path in payload:
            assert digest(payload[path]) == sha, path
    assert read_json(ROOT / "runs/review22_pair/results.json")["images"] == 90
    for name in payload:
        assert not Path(name).is_absolute() and ".." not in Path(name).parts
        assert Path(name).suffix.lower() not in {".pt", ".pth", ".npy", ".pkl", ".png", ".zip"}
        assert not name.startswith("data/")
    return payload, run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", help="Create a new code ZIP and receipt; default only validates")
    args = parser.parse_args()
    payload, run = prepare()
    if not args.build:
        print(json.dumps({"ok": True, "read_only": True, "files": len(payload)}))
        return
    if OUTPUT.exists() or RECEIPT.exists():
        raise FileExistsError("Refuse to overwrite final code bundle or its receipt")
    records = [{"path": name, "size_bytes": len(data), "sha256": digest(data)}
               for name, data in sorted(payload.items())]
    with zipfile.ZipFile(OUTPUT, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, data in sorted(payload.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 10, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)
    with zipfile.ZipFile(OUTPUT) as archive:
        assert archive.testzip() is None
        assert archive.namelist() == [item["path"] for item in records]
        for item in records:
            data = archive.read(item["path"])
            assert len(data) == item["size_bytes"] and digest(data) == item["sha256"]
    result = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "status": "final_selected_code_bundle_not_officially_submitted",
              "zip_path": OUTPUT.relative_to(ROOT).as_posix(), "zip_size_bytes": OUTPUT.stat().st_size,
              "zip_sha256": digest(OUTPUT.read_bytes()), "file_count": len(records),
              "source_py_count": sum(item["path"].endswith(".py") for item in records),
              "files": records, "zip_crc_passed": True, "archive_file_hashes_verified": True,
              "test_generation_status": run["status"], "test_mask_zip_sha256": run["zip_sha256"],
              "fresh_environment_install_tested": False, "official_submission_performed": False,
              "raw_data_weights_probabilities_included": False,
              "old_sub6_submission_artifacts_or_application_figures_included": False,
              "live_final_model_selection_included": False}
    with RECEIPT.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("zip_path", "zip_size_bytes", "zip_sha256", "file_count")}, indent=2))


if __name__ == "__main__":
    main()
