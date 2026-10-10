#!/usr/bin/env python3
"""Reproduce the fixed sub6 test masks; default is a read-only preflight."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import struct
import subprocess
import sys
from datetime import datetime, timezone
import zipfile

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
EXPS = ("x2_mit_bw_r90", "x3_unetpp_bw_e120_r90")
COLORS = {(153, 127, 76), (25, 76, 153), (76, 178, 76), (204, 204, 204)}
CORE_PACKAGES = (
    "torch", "torchvision", "segmentation-models-pytorch", "timm",
    "numpy", "opencv-python-headless", "pillow", "scipy",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        head = stream.read(24)
    if len(head) != 24 or head[:8] != b"\x89PNG\r\n\x1a\n" or head[12:16] != b"IHDR":
        raise ValueError(f"Invalid PNG header: {path}")
    width, height = struct.unpack(">II", head[16:24])
    if not width or not height:
        raise ValueError(f"Invalid PNG size: {path}")
    return width, height


def read_manifest() -> dict:
    manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    if (manifest["experiments"] != list(EXPS) or manifest["folds"] != 5
            or manifest["split"] != "test" or manifest["test_count"] != 10
            or manifest["recipe"] != dict.fromkeys(EXPS, 1)):
        raise ValueError("This entrypoint only supports the recorded sub6 recipe.")
    expected = {f"runs/cv/{exp}/fold{k}/last.pt" for exp in EXPS for k in range(5)}
    if {x["path"] for x in manifest["checkpoints"]} != expected or len(manifest["checkpoints"]) != 10:
        raise ValueError("Manifest must contain exactly ten fixed checkpoints.")
    for section in ("source_files", "checkpoints", "test_images"):
        for item in manifest[section]:
            path = Path(item["path"])
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Manifest paths must remain inside the project.")
    return manifest


def preflight(root: Path, out: Path, manifest: dict) -> dict:
    """Only stdlib reads. Never import torch, initialize CUDA, or create files."""
    errors, warnings = [], []
    if out.exists() or out.is_symlink():
        errors.append(f"Output must be a NEW directory: {out}")
    if platform.system() != "Linux":
        errors.append("The recorded execution environment is Linux x86_64 with NVIDIA CUDA.")
    if sys.version_info[:2] != (3, 12):
        errors.append(f"Python 3.12 required; found {platform.python_version()}")
    elif platform.python_version() != manifest["python_version"]:
        warnings.append(f"Reference Python patch version is {manifest['python_version']}.")
    versions = {}
    for name in CORE_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
        if versions[name] != manifest["packages"][name]:
            errors.append(f"Package mismatch: {name}: expected {manifest['packages'][name]}, found {versions[name]}")
    verified = {key: 0 for key in ("source_files", "checkpoints", "test_images")}
    for section in verified:
        for item in manifest[section]:
            path = root / item["path"]
            if not path.is_file():
                errors.append(f"Missing file: {item['path']}")
            elif path.stat().st_size != item["size_bytes"] or sha256(path) != item["sha256"]:
                errors.append(f"File differs from reference: {item['path']}")
            else:
                verified[section] += 1
    expected_images = {Path(x["path"]).name for x in manifest["test_images"]}
    found_images = {p.name for p in (root / "data/test/images").glob("*_image.png") if p.is_file()}
    if len(expected_images) != 10 or found_images != expected_images:
        errors.append("data/test/images must contain exactly the ten recorded *_image.png inputs.")
    shapes = {}
    for item in manifest["test_images"]:
        path = root / item["path"]
        if path.is_file():
            try:
                shapes[path.name.removesuffix("_image.png")] = list(png_size(path))
            except ValueError as exc:
                errors.append(str(exc))
    return {"ok": not errors, "mode": "read_only_preflight", "project_root": str(root),
            "output": str(out), "verified_files": verified, "versions": versions,
            "image_sizes_width_height": shapes, "warnings": warnings, "errors": errors,
            "gpu_checked": False, "gpu_inference_performed": False}


def check_gpu(index: int) -> dict:
    """Require an explicitly selected idle GPU immediately before inference."""
    query = subprocess.run(
        ["nvidia-smi", f"--id={index}", "--query-gpu=uuid,name,memory.total,memory.used",
         "--format=csv,noheader,nounits"], text=True, capture_output=True, check=True)
    fields = [x.strip() for x in query.stdout.strip().split(",")]
    if len(fields) != 4:
        raise RuntimeError("Could not read the selected GPU.")
    uuid, name, total, used = fields
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=True)
    if any(line.split(",")[0].strip() == uuid for line in processes.stdout.splitlines()):
        raise RuntimeError(f"GPU {index} has an active compute process; choose an idle GPU.")
    if int(total) - int(used) < 30000:
        raise RuntimeError("At least 30000 MiB free GPU memory is required by this conservative preflight.")
    return {"index": index, "uuid": uuid, "name": name,
            "total_mib": int(total), "used_mib": int(used)}


def validate_parts(out: Path, shapes: dict) -> None:
    import numpy as np
    models = out / "parts/models"
    expected = {models / exp / f"fold{k}" / f"{name}.npy"
                for exp in EXPS for k in range(5) for name in shapes}
    if set(models.rglob("*.npy")) != expected:
        raise RuntimeError("Expected exactly 100 model/image probability arrays before combine.")
    for path in sorted(expected):
        width, height = shapes[path.stem]
        prob = np.load(path, mmap_mode="r", allow_pickle=False)
        if prob.shape != (4, height, width) or prob.dtype != np.float32:
            raise RuntimeError(f"Invalid probability shape/dtype: {path}")
        if (not np.isfinite(prob).all() or prob.min() < 0 or prob.max() > 1
                or np.max(np.abs(prob.sum(0) - 1)) > 0.005):
            raise RuntimeError(f"Invalid probability values: {path}")


def validate_masks(out: Path, shapes: dict) -> list[dict]:
    from PIL import Image
    expected = {f"{name}_mask.png" for name in shapes}
    files = sorted((out / "masks").iterdir())
    if {p.name for p in files} != expected or len(files) != 10:
        raise RuntimeError("Expected exactly ten mask PNGs.")
    receipts = []
    for path in files:
        name = path.name.removesuffix("_mask.png")
        with Image.open(path) as image:
            image.load()
            colors = image.getcolors(maxcolors=5)
            if (image.mode != "RGB" or image.format != "PNG"
                    or image.size != tuple(shapes[name]) or colors is None
                    or not {color for _, color in colors} <= COLORS):
                raise RuntimeError(f"Mask format/color/size mismatch: {path.name}")
        receipts.append({"name": path.name, "sha256": sha256(path), "size_bytes": path.stat().st_size})
    summary = json.loads((out / "summary_test.json").read_text(encoding="utf-8"))
    if (summary["n_images"] != 10 or summary["n_models"] != 10
            or summary["models_per_image"] != [10] or summary["problems"]
            or summary["models_per_exp"] != dict.fromkeys(EXPS, 5)
            or summary["recipe"] != dict.fromkeys(EXPS, 1.0)
            or summary["full_mix"] != 0 or summary["si_classifier"] is not None):
        raise RuntimeError("combine summary does not match the fixed sub6 recipe.")
    return receipts


def run(root: Path, out: Path, manifest: dict, report: dict, gpu: int) -> dict:
    gpu_info = check_gpu(gpu)
    out.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu_info["uuid"], "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    common = [sys.executable, str(root / "src/submit.py")]
    commands = [common + ["predict", "--exps", *EXPS, "--root", "runs/cv", "--folds", "5",
                          "--split", "test", "--out", str(out), "--shard", "0", "--nshards", "1"],
                common + ["combine", "--split", "test", "--parts", str(out), "--out", str(out),
                          "--recipe", *EXPS, "--full-mix", "0"]]
    receipt = {"started_utc": datetime.now(timezone.utc).isoformat(), "preflight": report,
               "gpu": gpu_info, "manifest_sha256": sha256(HERE / "manifest.json"),
               "commands": commands, "status": "running"}
    (out / "run_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    try:
        for index, command in enumerate(commands):
            with (out / ("predict.log" if index == 0 else "combine.log")).open("x", encoding="utf-8") as log:
                subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            if index == 0:
                validate_parts(out, report["image_sizes_width_height"])
        masks = validate_masks(out, report["image_sizes_width_height"])
        zip_path = out / "sub6_masks.zip"
        with zipfile.ZipFile(zip_path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for item in masks:
                archive.write(out / "masks" / item["name"], arcname=item["name"])
        with zipfile.ZipFile(zip_path) as archive:
            if archive.testzip() is not None or archive.namelist() != [m["name"] for m in masks]:
                raise RuntimeError("ZIP validation failed.")
            for item in masks:
                if hashlib.sha256(archive.read(item["name"])).hexdigest() != item["sha256"]:
                    raise RuntimeError("ZIP content differs from validated masks.")
        receipt.update(status="complete", masks=masks, zip_sha256=sha256(zip_path),
                       finished_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        (out / "run_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Read-only preflight (default); no CUDA initialization")
    mode.add_argument("--run", action="store_true", help="Generate ten masks and a ZIP in a new output directory")
    parser.add_argument("--project-root", type=Path, default=HERE.parents[1])
    parser.add_argument("--out", type=Path, default=Path("submissions/sub6_reproduced"))
    parser.add_argument("--gpu", type=int, help="Required with --run; an idle physical NVIDIA GPU index")
    args = parser.parse_args()
    if args.run and (args.gpu is None or args.gpu < 0):
        parser.error("--run requires an explicit non-negative --gpu index")
    root = args.project_root.resolve()
    out = (root / args.out).absolute() if not args.out.is_absolute() else args.out.absolute()
    # Check the lexical final path before resolve so an existing dangling symlink cannot be reused.
    if out.is_symlink():
        parser.error("Output cannot be an existing symlink")
    out = out.resolve()
    try:
        manifest = read_manifest()
        report = preflight(root, out, manifest)
        print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
        if not report["ok"]:
            return 2
        if args.run:
            receipt = run(root, out, manifest, report, args.gpu)
            print(json.dumps({"status": receipt["status"], "masks": 10,
                              "zip": str(out / "sub6_masks.zip")}, indent=2), flush=True)
        return 0
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
