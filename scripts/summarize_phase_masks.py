"""Summarize predicted 2D phase area fractions without changing source masks.

Usage: python scripts/summarize_phase_masks.py --mask-dir submissions/sub6/masks \
    --out-dir reports/phase_application --expected-count 10 --plot
Dependencies: numpy, Pillow; optional plot: matplotlib.
These estimates are not measured composition, volume fraction, modulus or accuracy.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


CLASSES = ("Al", "Al3Ni", "Primary Si", "Eutectic Si")
PALETTE = np.array(((153, 127, 76), (25, 76, 153),
                    (76, 178, 76), (204, 204, 204)), dtype=np.uint8)
LIMITATIONS = [
    "Predicted 2D area fractions are calculated from segmentation masks, not ground truth.",
    "They are not measured chemical composition, volume fraction, elastic modulus or accuracy.",
    "No physical pixel scale is supplied; pixel counts are not converted to square micrometres.",
    "Test labels are unavailable, so phase-fraction error and test mIoU cannot be calculated.",
    "Image-level percentages describe sampled fields of view; they are not bulk-alloy estimates.",
    "The pooled fraction is weighted by digital pixel count, not physical sampled area.",
    "Expert review and reference measurements are required for materials decisions.",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize_mask(path: Path) -> dict:
    before = sha256(path)
    with Image.open(path) as im:
        if im.format != "PNG" or im.mode != "RGB":
            raise ValueError(f"Expected an RGB PNG: {path} ({im.format}, {im.mode})")
        width, height = im.size
        if width <= 0 or height <= 0:
            raise ValueError(f"Empty image: {path}")
        rgb = np.asarray(im)
    counts = np.array([np.count_nonzero(np.all(rgb == color, axis=2))
                       for color in PALETTE], dtype=np.int64)
    total = width * height
    if int(counts.sum()) != total:
        valid = np.zeros((height, width), dtype=bool)
        for color in PALETTE:
            valid |= np.all(rgb == color, axis=2)
        examples = np.unique(rgb[~valid], axis=0)[:8].tolist()
        raise ValueError(f"Invalid mask colors in {path}: examples={examples}, "
                         f"invalid_pixels={total - int(counts.sum())}")
    if sha256(path) != before:
        raise RuntimeError(f"Source changed during reading: {path}")
    return {
        "file": path.name,
        "sha256": before,
        "width_pixels": width,
        "height_pixels": height,
        "total_pixels": total,
        "class_pixel_counts": dict(zip(CLASSES, map(int, counts))),
        "predicted_area_percent": dict(zip(CLASSES, map(float, counts / total * 100))),
    }


def make_plot(rows: list[dict], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.subplots_adjust(left=0.285, right=0.975, top=0.79, bottom=0.28)
    y = np.arange(len(rows))
    left = np.zeros(len(rows))
    for cls, color in zip(CLASSES, PALETTE / 255):
        values = np.array([r["predicted_area_percent"][cls] for r in rows])
        ax.barh(y, values, left=left, label=cls, color=color, height=0.70,
                edgecolor="white", linewidth=0.5)
        for index, (start, value) in enumerate(zip(left, values)):
            if value >= 5:
                ax.text(start + value / 2, index, f"{value:.1f}", ha="center",
                        va="center", fontsize=7.5, color="white" if cls in ("Al", "Al3Ni") else "#13202a")
        left += values
    ax.set_yticks(y, [r["file"].removesuffix("_mask.png") for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_xlabel("Predicted 2D area fraction (%)", labelpad=6)
    ax.grid(axis="x", color="#e3e8ee", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0, pad=5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.suptitle("Phase area fractions from test masks", x=0.04, y=0.98,
                 ha="left", fontsize=13, fontweight="bold", color="#172c49")
    fig.text(0.04, 0.905, f"{len(rows)} existing predictions | Original mask colors",
             fontsize=9, color="#536477")
    ax.legend(ncol=4, loc="lower left", bbox_to_anchor=(0, 1.015), frameon=False,
              borderaxespad=0, handlelength=1.1, columnspacing=1.0, fontsize=8)
    fig.text(0.04, 0.035,
             f"Mask pixels / total pixels; exact four-color validation passed for all {len(rows)} images.\n"
             "Predicted area is not measured composition, volume fraction, modulus or accuracy.\n"
             "No test labels or physical pixel scale; expert review is required.",
             fontsize=7.5, color="#536477", linespacing=1.65)
    fig.savefig(out_dir / "phase_area_fractions.png", dpi=300, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--overwrite", action="store_true",
                        help="Explicitly permit replacing this tool's existing output files")
    args = parser.parse_args()
    if not args.mask_dir.is_dir():
        parser.error("--mask-dir must be an existing directory")
    if args.out_dir.resolve() == args.mask_dir.resolve():
        parser.error("--out-dir must differ from --mask-dir to preserve source masks")
    paths = sorted((p for p in args.mask_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"),
                   key=lambda p: p.name)
    if not paths:
        parser.error("No PNG masks found")
    if args.expected_count is not None and len(paths) != args.expected_count:
        parser.error(f"Expected {args.expected_count} masks; found {len(paths)}")
    output_names = ["phase_area_summary.json"]
    if args.plot:
        output_names += ["phase_area_fractions.png"]
    for name in output_names:
        target = args.out_dir / name
        if target.exists() and not args.overwrite:
            parser.error(f"Output exists: {target}; use --overwrite explicitly")
        if target.is_symlink():
            parser.error(f"Refusing symlink output: {target}")
    rows = [summarize_mask(path) for path in paths]
    totals = {cls: sum(r["class_pixel_counts"][cls] for r in rows) for cls in CLASSES}
    total_pixels = sum(r["total_pixels"] for r in rows)
    assert sum(totals.values()) == total_pixels
    for row in rows:
        assert abs(sum(row["predicted_area_percent"].values()) - 100) < 1e-10
    try:
        source_reference = args.mask_dir.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        source_reference = str(args.mask_dir)
    result = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_mask_directory": source_reference,
        "script_sha256": sha256(Path(__file__)),
        "quantity": "predicted_2d_area_fraction",
        "formula": "100 * class_pixel_count / total_image_pixels",
        "mask_count": len(rows),
        "palette_rgb": dict(zip(CLASSES, PALETTE.tolist())),
        "all_masks_exact_four_color_valid": True,
        "images": rows,
        "pooled_digital_pixel_counts": totals,
        "pooled_digital_pixel_percent": {cls: 100 * count / total_pixels for cls, count in totals.items()},
        "image_mean_area_percent": {cls: float(np.mean([r["predicted_area_percent"][cls] for r in rows]))
                                    for cls in CLASSES},
        "limitations": LIMITATIONS,
        "source_masks_unchanged": True,
    }
    for path, row in zip(paths, rows):
        if sha256(path) != row["sha256"]:
            raise RuntimeError(f"Source changed before output: {path}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.plot:
        make_plot(rows, args.out_dir)
    target = args.out_dir / "phase_area_summary.json"
    with target.open("w" if args.overwrite else "x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"mask_count": len(rows), "total_pixels": total_pixels,
                      "output": str(target), "source_masks_unchanged": True}))


if __name__ == "__main__":
    main()
