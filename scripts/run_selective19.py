"""Run a CPU-only selective-blending review, then wait for the crop18 review.

This independent companion never modifies or launches training jobs. It runs one
fixed rule on existing OOF probabilities and later on both completed crop arms.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/review19_selective"
CROP = ROOT / "runs/review18_crop"
EVALUATOR = ROOT / "scripts/review_selective19.py"
SOURCES = (
    "scripts/run_selective19.py", "scripts/review_selective19.py",
    "scripts/review_crop18.py", "src/data.py", "src/infer.py",
    "src/review_grains15.py", "splits/folds5.json",
)
ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": "", **{
    key: "2" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                          "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}}
CHILD = None


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def state(stage, **extra):
    value = {"time": now(), "pid": os.getpid(), "stage": stage,
             "training_started": False, "gpu_inference_started": False,
             "submission_changed": False, **extra}
    save(OUT / "state.json", value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def preflight():
    hashes = {name: digest(ROOT / name) for name in SOURCES}
    old = read(ROOT / "runs/review17_tiles/all.done")
    assert not old["training"] and not old["submission_changed"]
    fold_of = read(ROOT / "splits/folds5.json")
    assert len(fold_of) == 90 and set(fold_of.values()) == set(range(5))
    roots = (ROOT / "runs/oof/x2_mit_bw_r90",
             ROOT / "runs/oof/x3_unetpp_bw_e120_r90",
             ROOT / "runs/review17_tiles/oof")
    for folder in roots:
        assert all((folder / f"{name}.npy").is_file() for name in fold_of), str(folder)
    crop_plan = read(CROP / "plan.json")
    assert len(crop_plan["jobs"]) == 4 and crop_plan["pilot_folds"] == [0, 1]
    assert crop_plan["automatic_remaining_folds"] == []
    assert not crop_plan["further_training_scheduled"]
    return {"source_sha256": hashes, "crop_plan_sha256": digest(CROP / "plan.json"),
            "fixed_policy": {"selection": "sub6 top1 minus top2 probability < 0.15",
                             "threshold": 0.15,
                             "selected_probability": "(x2_MiT + x3_UNetpp + x4_MiT) / 3",
                             "else_probability": "(x2_MiT + x3_UNetpp) / 2",
                             "uniform_control": "equal three everywhere",
                             "threshold_search": False},
            "stages": {"existing": {"images": 90, "folds": list(range(5))},
                       "crop": {"images": 36, "folds": [0, 1],
                                "arms": ["matched_crop1024", "crop1536"]}},
            "scope": "Stored-probability selective blending only; no ROI reinference or training",
            "cpu_threads": 2, "opencv_threads": 1,
            "submission_changed": False, "automatic_model_replacement": False}


def check_sources(plan):
    for name, expected in plan["source_sha256"].items():
        assert digest(ROOT / name) == expected, f"Source changed: {name}"
    assert digest(CROP / "plan.json") == plan["crop_plan_sha256"]


def run_stage(stage, plan):
    global CHILD
    check_sources(plan)
    output = OUT / stage
    if (output / "results.json").exists():
        raise FileExistsError(f"Refuse to overwrite a completed review: {output}")
    state(f"evaluating_{stage}_cpu")
    with (OUT / f"{stage}.log").open("x", encoding="utf-8") as log:
        CHILD = subprocess.Popen(
            [sys.executable, str(EVALUATOR), "--stage", stage,
             "--out-dir", str(output)], cwd=ROOT, env=ENV,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        code = CHILD.wait(timeout=1800)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"CPU review exceeded 30 minutes: {stage}")
    if code:
        raise RuntimeError(f"CPU review failed: {stage}; exit={code}; see {stage}.log")
    result = read(output / "results.json")
    assert result["stage"] == stage
    assert result["script_sha256"] == plan["source_sha256"]["scripts/review_selective19.py"]
    assert not result["submission_changed"]
    receipt = {"completed_utc": now(), "stage": stage,
               "results_sha256": digest(output / "results.json"),
               "training_started": False, "gpu_inference_started": False}
    save(output / "all.done", receipt)
    CHILD = None
    return receipt


def cleanup():
    if CHILD is not None and CHILD.poll() is None:
        os.killpg(CHILD.pid, signal.SIGTERM)
        try:
            CHILD.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(CHILD.pid, signal.SIGKILL)
            CHILD.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    plan = preflight()
    if args.check:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "driver.lock").open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (OUT / "started.json").exists():
            raise FileExistsError("This companion was already started; inspect its state before rerunning")
        os.nice(10)
        plan["created_utc"] = now()
        save(OUT / "plan.json", plan)
        save(OUT / "started.json", {"time": now(), "pid": os.getpid()})

        def interrupted(signum, frame):
            raise RuntimeError(f"Companion received signal {signum}")

        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        try:
            existing = run_stage("existing", plan)
            deadline = time.monotonic() + 8 * 3600
            while not (CROP / "all.done").exists():
                if (CROP / "error.json").exists():
                    raise RuntimeError(f"Crop18 failed: {read(CROP / 'error.json')}")
                if time.monotonic() > deadline:
                    raise TimeoutError("Crop18 completion not available within eight hours")
                state("waiting_for_crop18_completion", existing=existing,
                      crop_status=read(CROP / "state.json").get("stage"))
                time.sleep(30)
            cropped = run_stage("crop", plan)
            state("complete_and_stop", existing=existing, crop=cropped)
            save(OUT / "all.done", {"time": now(), "existing": existing, "crop": cropped,
                                    "training_started": False, "submission_changed": False})
        except BaseException as exc:
            cleanup()
            state("error", error=repr(exc))
            save(OUT / "error.json", {"time": now(), "error": repr(exc)})
            raise


if __name__ == "__main__":
    main()
