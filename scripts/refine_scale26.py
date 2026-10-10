"""Fixed, inference-only low-resolution scale check; never changes submission files.

prepare -> run (four idle GPUs, each owned worker joins) -> evaluate.
Workers reuse each held-out fold's MiT checkpoint for scale8/6/4. The existing
UNet++ OOF and all 64 non-675px cases remain unchanged. No model is trained.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import uuid

sys.dont_write_bytecode = True
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '2'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/review26_final_strengthening/scale_trials'
sys.path.insert(0, str(ROOT / 'src'))
MIT = 'x4_mit_c1536_bw_r90'
UNET = 'x3_unetpp_bw_e120_r90'
ARMS = {'fresh_scale8': 8, 'cap6': 6, 'cap4': 4}
EXPECTED_SCORE = 80.900653378248
GPUS = {
    0: 'GPU-76da22e2-40da-ebc0-098a-fa92ba030976',
    1: 'GPU-3fc53ad0-271b-ffd6-c83c-84575e82d19c',
    2: 'GPU-e800362c-2a5a-0208-8d32-28107bca6017',
    3: 'GPU-dce5fed1-b1cd-7a95-9c1b-1f281a29f713',
}
SOURCES = ['src/data.py', 'src/infer.py', 'src/train.py', 'src/review_grains15.py',
           'scripts/refine_scale26.py']
CHILDREN = []


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    return value


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def under_root(name):
    path = (ROOT / name).resolve()
    assert path.is_relative_to(ROOT.resolve()), name
    return path


def receipt(path):
    path = Path(path)
    return {'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path), 'bytes': path.stat().st_size}


def verify(record):
    path = under_root(record['path'])
    assert sha(path) == record['sha256'], f'Pinned input changed: {path}'
    size = record.get('bytes', record.get('size_bytes'))
    assert size is None or path.stat().st_size == size, path
    return receipt(path)


def write_new(path, value):
    path = Path(path)
    assert path.resolve().is_relative_to(OUT.resolve()), path
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    # Exclusive creation prevents resumed/racing commands from replacing evidence.
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def extra_scale(width, arm):
    assert arm in ARMS
    if width != 675:
        return 1.0
    base = float(max(1, round(2560 / width))) * 2.0
    assert base == 8.0
    value = ARMS[arm] / base
    assert base * value == ARMS[arm]
    return value


def assign_folds(counts):
    jobs = [{'gpu': gpu, 'uuid': GPUS[gpu], 'folds': [], 'images': 0} for gpu in sorted(GPUS)]
    for fold in sorted(counts, key=lambda f: (-counts[f], f)):
        job = min(jobs, key=lambda row: (row['images'], len(row['folds']), row['gpu']))
        job['folds'].append(fold)
        job['images'] += counts[fold]
    assert sorted(f for job in jobs for f in job['folds']) == list(range(5))
    assert all(job['images'] > 0 for job in jobs)
    return jobs


def items_and_folds():
    import numpy as np
    from data import load_split
    items = load_split(str(ROOT / 'data'), 'train') + load_split(str(ROOT / 'data'), 'valid')
    folds = read(ROOT / 'splits/folds5.json')
    assert len(items) == len(folds) == 90
    assert {item['name'] for item in items} == set(folds)
    groups = {}
    for item in items:
        assert item['lab'] is not None and item['gray'].shape == item['lab'].shape
        assert set(map(int, np.unique(item['lab']))) <= {0, 1, 2, 3}
        groups.setdefault(item['alloy'], set()).add(folds[item['name']])
    assert all(len(values) == 1 for values in groups.values())
    return items, folds


def probability(path, shape):
    import numpy as np
    value = np.load(path, allow_pickle=False).astype(np.float32)
    assert value.shape == (4, *shape) and np.isfinite(value).all(), path
    assert value.min() >= 0 and value.max() <= 1, path
    assert np.max(np.abs(value.sum(0) - 1)) < 0.005, path
    return value


def baseline_rows(items):
    from infer import image_iou, postprocess
    rows = {}
    for item in items:
        name, shape = item['name'], item['gray'].shape
        pm = probability(ROOT / 'runs/oof' / MIT / (name + '.npy'), shape)
        pu = probability(ROOT / 'runs/oof' / UNET / (name + '.npy'), shape)
        rows[name] = {'tta_pp': image_iou(item['lab'], postprocess((pm + pu) / 2)).tolist(),
                      'low': bool(item['low_contrast'])}
    return rows


def score(rows):
    import numpy as np
    return float(np.nanmean(np.asarray([r['tta_pp'] for r in rows.values()], float), axis=1).mean() * 100)


def prepare():
    assert not (OUT / 'plan.json').exists(), 'Plan already exists; refusing overwrite'
    assert not (OUT / 'started.json').exists()
    prior_dir = ROOT / 'runs/review22_pair'
    done, prior, old_plan = [read(prior_dir / name) for name in ('all.done', 'results.json', 'plan.json')]
    assert done['results_sha256'] == sha(prior_dir / 'results.json')
    assert done['plan_sha256'] == sha(prior_dir / 'plan.json')
    assert abs(prior['scopes']['full90']['scores']['x4_mit_x3_unetpp_pair'] - EXPECTED_SCORE) < 1e-9
    manifest_path = ROOT / 'delivery/final_pair_repro/manifest.json'
    manifest = read(manifest_path)
    data = [verify(row) for row in old_plan['data_receipts']]
    assert len(data) == 181 and sum(row['path'].startswith('data/') for row in data) == 180
    for row in manifest['source_files']:
        verify(row)
    checkpoints = [verify(row) for row in manifest['checkpoints']]
    assert {r['path'] for r in checkpoints} == {
        f'runs/cv/{exp}/fold{fold}/last.pt' for exp in (MIT, UNET) for fold in range(5)}
    historical = prior['source_files']
    oof = [verify(row) for row in historical if row['source'] in ('candidate', 'unet')]
    assert len(oof) == 180 and len({row['path'] for row in oof}) == 180
    sources = [receipt(ROOT / name) for name in SOURCES]
    for row in old_plan['source_receipts']:
        if row['path'] in SOURCES:
            verify(row)
    items, folds = items_and_folds()
    affected = sorted(item['name'] for item in items if item['gray'].shape[1] == 675)
    assert len(affected) == 26
    counts = {fold: sum(folds[name] == fold for name in affected) for fold in range(5)}
    assert all(counts.values())
    baseline = baseline_rows(items)
    base_score = score(baseline)
    assert abs(base_score - EXPECTED_SCORE) < 1e-6, (base_score, EXPECTED_SCORE)
    args_receipts, args_by_fold = [], {}
    for fold in range(5):
        path = ROOT / f'runs/cv/{MIT}/fold{fold}/args.json'
        args = read(path)
        required = dict(arch='Unet', encoder='mit_b2', width=2560, scale_multiplier=2.0,
                        crop=1536, tile=1536, overlap=128, epochs=60, fold=fold, amp=True,
                        folds_file='splits/folds5.json')
        assert all(args.get(key) == value for key, value in required.items()), (fold, required)
        args_receipts.append(receipt(path))
        args_by_fold[str(fold)] = args
    plan = dict(created_utc=now(), nonce=uuid.uuid4().hex,
                experiment='width675_mit_inference_scale_only', training_started=False,
                submission_changed=False, automatic_adoption=False, test_inference=False,
                primary='cap6', secondary='cap4', control='fresh_scale8', arms=ARMS,
                affected_names=affected, affected_images=26, unchanged_images=64,
                fold_counts=counts, fold_of=folds, jobs=assign_folds(counts),
                data_receipts=data, checkpoint_receipts=checkpoints, oof_receipts=oof,
                source_receipts=sources, args_receipts=args_receipts, args_by_fold=args_by_fold,
                prerequisite_receipts=[receipt(prior_dir / n) for n in ('plan.json', 'results.json', 'all.done')]
                                      + [receipt(manifest_path)],
                old_baseline_full90=base_score, reference_score=EXPECTED_SCORE,
                fixed_settings=dict(tile=1536, overlap=128, sliding_batch=4, tta=8,
                                    soft_voting=[0.5, 0.5], postprocess='unchanged',
                                    probability_storage='float16 -> float32',
                                    checkpoint='held-out fold last.pt at epoch60',
                                    cudnn_benchmark=True, matmul_allow_tf32=True, cudnn_allow_tf32=True),
                comparisons=['each arm vs saved baseline', 'cap6 and cap4 vs fresh_scale8'],
                metrics=['full90', 'affected26', 'per_fold', 'per_class_with_valid_counts',
                         'alloy_group_bootstrap_10000', 'per_image_changes', 'drops_below_minus2pp'],
                fixed_review_flags=dict(full90_gain_at_least_pp=0.05, affected_gain_positive=True,
                                        gain_vs_fresh_control_positive=True,
                                        eutectic_nonnegative=True, other_class_floor_pp=-0.10,
                                        no_image_below_pp=-2.0),
                caveats=['Exploratory, same repeatedly examined dataset; no independent test labels.',
                         'Two fixed scale candidates; bootstrap is descriptive and excludes selection uncertainty.',
                         'Width675 is not proof of lower physical resolution or matched field of view.',
                         'Fresh scale8 estimates backend/numerical variation from saved OOF.',
                         'No automatic adoption, report rewrite, mask replacement or new training.'])
    write_new(OUT / 'plan.json', plan)
    write_new(OUT / 'baseline_rows.json', baseline)
    print(json.dumps({'prepared': True, 'plan_sha256': sha(OUT / 'plan.json'),
                      'old_baseline_full90': base_score, 'fold_counts': counts, 'jobs': plan['jobs']}), flush=True)


def validate_plan(plan, all_inputs=True):
    for key in ('source_receipts', 'args_receipts', 'prerequisite_receipts'):
        for row in plan[key]:
            verify(row)
    assert plan['arms'] == ARMS and plan['primary'] == 'cap6' and plan['secondary'] == 'cap4'
    assert plan['submission_changed'] is False and plan['training_started'] is False
    if all_inputs:
        for key in ('data_receipts', 'checkpoint_receipts', 'oof_receipts'):
            for row in plan[key]:
                verify(row)


def inventory():
    lines = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
                                     '--format=csv,noheader,nounits'], text=True, timeout=20)
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                    '--format=csv,noheader,nounits'], text=True, timeout=20)
    busy = {line.split(',')[0].strip() for line in apps.splitlines() if line.strip()}
    rows = {}
    for line in lines.splitlines():
        index, gpu_uuid, memory, util = [part.strip() for part in line.split(',')]
        rows[int(index)] = dict(index=int(index), uuid=gpu_uuid, memory_mib=int(memory), utilization=int(util),
                               idle=int(memory) < 1000 and int(util) < 10 and gpu_uuid not in busy)
    return rows


def worker(gpu, nonce):
    plan = read(OUT / 'plan.json')
    assert nonce == plan['nonce']
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == GPUS[gpu]
    validate_plan(plan, all_inputs=False)
    job = next(row for row in plan['jobs'] if row['gpu'] == gpu)
    started = read(OUT / 'started.json')
    assert started['nonce'] == nonce and started['plan_sha256'] == sha(OUT / 'plan.json')
    worker_out = OUT / f'workers/gpu{gpu}'
    worker_out.mkdir(parents=True, exist_ok=False)
    import cv2
    import numpy as np
    import torch
    from data import scale_for
    from infer import predict_probs
    from train import autocast, build_from_args
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    assert torch.cuda.device_count() == 1
    items, fold_of = items_and_folds()
    lookup = {item['name']: item for item in items}
    timings, outputs, drift = [], [], []
    started_time = time.monotonic()
    for fold in job['folds']:
        relative = f'runs/cv/{MIT}/fold{fold}/last.pt'
        record = next(row for row in plan['checkpoint_receipts'] if row['path'] == relative)
        verify(record)
        checkpoint = torch.load(ROOT / relative, map_location='cpu', weights_only=False)
        args = checkpoint['args']
        assert checkpoint['epoch'] == 60 and args == plan['args_by_fold'][str(fold)]
        assert args['fold'] == fold
        model = build_from_args(args, weights=None)
        model.load_state_dict(checkpoint['model'], strict=True)
        model = model.cuda().eval()
        del checkpoint
        names = sorted(name for name in plan['affected_names'] if fold_of[name] == fold)
        for name in names:
            item = lookup[name]
            assert item['gray'].shape[1] == 675
            baseline = probability(ROOT / 'runs/oof' / MIT / (name + '.npy'), item['gray'].shape)
            for arm, actual_scale in ARMS.items():
                scale = extra_scale(item['gray'].shape[1], arm)
                assert scale_for(675, args['width'], args['scale_multiplier']) * scale == actual_scale
                torch.cuda.synchronize()
                before = time.monotonic()
                with torch.no_grad(), autocast(args['amp']):
                    value = predict_probs(model, item['gray'], args['width'], args['tile'], args['overlap'],
                                          True, 'cuda', extra_scale=scale,
                                          scale_multiplier=args['scale_multiplier'])
                torch.cuda.synchronize()
                elapsed = time.monotonic() - before
                assert value.shape == baseline.shape and np.isfinite(value).all()
                assert value.min() >= 0 and value.max() <= 1
                assert np.max(np.abs(value.sum(0) - 1)) < 0.005
                path = worker_out / arm / (name + '.npy')
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('xb') as stream:
                    np.save(stream, value.astype(np.float16), allow_pickle=False)
                restored = probability(path, item['gray'].shape)
                outputs.append(dict(name=name, fold=fold, arm=arm, actual_scale=actual_scale, **receipt(path)))
                timings.append(dict(name=name, fold=fold, arm=arm, seconds=elapsed))
                if arm == 'fresh_scale8':
                    drift.append(dict(name=name, max_abs_probability_difference=float(np.abs(restored-baseline).max()),
                                      mean_abs_probability_difference=float(np.abs(restored-baseline).mean()),
                                      changed_argmax_pixels=int((restored.argmax(0) != baseline.argmax(0)).sum())))
                print(json.dumps(dict(gpu=gpu, name=name, fold=fold, arm=arm, seconds=round(elapsed, 3))), flush=True)
        del model
        torch.cuda.empty_cache()
    assert len(outputs) == job['images'] * len(ARMS)
    validate_plan(plan, all_inputs=False)
    write_new(worker_out / 'complete.json', dict(nonce=nonce, plan_sha256=sha(OUT / 'plan.json'),
              completed_utc=now(), gpu=gpu, uuid=GPUS[gpu], folds=job['folds'], images=job['images'],
              outputs=outputs, timings=timings, fresh_control_probability_drift=drift,
              elapsed_seconds=time.monotonic()-started_time, torch_version=torch.__version__,
              cuda_version=torch.version.cuda, gpu_name=torch.cuda.get_device_name(0),
              training_started=False, submission_changed=False))


def arm_summary(rows, fold_of):
    import numpy as np
    from data import CLASSES
    names = sorted(rows)
    values = np.asarray([rows[name]['tta_pp'] for name in names], float) * 100
    miou = np.nanmean(values, axis=1)
    return dict(images=len(names), mIoU=float(miou.mean()),
                class_IoU=dict(zip(CLASSES, np.nanmean(values, axis=0).tolist())),
                class_valid_images=dict(zip(CLASSES, np.isfinite(values).sum(0).tolist())),
                per_fold={str(fold): float(miou[[fold_of[name] == fold for name in names]].mean())
                          for fold in sorted({fold_of[name] for name in names})})


def compare(rows, baseline, fold_of):
    import numpy as np
    from data import CLASSES
    from review_grains15 import metrics
    result = metrics(rows, baseline, fold_of, repeats=10000)
    changes = {name: float(100 * (np.nanmean(row['tta_pp']) - np.nanmean(baseline[name]['tta_pp'])))
               for name, row in rows.items()}
    result['worst10'] = [{'name': name, 'delta_pp': delta} for name, delta in sorted(changes.items(), key=lambda x: x[1])[:10]]
    result['drops_below_minus2pp'] = {name: delta for name, delta in changes.items() if delta < -2}
    result['per_image_delta_pp'] = changes
    names = sorted(rows)
    a = np.asarray([rows[name]['tta_pp'] for name in names], float)
    b = np.asarray([baseline[name]['tta_pp'] for name in names], float)
    result['paired_class_valid_images'] = dict(zip(CLASSES, (np.isfinite(a) & np.isfinite(b)).sum(0).tolist()))
    return result


def evaluate():
    import cv2
    import numpy as np
    import torch
    from infer import image_iou, postprocess
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    plan = read(OUT / 'plan.json')
    assert not (OUT / 'all.done').exists() and not (OUT / 'results.json').exists()
    validate_plan(plan)
    start = read(OUT / 'started.json')
    assert start['nonce'] == plan['nonce'] and start['plan_sha256'] == sha(OUT / 'plan.json')
    items, fold_of = items_and_folds()
    original = baseline_rows(items)
    assert abs(score(original) - EXPECTED_SCORE) < 1e-6
    # Check saved preflight rows, including NaN locations, without relying on JSON NaN.
    prepared = read(OUT / 'baseline_rows.json')
    for name in original:
        assert original[name]['low'] == prepared[name]['low']
        np.testing.assert_allclose(original[name]['tta_pp'], np.asarray(prepared[name]['tta_pp'], float), rtol=0, atol=0, equal_nan=True)
    rows = {'saved_baseline': original, **{arm: dict(original) for arm in ARMS}}
    records, worker_receipts, timings, drift = {}, [], [], []
    for job in plan['jobs']:
        path = OUT / f"workers/gpu{job['gpu']}/complete.json"
        completed = read(path)
        assert completed['nonce'] == plan['nonce'] and completed['plan_sha256'] == sha(OUT / 'plan.json')
        assert completed['uuid'] == job['uuid'] and completed['folds'] == job['folds']
        assert completed['images'] == job['images'] and len(completed['outputs']) == job['images'] * 3
        worker_receipts.append(receipt(path))
        timings.extend(completed['timings'])
        drift.extend(completed['fresh_control_probability_drift'])
        for record in completed['outputs']:
            key = (record['arm'], record['name'])
            assert key not in records and record['fold'] == fold_of[record['name']]
            assert record['actual_scale'] == ARMS[record['arm']]
            verify(record)
            records[key] = record
    assert set(records) == {(arm, name) for arm in ARMS for name in plan['affected_names']}
    for item in items:
        name = item['name']
        if name not in plan['affected_names']:
            continue
        pu = probability(ROOT / 'runs/oof' / UNET / (name + '.npy'), item['gray'].shape)
        for arm in ARMS:
            pm = probability(under_root(records[(arm, name)]['path']), item['gray'].shape)
            rows[arm][name] = {'tta_pp': image_iou(item['lab'], postprocess((pm + pu) / 2)).tolist(),
                               'low': bool(item['low_contrast'])}
    scopes = {}
    for scope, names in [('full90', set(fold_of)), ('affected26', set(plan['affected_names']))]:
        selected = {arm: {name: value for name, value in values.items() if name in names} for arm, values in rows.items()}
        scopes[scope] = dict(arms={arm: arm_summary(values, fold_of) for arm, values in selected.items()},
                              comparisons_vs_saved={arm: compare(selected[arm], selected['saved_baseline'], fold_of) for arm in ARMS},
                              comparisons_vs_fresh={arm: compare(selected[arm], selected['fresh_scale8'], fold_of) for arm in ('cap6', 'cap4')})
    unchanged = set(fold_of) - set(plan['affected_names'])
    assert len(unchanged) == 64
    for arm in ARMS:
        for name in unchanged:
            np.testing.assert_allclose(rows[arm][name]['tta_pp'], original[name]['tta_pp'], rtol=0, atol=0, equal_nan=True)
    flags = {}
    for arm in ('cap6', 'cap4'):
        full = scopes['full90']['comparisons_vs_saved'][arm]
        affected = scopes['affected26']['comparisons_vs_saved'][arm]
        fresh = scopes['affected26']['comparisons_vs_fresh'][arm]
        delta = full['delta_per_class']
        flags[arm] = dict(full90_gain_at_least_0_05pp=full['delta'] >= 0.05,
                          affected_gain_positive=affected['delta'] > 0,
                          gain_vs_fresh_control_positive=fresh['delta'] > 0,
                          eutectic_nonnegative=delta['Eutectic Si'] >= 0,
                          other_class_drop_at_most_0_10pp=all(delta[c] >= -0.1 for c in ('Al', 'Al3Ni', 'Primary Si')),
                          no_image_drop_below_minus2pp=not full['drops_below_minus2pp'])
    validate_plan(plan)
    write_new(OUT / 'per_image_rows.json', rows)
    result = dict(created_utc=now(), nonce=plan['nonce'], plan_sha256=sha(OUT / 'plan.json'),
                  scopes=scopes, diagnostic_review_flags=flags, worker_receipts=worker_receipts,
                  output_receipts=list(records.values()), inference_timings=timings,
                  fresh_control_probability_drift=drift, unchanged64_verified=True,
                  caveats=plan['caveats'], automatic_adoption=False,
                  training_started=False, submission_changed=False, test_inference=False)
    write_new(OUT / 'results.json', result)
    write_new(OUT / 'all.done', dict(completed_utc=now(), nonce=plan['nonce'],
              outcome='fixed_low_resolution_inference_scale_review_complete',
              plan_sha256=sha(OUT / 'plan.json'), results_sha256=sha(OUT / 'results.json'),
              rows_sha256=sha(OUT / 'per_image_rows.json'), images=90, affected_images=26,
              automatic_adoption=False, training_started=False, submission_changed=False))
    print(json.dumps({'completed': True, 'full90': scopes['full90']['arms'], 'review_flags': flags}), flush=True)


def stop_owned(signum=None, frame=None):
    for child in CHILDREN:
        if child.poll() is None:
            try:
                child.terminate()
            except ProcessLookupError:
                pass
    if signum is not None:
        raise KeyboardInterrupt(f'Controller received signal {signum}; terminated owned children only')


def run():
    # Linux server lock is released by OS on exit; never touches sibling diagnostics.
    import fcntl
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = read(OUT / 'plan.json')
        assert not (OUT / 'started.json').exists() and not (OUT / 'error.json').exists()
        validate_plan(plan)
        first = inventory()
        time.sleep(3)
        second = inventory()
        for gpu, expected in GPUS.items():
            assert first[gpu]['uuid'] == second[gpu]['uuid'] == expected
            assert first[gpu]['idle'] and second[gpu]['idle'], f'GPU {gpu} is busy'
        write_new(OUT / 'started.json', dict(started_utc=now(), nonce=plan['nonce'], pid=os.getpid(),
                  plan_sha256=sha(OUT / 'plan.json'), gpus=list(second.values()), training_started=False))
        signal.signal(signal.SIGTERM, stop_owned)
        signal.signal(signal.SIGINT, stop_owned)
        launched = time.monotonic()
        try:
            for job in plan['jobs']:
                with (OUT / f"gpu{job['gpu']}.log").open('x', encoding='utf-8') as log:
                    command = [sys.executable, str(Path(__file__).resolve()), 'worker', '--gpu', str(job['gpu']), '--nonce', plan['nonce']]
                    child = subprocess.Popen(command, cwd=ROOT,
                              env={**os.environ, 'CUDA_VISIBLE_DEVICES': job['uuid']},
                              stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                CHILDREN.append(child)
            while any(child.poll() is None for child in CHILDREN):
                assert time.monotonic()-launched < 3600, 'Inference exceeded one hour'
                time.sleep(5)
            codes = [child.returncode for child in CHILDREN]
            assert codes == [0] * len(plan['jobs']), f'Workers failed: {codes}'
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            evaluate()
        except BaseException:
            stop_owned()
            raise


def self_test():
    # CPU-only contract checks; does not import torch or touch the real output directory.
    assert [extra_scale(675, arm) for arm in ARMS] == [1.0, 0.75, 0.5]
    # Compile the actual pure scale function without importing torch/CUDA.
    import ast
    tree = ast.parse((ROOT / 'src/data.py').read_text(encoding='utf-8'))
    definition = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'scale_for')
    namespace = {'math': math}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), 'data.scale_for', 'exec'), namespace)
    actual = namespace['scale_for']
    assert all(actual(675, 2560, 2) * extra_scale(675, arm) == scale for arm, scale in ARMS.items())
    for width in (674, 676, 1096, 1230, 1278, 1440):
        assert all(extra_scale(width, arm) == 1 for arm in ARMS)
    counts = {0: 6, 1: 4, 2: 5, 3: 6, 4: 5}
    jobs = assign_folds(counts)
    assert sum(job['images'] for job in jobs) == 26
    assert all(sum(counts[fold] for fold in job['folds']) == job['images'] for job in jobs)
    assert clean({'x': [float('nan'), float('inf'), 1.0]}) == {'x': [None, None, 1.0]}
    original_out = globals()['OUT']
    with tempfile.TemporaryDirectory() as directory:
        globals()['OUT'] = Path(directory)
        try:
            path = Path(directory) / 'exclusive.json'
            write_new(path, {'nonce': 'fixture'})
            digest = sha(path)
            try:
                write_new(path, {'nonce': 'overwritten'})
            except FileExistsError:
                pass
            else:
                raise AssertionError('Overwrite protection failed')
            assert sha(path) == digest and read(path)['nonce'] == 'fixture'
            try:
                write_new(Path(directory).parent / 'outside.json', {})
            except AssertionError:
                pass
            else:
                raise AssertionError('Output scope guard failed')
        finally:
            globals()['OUT'] = original_out
    print(json.dumps({'self_test': 'passed', 'gpu_used': False,
                      'checks': ['exact scale factors', '675-only gate', 'balanced exhaustive fold assignment',
                                 'finite JSON', 'exclusive writes', 'output scope']}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'run', 'worker', 'evaluate', 'verify', 'self-test'])
    parser.add_argument('--gpu', type=int, choices=list(GPUS))
    parser.add_argument('--nonce')
    args = parser.parse_args()
    os.chdir(ROOT)
    try:
        if args.mode == 'self-test':
            self_test()
        elif args.mode == 'prepare':
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            prepare()
        elif args.mode == 'run':
            run()
        elif args.mode == 'worker':
            assert args.gpu is not None and args.nonce
            worker(args.gpu, args.nonce)
        elif args.mode == 'evaluate':
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            evaluate()
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            validate_plan(read(OUT / 'plan.json'))
            print(json.dumps({'verified': True, 'plan_sha256': sha(OUT / 'plan.json'), 'gpu_used': False}))
    except BaseException as exc:
        if args.mode not in ('self-test', 'verify'):
            path = OUT / (f'worker{args.gpu}_error.json' if args.mode == 'worker' else 'error.json')
            if not path.exists():
                write_new(path, dict(time_utc=now(), mode=args.mode, error=repr(exc), traceback=traceback.format_exc(),
                                    training_started=False, submission_changed=False))
        raise


if __name__ == '__main__':
    main()
