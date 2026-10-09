"""Evaluate wider inference tiles on existing x4 MiT checkpoints; never train."""
import argparse
from datetime import datetime, timezone
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/review17_tiles'
EXP = 'x4_mit_bw_r90'
BASE = 'x2_mit_bw_r90'
UNET = 'x3_unetpp_bw_e120_r90'
TILE = 1536
BATCH = 2
sys.path.insert(0, str(ROOT / 'src'))


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, obj):
    def clean(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        return value
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(clean(obj), indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def returned_paths():
    state = json.loads((ROOT / 'runs/runpod_return_20261010/state.json').read_text())
    assert state['stage'] == 'complete' and state['all_sha256_verified']
    receipt = json.loads((ROOT / 'runs/runpod_return_20261010/verified.json').read_text())
    return {r['path']: ROOT / r['stored'] for r in receipt['files']}


def old_probability(exp, name):
    rel = f'runs/oof/{exp}/{name}.npy'
    return returned_paths().get(rel, ROOT / rel)


def prepare_gpu():
    import cv2
    import torch
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return torch


def load_model(fold):
    torch = prepare_gpu()
    from train import build_from_args
    path = ROOT / f'runs/cv/{EXP}/fold{fold}/last.pt'
    ck = torch.load(path, map_location='cpu', weights_only=False)
    a = ck['args']
    assert a['fold'] == fold and a['seed'] == 0 and a['epochs'] == 60
    assert a['crop'] == a['tile'] == 1024 and a['scale_multiplier'] == 2.0
    assert a['width'] == 2560 and a['amp']
    model = build_from_args(a, weights=None)
    model.load_state_dict(ck['model'], strict=True)
    del ck
    return model.cuda().eval(), a


def probability(model, item, a, tile, batch):
    import numpy as np
    import infer
    from train import autocast
    torch = __import__('torch')
    original = infer._sliding
    infer._sliding = functools.partial(original, batch=batch)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    try:
        with torch.inference_mode(), autocast(a['amp']):
            p = infer.predict_probs(model, item['gray'], a['width'], tile, a['overlap'], True,
                                    'cuda', scale_multiplier=a['scale_multiplier'])
    finally:
        infer._sliding = original
    validate(p, item)
    return p, {'seconds': time.monotonic() - started,
               'peak_memory_gb': torch.cuda.max_memory_allocated() / 2**30,
               'min': float(p.min()), 'max': float(p.max()),
               'max_sum_error': float(np.max(np.abs(p.sum(0) - 1)))}


def validate(p, item):
    import numpy as np
    assert p.shape == (4, *item['gray'].shape) and np.isfinite(p).all()
    assert p.min() >= 0 and p.max() <= 1
    assert np.max(np.abs(p.sum(0) - 1)) < .005


def preflight():
    import numpy as np
    from data import load_cv
    from infer import image_iou, postprocess
    returned_paths()
    assert not (OUT / 'all.done').exists()
    model, a = load_model(0)
    _, items = load_cv('data', 0, 'splits/folds5.json')
    item = max(items, key=lambda it: it['gray'].shape[0] * it['gray'].shape[1])
    result = {'created_utc': now(), 'image': item['name'], 'shape': item['gray'].shape,
              'training': False, 'fixed_tile': TILE, 'inference_batch': BATCH, 'cases': {}}
    outputs = {}
    for tile, batch in [(1024, 4), (1024, BATCH), (TILE, BATCH)]:
        p, info = probability(model, item, a, tile, batch)
        key = f'tile{tile}_batch{batch}'
        outputs[key] = p
        info['miou'] = float(np.nanmean(image_iou(item['lab'], postprocess(p))) * 100)
        result['cases'][key] = info
        print(key, json.dumps(info), flush=True)
    stored = np.load(old_probability(EXP, item['name'])).astype(np.float32)
    validate(stored, item)
    reference = outputs['tile1024_batch4']
    result['canonical_baseline_path'] = str(old_probability(EXP, item['name']).relative_to(ROOT))
    result['canonical_vs_rerun_max_probability_diff'] = float(np.max(np.abs(stored - reference)))
    result['canonical_vs_rerun_miou_delta_pp'] = result['cases']['tile1024_batch4']['miou'] - float(np.nanmean(image_iou(item['lab'], postprocess(stored))) * 100)
    result['batch_only_max_probability_diff'] = float(np.max(np.abs(reference - outputs['tile1024_batch2'])))
    result['batch_only_miou_delta_pp'] = result['cases']['tile1024_batch2']['miou'] - result['cases']['tile1024_batch4']['miou']
    assert max(v['peak_memory_gb'] for v in result['cases'].values()) < 42
    save(OUT / 'preflight.json', result)
    print(json.dumps(result), flush=True)


def worker(fold):
    import numpy as np
    from data import load_cv
    from infer import image_iou, postprocess
    returned_paths()
    model, a = load_model(fold)
    _, items = load_cv('data', fold, 'splits/folds5.json')
    assert len(items) == 18
    target = OUT / 'oof'
    target.mkdir(parents=True, exist_ok=True)
    rows = {}
    started = time.monotonic()
    for i, item in enumerate(items, 1):
        path = target / f"{item['name']}.npy"
        if path.exists():
            stored = np.load(path).astype(np.float32)
            validate(stored, item)
            info = {'reused_existing_validated': True}
        else:
            p, info = probability(model, item, a, TILE, BATCH)
            partial = path.with_suffix('.npy.partial')
            with partial.open('wb') as stream:
                np.save(stream, p.astype(np.float16))
            partial.rename(path)
            stored = np.load(path).astype(np.float32)
        validate(stored, item)
        rows[item['name']] = {'tta_pp': image_iou(item['lab'], postprocess(stored)).tolist(),
                              'low': item['low_contrast'], **info}
        status = {'fold': fold, 'pid': os.getpid(), 'stage': 'inference', 'completed_images': i,
                  'total_images': len(items), 'elapsed_seconds': time.monotonic() - started,
                  'latest': item['name'], 'updated_utc': now(), 'training': False}
        save(OUT / f'fold{fold}_state.json', status)
        print(json.dumps(status), flush=True)
    save(OUT / f'fold{fold}_result.json', {'fold': fold, 'tile': TILE, 'batch': BATCH,
                                        'rows': rows, 'elapsed_seconds': time.monotonic() - started})


def compare():
    import numpy as np
    from data import load_split
    from infer import image_iou, postprocess
    from review_grains15 import metrics
    folds = json.loads((ROOT / 'splits/folds5.json').read_text())
    items = load_split('data', 'train') + load_split('data', 'valid')
    all_rows = {k: {} for k in ['sub6', 'old_x4', 'new_x4', 'old_add', 'new_add']}
    for item in items:
        n = item['name']
        probs = [np.load(old_probability(e, n)).astype(np.float32) for e in [BASE, UNET, EXP]]
        pb, pu, po = probs
        pn = np.load(OUT / 'oof' / f'{n}.npy').astype(np.float32)
        for p in [*probs, pn]:
            validate(p, item)
        for key, p in [('sub6', (pb + pu) / 2), ('old_x4', po), ('new_x4', pn),
                       ('old_add', (pb + pu + po) / 3), ('new_add', (pb + pu + pn) / 3)]:
            all_rows[key][n] = {'tta_pp': image_iou(item['lab'], postprocess(p)).tolist(),
                                'low': item['low_contrast']}
    def pair(a, b):
        return {'full': metrics(all_rows[a], all_rows[b], folds),
                'previous_nonpilot': metrics({n: r for n, r in all_rows[a].items() if folds[n] in [2, 3, 4]}, all_rows[b], folds)}
    previous = ROOT / 'runs/review16_additional/seed0_full_results.json'
    historical = json.loads(previous.read_text()) if previous.exists() else None
    result = {'created_utc': now(), 'training': False, 'submission_changed': False,
              'tile1536_vs_tile1024_single': pair('new_x4', 'old_x4'),
              'tile1536_addition_vs_sub6': pair('new_add', 'sub6'),
              'tile1536_addition_vs_tile1024_addition': pair('new_add', 'old_add'),
              'tile1024_addition_vs_sub6': pair('old_add', 'sub6'),
              'baseline': metrics(all_rows['sub6'], all_rows['sub6'], folds),
              'previous_reference_baseline_miou': historical['baseline']['cv_mIoU'] if historical else None,
              'selection_note': 'All 90 images were previously examined. Historical nonpilot folds are no longer untouched. Exploratory fixed inference change, not independent validation. Wider tiles also change blending windows and boundary placement.',
              'recipe': 'x2 MiT 1/3 + x3 UNet++ 1/3 + x4 MiT tile1536 1/3; fixed before inference',
              'preflight': json.loads((OUT / 'preflight.json').read_text())}
    if historical:
        result['baseline_drift_pp'] = result['baseline']['cv_mIoU'] - historical['baseline']['cv_mIoU']
    save(OUT / 'results.json', result)
    save(OUT / 'per_image_rows.json', all_rows)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def controller(resume=False):
    import fcntl
    returned_paths()
    pre = json.loads((OUT / 'preflight.json').read_text())
    assert pre['fixed_tile'] == TILE and pre['inference_batch'] == BATCH
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if resume:
            prior = json.loads((OUT / 'started.json').read_text())
            assert not Path(f"/proc/{prior['pid']}").exists(), 'Prior controller still alive'
            save(OUT / 'previous_started.json', prior)
            if (OUT / 'plan.json').exists():
                save(OUT / 'previous_plan.json', json.loads((OUT / 'plan.json').read_text()))
            if (OUT / 'error.json').exists():
                (OUT / 'error.json').rename(OUT / 'previous_error.json')
        else:
            assert not (OUT / 'started.json').exists(), 'Controller already started'
            assert not (OUT / 'oof').exists(), 'Refuse existing outputs'
        gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
        for line in gpu.splitlines():
            idx, mem, util = [int(v.strip()) for v in line.split(',')]
            if idx in range(4):
                assert mem < 1000 and util < 10, f'GPU {idx} busy'
        checkpoints = [{'fold': f, 'path': f'runs/cv/{EXP}/fold{f}/last.pt',
                         'sha256': digest(ROOT / f'runs/cv/{EXP}/fold{f}/last.pt')} for f in range(5)]
        save(OUT / 'plan.json', {'created_utc': now(), 'folds': list(range(5)), 'tile': TILE,
                               'batch': BATCH, 'epochs_trained': 0, 'training': False,
                               'checkpoints': checkpoints, 'overlap': 128, 'tta': 8,
                               'candidate_weights': [1/3, 1/3, 1/3], 'script_sha256': digest(__file__),
                               'preserve_existing_oof': True, 'prior90_seen': True})
        save(OUT / 'started.json', {'pid': os.getpid(), 'started_utc': now(), 'training': False})
        waiting, active, completed = list(range(5)), {}, []
        try:
            while waiting or active:
                for gpu_id in range(4):
                    if gpu_id not in active and waiting:
                        fold = waiting.pop(0)
                        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu_id), OMP_NUM_THREADS='4',
                                   OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', HF_HUB_OFFLINE='1')
                        log = (OUT / f'fold{fold}.log').open('w')
                        proc = subprocess.Popen([sys.executable, __file__, '--worker', str(fold)], cwd=ROOT,
                                                env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                        active[gpu_id] = (fold, proc, log)
                for gpu_id, (fold, proc, log) in list(active.items()):
                    if proc.poll() is not None:
                        log.close()
                        if proc.returncode:
                            raise RuntimeError(f'Inference fold {fold} failed: {proc.returncode}')
                        assert (OUT / f'fold{fold}_result.json').exists()
                        completed.append(fold)
                        del active[gpu_id]
                save(OUT / 'state.json', {'stage': 'inference', 'pid': os.getpid(), 'completed_folds': completed,
                                         'active': [{'gpu': g, 'fold': f, 'pid': p.pid} for g, (f, p, _) in active.items()],
                                         'waiting': waiting, 'training': False, 'updated_utc': now()})
                time.sleep(5)
            save(OUT / 'state.json', {'stage': 'comparing', 'pid': os.getpid(), 'training': False, 'updated_utc': now()})
            compare()
            save(OUT / 'state.json', {'stage': 'complete', 'pid': os.getpid(), 'training': False, 'updated_utc': now()})
            save(OUT / 'all.done', {'created_utc': now(), 'training': False, 'submission_changed': False})
        except BaseException as e:
            for _, proc, log in active.values():
                if proc.poll() is None:
                    proc.terminate()
                log.close()
            save(OUT / 'error.json', {'error': repr(e), 'created_utc': now(), 'training': False})
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--preflight', action='store_true')
    modes.add_argument('--worker', type=int, choices=range(5))
    modes.add_argument('--controller', action='store_true')
    modes.add_argument('--compare', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.preflight:
        preflight()
    elif args.worker is not None:
        worker(args.worker)
    elif args.controller:
        controller(args.resume)
    else:
        compare()
