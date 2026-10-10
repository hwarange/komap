"""Optional paired six-epoch continuation; preparation never launches training.

Only --run launches the two fixed fold0 jobs. No automatic expansion/adoption.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from unittest.mock import patch

sys.dont_write_bytecode = True
import run_crop18 as common

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/review26_final_strengthening/finetune_plan'
BASE = 'x3_unetpp_bw_e120_r90'
MIT = 'x4_mit_c1536_bw_r90'
SOURCES = ['src/train.py', 'src/data.py', 'src/infer.py', 'src/review_grains15.py',
           'scripts/run_crop18.py', 'scripts/finetune_final26.py']
ENV = {**common.ENV, 'PYTHONDONTWRITEBYTECODE': '1'}
CHILDREN = []


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    return value


def save(name, value):
    path = OUT / name
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def relative(path):
    return Path(path).relative_to(ROOT).as_posix()


def pin(paths):
    return {relative(path): common.sha256(path) for path in paths}


def verify(records):
    for name, digest in records.items():
        assert common.sha256(common.under_root(name)) == digest, f'Pinned input changed: {name}'


def status(stage, **fields):
    record = dict(time=common.now(), pid=os.getpid(), stage=stage, fixed_holdout_fold=0,
                  training_images=72, validation_images=18, epochs=6,
                  repeated_cross_validation=False, submission_changed=False,
                  further_training_scheduled=False, **fields)
    save('state.json', record)
    print(json.dumps(clean(record), ensure_ascii=False), flush=True)


def no_network(*args, **kwargs):
    raise RuntimeError('Cached initialization required; network disabled')


def actual_parser(values):
    import train
    captured = {}
    original = argparse.ArgumentParser.parse_args

    class Captured(Exception):
        pass

    def intercept(parser, *args, **kwargs):
        captured.update(vars(original(parser, common.cli_args(values))))
        raise Captured()

    with patch.object(argparse.ArgumentParser, 'parse_args', intercept):
        try:
            train.main()
        except Captured:
            pass
    assert all(captured.get(k) == v for k, v in values.items())
    assert captured and len(captured) == 56
    return captured


def prepare():
    """CPU-only inspection and GT train-only map creation; no GPU allocation."""
    import cv2
    import numpy as np
    import torch
    from data import load_cv
    from train import build_from_args

    assert not OUT.exists(), f'Refusing to overwrite preparation: {OUT}'
    source_dir = ROOT / f'runs/cv/{BASE}/fold0'
    source_args = read(source_dir / 'args.json')
    expected = dict(arch='UnetPlusPlus', encoder='resnet34', epochs=120, fold=0,
                    width=3840, crop=1536, tile=1536, bs=4, spi=8, bw=2.0,
                    rot='rot90', seed=0, amp=True, enc_lr_mult=1.0,
                    full=False, init='', hard_crop=0.0)
    assert all(source_args.get(k) == v for k, v in expected.items())
    fold_of = read(ROOT / 'splits/folds5.json')
    train_items, valid_items = load_cv('data', 0, 'splits/folds5.json')
    train_names = sorted(it['name'] for it in train_items)
    valid_names = sorted(it['name'] for it in valid_items)
    assert len(train_names) == 72 and len(valid_names) == 18 and len(fold_of) == 90
    assert set(train_names).isdisjoint(valid_names)
    assert {it['alloy'] for it in train_items}.isdisjoint({it['alloy'] for it in valid_items})
    assert all(fold_of[n] == 0 for n in valid_names)
    checkpoint = source_dir / 'last.pt'
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert ckpt['epoch'] == 120 and ckpt['args'] == source_args
    # The shared trainer uses strict=False; prove complete key/shape equality here.
    with patch.object(socket.socket, 'connect', no_network), patch.object(socket, 'create_connection', no_network):
        model = build_from_args(source_args, weights='imagenet').cpu()
    own, weights = model.state_dict(), ckpt['model']
    assert own.keys() == weights.keys()
    assert all(own[k].shape == v.shape and own[k].dtype == v.dtype for k, v in weights.items())
    assert all(torch.isfinite(v).all().item() for v in weights.values() if v.is_floating_point())
    model.load_state_dict(weights, strict=True)
    tensor_count = len(weights)
    del ckpt, model, own, weights
    diagnostic = ROOT / 'runs/review26_final_strengthening/diagnostics/results.json'
    assert read(diagnostic)['images'] == 90
    historical = ROOT / 'runs/review9/upp_hard.txt'
    args = dict(source_args)
    args.update(epochs=6, lr=3e-5, workers=4, accum_steps=1, scale_jitter=0.15,
                scale_multiplier=1.0, val_every=6, init=relative(checkpoint),
                init_skip_head=False, hard_maps=relative(OUT / 'train_interface_maps'))
    jobs = []
    for arm, probability in [('control', 0.0), ('interface', 0.35)]:
        values = actual_parser(dict(args, hard_crop=probability,
                                   out=relative(OUT / arm), save_oof=relative(OUT / 'oof' / arm)))
        assert not common.under_root(values['out']).exists()
        jobs.append(dict(id=arm, arm=arm, fold=0, args=values, validation_names=valid_names))
    differences = {k for k in jobs[0]['args'] if jobs[0]['args'][k] != jobs[1]['args'][k]}
    assert differences == {'out', 'save_oof', 'hard_crop'}
    paths = [ROOT / 'splits/folds5.json', checkpoint, source_dir / 'args.json',
             source_dir / 'result.json', diagnostic, historical]
    paths += [ROOT / name for name in SOURCES]
    for it in train_items + valid_items:
        paths.extend([ROOT / f"data/{it['split']}/images/{it['name']}_image.png",
                      ROOT / f"data/{it['split']}/masks/{it['name']}_mask.png"])
    for name in valid_names:
        paths += [ROOT / f'runs/oof/{experiment}/{name}.npy' for experiment in [BASE, MIT]]
    inputs = pin(paths)
    OUT.mkdir(parents=True)
    maps = OUT / 'train_interface_maps'
    maps.mkdir()
    map_rows = []
    from PIL import Image
    for it in train_items:
        label = it['lab']
        a, e = (label == 1).astype(np.uint8), (label == 3).astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        focus = ((a > 0) & (cv2.dilate(e, kernel) > 0)) | ((e > 0) & (cv2.dilate(a, kernel) > 0))
        path = maps / f"{it['name']}.png"
        Image.fromarray(focus.astype(np.uint8) * 255).save(path)
        assert path.stem in train_names and path.stem not in valid_names
        inputs[relative(path)] = common.sha256(path)
        map_rows.append(dict(name=it['name'], selected_pixels=int(focus.sum()), total_pixels=int(focus.size),
                             al3ni_side=int((focus & (label == 1)).sum()),
                             eutectic_side=int((focus & (label == 3)).sum())))
    assert len(list(maps.glob('*.png'))) == 72
    plan = dict(created=common.now(), jobs=jobs, input_hashes=inputs,
                source_hashes=pin([ROOT / name for name in SOURCES]),
                initial_checkpoint=relative(checkpoint), strict_initialization_tensors=tensor_count,
                train_names=train_names, validation_names=valid_names, split_sha256=common.sha256(ROOT / 'splits/folds5.json'),
                focus_definition='(GT==Al3Ni & dilate(GT==EutecticSi,5x5)) OR (GT==EutecticSi & dilate(GT==Al3Ni,5x5))',
                focus_probability=0.35, radius_original_pixels=2, map_rows=map_rows,
                map_images_with_focus=sum(row['selected_pixels'] > 0 for row in map_rows),
                empty_map_behavior='uniform crop, as implemented by existing CropDataset',
                map_uses_validation_labels=False, map_uses_prediction_errors=False,
                paired_difference='only crop center sampling policy; same seed, not identical augmentation RNG draws',
                initial_training_epochs=120, continuation_epochs=6, lr=3e-5,
                optimizer_state_resumed=False, new_adamw_and_cosine_schedule=True,
                schedule='one warmup epoch then cosine to zero; last continuation epoch6 only',
                effective_batch=4, optimizer_updates_per_epoch=144, total_updates=864,
                evaluation='fixed existing fold0 18 previously examined images; equal pair with unchanged MiT4 OOF',
                recipes=['baseline_UNet3', 'control_UNet3', 'interface_UNet3',
                         'baseline_pair', 'control_pair', 'interface_pair'],
                hypothesis='Al3Ni/Eutectic confusion may benefit from brief two-sided interface exposure; uncertain expected benefit',
                prior_failed_recipe='upp_hard: old1x60epoch,50%interior OOF-error focus; full90 delta -0.24pp',
                estimated_parallel_minutes=[10, 15], automatic_remaining_folds=[],
                independent_validation=False, submission_changed=False, further_training_scheduled=False)
    save('plan.json', plan)
    status('prepared_not_started', map_images_with_focus=plan['map_images_with_focus'],
           strict_initialization_tensors=tensor_count, expected_minutes=[10, 15])


def inventory():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
                                   '--format=csv,noheader,nounits'], text=True, timeout=15)
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                    '--format=csv,noheader,nounits'], text=True, timeout=15)
    busy = {line.split(',')[0].strip() for line in apps.splitlines() if line.strip()}
    rows = []
    for line in raw.splitlines():
        index, uuid, memory, use = [part.strip() for part in line.split(',')]
        if int(memory) < 1000 and int(use) < 10 and uuid not in busy:
            rows.append(dict(index=int(index), uuid=uuid))
    return rows


def completed(job):
    import torch
    target = common.under_root(job['args']['out'])
    rows = common.read_epoch_logs(target / 'log.jsonl')
    assert [row['epoch'] for row in rows] == list(range(1, 7))
    assert all(math.isfinite(row['loss']) and row['microbatches'] == 144 and row['optimizer_updates'] == 144
               and row['effective_batch'] == 4 and row['accum_steps'] == 1 for row in rows)
    assert read(target / 'args.json') == job['args']
    result = read(target / 'result.json')
    assert result['args'] == job['args'] and sorted(row['name'] for row in result['rows']) == job['validation_names']
    ckpt = torch.load(target / 'last.pt', map_location='cpu', weights_only=False)
    assert ckpt['epoch'] == 6 and ckpt['args'] == job['args']
    assert all(torch.isfinite(v).all().item() for v in ckpt['model'].values() if v.is_floating_point())
    del ckpt
    paths = [target / filename for filename in ['args.json', 'log.jsonl', 'last.pt', 'result.json']]
    paths += [common.under_root(job['args']['save_oof']) / f'{name}.npy' for name in job['validation_names']]
    return dict(id=job['id'], completed=common.now(), train_seconds=sum(row['train_sec'] for row in rows), hashes=pin(paths))


def evaluate(plan, receipts):
    import numpy as np
    from data import CLASSES, load_cv
    from infer import image_iou, postprocess
    from review_grains15 import metrics

    _, items = load_cv('data', 0, 'splits/folds5.json')
    folds = read(ROOT / 'splits/folds5.json')
    rows = {recipe: {} for recipe in plan['recipes']}
    for it in items:
        name = it['name']
        paths = {'mit': ROOT / f'runs/oof/{MIT}/{name}.npy', 'baseline': ROOT / f'runs/oof/{BASE}/{name}.npy'}
        paths.update({job['arm']: common.under_root(job['args']['save_oof']) / f'{name}.npy' for job in plan['jobs']})
        probs = {key: np.load(path).astype(np.float32) for key, path in paths.items()}
        assert all(p.shape == (4, *it['gray'].shape) and np.isfinite(p).all() and p.min() >= 0 and p.max() <= 1
                   and np.max(np.abs(p.sum(0) - 1)) < 0.005 for p in probs.values())
        for arm in ['baseline', 'control', 'interface']:
            for label, p in [(f'{arm}_UNet3', probs[arm]), (f'{arm}_pair', (probs[arm] + probs['mit']) / 2)]:
                rows[label][name] = dict(tta_pp=image_iou(it['lab'], postprocess(p)).tolist(), low=it['low_contrast'])
    scores = {}
    for label, data in rows.items():
        a = np.asarray([data[name]['tta_pp'] for name in sorted(data)], dtype=float) * 100
        scores[label] = dict(mIoU=float(np.nanmean(a, axis=1).mean()), per_class=dict(zip(CLASSES, np.nanmean(a, axis=0).tolist())))
    comparisons = {f'{a}_vs_{b}': metrics(rows[a], rows[b], folds) for a, b in [
        ('control_pair', 'baseline_pair'), ('interface_pair', 'baseline_pair'), ('interface_pair', 'control_pair'),
        ('control_UNet3', 'baseline_UNet3'), ('interface_UNet3', 'baseline_UNet3'), ('interface_UNet3', 'control_UNet3')]}
    for receipt in receipts:
        verify(receipt['hashes'])
    verify(plan['input_hashes'])
    save('per_image_rows.json', rows)
    save('results.json', dict(time=common.now(), images=18, fold=0, scores=scores, comparisons=comparisons,
                             plan_sha256=common.sha256(OUT / 'plan.json'), completion_receipts=receipts,
                             independent_validation=False, submission_changed=False,
                             note='Exploratory single-holdout paired continuation; no full90 claim or automatic adoption/expansion'))
    save('all.done', dict(time=common.now(), outcome='paired_six_epoch_continuation_complete',
                         results_sha256=common.sha256(OUT / 'results.json'),
                         plan_sha256=common.sha256(OUT / 'plan.json'), submission_changed=False,
                         further_training_scheduled=False))


def run():
    assert (OUT / 'plan.json').is_file()
    assert not any((OUT / marker).exists() for marker in ['started.json', 'results.json', 'error.json', 'all.done'])
    plan = read(OUT / 'plan.json')
    verify(plan['input_hashes'])
    first = {row['uuid'] for row in inventory()}
    time.sleep(3)
    gpus = [row for row in inventory() if row['uuid'] in first]
    assert len(gpus) >= 2, 'Two idle GPUs required; existing tasks will not be interrupted'
    (OUT / 'logs').mkdir()
    save('started.json', dict(time=common.now(), pid=os.getpid(), plan_sha256=common.sha256(OUT / 'plan.json'), gpus=gpus[:2]))
    running, receipts = {}, []
    for job, gpu in zip(plan['jobs'], gpus[:2]):
        assert not common.under_root(job['args']['out']).exists()
        command = [sys.executable, str(Path(__file__).resolve()), '--train-worker', *common.cli_args(job['args'])]
        with (OUT / 'logs' / f"{job['id']}.log").open('x', encoding='utf-8') as stream:
            child = subprocess.Popen(command, cwd=ROOT, env={**ENV, 'CUDA_VISIBLE_DEVICES': gpu['uuid']},
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        CHILDREN.append(child)
        running[job['id']] = (job, child, gpu['index'], time.monotonic())
    while running:
        for identifier, values in list(running.items()):
            job, child, gpu, started = values
            assert time.monotonic() - started < 25 * 60, f'Budget overrun: {identifier}'
            logs = common.read_epoch_logs(common.under_root(job['args']['out']) / 'log.jsonl')
            assert all(math.isfinite(row['loss']) for row in logs)
            if child.poll() is None:
                continue
            assert child.returncode == 0, f'{identifier} exited {child.returncode}'
            receipts.append(completed(job))
            save('completed_jobs.json', receipts)
            del running[identifier]
        status('paired_continuation' if running else 'evaluating_cpu',
               jobs=[common.job_progress(*value) for value in running.values()], completed_jobs=receipts)
        if running:
            time.sleep(15)
    evaluate(plan, receipts)
    status('complete_and_stop', results=relative(OUT / 'results.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', action='store_true', help='CPU-only preflight and train-label map generation')
    action.add_argument('--run', action='store_true', help='Explicitly launch the prepared two-job experiment')
    args = parser.parse_args()
    os.chdir(ROOT)
    os.environ.update(common.THREAD_ENV)
    sys.path.insert(0, str(ROOT / 'src'))
    import cv2
    import torch
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if args.prepare:
        prepare()
        return
    with (OUT / 'driver.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def interrupted(signum, frame):
            raise RuntimeError(f'Controller received signal {signum}')

        signal.signal(signal.SIGTERM, interrupted)
        common.CHILDREN = CHILDREN
        try:
            run()
        except BaseException as error:
            common.cleanup_children()
            status('error', message=repr(error))
            save('error.json', dict(time=common.now(), error=repr(error)))
            raise


if __name__ == '__main__':
    if sys.argv[1:2] == ['--train-worker']:
        common.train_worker()
    else:
        main()
