"""Fresh train70/valid20 runs on GPU1-3, preserving the independent GPU0 pilot."""
import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/submission31_train70'
SOURCES = ['scripts/run_submission31.py', 'scripts/review_submission31.py',
           'src/train_submission31.py', 'src/train.py', 'src/data.py', 'src/infer.py',
           'src/context30.py', 'src/hybrid30.py', 'src/hybrid29.py', 'scripts/run_crop18.py']
GPUS = {1: 'GPU-3fc53ad0-271b-ffd6-c83c-84575e82d19c',
        2: 'GPU-e800362c-2a5a-0208-8d32-28107bca6017',
        3: 'GPU-dce5fed1-b1cd-7a95-9c1b-1f281a29f713'}
ENV = dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
           NUMEXPR_NUM_THREADS='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
           PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')


def now():
    return datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def save(name, value):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf8')
    temp.replace(path)


def relative(path):
    return Path(path).relative_to(ROOT).as_posix()


def verify(plan):
    for name, digest in plan['input_hashes'].items():
        assert sha(ROOT / name) == digest, name
    for row in plan['image_net_cache']:
        assert sha(row['path']) == row['sha256'], row['path']


def parsed(values):
    import train
    from run_crop18 import cli_args
    original = argparse.ArgumentParser.parse_args
    result = {}
    class Captured(Exception):
        pass
    def capture(parser, *unused, **kwargs):
        result.update(vars(original(parser, cli_args(values))))
        raise Captured()
    with patch.object(argparse.ArgumentParser, 'parse_args', capture):
        try:
            train.main()
        except Captured:
            pass
    assert result and all(result.get(k) == v for k, v in values.items())
    return result


def prepare():
    import cv2
    import numpy as np
    from PIL import Image
    from data import load_split
    assert not (OUT / 'plan.json').exists(), 'Refusing to overwrite a prepared run'
    assert (OUT / 'stop_previous.json').exists(), 'GPU1-3 switch receipt required'
    train_items, valid_items = load_split('data', 'train'), load_split('data', 'valid')
    training_names = sorted(x['name'] for x in train_items)
    validation_names = sorted(x['name'] for x in valid_items)
    assert len(training_names) == 70 and len(validation_names) == 20
    assert set(training_names).isdisjoint(validation_names)
    assert all(x['split'] == 'train' for x in train_items)
    assert all(x['split'] == 'valid' for x in valid_items)
    references = {'mit4': 'runs/cv/x4_mit_c1536_bw_r90/fold0/args.json',
                  'unet3': 'runs/cv/x3_unetpp_bw_e120_r90/fold0/args.json'}
    reference_args = {k: read(ROOT / p) for k, p in references.items()}
    assert reference_args['mit4']['arch'] == 'Unet' and reference_args['mit4']['epochs'] == 60
    assert reference_args['unet3']['arch'] == 'UnetPlusPlus' and reference_args['unet3']['epochs'] == 120
    maps = OUT / 'train_interface_maps'
    assert not maps.exists()
    maps.mkdir(parents=True)
    map_rows = []
    for item in train_items:
        lab = item['lab']; a = (lab == 1).astype(np.uint8); e = (lab == 3).astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        focus = ((a > 0) & (cv2.dilate(e, kernel) > 0)) | ((e > 0) & (cv2.dilate(a, kernel) > 0))
        Image.fromarray(focus.astype(np.uint8) * 255).save(maps / (item['name'] + '.png'))
        map_rows.append(dict(name=item['name'], focus_pixels=int(focus.sum())))
    assert sorted(p.stem for p in maps.glob('*.png')) == training_names
    jobs = []
    for identifier, gpu in [('mit4', 1), ('unet3', 2), ('context4', 3)]:
        values = dict(reference_args['unet3' if identifier == 'unet3' else 'mit4'])
        values.update(fold=-1, train_splits=['train'], val_split='valid', full=False, init='',
                      init_skip_head=False, train_frac=1.0, seed=0, workers=4, max_iters=0,
                      hard_crop=0.0, hard_maps=relative(maps), val_every=5,
                      out=relative(OUT / 'models' / identifier / 'base'),
                      save_oof=relative(OUT / 'valid_probs' / identifier / 'base'))
        if identifier == 'context4':
            values['arch'] = 'GlobalContextMiTCNN30'
        if identifier == 'unet3':
            values.update(epochs=120, bs=4, accum_steps=1, width=3840, scale_multiplier=1.0)
        else:
            values.update(epochs=60, bs=2, accum_steps=2, width=2560, scale_multiplier=2.0)
        values = parsed(values)
        assert values['crop'] == values['tile'] == 1536 and values['spi'] == 8
        assert values['bw'] == 2 and values['rot'] == 'rot90' and values['lr'] == 3e-4
        assert not (ROOT / values['out']).exists()
        job = dict(id=identifier, gpu_index=gpu, gpu_uuid=GPUS[gpu], args=values,
                   training_names=training_names, validation_names=validation_names,
                   microbatches_per_epoch=560 // values['bs'], optimizer_updates_per_epoch=140,
                   command=[str(ROOT / '.venv/bin/python'), 'src/train_submission31.py',
                            '--job', identifier, '--stage', 'base'])
        if identifier == 'unet3':
            continuation = dict(values, epochs=6, lr=3e-5, hard_crop=.35, val_every=6,
                                init=values['out'] + '/last.pt',
                                out=relative(OUT / 'models' / identifier / 'interface6'),
                                save_oof=relative(OUT / 'valid_probs' / identifier / 'interface6'))
            job['continuation'] = dict(args=parsed(continuation), command=[str(ROOT / '.venv/bin/python'),
                    'src/train_submission31.py', '--job', identifier, '--stage', 'continuation'])
        jobs.append(job)
    paths = [ROOT / name for name in SOURCES] + [ROOT / name for name in references.values()]
    for item in train_items + valid_items:
        for kind, suffix in [('images', 'image'), ('masks', 'mask')]:
            paths.append(ROOT / 'data' / item['split'] / kind / (item['name'] + '_' + suffix + '.png'))
    paths += sorted(maps.glob('*.png'))
    cache = []
    for encoder, revision in [('mit_b2', '868ab6f13871dcf8c3d9f90ee4519403475b65ef'),
                              ('resnet34', '7a57b34f723329ff020b3f8bc41771163c519d0c')]:
        base = Path.home() / '.cache/huggingface/hub' / ('models--smp-hub--' + encoder + '.imagenet') / 'snapshots' / revision
        for name in ['config.json', 'model.safetensors']:
            path = base / name
            cache.append(dict(path=str(path), encoder=encoder, revision=revision, sha256=sha(path)))
    old_plan = read(ROOT / 'runs/review29_hybrid_pilot/plan.json')
    protected = {name: digest for name, digest in old_plan['input_hashes'].items() if name.startswith(('src/', 'scripts/'))}
    assert all(sha(ROOT / name) == digest for name, digest in protected.items())
    plan = dict(created=now(), purpose='Fresh submission-condition training on provided70; evaluate provided20',
                jobs=jobs, training_names=training_names, validation_names=validation_names,
                input_hashes={relative(p): sha(p) for p in paths}, image_net_cache=cache,
                gpu0_protected_source_hashes=protected, gpu0_training_unchanged=True,
                training_images=70, validation_images=20, fresh_imagenet=True,
                historical_project_checkpoints_allowed=False,
                continuation_policy='Only this new train70 UNet120 last.pt may initialize its six-epoch continuation',
                map_rows=map_rows, maps_use_only_train70_ground_truth=True,
                shared_alloy_ids=sorted({x['alloy'] for x in train_items} & {x['alloy'] for x in valid_items}),
                independent_validation=False, previously_viewed_validation=True,
                checkpoint_selection='Fixed final epochs, never validation-selected best.pt',
                evaluation='Same20, TTA8, fixed original/cap6 policies and 50:50 two-model ensembles',
                automatic_adoption=False, automatic_test_prediction=False, further_training_scheduled=False,
                maximum_controller_hours=9, expected_total_minutes=[225, 255],
                protected_previous_deliverables=True, test_images_or_labels_used=False)
    save('plan.json', plan)
    save('state.json', dict(time=now(), stage='prepared_not_started', jobs=[j['id'] for j in jobs]))
    print(json.dumps(dict(prepared=True, images=[70, 20], input_hashes=len(plan['input_hashes']),
                         plan_sha256=sha(OUT / 'plan.json'), jobs=[j['id'] for j in jobs])), flush=True)


def logs(args):
    path = ROOT / args['out'] / 'log.jsonl'
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().split('\n')[:-1] if line]


def idle(job):
    line = subprocess.check_output(['nvidia-smi', '-i', job['gpu_uuid'],
            '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip()
    index, uuid, memory, utilization = map(str.strip, line.split(','))
    assert int(index) == job['gpu_index'] and uuid == job['gpu_uuid']
    assert int(memory) < 1000 and int(utilization) < 10, line


def training_complete(job, stage):
    args = job['args'] if stage == 'base' else job['continuation']['args']
    rows = logs(args)
    assert [x['epoch'] for x in rows] == list(range(1, args['epochs'] + 1))
    assert all(math.isfinite(x['loss']) and x['microbatches'] == 560 // args['bs']
               and x['optimizer_updates'] == 140 for x in rows)
    assert read(ROOT / args['out'] / 'args.json') == args
    result = read(ROOT / args['out'] / 'result.json')
    assert result['args'] == args
    assert sorted(x['name'] for x in result['rows']) == job['validation_names']
    assert sorted(p.stem for p in (ROOT / args['save_oof']).glob('*.npy')) == job['validation_names']
    assert (ROOT / args['out'] / 'last.pt').exists()
    return dict(completed=now(), epochs=args['epochs'], checkpoint_sha256=sha(ROOT / args['out'] / 'last.pt'))


def run():
    plan = read(OUT / 'plan.json'); verify(plan)
    lock = (OUT / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not any((OUT / name).exists() for name in ['started.json', 'error.json', 'all.done'])
    for job in plan['jobs']:
        idle(job)
        assert not (ROOT / job['args']['out']).exists()
    children, streams, state = {}, [], {}
    started = time.monotonic()
    def interrupted(*unused):
        raise RuntimeError('Submission31 controller interrupted')
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    def spawn(job, stage):
        identifier = job['id']
        command = (job['command'] if stage == 'base' else job['continuation']['command']) if stage != 'review' else [
                str(ROOT / '.venv/bin/python'), 'scripts/review_submission31.py', '--job', identifier]
        stream = (OUT / (identifier + '_' + stage + '.log')).open('x'); streams.append(stream)
        process = subprocess.Popen(command, cwd=ROOT, env=dict(ENV, CUDA_VISIBLE_DEVICES=job['gpu_uuid']),
                    stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        children[identifier] = process
        receipts = state.get(identifier, {}).get('stage_receipts', {})
        state[identifier] = dict(stage=stage, pid=process.pid, gpu_index=job['gpu_index'],
                                gpu_uuid=job['gpu_uuid'], started=now(), command=command, stage_receipts=receipts)
    try:
        for job in plan['jobs']:
            spawn(job, 'base')
        save('started.json', dict(time=now(), controller_pid=os.getpid(), plan_sha256=sha(OUT / 'plan.json'), jobs=state))
        while not all(s['stage'] in ['complete', 'failed'] for s in state.values()):
            assert time.monotonic() - started < 3600 * plan['maximum_controller_hours']
            for job in plan['jobs']:
                identifier = job['id']; current = state[identifier]; process = children[identifier]
                stage = current['stage']
                if stage in ['complete', 'failed']:
                    continue
                if stage in ['base', 'continuation']:
                    args = job['args'] if stage == 'base' else job['continuation']['args']
                    rows = logs(args)
                    current.update(completed_epochs=len(rows), total_epochs=args['epochs'], latest_epoch=rows[-1] if rows else None)
                    assert all(math.isfinite(x['loss']) for x in rows), identifier
                if process.poll() is None:
                    continue
                if process.returncode != 0:
                    current.update(stage='failed', failed_stage=stage, exit_code=process.returncode, finished=now())
                    continue
                if stage == 'review':
                    assert (OUT / 'reviews' / identifier / 'all.done').exists()
                    current.update(stage='complete', finished=now())
                else:
                    current['stage_receipts'][stage] = training_complete(job, stage)
                    spawn(job, 'continuation' if stage == 'base' and 'continuation' in job else 'review')
            save('state.json', dict(time=now(), controller_pid=os.getpid(), jobs=state,
                                   elapsed_minutes=(time.monotonic() - started) / 60))
            if not all(s['stage'] in ['complete', 'failed'] for s in state.values()):
                time.sleep(15)
        assert all(s['stage'] == 'complete' for s in state.values()), 'One or more jobs failed; inspect logs'
        verify(plan)
        with (OUT / 'aggregate.log').open('x') as stream:
            subprocess.run([str(ROOT / '.venv/bin/python'), 'scripts/review_submission31.py', '--aggregate'],
                           cwd=ROOT, env=dict(ENV, CUDA_VISIBLE_DEVICES=''), stdout=stream,
                           stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, check=True, timeout=900)
        assert (OUT / 'all.done').exists()
        save('state.json', dict(time=now(), stage='complete_and_stop', jobs=state, automatic_adoption=False))
    except BaseException as error:
        save('error.json', dict(time=now(), error=f'{type(error).__name__}: {error}', jobs=state))
        raise
    finally:
        for child in children.values():
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); child.wait()
        for stream in streams:
            stream.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'run'])
    opts = parser.parse_args()
    os.chdir(ROOT); sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
    os.environ.update(ENV); os.environ['CUDA_VISIBLE_DEVICES'] = ''
    prepare() if opts.mode == 'prepare' else run()
