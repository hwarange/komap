"""Four fresh model families on five common alloy-group folds within provided train70."""
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

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/review34_train70_group5fold'
FAMILIES = ('mit2', 'mit4', 'unet3', 'context4')
GPUS = {
    0: 'GPU-76da22e2-40da-ebc0-098a-fa92ba030976',
    1: 'GPU-3fc53ad0-271b-ffd6-c83c-84575e82d19c',
    2: 'GPU-e800362c-2a5a-0208-8d32-28107bca6017',
    3: 'GPU-dce5fed1-b1cd-7a95-9c1b-1f281a29f713',
}
ENV = dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
           NUMEXPR_NUM_THREADS='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
           PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
sys.dont_write_bytecode = True


def now():
    return datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf8')
    temp.replace(path)


def idle(index, uuid):
    line = subprocess.check_output(['nvidia-smi', '-i', uuid,
        '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip()
    got_index, got_uuid, memory, utilization = map(str.strip, line.split(','))
    assert int(got_index) == index and got_uuid == uuid, line
    assert int(memory) < 1000 and int(utilization) < 10, line
    return line


def verify(plan):
    assert len(plan['jobs']) == 20
    assert {(j['family'], j['fold']) for j in plan['jobs']} == {(f, k) for f in FAMILIES for k in range(5)}
    for name, digest in plan['input_hashes'].items():
        path = (ROOT / name).resolve()
        assert path.is_relative_to(ROOT) and not Path(name).is_absolute(), name
        assert sha(path) == digest, name
    for record in plan['image_net_cache']:
        assert sha(record['path']) == record['sha256'], record['path']
    assert sha(ROOT / plan['folds_file']) == plan['fold_assignment_sha256']
    assert set(plan['train70_names']).isdisjoint(plan['official_validation_names'])
    for job in plan['jobs']:
        assert set(job['training_names']).isdisjoint(job['validation_names'])
        assert sorted(job['training_names'] + job['validation_names']) == plan['train70_names']
        assert set(job['training_alloys']).isdisjoint(job['validation_alloys'])


def prepare():
    from data import load_split, make_folds
    from run_submission31 import parsed
    assert not OUT.exists(), 'Refusing to overwrite group34'
    parents = {'mit2': ROOT / 'runs/review32_mit2_train70/plan.json',
               'mit4': ROOT / 'runs/submission31_train70/plan.json'}
    plans = {key: read(path) for key, path in parents.items()}
    for key, path in parents.items():
        done = read(path.parent / 'all.done')
        assert done['plan_sha256'] == sha(path)
        assert done['results_sha256'] == sha(path.parent / 'results.json')
        assert done['rows_sha256'] == sha(path.parent / 'per_image_rows.json')
    originals = {j['id']: j['args'] for j in plans['mit4']['jobs']}
    originals['mit2'] = plans['mit2']['jobs'][0]['args']
    items = load_split('data', 'train')
    names = sorted(it['name'] for it in items)
    assert len(names) == len(set(names)) == 70 and all(it['lab'] is not None for it in items)
    assert names == plans['mit4']['training_names'] == plans['mit2']['training_names']
    official = plans['mit4']['validation_names']
    assert len(official) == len(set(official)) == 20 and set(names).isdisjoint(official)
    assignment = make_folds(items, n_folds=5, seed=0)
    assert set(assignment) == set(names) and set(assignment.values()) == set(range(5))
    for alloy in {it['alloy'] for it in items}:
        assert len({assignment[it['name']] for it in items if it['alloy'] == alloy}) == 1
    for index, uuid in GPUS.items():
        idle(index, uuid)
    folds_path = OUT / 'folds.json'
    jobs, fold_rows = [], []
    reference = {}
    source_paths = list((ROOT / 'src').glob('*.py')) + [
        ROOT / 'scripts/run_group34.py', ROOT / 'scripts/review_group34.py',
        ROOT / 'scripts/run_submission31.py', ROOT / 'scripts/run_crop18.py']
    source_paths += list(parents.values())
    for family in FAMILIES:
        args = originals[family]
        audit_path = ROOT / args['out'] / 'initialization_audit.json'
        audit = read(audit_path)
        parent_path = parents['mit2' if family == 'mit2' else 'mit4']
        assert audit['args'] == args and audit['plan_sha256'] == sha(parent_path)
        assert audit['inherited_checkpoint'] is None and not audit['previous_experiment_checkpoint_used']
        assert audit['fresh_imagenet_state_sha256'] == audit['full_initial_state_sha256']
        reference[family] = dict(path=relative(audit_path), sha256=sha(audit_path),
                                 state_sha256=audit['full_initial_state_sha256'])
        source_paths.append(audit_path)
    assert reference['mit2']['state_sha256'] == reference['mit4']['state_sha256']
    input_paths = source_paths + sorted((ROOT / 'data/train/images').glob('*_image.png')) + sorted(
        (ROOT / 'data/train/masks').glob('*_mask.png'))
    assert len(list((ROOT / 'data/train/images').glob('*_image.png'))) == 70
    assert len(list((ROOT / 'data/train/masks').glob('*_mask.png'))) == 70
    assert all(p.exists() for p in input_paths)
    hashes = {relative(path): sha(path) for path in input_paths}
    for path, digest in hashes.items():
        for parent in plans.values():
            if path in parent['input_hashes']:
                assert digest == parent['input_hashes'][path], f'Changed reference input: {path}'
    for fold in range(5):
        train_items = [it for it in items if assignment[it['name']] != fold]
        val_items = [it for it in items if assignment[it['name']] == fold]
        train_names = sorted(it['name'] for it in train_items)
        val_names = sorted(it['name'] for it in val_items)
        train_alloys = sorted({it['alloy'] for it in train_items})
        val_alloys = sorted({it['alloy'] for it in val_items})
        assert train_names and val_names and set(train_alloys).isdisjoint(val_alloys)
        fold_rows.append(dict(fold=fold, training_images=len(train_names), validation_images=len(val_names),
            training_alloys=train_alloys, validation_alloys=val_alloys,
            low_contrast_train=sum(it['low_contrast'] for it in train_items),
            low_contrast_validation=sum(it['low_contrast'] for it in val_items)))
        for family in FAMILIES:
            base = originals[family]
            args = parsed(dict(base, fold=fold, folds_file=relative(folds_path),
                out=relative(OUT / 'models' / family / f'fold{fold}'),
                save_oof=relative(OUT / 'probs' / family / f'fold{fold}')))
            assert {key for key in args if args[key] != base[key]} == {'fold', 'folds_file', 'out', 'save_oof'}
            assert args['init'] == '' and not args['full'] and args['hard_crop'] == 0
            assert args['epochs'] == (120 if family == 'unet3' else 60)
            micro = len(train_items) * args['spi'] // args['bs']
            job_id = f'{family}_fold{fold}'
            jobs.append(dict(id=job_id, family=family, fold=fold, args=args,
                training_names=train_names, validation_names=val_names,
                training_alloys=train_alloys, validation_alloys=val_alloys,
                microbatches_per_epoch=micro, optimizer_updates_per_epoch=math.ceil(micro / args['accum_steps']),
                command=[str(ROOT / '.venv/bin/python'), 'src/train_group34.py', '--job', job_id]))
    cache = plans['mit4']['image_net_cache']
    assert cache == plans['mit2']['image_net_cache']
    for record in cache:
        assert sha(record['path']) == record['sha256']
    save(folds_path, dict(sorted(assignment.items())))
    hashes[relative(folds_path)] = sha(folds_path)
    plan = dict(created=now(), purpose='Final split-sensitivity comparison within provided train70',
        jobs=jobs, families=list(FAMILIES), train70_names=names, official_validation_names=official,
        folds_file=relative(folds_path), fold_assignment_sha256=sha(folds_path), folds=fold_rows,
        grouping='alloy ID; all images from one alloy stay in one fold', split_seed=0, training_seed=0,
        input_hashes=hashes, image_net_cache=cache, initialization_reference=reference,
        gpu_pool=[dict(index=index, uuid=uuid) for index, uuid in GPUS.items()],
        scheduling='Four independent GPU slots; fold-major FIFO jobs; next job starts after CPU review',
        checkpoint_selection='Fixed last.pt at60/120epochs; never best.pt',
        evaluation='Held-out train70 OOF; TTA8, original scales, fixed PrimarySi postprocess',
        ensemble_recipes={'mit2_plus_unet3': ['mit2', 'unet3'],
                          'mit4_plus_unet3': ['mit4', 'unet3'],
                          'context4_plus_unet3': ['context4', 'unet3']},
        ensemble_weights=[.5, .5], cap6_evaluated=False, continuation_evaluated=False,
        fresh_imagenet=True, historical_project_checkpoints_allowed=False,
        official_validation_images_or_labels_read=False, test_images_or_labels_used=False,
        previous_data_used_for_model_development=True, independent_validation=False,
        new_seed_stability_evaluation=False, automatic_adoption=False, automatic_test_prediction=False,
        maximum_controller_hours=30, expected_total_hours=[14, 17],
        expected_time_basis='Prior measured epochs scaled by train-fold image count; four GPU scheduling estimate',
        protected_previous_deliverables=True)
    save(OUT / 'plan.json', plan)
    verify(plan)
    save(OUT / 'state.json', dict(time=now(), stage='prepared_not_started', total_jobs=len(jobs)))
    print(json.dumps(dict(prepared=True, plan_sha256=sha(OUT / 'plan.json'),
        fold_images=[(row['training_images'], row['validation_images']) for row in fold_rows],
        jobs=len(jobs), inputs=len(hashes))), flush=True)


def logs(job):
    path = ROOT / job['args']['out'] / 'log.jsonl'
    if not path.exists():
        return []
    # Only complete lines: the child may currently be writing the last record.
    return [json.loads(line) for line in path.read_text(encoding='utf8').split('\n')[:-1] if line]


def training_complete(job):
    args = job['args']
    rows = logs(job)
    assert [r['epoch'] for r in rows] == list(range(1, args['epochs'] + 1))
    assert all(math.isfinite(r['loss']) and r['microbatches'] == job['microbatches_per_epoch']
        and r['optimizer_updates'] == job['optimizer_updates_per_epoch'] for r in rows)
    output = ROOT / args['out']
    assert read(output / 'args.json') == args
    result = read(output / 'result.json')
    assert result['args'] == args and sorted(r['name'] for r in result['rows']) == job['validation_names']
    assert sorted(p.stem for p in (ROOT / args['save_oof']).glob('*.npy')) == job['validation_names']
    audit = read(output / 'initialization_audit.json')
    assert audit['training_names'] == job['training_names'] and audit['validation_names'] == job['validation_names']
    return dict(completed=now(), epochs=args['epochs'], checkpoint_sha256=sha(output / 'last.pt'))


def run():
    plan = read(OUT / 'plan.json')
    verify(plan)
    preflight = read(OUT / 'preflight.json')
    assert preflight['plan_sha256'] == sha(OUT / 'plan.json') and preflight['passed']
    assert sorted(preflight['checked_families']) == sorted(FAMILIES)
    lock = (OUT / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not any((OUT / name).exists() for name in ('started.json', 'error.json', 'all.done'))
    for index, uuid in GPUS.items():
        idle(index, uuid)
    for job in plan['jobs']:
        assert not (ROOT / job['args']['out']).exists() and not (ROOT / job['args']['save_oof']).exists()
    queue = list(plan['jobs'])
    states = {j['id']: dict(stage='queued', family=j['family'], fold=j['fold']) for j in queue}
    active, children, streams = {}, [], []
    started = time.monotonic()

    def interrupted(*unused):
        raise RuntimeError('Group34 controller interrupted')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    def spawn(index, job, stage):
        job_id = job['id']
        command = job['command'] if stage == 'training' else [str(ROOT / '.venv/bin/python'),
            'scripts/review_group34.py', '--job', job_id]
        path = OUT / 'logs' / f'{job_id}_{stage}.log'
        path.parent.mkdir(parents=True, exist_ok=True)
        stream = path.open('x')
        streams.append(stream)
        process = subprocess.Popen(command, cwd=ROOT,
            env=dict(ENV, CUDA_VISIBLE_DEVICES=GPUS[index] if stage == 'training' else ''),
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        children.append(process)
        active[index] = (job, process, stage)
        states[job_id].update(stage=stage, pid=process.pid, gpu_index=index, gpu_uuid=GPUS[index],
            stage_started=now(), command=command)
        if stage == 'training':
            states[job_id]['training_started'] = now()

    def snapshot(stage='running'):
        save(OUT / 'state.json', dict(time=now(), stage=stage, controller_pid=os.getpid(),
            plan_sha256=sha(OUT / 'plan.json'), jobs=states,
            completed_jobs=sum(s['stage'] == 'complete' for s in states.values()), total_jobs=20,
            queued_jobs=len(queue), elapsed_minutes=(time.monotonic() - started) / 60))

    try:
        for index in GPUS:
            spawn(index, queue.pop(0), 'training')
        save(OUT / 'started.json', dict(time=now(), controller_pid=os.getpid(),
            plan_sha256=sha(OUT / 'plan.json'), initial_jobs={k: dict(v) for k, v in states.items() if v['stage'] != 'queued'}))
        snapshot()
        while active or queue:
            assert time.monotonic() - started < plan['maximum_controller_hours'] * 3600
            for index, (job, process, stage) in list(active.items()):
                state = states[job['id']]
                if stage == 'training':
                    rows = logs(job)
                    assert all(math.isfinite(row['loss']) for row in rows), job['id']
                    state.update(completed_epochs=len(rows), total_epochs=job['args']['epochs'],
                        latest_epoch=rows[-1] if rows else None)
                if process.poll() is None:
                    continue
                assert process.returncode == 0, f"{job['id']} {stage} failed: {process.returncode}"
                if stage == 'training':
                    state['training_receipt'] = training_complete(job)
                    spawn(index, job, 'review')
                else:
                    done_path = OUT / 'reviews' / job['id'] / 'all.done'
                    done = read(done_path)
                    assert done['plan_sha256'] == sha(OUT / 'plan.json')
                    assert done['results_sha256'] == sha(done_path.parent / 'results.json')
                    assert done['rows_sha256'] == sha(done_path.parent / 'per_image_rows.json')
                    state.update(stage='complete', pid=None, finished=now(), review_done_sha256=sha(done_path))
                    del active[index]
                    if queue:
                        idle(index, GPUS[index])
                        spawn(index, queue.pop(0), 'training')
            snapshot()
            if active or queue:
                time.sleep(15)
        verify(plan)
        snapshot('aggregate')
        path = OUT / 'logs/aggregate.log'
        with path.open('x') as stream:
            child = subprocess.Popen([str(ROOT / '.venv/bin/python'), 'scripts/review_group34.py', '--aggregate'],
                cwd=ROOT, env=dict(ENV, CUDA_VISIBLE_DEVICES=''), stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            children.append(child)
            child.wait(timeout=1800)
            assert child.returncode == 0, 'Aggregate review failed'
        done = read(OUT / 'all.done')
        assert done['plan_sha256'] == sha(OUT / 'plan.json')
        assert done['results_sha256'] == sha(OUT / 'results.json')
        assert done['rows_sha256'] == sha(OUT / 'per_image_rows.json')
        snapshot('complete_and_stop')
    except BaseException as error:
        save(OUT / 'error.json', dict(time=now(), error=f'{type(error).__name__}: {error}', jobs=states))
        raise
    finally:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
        for stream in streams:
            stream.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'run'])
    options = parser.parse_args()
    os.chdir(ROOT)
    os.environ.update(ENV)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    prepare() if options.mode == 'prepare' else run()
