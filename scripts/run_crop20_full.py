"""Finish only the six authorized 60-epoch crop runs, review all five folds, stop.

Folds 0/1 are immutable pilot evidence. GPU 0 trains control folds 2/3/4 in
sequence, while GPUs 1/2/3 each train one candidate fold. No submission changes.
"""
import argparse
import copy
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

# In particular, --check must not create even a script import cache.
sys.dont_write_bytecode = True
import run_crop18 as pilot

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/review20_crop_full'
PILOT = ROOT / 'runs/review18_crop'
ENV = {**pilot.ENV, 'PYTHONDONTWRITEBYTECODE': '1'}
CHILDREN = []
WAIT_TIMEOUT = 15 * 60
JOB_TIMEOUT = 12 * 60 * 60
STALE_TIMEOUT = 90 * 60
EVALUATION_TIMEOUT = 60 * 60
SOURCE_FILES = [
    'src/train.py', 'src/data.py', 'src/infer.py', 'src/review_grains15.py',
    'scripts/run_crop18.py', 'scripts/review_crop18.py',
    'scripts/review_selective19.py', 'scripts/run_crop20_full.py',
    'scripts/review_crop20_full.py',
]


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(name, value):
    path = OUT / name
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def status(stage, **fields):
    value = {'time': pilot.now(), 'pid': os.getpid(), 'stage': stage,
             'authorized_new_folds': [2, 3, 4], 'preserved_pilot_folds': [0, 1],
             'epochs': 60, 'further_training_scheduled': False,
             'submission_changed': False, **fields}
    save('state.json', value)
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


def verify_hashes(receipts):
    for relative, expected in receipts.items():
        path = pilot.under_root(relative)
        if not path.is_file() or pilot.sha256(path) != expected:
            raise RuntimeError(f'Pinned input changed or missing: {relative}')


def add_receipt(receipts, path, expected=None):
    path = Path(path).resolve()
    relative = str(path.relative_to(ROOT.resolve()))
    actual = pilot.sha256(path)
    if expected is not None and actual != expected:
        raise RuntimeError(f'Prior evidence SHA256 mismatch: {relative}')
    if relative in receipts and receipts[relative] != actual:
        raise RuntimeError(f'Input changed during preflight: {relative}')
    receipts[relative] = actual
    return actual


def gpu_inventory():
    response = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
         '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True, timeout=15)
    applications = subprocess.run(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'],
        check=True, capture_output=True, text=True, timeout=15)
    pids = {}
    for line in applications.stdout.splitlines():
        if not line.strip():
            continue
        uuid, process = [value.strip() for value in line.split(',')]
        pids.setdefault(uuid, []).append(int(process))
    rows = []
    for line in response.stdout.splitlines():
        index, uuid, memory, utilization = [value.strip() for value in line.split(',')]
        if int(index) in range(4):
            rows.append({'index': int(index), 'uuid': uuid, 'memory_mib': int(memory),
                         'utilization_percent': int(utilization), 'compute_pids': pids.get(uuid, [])})
    if sorted(row['index'] for row in rows) != [0, 1, 2, 3]:
        raise RuntimeError('Expected GPU indices 0, 1, 2, 3')
    return rows


def idle_indices(rows):
    return [row['index'] for row in rows if row['memory_mib'] < 1000
            and row['utilization_percent'] < 10 and not row['compute_pids']]


def imagenet_receipt():
    os.environ.update(pilot.THREAD_ENV)
    from huggingface_hub import try_to_load_from_cache
    from segmentation_models_pytorch.encoders import encoders
    expected = {'repo_id': 'smp-hub/mit_b2.imagenet', 'revision': pilot.IMAGE_NET_REVISION}
    if encoders['mit_b2']['pretrained_settings']['imagenet'] != expected:
        raise RuntimeError('Installed encoder changed the locked ImageNet revision')
    receipts = []
    for filename, expected_sha in pilot.IMAGE_NET_HASHES.items():
        cached = try_to_load_from_cache('smp-hub/mit_b2.imagenet', filename,
                                        revision=pilot.IMAGE_NET_REVISION)
        if not isinstance(cached, str) or not Path(cached).is_file() or pilot.sha256(cached) != expected_sha:
            raise RuntimeError(f'Original ImageNet initialization is unavailable: {filename}')
        receipts.append({'file': filename, 'sha256': expected_sha, 'cache_path': cached})
    return {'revision': pilot.IMAGE_NET_REVISION, 'files': receipts}


def validate_split(values, previous_jobs):
    fold_path = pilot.under_root(values['folds_file'])
    fold_of = read_json(fold_path)
    if len(fold_of) != 90 or set(fold_of.values()) != set(range(5)):
        raise RuntimeError('Expected the unchanged 90-image, five-fold split')
    if any(sum(f == fold for f in fold_of.values()) != 18 for fold in range(5)):
        raise RuntimeError('Every held-out fold must contain exactly 18 images')
    digest = pilot.sha256(fold_path)
    if any(job['folds_sha256'] != digest for job in previous_jobs):
        raise RuntimeError('Fold assignment differs from the completed pilot')
    alloy_folds = {}
    for name, fold in fold_of.items():
        match = re.fullmatch(r'A(\d+)_step\d_sample\d', name)
        if match is None:
            raise RuntimeError(f'Unexpected image identity: {name}')
        alloy_folds.setdefault(match[1], set()).add(fold)
    if any(len(folds) != 1 for folds in alloy_folds.values()):
        raise RuntimeError('An alloy appears in both training and validation folds')
    found, receipts = {}, {}
    data = pilot.under_root(values['data'])
    for split in ('train', 'valid'):
        for image in (data / split / 'images').glob('*_image.png'):
            name = image.name.removesuffix('_image.png')
            mask = data / split / 'masks' / f'{name}_mask.png'
            if name in found or not mask.is_file():
                raise RuntimeError(f'Duplicate image or missing mask: {name}')
            found[name] = split
            add_receipt(receipts, image)
            add_receipt(receipts, mask)
    if set(found) != set(fold_of):
        raise RuntimeError('Available train/valid images differ from the fixed fold assignment')
    add_receipt(receipts, fold_path)
    return fold_of, receipts


def assert_new_outputs(job):
    target = pilot.under_root(job['args']['out'])
    if target.exists():
        raise FileExistsError(f'Refusing to reuse/overwrite a new fold directory: {target}')
    oof = pilot.under_root(job['args']['save_oof'])
    for name in job['validation_names']:
        if (oof / f'{name}.npy').exists():
            raise FileExistsError(f'Refusing to overwrite OOF prediction: {oof / name}')


def build_plan():
    # All operations here are reads. In particular, no output directory is made.
    for marker in ('plan.json', 'started.json', 'all.done', 'error.json'):
        if (OUT / marker).exists():
            raise FileExistsError(f'Full-fold controller already initialized: {OUT / marker}')
    receipts = {}
    done = read_json(PILOT / 'all.done')
    if done.get('pilot_folds') != [0, 1] or done.get('outcome') != 'four_job_crop_pilot_complete':
        raise RuntimeError('Expected the successfully completed four-job crop pilot')
    result_sha = add_receipt(receipts, PILOT / 'results.json', done['results_sha256'])
    done_sha = add_receipt(receipts, PILOT / 'all.done')
    plan_sha = add_receipt(receipts, PILOT / 'plan.json')
    old_plan = read_json(PILOT / 'plan.json')
    results = read_json(PILOT / 'results.json')
    old_jobs = old_plan['jobs']
    if {(job['arm'], job['fold']) for job in old_jobs} != {(arm, fold) for arm in pilot.ARMS for fold in (0, 1)} or len(old_jobs) != 4:
        raise RuntimeError('Pilot job identities differ from the fixed four-job comparison')
    # Same training/data implementations as the pilot, and same pilot evaluator.
    verify_hashes(old_plan['source_sha256'])
    if results['script_sha256'] != old_plan['source_sha256']['scripts/review_crop18.py']:
        raise RuntimeError('Pilot result uses an unrecognized evaluator revision')
    source_hashes = {name: pilot.sha256(ROOT / name) for name in SOURCE_FILES}
    fold_of, data_receipts = validate_split(old_jobs[0]['args'], old_jobs)
    preserved, base_configs, timing = [], {}, {}
    for job in old_jobs:
        arm, fold = job['arm'], job['fold']
        name, crop = pilot.ARMS[arm]
        values = read_json(pilot.under_root(job['args']['out']) / 'args.json')
        if values != job['args']:
            raise RuntimeError(f'Pilot actual arguments differ from its fixed plan: {job["id"]}')
        expected = {'epochs': 60, 'fold': fold, 'seed': 0, 'crop': crop, 'tile': 1536,
                    'bs': 2, 'accum_steps': 2, 'arch': 'Unet', 'encoder': 'mit_b2',
                    'width': 2560, 'scale_multiplier': 2.0, 'spi': 8, 'workers': 4,
                    'lr': 0.0003, 'bw': 2.0, 'rot': 'rot90', 'amp': True,
                    'out': f'runs/cv/{name}/fold{fold}', 'save_oof': f'runs/oof/{name}'}
        if any(values.get(key) != value for key, value in expected.items()) or values.get('init') or values.get('full'):
            raise RuntimeError(f'Pilot configuration is not the authorized matched comparison: {job["id"]}')
        names = sorted(name for name, assigned in fold_of.items() if assigned == fold)
        if job['validation_names'] != names:
            raise RuntimeError('Pilot held-out image identities changed')
        preserved.append(pilot.validate_completed(job))
        target = pilot.under_root(values['out'])
        for filename in ('args.json', 'log.jsonl', 'result.json', 'best.pt', 'last.pt'):
            add_receipt(receipts, target / filename)
        logs = pilot.read_epoch_logs(target / 'log.jsonl')
        validation_times = [row['val_sec'] for row in logs if 'val_sec' in row]
        duration = sum(row['train_sec'] for row in logs) + sum(validation_times)
        duration += 8 * sum(validation_times[-3:]) / len(validation_times[-3:])
        timing.setdefault(arm, []).append(duration)
        if fold == 0:
            base_configs[arm] = copy.deepcopy(values)
        else:
            differences = {key for key in values if values[key] != base_configs[arm].get(key)}
            if differences != {'fold', 'out'}:
                raise RuntimeError(f'Pilot fold configurations unexpectedly differ: {arm} {differences}')
    for record in results['source_files']:
        # Preserve the pilot predictions (both arms), plus the baseline evidence
        # actually used by its completed review.
        add_receipt(receipts, pilot.under_root(record['path']), record['sha256'])
    baseline_dir = ROOT / 'runs/review19_selective/existing'
    baseline_done = read_json(baseline_dir / 'all.done')
    add_receipt(receipts, baseline_dir / 'all.done')
    add_receipt(receipts, baseline_dir / 'results.json', baseline_done['results_sha256'])
    allowed = {'crop', 'out', 'save_oof'}
    if {key for key in base_configs['control'] if base_configs['control'][key] != base_configs['candidate'].get(key)} - allowed:
        raise RuntimeError('Control/candidate differ in more than crop and output paths')
    jobs = []
    for arm in ('candidate', 'control'):
        name, _ = pilot.ARMS[arm]
        for fold in (2, 3, 4):
            values = copy.deepcopy(base_configs[arm])
            values.update(fold=fold, out=f'runs/cv/{name}/fold{fold}')
            job = {'id': f'{arm}_fold{fold}', 'arm': arm, 'fold': fold,
                   'gpu': fold - 1 if arm == 'candidate' else 0, 'args': values,
                   'source': {'path': f'runs/cv/{name}/fold0/args.json',
                              'sha256': receipts[f'runs/cv/{name}/fold0/args.json']},
                   'folds_sha256': pilot.sha256(pilot.under_root(values['folds_file'])),
                   'validation_names': sorted(name for name, assigned in fold_of.items() if assigned == fold)}
            assert_new_outputs(job)
            jobs.append(job)
    imagenet = imagenet_receipt()
    if imagenet['revision'] != old_plan['imagenet']['revision'] or {
            row['file']: row['sha256'] for row in imagenet['files']} != {
            row['file']: row['sha256'] for row in old_plan['imagenet']['files']}:
        raise RuntimeError('ImageNet initialization differs from the pilot')
    timing = {arm: sum(durations) / len(durations) for arm, durations in timing.items()}
    inventory = gpu_inventory()
    return {'created': pilot.now(), 'jobs': jobs, 'preserved_jobs': preserved,
            'source_sha256': source_hashes, 'pilot_receipts': receipts,
            'data_receipts': data_receipts, 'pilot_plan_sha256': plan_sha,
            'pilot_results_sha256': result_sha, 'pilot_done_sha256': done_sha,
            'imagenet': imagenet, 'epochs': 60, 'seed': 0, 'new_folds': [2, 3, 4],
            'preserved_folds': [0, 1], 'evaluation_folds': [0, 1, 2, 3, 4],
            'validation_images': 90, 'new_validation_images': 54,
            'microbatch': 2, 'accumulation_steps': 2, 'effective_batch': 4,
            'fresh_training': True, 'checkpoint_initialization': None,
            'checkpoint_for_comparison': 'last.pt at epoch 60', 'evaluation_tile': 1536,
            'gpu_schedule': {'0': ['control_fold2', 'control_fold3', 'control_fold4'],
                             '1': ['candidate_fold2'], '2': ['candidate_fold3'], '3': ['candidate_fold4']},
            'preflight_gpus': inventory, 'preflight_idle_gpu_indices': idle_indices(inventory),
            'pilot_based_seconds_per_job': timing,
            'estimated_training_schedule_seconds': max(timing['candidate'], 3 * timing['control']),
            'eta_method': 'pilot total training/validation seconds plus eight validation passes for final TTA; not a guaranteed duration',
            'comparison': 'Only training crop differs; original batch2/accum2 and all other pilot options retained',
            'submission_changed': False, 'further_training_scheduled': False}


def verify_runtime_inputs(plan, include_data=False):
    verify_saved_plan()
    verify_hashes(plan['source_sha256'])
    if include_data:
        verify_hashes(plan['data_receipts'])
    for cached in plan['imagenet']['files']:
        if pilot.sha256(cached['cache_path']) != cached['sha256']:
            raise RuntimeError('Locked ImageNet cache changed after preflight')


def verify_saved_plan():
    marker = read_json(OUT / 'started.json')
    if pilot.sha256(OUT / 'plan.json') != marker['plan_sha256']:
        raise RuntimeError('The immutable launch plan changed after startup')


def wait_for_initial_gpus(plan):
    deadline = time.monotonic() + WAIT_TIMEOUT
    while True:
        inventory = gpu_inventory()
        if idle_indices(inventory) == [0, 1, 2, 3]:
            time.sleep(5)
            if idle_indices(gpu_inventory()) == [0, 1, 2, 3]:
                return
        if time.monotonic() > deadline:
            raise TimeoutError('Four idle GPUs did not become available within 15 minutes')
        status('waiting_for_four_idle_gpus', jobs=[], queued_jobs=[job['id'] for job in plan['jobs']],
               completed_jobs=[], gpu_inventory=inventory)
        time.sleep(15)


def start_job(job, plan):
    verify_runtime_inputs(plan, include_data=True)
    assert_new_outputs(job)
    if job['gpu'] not in idle_indices(gpu_inventory()):
        raise RuntimeError(f'GPU {job["gpu"]} is no longer idle before launch')
    command = [sys.executable, str(Path(__file__).resolve()), '--train-worker',
               *pilot.cli_args(job['args'])]
    with open(OUT / 'logs' / f'{job["id"]}.log', 'x', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=ROOT,
                                   env={**ENV, 'CUDA_VISIBLE_DEVICES': str(job['gpu'])},
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    CHILDREN.append(process)
    return job, process, job['gpu'], time.monotonic()


def run_jobs(plan):
    wait_for_initial_gpus(plan)
    queues = {gpu: [job for job in plan['jobs'] if job['gpu'] == gpu] for gpu in range(4)}
    running, completed, waiting_since = {}, [], {}
    while running or any(queues.values()):
        verify_saved_plan()
        verify_hashes(plan['source_sha256'])
        for identifier, (job, process, gpu, started) in list(running.items()):
            if process.poll() is None:
                if time.monotonic() - started > JOB_TIMEOUT:
                    raise TimeoutError(f'{identifier} exceeded the 12-hour job timeout')
                log = pilot.under_root(job['args']['out']) / 'log.jsonl'
                if log.exists() and time.time() - log.stat().st_mtime > STALE_TIMEOUT:
                    raise TimeoutError(f'{identifier} has made no logged progress for 90 minutes')
                continue
            if process.returncode:
                raise RuntimeError(f'{identifier} exited with status {process.returncode}')
            receipt = pilot.validate_completed(job)
            receipt.update(gpu=gpu, elapsed_seconds=round(time.monotonic() - started, 1))
            completed.append(receipt)
            save('completed_jobs.json', completed)
            del running[identifier]
        busy = {entry[2] for entry in running.values()}
        idle = idle_indices(gpu_inventory())
        for gpu in range(4):
            if gpu in busy or not queues[gpu]:
                continue
            if gpu not in idle:
                waiting_since.setdefault(gpu, time.monotonic())
                if time.monotonic() - waiting_since[gpu] > WAIT_TIMEOUT:
                    raise TimeoutError(f'GPU {gpu} has another workload; queued run was not started')
                continue
            waiting_since.pop(gpu, None)
            job = queues[gpu].pop(0)
            running[job['id']] = start_job(job, plan)
        progress = [pilot.job_progress(*entry) for entry in running.values()]
        queue_rows = [{'id': job['id'], 'arm': job['arm'], 'fold': job['fold'], 'gpu': gpu}
                      for gpu, jobs in queues.items() for job in jobs]
        per_gpu = {gpu: sum(plan['pilot_based_seconds_per_job'][job['arm']] for job in jobs)
                   for gpu, jobs in queues.items()}
        for row in progress:
            remaining = row['remaining_total_sec_estimate']
            if remaining is None:
                remaining = max(0, plan['pilot_based_seconds_per_job'][row['arm']] - row['elapsed_sec'])
            per_gpu[row['gpu']] += remaining
        status('full_fold_training' if running or queue_rows else 'full_fold_training_complete',
               jobs=progress, queued_jobs=queue_rows, completed_jobs=completed,
               remaining_total_sec_estimate=max(per_gpu.values()),
               eta_excludes_cpu_final_review=True,
               waiting_gpu_indices=sorted(waiting_since))
        if running or queue_rows:
            time.sleep(15)
    if len(completed) != 6:
        raise RuntimeError('Expected exactly six newly completed jobs')
    verify_hashes(plan['pilot_receipts'])


def evaluate(plan):
    verify_runtime_inputs(plan, include_data=True)
    verify_hashes(plan['pilot_receipts'])
    status('evaluating_all_five_folds_cpu', jobs=[], queued_jobs=[],
           completed_jobs=read_json(OUT / 'completed_jobs.json'))
    command = [sys.executable, 'scripts/review_crop20_full.py']
    with open(OUT / 'logs' / 'evaluation.log', 'x', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=ROOT, env={**ENV, 'CUDA_VISIBLE_DEVICES': ''},
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    CHILDREN.append(process)
    deadline = time.monotonic() + EVALUATION_TIMEOUT
    while process.poll() is None:
        if time.monotonic() > deadline:
            raise TimeoutError('Final CPU review exceeded one hour')
        time.sleep(5)
    if process.returncode:
        raise RuntimeError(f'Full-fold evaluator exited with status {process.returncode}')
    results = read_json(OUT / 'results.json')
    if not isinstance(results, dict) or not (OUT / 'summary.txt').is_file():
        raise RuntimeError('Final review did not produce results.json and summary.txt')
    verify_hashes(plan['source_sha256'])
    verify_hashes(plan['pilot_receipts'])


def cleanup_children():
    # Only process groups started and owned by this controller are signalled.
    for process in CHILDREN:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 10
    for process in CHILDREN:
        if process.poll() is None:
            try:
                process.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Read-only preflight; no training or file writes')
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.check:
        print(json.dumps(build_plan(), indent=2, ensure_ascii=False, allow_nan=False), flush=True)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / 'driver.lock', 'a', encoding='utf-8') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = build_plan()
        (OUT / 'logs').mkdir(exist_ok=True)
        save('plan.json', plan)
        with open(OUT / 'started.json', 'x', encoding='utf-8') as marker:
            json.dump({'time': pilot.now(), 'pid': os.getpid(),
                       'plan_sha256': pilot.sha256(OUT / 'plan.json')}, marker)
        def interrupted(signum, frame):
            raise RuntimeError(f'Controller received signal {signum}')
        signal.signal(signal.SIGTERM, interrupted)
        try:
            run_jobs(plan)
            evaluate(plan)
            status('complete_full_validation_and_stop', jobs=[], queued_jobs=[],
                   completed_jobs=read_json(OUT / 'completed_jobs.json'),
                   results='runs/review20_crop_full/results.json')
            save('all.done', {'time': pilot.now(), 'outcome': 'matched_crop_five_fold_validation_complete',
                              'preserved_folds': [0, 1], 'new_folds': [2, 3, 4],
                              'new_training_jobs': 6, 'epochs': 60, 'validation_images': 90,
                              'further_training_scheduled': False, 'submission_changed': False,
                              'plan_sha256': pilot.sha256(OUT / 'plan.json'),
                              'results_sha256': pilot.sha256(OUT / 'results.json')})
        except BaseException as exc:
            cleanup_children()
            status('error', message=repr(exc))
            save('error.json', {'time': pilot.now(), 'error': repr(exc)})
            raise


if __name__ == '__main__':
    if sys.argv[1:2] == ['--train-worker']:
        pilot.train_worker()
    else:
        main()
