"""Run the fixed four-job crop pilot; never expand to the remaining folds."""
import argparse
import fcntl
import hashlib
import json
import math
import os
import runpy
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/review18_crop'
ARMS = {'control': ('x4_mit_c1024_acc2_bw_r90', 1024),
        'candidate': ('x4_mit_c1536_bw_r90', 1536)}
THREAD_ENV = {'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '1',
              'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1',
              'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
              'PYTHONUNBUFFERED': '1'}
ENV = {**os.environ, **THREAD_ENV}
CHILDREN = []
IMAGE_NET_REVISION = '868ab6f13871dcf8c3d9f90ee4519403475b65ef'
IMAGE_NET_HASHES = {
    'config.json': '13c350b0a57ad271642f941fedd790491f9fa47c70e3aaddfa5cd18de212540d',
    'model.safetensors': 'ab18b54f81748427bae1183160822903470bcca0e3121944c0d185b4b1816a95',
}


def now():
    return datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds')


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save(name, value):
    path = OUT / name
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def status(stage, **fields):
    value = {'time': now(), 'pid': os.getpid(), 'stage': stage,
             'pilot_only': True, 'further_training_scheduled': False, **fields}
    save('state.json', value)
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


def under_root(relative):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError(f'Path is outside project: {relative}')
    return path


def cv_args(arm, fold):
    source = ROOT / f'runs/cv/x4_mit_bw_r90/fold{fold}/args.json'
    values = json.loads(source.read_text(encoding='utf-8'))
    expected = {'epochs': 60, 'arch': 'Unet', 'encoder': 'mit_b2', 'seed': 0,
                'fold': fold, 'width': 2560, 'crop': 1024, 'tile': 1024,
                'bs': 4, 'scale_multiplier': 2.0, 'scale_jitter': 0.15,
                'loss': 'dice', 'bw': 2.0, 'spi': 8, 'rot': 'rot90',
                'amp': True, 'max_iters': 0}
    for key, value in expected.items():
        if values.get(key) != value:
            raise ValueError(f'Unexpected original {source}: {key}={values.get(key)!r}')
    if values.get('init') or values.get('full') or values.get('accum_steps', 1) != 1:
        raise ValueError('Original must be a fresh ImageNet initialization and ordinary CV run')
    name, crop = ARMS[arm]
    values.update(out=f'runs/cv/{name}/fold{fold}', save_oof=f'runs/oof/{name}',
                  crop=crop, tile=1536, bs=2, accum_steps=2, workers=4)
    return values, {'path': str(source.relative_to(ROOT)), 'sha256': sha256(source)}


def cli_args(values):
    command = []
    for key, value in values.items():
        flag = '--' + key.replace('_', '-')
        if value is None or value == '' or value is False:
            continue
        if value is True:
            command.append(flag)
        elif isinstance(value, list):
            command.extend([flag, *map(str, value)])
        else:
            command.extend([flag, str(value)])
    return command


def build_plan():
    # The check path only reads files/cache and never creates output directories.
    for name, _ in ARMS.values():
        for prefix in ['runs/cv', 'runs/oof']:
            path = under_root(f'{prefix}/{name}')
            if path.exists():
                raise FileExistsError(f'Refusing to overwrite experiment directory: {path}')
    for marker in ['started.json', 'all.done', 'error.json']:
        if (OUT / marker).exists():
            raise FileExistsError(f'Pilot already started or finished: {OUT / marker}')
    sources = ['src/train.py', 'src/data.py', 'scripts/run_crop18.py', 'scripts/review_crop18.py']
    hashes = {name: sha256(ROOT / name) for name in sources}
    if '--accum-steps' not in (ROOT / 'src/train.py').read_text(encoding='utf-8'):
        raise RuntimeError('Trainer does not support gradient accumulation')
    os.environ.update(THREAD_ENV)
    from huggingface_hub import try_to_load_from_cache
    from segmentation_models_pytorch.encoders import encoders
    settings = encoders['mit_b2']['pretrained_settings']['imagenet']
    if settings != {'repo_id': 'smp-hub/mit_b2.imagenet', 'revision': IMAGE_NET_REVISION}:
        raise RuntimeError('Installed encoder does not use the locked ImageNet initialization')
    cached = []
    for filename, expected in IMAGE_NET_HASHES.items():
        path = try_to_load_from_cache('smp-hub/mit_b2.imagenet', filename,
                                     revision=IMAGE_NET_REVISION)
        if not isinstance(path, str) or not Path(path).is_file() or sha256(path) != expected:
            raise RuntimeError(f'Exact original ImageNet cache unavailable: {filename}')
        cached.append({'file': filename, 'sha256': expected})
    jobs = []
    for fold in [0, 1]:
        for arm in ARMS:
            values, source = cv_args(arm, fold)
            fold_path = under_root(values['folds_file'])
            fold_of = json.loads(fold_path.read_text(encoding='utf-8'))
            names = sorted(name for name, value in fold_of.items() if value == fold)
            if len(fold_of) != 90 or len(names) != 18:
                raise ValueError('Expected the unchanged 90-image split with 18 held-out images')
            data = under_root(values['data'])
            found = {}
            for split in ['train', 'valid']:
                for image in (data / split / 'images').glob('*_image.png'):
                    name = image.name.removesuffix('_image.png')
                    mask = data / split / 'masks' / f'{name}_mask.png'
                    if not mask.is_file() or name in found:
                        raise ValueError(f'Missing mask or duplicate training image: {name}')
                    found[name] = split
            if set(found) != set(fold_of):
                raise ValueError('Training data and the recorded 90-image fold split differ')
            jobs.append({'id': f'{arm}_fold{fold}', 'arm': arm, 'fold': fold,
                         'args': values, 'source': source, 'validation_names': names,
                         'folds_sha256': sha256(fold_path)})
    return {'created': now(), 'jobs': jobs, 'source_sha256': hashes,
            'imagenet': {'revision': IMAGE_NET_REVISION, 'files': cached},
            'fresh_training': True, 'checkpoint_initialization': None,
            'pilot_folds': [0, 1], 'automatic_remaining_folds': [],
            'epochs': 60, 'microbatch': 2, 'accumulation_steps': 2, 'effective_batch': 4,
            'comparison': 'Matched microbatch/accumulation; only training crop differs between arms',
            'evaluation_tile': 1536, 'checkpoint_for_comparison': 'last.pt at epoch 60',
            'cpu_threads_per_job': 4, 'data_workers_per_job': 4,
            'gpu_policy': 'Four idle GPUs, indices 0..3, checked twice five seconds apart',
            'submission_changed': False, 'further_training_scheduled': False}


def free_gpus():
    response = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                               '--format=csv,noheader,nounits'], check=True, capture_output=True,
                              text=True, timeout=15)
    return [gpu for line in response.stdout.splitlines()
            for gpu, memory, utilization in [map(int, line.split(','))]
            if gpu in range(4) and memory < 1000 and utilization < 10]


def read_epoch_logs(path):
    if not path.exists():
        return []
    lines = path.read_text(encoding='utf-8').splitlines()
    rows = []
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
            # The trainer may be in the middle of writing the newest line.
    return rows


def job_progress(job, process, gpu, started):
    rows = read_epoch_logs(under_root(job['args']['out']) / 'log.jsonl')
    latest = rows[-1] if rows else {}
    train_seconds = [float(row['train_sec']) for row in rows[-3:] if 'train_sec' in row]
    mean = sum(train_seconds) / len(train_seconds) if train_seconds else None
    epoch = latest.get('epoch', 0)
    validations = [float(row['val_sec']) for row in rows if 'val_sec' in row][-3:]
    val_mean = sum(validations) / len(validations) if validations else None
    remaining_train = (60 - epoch) * mean if mean is not None else None
    remaining_validations = sum(e % job['args']['val_every'] == 0 for e in range(epoch + 1, 61))
    eta = (remaining_train + (remaining_validations + 8) * val_mean
           if remaining_train is not None and val_mean is not None else None)
    return {'id': job['id'], 'arm': job['arm'], 'fold': job['fold'], 'gpu': gpu,
            'pid': process.pid, 'epoch': epoch, 'total_epochs': 60,
            'phase': 'final_tta' if epoch == 60 else 'training',
            'train_sec': latest.get('train_sec'), 'recent_three_mean_train_sec': mean,
            'elapsed_sec': round(time.monotonic() - started, 1),
            'remaining_train_sec_estimate': remaining_train,
            'remaining_total_sec_estimate': eta,
            'eta_method': 'recent up-to-3 training/validation means; final 8-TTA estimated as 8 validations',
            'loss': latest.get('loss'), 'gpu_mem_gb': latest.get('gpu_mem_gb')}


def validate_completed(job):
    target = under_root(job['args']['out'])
    rows = read_epoch_logs(target / 'log.jsonl')
    if len(rows) != 60 or [row['epoch'] for row in rows] != list(range(1, 61)):
        raise RuntimeError(f'{job["id"]}: incomplete 60-epoch log')
    if not all(math.isfinite(row['loss']) for row in rows):
        raise RuntimeError(f'{job["id"]}: nonfinite training loss')
    if not all(row['microbatches'] == 288 and row['optimizer_updates'] == 144
               and row['accum_steps'] == 2 and row['effective_batch'] == 4 for row in rows):
        raise RuntimeError(f'{job["id"]}: optimizer update counts differ from fixed plan')
    actual = json.loads((target / 'args.json').read_text(encoding='utf-8'))
    if any(actual.get(key) != value for key, value in job['args'].items()):
        raise RuntimeError(f'{job["id"]}: recorded arguments differ from fixed plan')
    result = json.loads((target / 'result.json').read_text(encoding='utf-8'))
    names = [row['name'] for row in result['rows']]
    if len(names) != 18 or sorted(names) != job['validation_names']:
        raise RuntimeError(f'{job["id"]}: validation identities differ')
    for name in ['best.pt', 'last.pt']:
        if (target / name).stat().st_size < 1000000:
            raise RuntimeError(f'{job["id"]}: checkpoint missing or too small')
    for name in names:
        if not (under_root(job['args']['save_oof']) / f'{name}.npy').is_file():
            raise RuntimeError(f'{job["id"]}: missing OOF probability {name}')
    return {'id': job['id'], 'completed': now(), 'epochs': 60, 'images': 18,
            'train_seconds': sum(row['train_sec'] for row in rows),
            'result_sha256': sha256(target / 'result.json')}


def run_jobs(plan):
    while True:
        first = free_gpus()
        if len(first) == 4:
            time.sleep(5)
            second = free_gpus()
            stable = [gpu for gpu in first if gpu in second]
            if len(stable) == 4:
                break
        status('waiting_for_four_idle_gpus', idle_gpu_indices=first, jobs=[])
        time.sleep(15)
    running, complete = {}, []
    for job, gpu in zip(plan['jobs'], sorted(stable)):
        target = under_root(job['args']['out'])
        if target.exists():
            raise FileExistsError(f'Refusing to overwrite: {target}')
        oof = under_root(job['args']['save_oof'])
        if any((oof / f'{name}.npy').exists() for name in job['validation_names']):
            raise FileExistsError(f'Refusing to overwrite OOF files for {job["id"]}')
        command = [sys.executable, str(Path(__file__).resolve()), '--train-worker', *cli_args(job['args'])]
        with open(OUT / 'logs' / f'{job["id"]}.log', 'x', encoding='utf-8') as log:
            process = subprocess.Popen(command, cwd=ROOT, env={**ENV, 'CUDA_VISIBLE_DEVICES': str(gpu)},
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        CHILDREN.append(process)
        running[job['id']] = (job, process, gpu, time.monotonic())
    while running:
        for identifier, (job, process, gpu, started) in list(running.items()):
            if process.poll() is None:
                continue
            if process.returncode:
                raise RuntimeError(f'{identifier} exited with status {process.returncode}')
            complete.append(validate_completed(job))
            save('completed_jobs.json', complete)
            del running[identifier]
        progress = [job_progress(*values) for values in running.values()]
        estimates = [row['remaining_total_sec_estimate'] for row in progress]
        eta = max(estimates) if estimates and all(value is not None for value in estimates) else None
        status('pilot_training' if running else 'pilot_training_complete',
               jobs=progress, completed_jobs=complete, remaining_total_sec_estimate=eta)
        if running:
            time.sleep(15)


def evaluate():
    status('evaluating_pilot_cpu')
    command = [sys.executable, 'scripts/review_crop18.py', '--out', 'runs/review18_crop/results.json']
    with open(OUT / 'logs' / 'evaluation.log', 'x', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=ROOT, env={**ENV, 'CUDA_VISIBLE_DEVICES': ''},
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    CHILDREN.append(process)
    while process.poll() is None:
        time.sleep(5)
    if process.returncode:
        raise RuntimeError(f'Pilot evaluation exited with status {process.returncode}')
    json.loads((OUT / 'results.json').read_text(encoding='utf-8'))


def cleanup_children():
    # All targets were started by this controller in their own sessions.
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
    parser.add_argument('--check', action='store_true', help='Read-only preflight; never starts training')
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.check:
        print(json.dumps(build_plan(), indent=2, ensure_ascii=False), flush=True)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / 'driver.lock', 'a', encoding='utf-8') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = build_plan()
        (OUT / 'logs').mkdir(exist_ok=True)
        save('plan.json', plan)
        with open(OUT / 'started.json', 'x', encoding='utf-8') as marker:
            json.dump({'time': now(), 'pid': os.getpid()}, marker)
        def interrupted(signum, frame):
            raise RuntimeError(f'Controller received signal {signum}')
        signal.signal(signal.SIGTERM, interrupted)
        try:
            run_jobs(plan)
            evaluate()
            status('complete_pilot_and_stop', results='runs/review18_crop/results.json',
                   submission_changed=False)
            save('all.done', {'time': now(), 'outcome': 'four_job_crop_pilot_complete',
                              'pilot_folds': [0, 1], 'further_training_scheduled': False,
                              'submission_changed': False,
                              'results_sha256': sha256(OUT / 'results.json')})
        except BaseException as exc:
            cleanup_children()
            status('error', message=repr(exc))
            save('error.json', {'time': now(), 'error': repr(exc)})
            raise


def train_worker():
    os.environ.update(THREAD_ENV)
    import cv2
    import torch
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(ROOT / 'src'))
    sys.argv = [str(ROOT / 'src/train.py'), *sys.argv[2:]]
    runpy.run_path(sys.argv[0], run_name='__main__')


if __name__ == '__main__':
    if sys.argv[1:2] == ['--train-worker']:
        train_worker()
    else:
        main()
