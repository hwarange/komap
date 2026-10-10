"""Run only the authorized two-fold 4x UNet++ pilot, then compare saved OOF."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import run_crop18 as common

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/review23_unetpp4'
EXPERIMENT = 'x4_unetpp_c1536_bw_r90'
ENV = {**common.ENV, 'PYTHONDONTWRITEBYTECODE': '1'}
CHILDREN = []
REVISION = '7a57b34f723329ff020b3f8bc41771163c519d0c'
CACHE_HASHES = {
    'config.json': '01bf2cf24eb29b405c28c159f46ceda92c098ab85868be88fb100967db47166e',
    'model.safetensors': '6549ccd46a2d20a0a74fca866cfc5c6f9b3672d468b0a92aa94a58545a61103e',
}
CONFIG = dict(arch='UnetPlusPlus', encoder='resnet34', width=2560,
              scale_multiplier=2.0, crop=1536, tile=1536, epochs=60,
              bs=4, accum_steps=1, seed=0)
SOURCES = [
    'src/train.py', 'src/data.py', 'src/infer.py', 'src/review_grains15.py',
    'scripts/run_crop18.py', 'scripts/run_unetpp23.py',
    'scripts/review_unetpp23.py', 'scripts/benchmark_unetpp23.py',
]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(name, value):
    target = OUT / name
    temp = target.with_suffix(target.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(target)


def status(stage, **values):
    result = dict(time=common.now(), pid=os.getpid(), stage=stage, folds=[0, 1],
                  epochs=60, further_training_scheduled=False, submission_changed=False, **values)
    save('state.json', result)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


def cv_args(fold):
    source = ROOT / f'runs/cv/x3_unetpp_bw_r90/fold{fold}/args.json'
    values = read(source)
    for key, expected in dict(arch='UnetPlusPlus', encoder='resnet34', fold=fold, epochs=60,
                              width=3840, crop=1536, tile=1536, bs=4, seed=0,
                              spi=8, bw=2.0, rot='rot90', amp=True).items():
        assert values[key] == expected, (source, key, values[key])
    assert not values.get('full') and not values.get('init') and not values.get('attention')
    values.update(CONFIG, fold=fold, workers=4, scale_jitter=0.15,
                  out=f'runs/cv/{EXPERIMENT}/fold{fold}', save_oof=f'runs/oof/{EXPERIMENT}')
    return values


def inventory():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
                                      '--format=csv,noheader,nounits'], text=True, timeout=15)
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                         '--format=csv,noheader,nounits'], text=True, timeout=15)
    busy = {line.split(',')[0].strip() for line in processes.splitlines() if line.strip()}
    records = []
    for line in output.splitlines():
        index, uuid, memory, utilization = [part.strip() for part in line.split(',')]
        records.append(dict(index=int(index), uuid=uuid, memory_mib=int(memory), utilization=int(utilization),
                            idle=int(memory) < 1000 and int(utilization) < 10 and uuid not in busy))
    return records


def verify_hashes(records):
    for name, expected in records.items():
        if common.sha256(common.under_root(name)) != expected:
            raise RuntimeError(f'Pinned input changed: {name}')


def build_plan():
    for path in [ROOT / f'runs/cv/{EXPERIMENT}', ROOT / f'runs/oof/{EXPERIMENT}']:
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite {path}')
    for name in ['plan.json', 'started.json', 'all.done', 'error.json', 'results.json']:
        if (OUT / name).exists():
            raise FileExistsError(f'Already started or completed: {OUT / name}')
    os.environ.update(common.THREAD_ENV)
    from huggingface_hub import try_to_load_from_cache
    from segmentation_models_pytorch.encoders import encoders
    assert encoders['resnet34']['pretrained_settings']['imagenet'] == {
        'repo_id': 'smp-hub/resnet34.imagenet', 'revision': REVISION}
    cache = []
    for name, expected in CACHE_HASHES.items():
        path = try_to_load_from_cache('smp-hub/resnet34.imagenet', name, revision=REVISION)
        assert isinstance(path, str) and Path(path).is_file() and common.sha256(path) == expected
        cache.append(dict(file=name, sha256=expected, path=path))
    fold_of = read(ROOT / 'splits/folds5.json')
    assert len(fold_of) == 90 and set(fold_of.values()) == set(range(5))
    grouped = {}
    for name, fold in fold_of.items():
        grouped.setdefault(name.split('_')[0], set()).add(fold)
    assert all(len(folds) == 1 for folds in grouped.values())
    data_hashes, found = {}, set()
    for split in ['train', 'valid']:
        for image in (ROOT / 'data' / split / 'images').glob('*_image.png'):
            name = image.name.removesuffix('_image.png')
            mask = image.parent.parent / 'masks' / f'{name}_mask.png'
            assert name not in found and mask.is_file()
            found.add(name)
            for path in [image, mask]:
                data_hashes[path.relative_to(ROOT).as_posix()] = common.sha256(path)
    assert found == set(fold_of) and len(data_hashes) == 180
    previous_path = ROOT / 'runs/review20_crop_full/results.json'
    previous = read(previous_path)
    assert common.sha256(previous_path) == read(ROOT / 'runs/review20_crop_full/all.done')['results_sha256']
    selected = {name for name, fold in fold_of.items() if fold in [0, 1]}
    baseline = {row['path']: row['sha256'] for row in previous['provenance']['source_files']
                if row['name'] in selected and row['source'] in ['x2', 'unet', 'candidate']}
    assert len(selected) == 36 and len(baseline) == 108
    baseline_runs = {}
    for model in ['x2_mit_bw_r90', 'x3_unetpp_bw_r90', 'x3_unetpp_bw_e120_r90', 'x4_mit_c1536_bw_r90']:
        for fold in [0, 1]:
            run_dir = ROOT / f'runs/cv/{model}/fold{fold}'
            original = read(run_dir / 'args.json')
            result = read(run_dir / 'result.json')
            expected_names = sorted(n for n, f in fold_of.items() if f == fold)
            assert original['fold'] == fold and original['folds_file'] == 'splits/folds5.json'
            assert sorted(r['name'] for r in result['rows']) == expected_names
            assert original['epochs'] == (120 if 'e120' in model else 60)
            for filename in ['args.json', 'result.json']:
                path = run_dir / filename
                baseline_runs[path.relative_to(ROOT).as_posix()] = common.sha256(path)
            if model == 'x3_unetpp_bw_r90':
                for name in expected_names:
                    path = ROOT / f'runs/oof/{model}/{name}.npy'
                    baseline[path.relative_to(ROOT).as_posix()] = common.sha256(path)
    assert len(baseline) == 144 and len(baseline_runs) == 16
    verify_hashes(baseline)
    jobs = []
    for fold in [0, 1]:
        names = sorted(name for name, value in fold_of.items() if value == fold)
        assert len(names) == 18
        jobs.append(dict(id=f'unetpp_fold{fold}', arm='unetpp', fold=fold, args=cv_args(fold),
                         validation_names=names))
    benchmark = read(OUT / 'benchmark.json')
    assert benchmark['status'] == 'complete'
    assert all(benchmark['benchmark_args'].get(key) == value for key, value in CONFIG.items())
    assert benchmark['script_sha256'] == common.sha256(ROOT / 'scripts/benchmark_unetpp23.py')
    return dict(created=common.now(), experiment=EXPERIMENT, folds=[0, 1], config=CONFIG,
                jobs=jobs, split_sha256=common.sha256(ROOT / 'splits/folds5.json'),
                source_hashes={name: common.sha256(ROOT / name) for name in SOURCES},
                data_hashes=data_hashes, baseline_oof_hashes=baseline, baseline_run_hashes=baseline_runs,
                previous_results_sha256=common.sha256(previous_path),
                imagenet=dict(repo_id='smp-hub/resnet34.imagenet', revision=REVISION, files=cache),
                architecture_defaults=dict(encoder_depth=5, decoder_channels=[256, 128, 64, 32, 16]),
                benchmark_sha256=common.sha256(OUT / 'benchmark.json'),
                checkpoint='last.pt at epoch60', ensemble_weights=[0.5, 0.5],
                new_jobs=2, fresh_imagenet_training=True, further_training_scheduled=False,
                submission_changed=False, independent_validation=False,
                note='Pilot36 previously examined; no threshold or weight search or automatic expansion')


def validate_completed(job):
    target = common.under_root(job['args']['out'])
    logs = common.read_epoch_logs(target / 'log.jsonl')
    assert len(logs) == 60 and [r['epoch'] for r in logs] == list(range(1, 61))
    assert all(math.isfinite(r['loss']) and r['microbatches'] == 144
               and r['optimizer_updates'] == 144 and r['accum_steps'] == 1
               and r['effective_batch'] == 4 for r in logs)
    actual = read(target / 'args.json')
    assert all(actual.get(k) == v for k, v in job['args'].items())
    result = read(target / 'result.json')
    assert sorted(r['name'] for r in result['rows']) == job['validation_names']
    assert len(result['rows']) == 18
    assert (target / 'last.pt').stat().st_size > 1000000
    for row in result['rows']:
        assert 'tta' in row and 'tta_pp' in row
        assert (common.under_root(job['args']['save_oof']) / (row['name'] + '.npy')).is_file()
    return dict(id=job['id'], fold=job['fold'], completed=common.now(), epochs=60, images=18,
                train_seconds=sum(r['train_sec'] for r in logs),
                result_sha256=common.sha256(target / 'result.json'))


def run(plan):
    verify_hashes(plan['source_hashes'])
    verify_hashes(plan['data_hashes'])
    verify_hashes(plan['baseline_run_hashes'])
    first = {r['uuid']: r for r in inventory() if r['idle']}
    time.sleep(3)
    available = [r for r in inventory() if r['idle'] and r['uuid'] in first]
    if len(available) < 2:
        raise RuntimeError('Two idle GPUs required; no existing task will be interrupted')
    running, completed = {}, []
    for job, gpu in zip(plan['jobs'], available[:2]):
        assert not common.under_root(job['args']['out']).exists()
        command = [sys.executable, str(Path(__file__).resolve()), '--train-worker', *common.cli_args(job['args'])]
        with (OUT / 'logs' / (job['id'] + '.log')).open('x', encoding='utf-8') as log:
            child = subprocess.Popen(command, cwd=ROOT, env={**ENV, 'CUDA_VISIBLE_DEVICES': gpu['uuid']},
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        CHILDREN.append(child)
        running[job['id']] = (job, child, gpu['index'], time.monotonic())
    while running:
        for identifier, values in list(running.items()):
            job, child, gpu, started = values
            if time.monotonic() - started > 8 * 3600:
                raise TimeoutError(f'Training exceeded eight hours: {identifier}')
            if child.poll() is None:
                continue
            if child.returncode:
                raise RuntimeError(f'{identifier} exited {child.returncode}')
            completed.append(validate_completed(job))
            save('completed_jobs.json', completed)
            del running[identifier]
        progress = [common.job_progress(*values) for values in running.values()]
        estimates = [r['remaining_total_sec_estimate'] for r in progress]
        eta = max(estimates) if estimates and all(v is not None for v in estimates) else None
        status('pilot_training' if running else 'pilot_training_complete', jobs=progress,
               completed_jobs=completed, remaining_total_sec_estimate=eta)
        if running:
            time.sleep(15)
    verify_hashes(plan['source_hashes'])
    verify_hashes(plan['baseline_oof_hashes'])
    verify_hashes(plan['baseline_run_hashes'])
    verify_hashes(plan['data_hashes'])
    status('evaluating_pilot_cpu', jobs=[], completed_jobs=completed)
    with (OUT / 'logs/evaluation.log').open('x', encoding='utf-8') as log:
        child = subprocess.Popen([sys.executable, 'scripts/review_unetpp23.py'], cwd=ROOT,
                                 env={**ENV, 'CUDA_VISIBLE_DEVICES': ''}, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        CHILDREN.append(child)
        result_code = child.wait(timeout=3600)
    if result_code:
        raise RuntimeError(f'CPU review failed: {result_code}')
    verify_hashes(plan['source_hashes'])
    for filename in ['results.json', 'summary.txt', 'per_image_rows.json']:
        assert (OUT / filename).is_file()
    status('complete_pilot_and_stop', jobs=[], completed_jobs=completed)
    save('all.done', dict(time=common.now(), outcome='unetpp4_two_fold_pilot_complete',
                         folds=[0, 1], images=36, epochs=60, new_training_jobs=2,
                         results_sha256=common.sha256(OUT / 'results.json'),
                         plan_sha256=common.sha256(OUT / 'plan.json'),
                         further_training_scheduled=False, submission_changed=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.check:
        plan = build_plan()
        print(json.dumps(dict(status='preflight_passed', jobs=len(plan['jobs']), config=plan['config'],
                              data_files=len(plan['data_hashes']), baseline_oof=len(plan['baseline_oof_hashes']),
                              source_hashes=plan['source_hashes']), indent=2))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'driver.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = build_plan()
        (OUT / 'logs').mkdir(exist_ok=True)
        save('plan.json', plan)
        with (OUT / 'started.json').open('x') as stream:
            json.dump(dict(time=common.now(), pid=os.getpid(),
                           plan_sha256=common.sha256(OUT / 'plan.json')), stream)
        def interrupted(signum, frame):
            raise RuntimeError(f'Controller received signal {signum}')
        signal.signal(signal.SIGTERM, interrupted)
        common.CHILDREN = CHILDREN
        try:
            run(plan)
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
