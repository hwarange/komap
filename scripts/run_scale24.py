"""Run two single-holdout scale trials on GPUs 2/3, then review all completed jobs."""
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
OUT = ROOT / 'runs/review24_single_scale'
ENV = {**common.ENV, 'PYTHONDONTWRITEBYTECODE': '1'}
CHILDREN = []
DEFINITIONS = {
    'mit5': dict(experiment='x5_mit_c1536_bw_r90', reference='x4_mit_c1536_bw_r90', gpu=2,
                 config=dict(arch='Unet', encoder='mit_b2', width=2560, scale_multiplier=2.5,
                             crop=1536, tile=1536, bs=2, accum_steps=2, epochs=60, seed=0)),
    'deeplab4': dict(experiment='x4_deeplabv3plus_r34_bw_r90',
                     reference='x2_deeplabv3plus_r34_bw_r90', gpu=3,
                     config=dict(arch='DeepLabV3Plus', encoder='resnet34', width=2560,
                                 scale_multiplier=2.0, crop=1024, tile=1024,
                                 bs=4, accum_steps=1, epochs=60, seed=0)),
}
BASELINES = ['x2_mit_bw_r90', 'x4_mit_c1536_bw_r90',
             'x3_unetpp_bw_e120_r90', 'x2_deeplabv3plus_r34_bw_r90']
SOURCES = ['src/train.py', 'src/data.py', 'src/infer.py', 'src/review_grains15.py',
           'scripts/run_crop18.py', 'scripts/run_scale24.py',
           'scripts/review_scale24.py', 'scripts/check_scale24_runtime.py']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(name, value):
    path = OUT / name
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def status(stage, **values):
    record = dict(time=common.now(), pid=os.getpid(), stage=stage, fixed_holdout_fold=0,
                  repeated_cross_validation=False, training_images=72, validation_images=18,
                  further_training_scheduled=False, submission_changed=False, **values)
    save('state.json', record)
    print(json.dumps(record, ensure_ascii=False, allow_nan=False), flush=True)


def verify_hashes(records):
    for name, digest in records.items():
        assert common.sha256(common.under_root(name)) == digest, f'Pinned input changed: {name}'


def inventory():
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
                                    '--format=csv,noheader,nounits'], text=True, timeout=15)
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                    '--format=csv,noheader,nounits'], text=True, timeout=15)
    busy = {line.split(',')[0].strip() for line in apps.splitlines() if line.strip()}
    records = {}
    for line in rows.splitlines():
        index, uuid, memory, utilization = [part.strip() for part in line.split(',')]
        records[int(index)] = dict(index=int(index), uuid=uuid, memory_mib=int(memory),
                                   utilization=int(utilization),
                                   idle=int(memory) < 1000 and int(utilization) < 10 and uuid not in busy)
    return records


def cv_args(arm):
    spec = DEFINITIONS[arm]
    args = read(ROOT / f"runs/cv/{spec['reference']}/fold0/args.json")
    config = spec['config']
    for key in ['arch', 'encoder', 'width', 'crop', 'tile', 'bs', 'accum_steps', 'epochs', 'seed']:
        assert args.get(key, 1 if key == 'accum_steps' else None) == config[key], (arm, key)
    assert args['fold'] == 0 and args['spi'] == 8 and args['bw'] == 2 and args['rot'] == 'rot90'
    assert args['scale_multiplier'] == (2.0 if arm == 'mit5' else 1.0)
    assert args['amp'] and not args.get('init') and not args.get('full') and not args.get('attention')
    args.update(config, fold=0, workers=4, scale_jitter=0.15,
                out=f"runs/cv/{spec['experiment']}/fold0", save_oof=f"runs/oof/{spec['experiment']}")
    return args


def build_plan():
    for spec in DEFINITIONS.values():
        for directory in ['cv', 'oof']:
            assert not (ROOT / f"runs/{directory}/{spec['experiment']}").exists(), 'Refusing experiment overwrite'
    for name in ['plan.json', 'started.json', 'all.done', 'error.json', 'results.json']:
        assert not (OUT / name).exists(), f'Review already started: {name}'
    fold_of = read(ROOT / 'splits/folds5.json')
    assert len(fold_of) == 90 and set(fold_of.values()) == set(range(5))
    grouped = {}
    for name, fold in fold_of.items():
        grouped.setdefault(name.split('_')[0], set()).add(fold)
    assert all(len(values) == 1 for values in grouped.values())
    names = sorted(name for name, fold in fold_of.items() if fold == 0)
    assert len(names) == 18
    prerequisite_hashes = {f'runs/review23_unetpp4/{name}': common.sha256(ROOT / f'runs/review23_unetpp4/{name}')
                           for name in ['plan.json', 'started.json']}
    prior = read(ROOT / 'runs/review23_unetpp4/plan.json')
    assert prior['split_sha256'] == common.sha256(ROOT / 'splits/folds5.json')
    assert prior['folds'] == [0, 1] and prior['config']['epochs'] == 60
    assert read(ROOT / 'runs/review23_unetpp4/started.json')['plan_sha256'] == prerequisite_hashes['runs/review23_unetpp4/plan.json']
    assert len(prior['data_hashes']) == 180
    verify_hashes(prior['data_hashes'])
    verify_hashes(prior['source_hashes'])
    deep_path = ROOT / 'runs/review21_deeplab/results.json'
    assert common.sha256(deep_path) == read(deep_path.parent / 'all.done')['results_sha256']
    deep_receipts = {row['path']: row['sha256'] for row in read(deep_path)['provenance']['source_files']
                     if row['source'] == 'deeplab'}
    oof, runs = {}, {}
    for model in BASELINES:
        run_dir = ROOT / f'runs/cv/{model}/fold0'
        args, result = read(run_dir / 'args.json'), read(run_dir / 'result.json')
        assert args['fold'] == 0 and args['folds_file'] == 'splits/folds5.json'
        assert args['epochs'] == (120 if 'e120' in model else 60)
        assert sorted(row['name'] for row in result['rows']) == names
        for filename in ['args.json', 'result.json']:
            path = run_dir / filename
            runs[path.relative_to(ROOT).as_posix()] = common.sha256(path)
        for name in names:
            relative = f'runs/oof/{model}/{name}.npy'
            expected = deep_receipts[relative] if model.startswith('x2_deeplab') else prior['baseline_oof_hashes'][relative]
            assert common.sha256(ROOT / relative) == expected
            oof[relative] = expected
    assert len(oof) == 72 and len(runs) == 8
    os.environ.update(common.THREAD_ENV)
    from huggingface_hub import try_to_load_from_cache
    from segmentation_models_pytorch.encoders import encoders
    caches, measurements, jobs = {}, {}, []
    cache_specs = {
        'mit_b2': ('smp-hub/mit_b2.imagenet', common.IMAGE_NET_REVISION, common.IMAGE_NET_HASHES),
        'resnet34': ('smp-hub/resnet34.imagenet', '7a57b34f723329ff020b3f8bc41771163c519d0c',
                     {'config.json': '01bf2cf24eb29b405c28c159f46ceda92c098ab85868be88fb100967db47166e',
                      'model.safetensors': '6549ccd46a2d20a0a74fca866cfc5c6f9b3672d468b0a92aa94a58545a61103e'}),
    }
    for arm, spec in DEFINITIONS.items():
        args = cv_args(arm)
        repo, revision, expected_files = cache_specs[args['encoder']]
        assert encoders[args['encoder']]['pretrained_settings']['imagenet'] == dict(repo_id=repo, revision=revision)
        files = []
        for filename, digest in expected_files.items():
            path = try_to_load_from_cache(repo, filename, revision=revision)
            assert isinstance(path, str) and common.sha256(path) == digest
            files.append(dict(file=filename, path=path, sha256=digest))
        caches[arm] = dict(repo_id=repo, revision=revision, files=files)
        measure_path = OUT / f'runtime_{arm}.json'
        measurement = read(measure_path)
        assert measurement['status'] == 'complete'
        assert all(measurement['benchmark_args'].get(key) == value for key, value in spec['config'].items())
        assert measurement['script_sha256'] == common.sha256(ROOT / 'scripts/check_scale24_runtime.py')
        measurements[arm] = dict(path=measure_path.relative_to(ROOT).as_posix(), sha256=common.sha256(measure_path))
        jobs.append(dict(id=arm, arm=arm, experiment=spec['experiment'], gpu=spec['gpu'], fold=0,
                         args=args, validation_names=names, expected_microbatches=576 // args['bs'],
                         expected_updates=144))
    return dict(created=common.now(), fixed_holdout_fold=0, repeated_cross_validation=False,
                training_images=72, validation_images=18, jobs=jobs,
                split_sha256=common.sha256(ROOT / 'splits/folds5.json'),
                source_hashes={name: common.sha256(ROOT / name) for name in SOURCES},
                data_hashes=prior['data_hashes'], baseline_oof_hashes=oof, baseline_run_hashes=runs,
                prerequisite_hashes=prerequisite_hashes, imagenet=caches, runtime_checks=measurements,
                new_training_jobs=2, new_seeds=[0], checkpoint='last.pt at epoch60',
                wait_for='runs/review23_unetpp4/all.done', further_training_scheduled=False,
                submission_changed=False, independent_validation=False,
                note='One fixed holdout per model, no repeated folds; compare all models on the same 18 images after all four jobs complete')


def validate_completed(job):
    run = common.under_root(job['args']['out'])
    logs = common.read_epoch_logs(run / 'log.jsonl')
    assert [row['epoch'] for row in logs] == list(range(1, 61))
    assert all(math.isfinite(row['loss']) and row['microbatches'] == job['expected_microbatches']
               and row['optimizer_updates'] == 144 and row['accum_steps'] == job['args']['accum_steps']
               and row['effective_batch'] == 4 for row in logs)
    args = read(run / 'args.json')
    assert args == job['args']
    result = read(run / 'result.json')
    assert result['args'] == args and sorted(row['name'] for row in result['rows']) == job['validation_names']
    assert (run / 'last.pt').stat().st_size > 1000000
    for name in job['validation_names']:
        assert (common.under_root(args['save_oof']) / f'{name}.npy').is_file()
    return dict(id=job['id'], arm=job['arm'], completed=common.now(), epochs=60, images=18,
                train_seconds=sum(row['train_sec'] for row in logs),
                result_sha256=common.sha256(run / 'result.json'))


def run(plan):
    for key in ['source_hashes', 'data_hashes', 'baseline_oof_hashes', 'baseline_run_hashes', 'prerequisite_hashes']:
        verify_hashes(plan[key])
    first = inventory()
    time.sleep(3)
    second = inventory()
    for index in [2, 3]:
        assert first[index]['idle'] and second[index]['idle'] and first[index]['uuid'] == second[index]['uuid'], f'GPU {index} is not idle'
    running, completed, failures = {}, [], []
    for job in plan['jobs']:
        gpu = second[job['gpu']]
        with (OUT / 'logs' / (job['id'] + '.log')).open('x', encoding='utf-8') as stream:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--train-worker', *common.cli_args(job['args'])],
                                     cwd=ROOT, env={**ENV, 'CUDA_VISIBLE_DEVICES': gpu['uuid']},
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        CHILDREN.append(child)
        running[job['id']] = (job, child, gpu['index'], time.monotonic())
    while running:
        for identifier, values in list(running.items()):
            job, child, gpu, started = values
            if time.monotonic() - started > 10 * 3600:
                raise TimeoutError(f'Training exceeded ten hours: {identifier}')
            if child.poll() is None:
                continue
            if child.returncode:
                failures.append(dict(id=identifier, returncode=child.returncode))
            else:
                completed.append(validate_completed(job))
            save('completed_jobs.json', completed)
            del running[identifier]
        progress = [common.job_progress(*values) for values in running.values()]
        status('single_holdout_training' if running else 'new_training_complete', jobs=progress,
               completed_jobs=completed, failed_jobs=failures)
        if running:
            time.sleep(15)
    assert not failures, f'Training failures: {failures}'
    wait_started = time.monotonic()
    prerequisite = ROOT / 'runs/review23_unetpp4'
    while not (prerequisite / 'all.done').exists():
        assert not (prerequisite / 'error.json').exists(), 'Existing UNet++ experiment has an error'
        assert time.monotonic() - wait_started < 8 * 3600, 'Timed out waiting for existing UNet++ jobs'
        status('waiting_for_existing_unetpp_jobs', jobs=[], completed_jobs=completed,
               prerequisite_state=read(prerequisite / 'state.json'))
        time.sleep(30)
    verify_hashes(plan['prerequisite_hashes'])
    assert read(prerequisite / 'all.done')['results_sha256'] == common.sha256(prerequisite / 'results.json')
    for key in ['source_hashes', 'data_hashes', 'baseline_oof_hashes', 'baseline_run_hashes']:
        verify_hashes(plan[key])
    status('all_four_jobs_complete_evaluating_cpu', jobs=[], completed_jobs=completed)
    with (OUT / 'logs/evaluation.log').open('x', encoding='utf-8') as stream:
        child = subprocess.Popen([sys.executable, 'scripts/review_scale24.py'], cwd=ROOT,
                                 env={**ENV, 'CUDA_VISIBLE_DEVICES': ''}, stdout=stream, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        CHILDREN.append(child)
        result_code = child.wait(timeout=3600)
    assert result_code == 0, f'CPU evaluation failed: {result_code}'
    verify_hashes(plan['source_hashes'])
    for name in ['results.json', 'per_image_rows.json', 'summary.txt']:
        assert (OUT / name).is_file()
    status('all_four_jobs_complete_and_reviewed', jobs=[], completed_jobs=completed)
    save('all.done', dict(time=common.now(), outcome='two_single_holdout_runs_and_all_four_jobs_reviewed',
                         new_training_jobs=2, validation_images=18, repeated_cross_validation=False,
                         results_sha256=common.sha256(OUT / 'results.json'),
                         plan_sha256=common.sha256(OUT / 'plan.json'),
                         prerequisite_results_sha256=common.sha256(prerequisite / 'results.json'),
                         further_training_scheduled=False, submission_changed=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.check:
        plan = build_plan()
        print(json.dumps(dict(status='preflight_passed', jobs=[dict(id=j['id'], gpu=j['gpu'], args=j['args']) for j in plan['jobs']],
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
            json.dump(dict(time=common.now(), pid=os.getpid(), plan_sha256=common.sha256(OUT / 'plan.json')), stream)
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
