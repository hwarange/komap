"""Run exactly six remaining 4x MiT jobs, then regenerate OOF and compare seeds."""
import argparse
from datetime import datetime, timezone
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

import run_resolution16 as original

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/review16_additional'
SEEDS = [(0, 'x4_mit_bw_r90'), (1, 'x4_mit_bw_r90_s1')]
ENV = {**os.environ, 'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1', 'PYTHONUNBUFFERED': '1', 'CUDA_VISIBLE_DEVICES': '0', 'HF_HUB_OFFLINE': '1'}


def save(name, value):
    p = OUT / name
    temporary = p.with_suffix(p.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(p)


def status(stage, **fields):
    value = {'time_utc': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(), 'stage': stage, **fields}
    save('state.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def config(seed, exp, fold):
    a = original.cv_args(fold)
    a.update(seed=seed, out=f'runs/cv/{exp}/fold{fold}', save_oof=f'runs/oof/{exp}')
    pilot = json.loads((ROOT / f'runs/cv/{exp}/fold0/args.json').read_text())
    assert {k:v for k,v in a.items() if k not in {'fold', 'out'}} == {
        k:v for k,v in pilot.items() if k not in {'fold', 'out'}}, 'Training config differs from its pilot'
    return a


def preflight():
    dataset = json.loads((ROOT / 'backup_metadata/additional_validation_dataset_verified.json').read_text())
    assert dataset['all_sha256_match'] and dataset['files'] == 180
    assert json.loads((ROOT / 'backup_metadata/imagenet_init_verified.json').read_text())['all_sha256_match']
    sys.path.insert(0, str(ROOT / 'src'))
    from data import load_split
    import numpy as np
    items = load_split('data', 'train') + load_split('data', 'valid')
    folds = json.loads((ROOT / 'splits/folds5.json').read_text())
    assert len(items) == len(folds) == 90 and {i['name'] for i in items} == set(folds)
    grouped = {}
    for it in items:
        assert it['lab'] is not None and it['gray'].shape == it['lab'].shape
        assert set(np.unique(it['lab'])).issubset({0, 1, 2, 3})
        grouped.setdefault(it['alloy'], set()).add(folds[it['name']])
    assert all(len(values) == 1 for values in grouped.values()), 'Alloy overlap between folds'
    assert all(sum(f == fold for f in folds.values()) == 18 for fold in range(5))
    jobs = [{'seed': seed, 'exp': exp, 'fold': fold, 'args': config(seed, exp, fold)}
        for fold in [2, 3, 4] for seed, exp in SEEDS]
    for job in jobs:
        assert not (ROOT / job['args']['out']).exists(), 'Existing experiment output must not be overwritten'
    return jobs, folds


def wait_gpu():
    while True:
        lines = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'], text=True).splitlines()
        assert len(lines) == 1, 'This controller is for the single-GPU Pod'
        index, memory, util = map(int, lines[0].split(','))
        if index == 0 and memory < 1000 and util < 10:
            return
        status('waiting_for_idle_gpu', memory_MiB=memory, utilization=util)
        time.sleep(15)


def validate_completed(job, folds):
    target = ROOT / job['args']['out']
    logs = [json.loads(line) for line in (target / 'log.jsonl').read_text().splitlines()]
    assert len(logs) == 60 and logs[-1]['epoch'] == 60 and all(math.isfinite(x['loss']) for x in logs)
    result = json.loads((target / 'result.json').read_text())
    assert result['args'] == job['args']
    names = [r['name'] for r in result['rows']]
    assert len(names) == len(set(names)) == 18 and set(names) == {n for n,f in folds.items() if f == job['fold']}
    for filename in ['best.pt', 'last.pt']:
        assert (target / filename).is_file()
    for name in names:
        assert (ROOT / job['args']['save_oof'] / f'{name}.npy').is_file()
    return {'seed':job['seed'], 'fold':job['fold'], 'exp':job['exp'],
        'train_seconds':sum(x['train_sec'] for x in logs), 'images':18}


def ensure_oof(exp, selected_folds):
    import cv2
    import numpy as np
    import torch
    from data import load_cv
    from infer import predict_probs
    from train import autocast, build_from_args
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    out = ROOT / 'runs/oof' / exp
    out.mkdir(parents=True, exist_ok=True)
    for fold in selected_folds:
        status('regenerating_oof', exp=exp, fold=fold, submission_changed=False)
        ck = torch.load(ROOT / f'runs/cv/{exp}/fold{fold}/last.pt', map_location='cpu', weights_only=False)
        a = ck['args']
        model = build_from_args(a, weights=None)
        model.load_state_dict(ck['model'], strict=True)
        del ck
        model = model.cuda().eval()
        _, items = load_cv(a['data'], fold, a['folds_file'])
        for it in items:
            target = out / f"{it['name']}.npy"
            if target.exists():
                raise RuntimeError('Regeneration would overwrite an OOF file: ' + str(target))
            with torch.inference_mode(), autocast(a.get('amp', False)):
                p = predict_probs(model, it['gray'], a['width'], a['tile'], a['overlap'], True,
                    'cuda', scale_multiplier=a.get('scale_multiplier',1.0))
            assert p.shape == (4,*it['gray'].shape) and np.isfinite(p).all()
            np.save(target, p.astype(np.float16))
        del model
        torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check', action='store_true')
    args = p.parse_args()
    os.chdir(ROOT)
    jobs, folds = preflight()
    plan = {'created_utc':datetime.now(timezone.utc).isoformat(), 'jobs':jobs,
        'reuse_existing_pilot_folds':[0,1], 'training_jobs':6, 'epochs_each':60,
        'order':'fold2 seed0/seed1, fold3 seed0/seed1, fold4 seed0/seed1',
        'initialization':'Original pinned ImageNet revision and source SHA256 verified; no pilot weights used for training',
        'data':'SHA256-matched original train/valid only; no test images or external grain data',
        'runtime_threads':{'OMP':4, 'BLAS':1, 'OpenCV':1, 'DataLoader':6},
        'scope':'Seed stability and generalization on the remaining three folds, followed by fixed full-CV comparison',
        'fixed_candidate':'x2 MiT 1/3 + x3 UNet++ 1/3 + x4 seed0 1/6 + x4 seed1 1/6',
        'submission_changed':False}
    if args.check:
        print(json.dumps(plan,ensure_ascii=False,indent=2),flush=True)
        return
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'driver.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (OUT/'started.json').exists(), 'Additional validation already started'
        save('plan.json',plan)
        save('started.json',{'pid':os.getpid(),'time_utc':datetime.now(timezone.utc).isoformat()})
        completed=[]
        child=None
        try:
            for number, job in enumerate(jobs,1):
                wait_gpu()
                command=[sys.executable,'scripts/train_runpod.py',*original.cli_args(job['args'])]
                with (OUT/f"seed{job['seed']}_fold{job['fold']}.log").open('w') as log:
                    child=subprocess.Popen(command,cwd=ROOT,env=ENV,stdin=subprocess.DEVNULL,
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    while child.poll() is None:
                        status('training',job_number=number,jobs_total=6,seed=job['seed'],fold=job['fold'],
                            training_pid=child.pid,completed=completed,submission_changed=False)
                        time.sleep(15)
                    if child.returncode:
                        raise RuntimeError(f"seed{job['seed']} fold{job['fold']} exit {child.returncode}")
                completed.append(validate_completed(job,folds))
                save('completed_jobs.json',completed)
            for exp, which in [('x2_mit_bw_r90',list(range(5))),
                ('x3_unetpp_bw_e120_r90',list(range(5))),('x4_mit_bw_r90',[0,1]),('x4_mit_bw_r90_s1',[0,1])]:
                ensure_oof(exp,which)
            status('comparing_full_cv',completed=completed,submission_changed=False)
            with (OUT/'full_comparison.log').open('w') as log:
                subprocess.run([sys.executable,'src/review_resolution16_full.py'],cwd=ROOT,env=ENV,
                    stdout=log,stderr=subprocess.STDOUT,check=True)
            status('complete_full_validation',completed=completed,submission_changed=False)
            save('all.done',{'time_utc':datetime.now(timezone.utc).isoformat(),'outcome':'six_jobs_and_full_cv_comparison_complete'})
        except BaseException as error:
            if child and child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try: child.wait(timeout=10)
                except subprocess.TimeoutExpired: os.killpg(child.pid,signal.SIGKILL)
            status('error',message=repr(error),completed=completed)
            save('error.json',{'error':repr(error)})
            raise


if __name__=='__main__':
    main()
