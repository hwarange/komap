"""Run a locked, gated 4x MiT experiment on idle GPUs without changing submissions."""
import argparse
import fcntl
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/review16'
BASE = 'x2_mit_bw_r90'
EXP = 'x4_mit_bw_r90'
ENV = {**os.environ, 'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '1',
       'MKL_NUM_THREADS': '1', 'PYTHONUNBUFFERED': '1'}
CHILDREN = []


def now():
    return datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds')


def save(name, value):
    p = OUT / name
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    tmp.replace(p)


def status(stage, **fields):
    save('state.json', {'time': now(), 'pid': os.getpid(), 'stage': stage, **fields})
    print(now(), stage, fields, flush=True)


def cv_args(fold):
    a = json.loads((ROOT / f'runs/cv/{BASE}/fold{fold}/args.json').read_text())
    assert a['epochs'] == 60 and a['encoder'] == 'mit_b2' and a['seed'] == 0
    assert a['width'] == 2560 and a['crop'] == a['tile'] == 1024 and a['bs'] == 4
    assert not a['init'] and not a['full']
    a.update(out=f'runs/cv/{EXP}/fold{fold}', save_oof=f'runs/oof/{EXP}',
             scale_jitter=0.15, scale_multiplier=2.0)
    return a


def cli_args(values):
    cmd = []
    for k, v in values.items():
        flag = '--' + k.replace('_', '-')
        if v is None or v == '' or v is False:
            continue
        if v is True:
            cmd.append(flag)
        elif isinstance(v, list):
            cmd += [flag, *map(str, v)]
        else:
            cmd += [flag, str(v)]
    return cmd


def free_gpus(reserved=()):
    r = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    return [g for line in r.stdout.splitlines() for g, mem, util in [map(int, line.split(','))]
            if g in range(4) and g not in reserved and mem < 1000 and util < 10]


def run_folds(folds):
    pending, running = list(folds), {}
    while pending or running:
        for f in list(running):
            proc, gpu = running[f]
            if proc.poll() is not None:
                if proc.returncode:
                    raise RuntimeError(f'fold{f} exit {proc.returncode}')
                target = ROOT / f'runs/cv/{EXP}/fold{f}'
                logs = [json.loads(l) for l in (target / 'log.jsonl').read_text().splitlines()]
                assert len(logs) == 60 and logs[-1]['epoch'] == 60
                assert all(math.isfinite(l['loss']) for l in logs)
                assert (target / 'result.json').exists()
                del running[f]
                print(now(), 'fold completed', f, flush=True)
        if pending:
            reserved = [g for _, g in running.values()]
            first = free_gpus(reserved)
            if first:
                time.sleep(5)
                stable = free_gpus(reserved)
                for gpu in [g for g in first if g in stable]:
                    if not pending:
                        break
                    f = pending.pop(0)
                    target = ROOT / f'runs/cv/{EXP}/fold{f}'
                    assert not target.exists(), f'{target} already exists'
                    cmd = [sys.executable, 'src/train.py', *cli_args(cv_args(f))]
                    with open(OUT / f'fold{f}.log', 'w') as log:
                        proc = subprocess.Popen(cmd, cwd=ROOT, env={**ENV, 'CUDA_VISIBLE_DEVICES': str(gpu)},
                                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    CHILDREN.append(proc)
                    running[f] = (proc, gpu)
                    print(now(), 'started', f, 'GPU', gpu, 'pid', proc.pid, flush=True)
        status('cv_pilot' if folds == [0, 1] else 'cv_remaining', pending=pending,
               running={str(f): {'gpu': g, 'pid': p.pid} for f, (p, g) in running.items()})
        if pending or running:
            time.sleep(15)


def evaluate(folds, filename):
    status('evaluating_pilot' if folds == [0, 1] else 'evaluating_full')
    cmd = [sys.executable, 'src/review_resolution16.py', '--folds', *map(str, folds),
           '--out', str(OUT / filename)]
    with open(OUT / filename.replace('.json', '.log'), 'w') as f:
        subprocess.run(cmd, cwd=ROOT, env=ENV, stdout=f, stderr=subprocess.STDOUT, check=True)
    return json.loads((OUT / filename).read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--check', action='store_true')
    args = p.parse_args()
    os.chdir(ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / 'src'))
    from review_resolution16 import RULES
    plan = {'created': now(), 'cv_args': {str(f): cv_args(f) for f in range(5)},
            'pilot_folds': [0, 1], 'remaining_folds_if_pass': [2, 3, 4], 'pilot_rules': RULES,
            'gate': 'single OR one fixed ensemble passes; no search for weights',
            'scale': 'scale_for(width,2560) times2 for both training and inference',
            'checkpoint': 'fixed last epoch60', 'submission_changed': False,
            'deadline_target': '2026-10-09T15:00:00+09:00; estimate, not a forced cutoff'}
    if args.check:
        assert not (ROOT / f'runs/cv/{EXP}').exists()
        print(json.dumps(plan, indent=2), flush=True)
        return
    with open(OUT / 'driver.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (OUT / 'started.json').exists(), 'Driver already started'
        assert not (ROOT / f'runs/cv/{EXP}').exists()
        save('plan.json', plan)
        save('started.json', {'time': now(), 'pid': os.getpid()})
        try:
            run_folds([0, 1])
            pilot = evaluate([0, 1], 'pilot.json')
            if not pilot['continue_full_cv']:
                status('stopped_pilot', single_delta=pilot['single']['delta'])
                save('all.done', {'time': now(), 'outcome': 'pilot_not_improved', 'submission_changed': False})
                return
            run_folds([2, 3, 4])
            evaluate([0, 1, 2, 3, 4], 'full.json')
            status('complete_full_cv', submission_changed=False)
            save('all.done', {'time': now(), 'outcome': 'full_cv_complete', 'submission_changed': False})
        except Exception as exc:
            for proc in CHILDREN:
                if proc.poll() is None:
                    proc.terminate()
            for proc in CHILDREN:
                if proc.poll() is None:
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
            status('error', message=repr(exc))
            save('error.json', {'time': now(), 'error': repr(exc)})
            raise


if __name__ == '__main__':
    main()
