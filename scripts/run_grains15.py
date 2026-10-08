"""Run source pretraining and gated CV only on GPUs freed by round14.

No current jobs are interrupted. A filesystem lock prevents duplicate drivers.
Failed or ambiguous results stop the experiment and preserve current submissions.
"""
import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/review15'
BASE = 'x2_mit_bw_r90'
EXP = 'x2_mit_grains20_r90'
PRE = 'runs/pretrain/mit_grains20'
ENV = {**os.environ, 'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '1',
       'MKL_NUM_THREADS': '1', 'PYTHONUNBUFFERED': '1'}
CHILDREN = []


def now():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def save(name, value):
    p = OUT / name
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(p)


def status(stage, **fields):
    save('state.json', {'time': now(), 'pid': os.getpid(), 'stage': stage, **fields})
    print(now(), stage, fields, flush=True)


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


def cv_args(fold):
    a = json.loads((ROOT / f'runs/cv/{BASE}/fold{fold}/args.json').read_text())
    assert a['epochs'] == 60 and a['encoder'] == 'mit_b2' and a['seed'] == 0
    a.update(out=f'runs/cv/{EXP}/fold{fold}', save_oof=f'runs/oof/{EXP}',
             init=f'{PRE}/last.pt', init_skip_head=True, scale_jitter=0.15)
    return a


def round14_queue_empty():
    p = ROOT / 'runs/launch_round14.log'
    text = p.read_text() if p.exists() else ''
    if 'FAIL' in text:
        raise RuntimeError('round14 has a failure; preserve resources and inspect manually')
    return sum(' start ' in line for line in text.splitlines()) == 10


def free_gpus(reserved=()):
    if not round14_queue_empty():
        return []
    r = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    free = []
    for line in r.stdout.splitlines():
        g, mem, util = [int(x.strip()) for x in line.split(',')]
        if g in [0, 1, 2, 3] and g not in reserved and mem < 1000 and util < 10:
            free.append(g)
    return free


def launch(cmd, gpu, logfile):
    f = open(OUT / logfile, 'w')
    try:
        proc = subprocess.Popen(cmd, cwd=ROOT, env={**ENV, 'CUDA_VISIBLE_DEVICES': str(gpu)},
                                stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    finally:
        f.close()
    print(now(), 'started', 'GPU', gpu, 'pid', proc.pid, 'cmd', cmd, flush=True)
    CHILDREN.append(proc)
    return proc


def cpu_run(cmd, logfile):
    with open(OUT / logfile, 'w') as f:
        subprocess.run(cmd, cwd=ROOT, env=ENV, stdout=f, stderr=subprocess.STDOUT, check=True)


def wait_source(cmd):
    status('waiting_for_first_idle_gpu', queue_empty_required=True)
    while True:
        first = free_gpus()
        if first:
            time.sleep(15)
            second = free_gpus()
            eligible = [g for g in first if g in second]
            if eligible:
                gpu = eligible[0]
                break
        time.sleep(15)
    proc = launch(cmd, gpu, 'pretrain.log')
    status('source_pretraining', gpu=gpu, child_pid=proc.pid)
    while proc.poll() is None:
        time.sleep(15)
    if proc.returncode != 0:
        raise RuntimeError(f'external pretraining failed with code {proc.returncode}')
    p = ROOT / PRE
    logs = [json.loads(s) for s in (p / 'log.jsonl').read_text().splitlines()]
    import math
    assert len(logs) == 20 and logs[-1]['epoch'] == 20
    assert all(math.isfinite(r['loss']) for r in logs)
    assert (p / 'last.pt').exists()
    result = json.loads((p / 'result.json').read_text())
    # The two output slots mean grain interior/boundary, not KoMaP phases.
    values = [row['tta'][1] for row in result['rows']]
    assert values and all(math.isfinite(v) for v in values)
    foreground_iou = sum(values) / len(values)
    save('source_sanity.json', {'last_epoch': 20, 'boundary_IoU': foreground_iou,
                              'minimum_boundary_IoU': 0.10, 'passed': foreground_iou >= 0.10})
    return foreground_iou >= 0.10


def run_folds(folds):
    pending, running = list(folds), {}
    while pending or running:
        for fold in list(running):
            proc, gpu = running[fold]
            if proc.poll() is not None:
                if proc.returncode != 0:
                    raise RuntimeError(f'fold{fold} failed with code {proc.returncode}')
                assert (ROOT / f'runs/cv/{EXP}/fold{fold}/result.json').exists()
                del running[fold]
                print(now(), 'fold completed', fold, flush=True)
        free = free_gpus([gpu for _, gpu in running.values()])
        if pending and free:
            time.sleep(15)
            stable = free_gpus([gpu for _, gpu in running.values()])
            for gpu in [g for g in free if g in stable]:
                if not pending:
                    break
                fold = pending.pop(0)
                target = ROOT / f'runs/cv/{EXP}/fold{fold}'
                if target.exists():
                    raise RuntimeError(f'{target} already exists; inspect before resuming')
                cmd = [sys.executable, 'src/train.py', *cli_args(cv_args(fold))]
                proc = launch(cmd, gpu, f'fold{fold}.log')
                running[fold] = proc, gpu
        status('cv_pilot' if folds == [0, 1] else 'cv_remaining', pending=pending,
               running={str(f): {'gpu': g, 'pid': p.pid} for f, (p, g) in running.items()})
        if pending or running:
            time.sleep(30)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true', help='Validate plan without starting work')
    args = parser.parse_args()
    os.chdir(ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    approval = json.loads((OUT / 'audit_approved.json').read_text())
    assert approval['approved'] is True
    audit_path = ROOT / 'external/kaggle_grains_seg/audit.json'
    assert hashlib.sha256(audit_path.read_bytes()).hexdigest() == approval['audit_sha256']
    audit = json.loads(audit_path.read_text())
    assert audit['ready_for_pretraining'] is True
    assert not (ROOT / PRE).exists(), 'Pretrain output exists; inspect before resuming.'
    source = {
        'data': 'external/kaggle_grains_seg', 'out': PRE, 'encoder': 'mit_b2',
        'binary_class': 1, 'amp': True, 'workers': 2, 'epochs': 20,
        'bs': 8, 'spi': 4, 'max_iters': 200, 'crop': 384, 'width': 0,
        'tile': 384, 'overlap': 128, 'bw': 2, 'rot': 'rot90', 'seed': 0,
    }
    plan = {
        'created': now(), 'dataset': 'peterwarren/voronoi-artificial-grains-gen',
        'version': 4, 'audit_approval': approval, 'source_args': source,
        'cv_args': {str(f): cv_args(f) for f in range(5)},
        'pilot_folds': [0, 1], 'remaining_folds_if_pass': [2, 3, 4],
        'checkpoint_policy': 'fixed last source epoch20 and last KoMaP epoch60',
        'submission_policy': 'report only; preserve sub6 pending full review',
        'gpu_policy': 'queue empty for round14, memory<1000MiB and utilization<10% twice15s apart',
    }
    save('plan.json', plan)
    if args.check:
        print('PLAN_CHECK_PASS', flush=True)
        return
    with open(OUT / 'driver.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (OUT / 'started.json').exists(), 'Driver has already started.'
        save('started.json', {'time': now(), 'pid': os.getpid()})
        try:
            if not wait_source([sys.executable, 'src/train.py', *cli_args(source)]):
                status('stopped_source_sanity', reason='boundary foreground IoU below predeclared minimum')
                save('all.done', {'time': now(), 'outcome': 'source_sanity_failed'})
                return
            run_folds([0, 1])
            cpu_run([sys.executable, 'src/review_grains15.py', '--out', 'runs/review15/pilot.json'], 'pilot.log')
            pilot = json.loads((OUT / 'pilot.json').read_text())
            if not pilot['continue_full_cv']:
                status('stopped_pilot', delta=pilot['delta'], checks=pilot['checks'])
                save('all.done', {'time': now(), 'outcome': 'pilot_not_improved'})
                return
            run_folds([2, 3, 4])
            cpu_run([sys.executable, 'src/review_grains15.py', '--folds', '0', '1', '2', '3', '4', '--ensemble',
                     '--out', 'runs/review15/full.json'], 'full.log')
            cpu_run([sys.executable, 'src/review.py', EXP, '--base', BASE], 'review.log')
            # CPU ensemble reads many OOF arrays; serialize after round14 evaluation.
            status('waiting_round14_evaluation')
            while not (ROOT / 'runs/review14/all.done').exists():
                if (ROOT / 'runs/review14/exit_code.txt').exists():
                    code = int((ROOT / 'runs/review14/exit_code.txt').read_text())
                    if code:
                        raise RuntimeError(f'round14 evaluation exit code {code}')
                time.sleep(30)
            cpu_run([sys.executable, 'src/ensemble.py', '--exps', BASE,
                     'x3_unetpp_bw_e120_r90', 'x2_mit_bw_e120_r90', 'x2_mit_bw_r90_sj0', EXP,
                     '--max-size', '4', '--out', 'runs/oof/ensemble_summary_r15.json'], 'ensemble.log')
            status('complete_full_cv', submission_changed=False)
            save('all.done', {'time': now(), 'outcome': 'full_cv_complete_submission_unchanged'})
        except Exception as exc:
            for child in CHILDREN:
                if child.poll() is None:
                    child.terminate()
            for child in CHILDREN:
                if child.poll() is None:
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill()
            status('error', message=repr(exc))
            save('error.json', {'time': now(), 'error': repr(exc)})
            raise


if __name__ == '__main__':
    main()
