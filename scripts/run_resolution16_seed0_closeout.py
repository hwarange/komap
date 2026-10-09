"""Adopt the current trainer, finish seed 0 folds 2/3/4, report, and stop."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import run_resolution16_additional as shared

ROOT, OUT, ENV = shared.ROOT, shared.OUT, shared.ENV


def alive(pid, ticks):
    try:
        stat = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()
        return stat[19] == ticks and stat[0] != 'Z'
    except FileNotFoundError:
        return False


def prepare():
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT / 'src'))
    handover = json.loads((OUT / 'scope_change_handover.json').read_text())
    assert json.loads((ROOT / 'backup_metadata/additional_validation_dataset_verified.json').read_text())['all_sha256_match']
    assert json.loads((ROOT / 'backup_metadata/imagenet_init_verified.json').read_text())['all_sha256_match']
    folds = json.loads((ROOT / 'splits/folds5.json').read_text())
    jobs = [{'seed': 0, 'exp': 'x4_mit_bw_r90', 'fold': fold,
             'args': shared.config(0, 'x4_mit_bw_r90', fold)} for fold in [2, 3, 4]]
    current = json.loads((ROOT / jobs[0]['args']['out'] / 'args.json').read_text())
    assert current == jobs[0]['args'], 'Adopted trainer config differs'
    for job in jobs[1:]:
        assert not (ROOT / job['args']['out']).exists(), 'Do not overwrite another output'
    assert not any((ROOT / f'runs/cv/x4_mit_bw_r90_s1/fold{fold}').exists() for fold in [2, 3, 4])
    return handover, folds, jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--resume-handover', action='store_true')
    args = parser.parse_args()
    handover, folds, jobs = prepare()
    if args.check:
        print(json.dumps({'jobs': jobs, 'adopt_pid': handover['adopt_training_pid'],
                          'training_continues': alive(handover['adopt_training_pid'], handover['child_start_ticks'])}))
        return
    # Use a new lock after handover: adopted descendants can retain the old lock.
    assert not Path(f"/proc/{handover['old_controller_pid']}").exists(), 'Old controller is still alive'
    lock_name = 'seed0_closeout_driver_v2.lock' if args.resume_handover else 'seed0_closeout_driver.lock'
    with (OUT / lock_name).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (OUT / 'seed0_closeout_started.json').exists():
            previous = json.loads((OUT / 'seed0_closeout_started.json').read_text())
            assert args.resume_handover and not Path(f"/proc/{previous['pid']}").exists()
            shared.save('seed0_closeout_previous_started.json', previous)
        shared.save('seed0_closeout_started.json', {'pid': os.getpid(), 'time_utc': datetime.now(timezone.utc).isoformat()})
        shared.save('plan_seed0_closeout.json', {
            'jobs': jobs, 'training_jobs': 3, 'epochs_each': 60,
            'reuse_existing_pilot_folds': [0, 1], 'adopted_trainer': handover,
            'scope': 'Complete seed0 full CV; pilot seed1 comparison only; write final report and stop.',
            'fixed_candidate': 'x2 MiT 1/3 + x3 UNet++ 1/3 + x4 MiT seed0 1/3',
            'weight_search': False, 'submission_changed': False,
            'no_more_training_after_report': True})
        completed = []
        child = None
        try:
            for number, job in enumerate(jobs, 1):
                if number == 1:
                    pid, ticks = handover['adopt_training_pid'], handover['child_start_ticks']
                    while alive(pid, ticks):
                        shared.status('training', job_number=number, jobs_total=3, seed=0,
                                      fold=job['fold'], training_pid=pid, completed=completed,
                                      scope='seed0_only_then_report_and_stop', submission_changed=False)
                        time.sleep(15)
                else:
                    shared.wait_gpu()
                    with (OUT / f"seed0_fold{job['fold']}.log").open('w') as log:
                        child = subprocess.Popen([sys.executable, 'scripts/train_runpod.py',
                            *shared.original.cli_args(job['args'])], cwd=ROOT, env=ENV,
                            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
                        while child.poll() is None:
                            shared.status('training', job_number=number, jobs_total=3, seed=0,
                                          fold=job['fold'], training_pid=child.pid, completed=completed,
                                          scope='seed0_only_then_report_and_stop', submission_changed=False)
                            time.sleep(15)
                        if child.returncode:
                            raise RuntimeError(f"fold{job['fold']} exit {child.returncode}")
                completed.append(shared.validate_completed(job, folds))
                shared.save('completed_jobs.json', completed)
            for exp, selected in [('x2_mit_bw_r90', list(range(5))),
                                  ('x3_unetpp_bw_e120_r90', list(range(5))),
                                  ('x4_mit_bw_r90', [0, 1])]:
                shared.ensure_oof(exp, selected)
            shared.status('comparing_seed0_full_cv', completed=completed, submission_changed=False)
            with (OUT / 'seed0_comparison.log').open('w') as log:
                subprocess.run([sys.executable, 'src/review_resolution16_seed0_final.py'],
                               cwd=ROOT, env=ENV, stdout=log, stderr=subprocess.STDOUT, check=True)
            artifacts = []
            for fold in range(5):
                for name in ['args.json', 'log.jsonl', 'result.json', 'best.pt', 'last.pt']:
                    path = ROOT / f'runs/cv/x4_mit_bw_r90/fold{fold}' / name
                    digest = hashlib.sha256()
                    with path.open('rb') as stream:
                        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b''):
                            digest.update(chunk)
                    artifacts.append({'path': path.relative_to(ROOT).as_posix(),
                                      'bytes': path.stat().st_size, 'sha256': digest.hexdigest()})
            shared.save('seed0_artifacts.json', artifacts)
            shared.status('complete_seed0_report_and_stop', completed=completed,
                          report='reports/round16_seed0_final.md', submission_changed=False,
                          further_training_scheduled=False)
            shared.save('all.done', {'time_utc': datetime.now(timezone.utc).isoformat(),
                        'outcome': 'three_seed0_jobs_full_cv_and_final_report_complete',
                        'report': 'reports/round16_seed0_final.md', 'further_training_scheduled': False})
        except BaseException as error:
            if child and child.poll() is None:
                import signal
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
            shared.status('error', message=repr(error), completed=completed)
            shared.save('error.json', {'error': repr(error), 'scope': 'seed0_closeout'})
            raise


if __name__ == '__main__':
    main()
