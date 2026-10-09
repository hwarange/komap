"""Run ONLY two folds of 4x MiT seed1, then stop irrespective of the pilot score."""
import fcntl
import json
import os
import subprocess
import sys

import run_resolution16 as runner

BASE_ARGS = runner.cv_args
runner.EXP = 'x4_mit_bw_r90_s1'
runner.OUT = runner.ROOT / 'runs/review16_seed1'


def seed_args(fold):
    values = BASE_ARGS(fold)
    values['seed'] = 1
    return values


runner.cv_args = seed_args


def main():
    os.chdir(runner.ROOT)
    runner.OUT.mkdir(parents=True, exist_ok=True)
    with open(runner.OUT / 'driver.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (runner.OUT / 'started.json').exists()
        assert not (runner.ROOT / f'runs/cv/{runner.EXP}').exists()
        runner.save('plan.json', {'exp': runner.EXP, 'created': runner.now(),
                    'folds': [0, 1], 'seed': 1, 'auto_remaining_folds': False,
                    'cv_args': {str(f): seed_args(f) for f in [0, 1]},
                    'reason': 'Repeat 4x at a different seed before migration; pilot-only by user instruction'})
        runner.save('started.json', {'time': runner.now(), 'pid': os.getpid()})
        try:
            runner.run_folds([0, 1])
            runner.status('evaluating_pilot')
            with open(runner.OUT / 'pilot.log', 'w') as log:
                subprocess.run([sys.executable, 'src/review_resolution16.py', '--exp', runner.EXP,
                                '--expected-seed', '1', '--out', str(runner.OUT / 'pilot.json')],
                               cwd=runner.ROOT, env=runner.ENV, stdout=log, stderr=subprocess.STDOUT, check=True)
            runner.status('complete_pilot_only', auto_remaining_folds=False)
            runner.save('all.done', {'time': runner.now(), 'outcome': 'pilot_complete_user_requested_stop'})
        except Exception as exc:
            for proc in runner.CHILDREN:
                if proc.poll() is None:
                    proc.terminate()
            runner.status('error', message=repr(exc))
            runner.save('error.json', {'time': runner.now(), 'error': repr(exc)})
            raise


if __name__ == '__main__':
    main()
