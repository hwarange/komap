"""Observe the retained seed0 children and seed1 pilot; report and stop, never launch folds."""
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import run_resolution16 as r

OUT = r.ROOT / 'runs/review16_closeout'


def save(name, value):
    p = OUT / name
    temp = p.with_suffix(p.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temp.replace(p)


def main():
    os.chdir(r.ROOT)
    OUT.mkdir(exist_ok=True)
    with open(OUT / 'driver.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (OUT / 'started.json').exists()
        save('started.json', {'time': r.now(), 'pid': os.getpid()})
        original = json.loads((r.OUT / 'scope_update.json').read_text())['pilot_children']
        try:
            while True:
                done = [f for f in [0, 1] if (r.ROOT / f'runs/cv/{r.EXP}/fold{f}/result.json').exists()]
                for f, row in original.items():
                    if int(f) not in done:
                        proc = Path(f"/proc/{row['pid']}/cmdline")
                        assert proc.exists() and b'src/train.py' in proc.read_bytes(), f'Original fold{f} stopped before result'
                if (r.ROOT / 'runs/review16_seed1/error.json').exists():
                    raise RuntimeError('Seed1 failed; inspect runs/review16_seed1/error.json')
                seed_done = (r.ROOT / 'runs/review16_seed1/all.done').exists()
                save('state.json', {'time': r.now(), 'pid': os.getpid(), 'stage': 'waiting_pilots',
                     'seed0_completed_folds': done, 'seed1_complete': seed_done, 'auto_remaining_folds': False})
                if len(done) == 2 and seed_done:
                    break
                time.sleep(30)
            with open(r.OUT / 'pilot.log', 'w') as log:
                subprocess.run([sys.executable, 'src/review_resolution16.py', '--out', str(r.OUT / 'pilot.json')],
                               env=r.ENV, stdout=log, stderr=subprocess.STDOUT, check=True)
            with open(OUT / 'comparison.log', 'w') as log:
                subprocess.run([sys.executable, 'src/review_resolution16_seeds.py'],
                               env=r.ENV, stdout=log, stderr=subprocess.STDOUT, check=True)
            result = {'time': r.now(), 'outcome': 'both_seed_pilots_complete_user_requested_stop',
                      'remaining_folds_run': False, 'submission_changed': False}
            save('all.done', result)
            save('state.json', {'pid': os.getpid(), 'stage': 'complete_pilot_only', **result})
            for parent in [r.OUT]:
                (parent / 'all.done').write_text(json.dumps(result, indent=2))
                (parent / 'state.json').write_text(json.dumps({'pid': os.getpid(), 'stage': 'complete_pilot_only', **result}, indent=2))
            print(json.dumps(result), flush=True)
        except Exception as exc:
            save('error.json', {'time': r.now(), 'error': repr(exc)})
            save('state.json', {'time': r.now(), 'stage': 'error', 'error': repr(exc)})
            raise


if __name__ == '__main__':
    main()
