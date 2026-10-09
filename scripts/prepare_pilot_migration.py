"""Mirror selected restore assets, wait for both pilots, then verify the final backup.

Uses the existing resumable SHA256 backup protocol and lock; never deletes server assets.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from backup_to_laptop import Backup, CRITICAL_MODELS, atomic_json, order, utc_now

WORKSPACE = Path(__file__).resolve().parents[1]
NEW_MODELS = {'x4_mit_bw_r90', 'x4_mit_bw_r90_s1'}


def selected(path):
    p = path.split('/')
    if p[0] in {'src', 'scripts', 'experiments', 'reports', 'splits', 'data', 'backup_metadata'}:
        return True
    if path in {'requirements.txt', '.gitignore', '.gitattributes'}:
        return True
    if len(p) > 1 and p[:2] == ['runs', 'si_clf']:
        return True
    if len(p) > 1 and p[0] == 'runs' and p[1] in {'review13', 'review14', 'review15', 'review16', 'review16_seed1', 'review16_closeout'}:
        return True
    if len(p) > 2 and p[:2] == ['runs', 'cv'] and p[2] in CRITICAL_MODELS | NEW_MODELS:
        return p[-1] != 'best.pt'
    if len(p) > 2 and p[:2] == ['runs', 'oof'] and p[2] in CRITICAL_MODELS | NEW_MODELS:
        return True
    if len(p) > 1 and p[0] == 'submissions' and p[1] in {'sub6', 'sub6_si'}:
        return 'parts' not in p
    return False


class MigrationBackup(Backup):
    def fetch_inventory(self):
        super().fetch_inventory()
        self.selected = sorted((f for f in self.inventory if selected(f['path'])), key=order)
        self.write_status(force=True, purpose='selected_restore_assets_after_two_seed_pilot',
                          state='migration_inventory_received')


def remote_completed(server):
    r = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', server,
                        'cd ~/komap && .venv/bin/python -c "from pathlib import Path; '
                        "print('ERROR' if Path('runs/review16_closeout/error.json').exists() else "
                        "'DONE' if Path('runs/review16_closeout/all.done').exists() else 'WAIT')\""],
                       capture_output=True, text=True, timeout=30, **Backup.process_options())
    if r.returncode:
        raise RuntimeError(f'Status SSH failed: {r.stderr.strip()}')
    return r.stdout.strip()


def write_migration_status(root, state, **fields):
    atomic_json(root / 'migration_status.json', {'updated_utc': utc_now(), 'state': state, **fields})


def final_verify(backup):
    required = {'runs/review16_closeout/all.done', 'runs/review16_closeout/results.json',
                'runs/review16_closeout/summary.md', 'splits/folds5.json', 'src/train.py',
                'requirements.txt', 'backup_metadata/runtime_metadata.json'}
    for model in CRITICAL_MODELS:
        required |= {f'runs/cv/{model}/fold{f}/last.pt' for f in range(5)}
    for model in NEW_MODELS:
        required |= {f'runs/cv/{model}/fold{f}/{name}' for f in [0, 1]
                     for name in ['last.pt', 'args.json', 'result.json', 'log.jsonl']}
    folds = json.loads((WORKSPACE / 'splits/folds5.json').read_text())
    for model in CRITICAL_MODELS | NEW_MODELS:
        required |= {f'runs/oof/{model}/{n}.npy' for n, f in folds.items()
                     if model in CRITICAL_MODELS or f in [0, 1]}
    complete = {f['path'] for f in backup.selected if backup.verified(f)}
    assert required <= complete, f'Missing restore files: {sorted(required - complete)}'
    checked = []
    for f in backup.selected:
        assert backup.verified(f), f['path']
        p = backup.target(f['path'])
        digest = hashlib.sha256()
        with p.open('rb') as source:
            for chunk in iter(lambda: source.read(4 * 1024 * 1024), b''):
                digest.update(chunk)
        assert digest.hexdigest() == backup.index[f['path']]['sha256'], f['path']
        checked.append({'path': f['path'], 'size': f['size'], 'sha256': digest.hexdigest()})
    atomic_json(backup.root / 'migration_verified_manifest.json',
                {'verified_utc': utc_now(), 'files': checked, 'required_files': sorted(required)})
    return len(checked), sum(f['size'] for f in checked)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--backup-root', default=r'D:\KoMaP_Backup_20261009')
    p.add_argument('--server', default='komap')
    p.add_argument('--check', action='store_true')
    args = p.parse_args()
    args.critical_only = False
    args.once = True
    args.poll_seconds = 60
    if args.check:
        assert selected('runs/cv/x4_mit_bw_r90_s1/fold0/last.pt')
        assert not selected('runs/cv/x4_mit_bw_r90_s1/fold0/best.pt')
        assert not selected('submissions/sub3_pred/parts/a.npy')
        print('MIGRATION_SELECTION_CHECK_PASS')
        return
    backup = MigrationBackup(args)
    backup.acquire_lock()
    try:
        backup.load_index()
        write_migration_status(backup.root, 'copying_existing_restore_assets', auto_remaining_folds=False)
        for attempt in range(3):
            if backup.sync_once():
                break
        write_migration_status(backup.root, 'waiting_for_pilot_results',
                               server_assets_verified=backup.status['verified_files'], auto_remaining_folds=False)
        while True:
            state = remote_completed(args.server)
            if state == 'ERROR':
                raise RuntimeError('Pilot closeout failed; preserve backup and inspect server logs')
            if state == 'DONE':
                break
            if state != 'WAIT':
                raise RuntimeError(f'Unexpected pilot status {state!r}')
            time.sleep(45)
        write_migration_status(backup.root, 'copying_completed_pilots')
        for attempt in range(3):
            if backup.sync_once():
                break
        count, size = final_verify(backup)
        shutil.copytree(WORKSPACE, backup.root / 'local_project', dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('.venv', 'Komap', '__pycache__', '*.pyc'))
        for f in ['results.json', 'summary.md']:
            shutil.copy2(backup.project / 'runs/review16_closeout' / f, WORKSPACE / 'runs/review16' / ('closeout_' + f))
        shutil.copy2(backup.project / 'runs/review16_closeout/summary.md', WORKSPACE / 'reports/round16_pilot_results.md')
        write_migration_status(backup.root, 'ready_selected_restore_assets', verified_files=count,
                               verified_bytes=size, pilot_results_local=True, server_gpu_jobs_finished=True,
                               scope='Current sub6 + both 4x seed pilots + data/code/OOF; excludes bulk old experiment assets',
                               new_server_transfer_done=False)
        print(f'MIGRATION_BACKUP_READY {count} files {size} bytes', flush=True)
    except Exception as exc:
        write_migration_status(backup.root, 'error', error=repr(exc))
        raise
    finally:
        backup.lock.close()


if __name__ == '__main__':
    main()
