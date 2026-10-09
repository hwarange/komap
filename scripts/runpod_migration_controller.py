"""Run rsync+SHA256 verification and uv restore concurrently on the source server.

Management logs live outside the project so the frozen asset manifest stays stable.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--state-dir', type=Path, required=True)
    p.add_argument('--destination', required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--identity', type=Path, required=True)
    p.add_argument('--transfer-script', type=Path)
    p.add_argument('--host-key-alias')
    p.add_argument('--known-hosts', type=Path)
    p.add_argument('--uv-already-verified', action='store_true')
    p.add_argument('--allow-ephemeral', action='store_true')
    args = p.parse_args()
    root = args.root.resolve()
    state_dir = args.state_dir.resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    state_file = state_dir / 'state.json'
    if state_file.exists():
        previous = json.loads(state_file.read_text())
        if previous.get('stage') not in {'failed', 'complete'}:
            raise RuntimeError('A controller run is already recorded; check its PID before retrying')
    manifest = json.loads((root / 'backup_metadata/transfer_manifest.json').read_text())
    ssh = ['ssh', '-p', str(args.port), '-i', str(args.identity),
           '-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes',
           '-o', 'StrictHostKeyChecking=yes', '-o', 'Compression=no']
    if args.host_key_alias:
        ssh += ['-o', 'HostKeyAlias=' + args.host_key_alias]
    if args.known_hosts:
        ssh += ['-o', 'UserKnownHostsFile=' + str(args.known_hosts)]
    ssh += [args.destination]
    if args.uv_already_verified:
        report = json.loads(subprocess.check_output(ssh + [
            'cat /workspace/komap/backup_metadata/uv_restore_verified.json'], text=True))
        if report.get('status') != 'environment_smoke_passed':
            raise RuntimeError('Existing uv restore has not passed verification')
    copy_command = [sys.executable, str(args.transfer_script or root / 'scripts/migration_assets.py'), 'sync',
        '--root', str(root), '--manifest', str(root / 'backup_metadata/transfer_manifest.json'),
        '--destination', args.destination, '--port', str(args.port),
        '--identity', str(args.identity), '--workers', '4']
    if args.host_key_alias:
        copy_command += ['--host-key-alias', args.host_key_alias]
    if args.known_hosts:
        copy_command += ['--known-hosts', str(args.known_hosts)]
    if args.allow_ephemeral:
        copy_command += ['--allow-ephemeral']
    state = {'started_utc': now(), 'stage': 'starting',
        'source_root': str(root), 'destination': args.destination, 'port': args.port,
        'destination_root': '/workspace/komap', 'files': len(manifest['files']),
        'bytes': manifest['bytes'], 'source_sha256_complete': True,
        'compression': False, 'source_preserved': True,
        'user_authorized_ephemeral_storage': args.allow_ephemeral}
    state['host_key_alias'] = args.host_key_alias
    state['uv_previously_verified'] = args.uv_already_verified
    state['controller_pid'] = __import__('os').getpid()
    save(state_file, state)
    with (state_dir / 'transfer.log').open('w') as copy_log, (state_dir / 'uv.log').open('w') as uv_log:
        uv_process = None if args.uv_already_verified else subprocess.Popen(ssh + [
            'bash /workspace/komap/scripts/restore_uv.sh /workspace/komap'],
            stdin=subprocess.DEVNULL, stdout=uv_log, stderr=subprocess.STDOUT)
        copy_process = subprocess.Popen(copy_command, stdin=subprocess.DEVNULL,
            stdout=copy_log, stderr=subprocess.STDOUT)
        state.update(copy_pid=copy_process.pid, uv_pid=uv_process.pid if uv_process else None)
        while True:
            copy_code = copy_process.poll()
            uv_code = uv_process.poll() if uv_process else 0
            if copy_code is None and uv_code is None:
                stage = 'transferring_and_restoring_environment'
            elif copy_code is None:
                stage = 'transferring_and_verifying_files'
            elif uv_code is None:
                stage = 'restoring_environment'
            else:
                stage = 'complete' if copy_code == uv_code == 0 else 'failed'
            state.update(updated_utc=now(), stage=stage,
                         transfer_returncode=copy_code, uv_returncode=uv_code)
            if copy_code is not None and uv_code is not None:
                state['finished_utc'] = now()
            save(state_file, state)
            if copy_code is not None and uv_code is not None:
                break
            time.sleep(5)
    if state['stage'] == 'complete':
        (state_dir / 'all.done').write_text(now())
        print('TRANSFER_SHA256_AND_UV_RESTORE_COMPLETE', flush=True)
    else:
        raise SystemExit('Migration task failed; inspect transfer.log and uv.log; source is intact')


if __name__ == '__main__':
    main()
