"""Upload frozen training assets through the authenticated RunPod Jupyter HTTPS API.

Large files are chunked without compression. Ambiguous chunk failures restart the
whole file, and a separate destination worker verifies every SHA256 before success.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
import urllib.parse
import urllib.request
import urllib.error


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--state-dir', type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    root, directory = args.root.resolve(), args.state_dir.resolve()
    sys.path.insert(0, str(root / 'scripts'))
    from migration_assets import read_manifest, safe_file
    manifest_path = root / 'backup_metadata/transfer_manifest.json'
    manifest = read_manifest(manifest_path)
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    auth_path = directory / 'jupyter_auth.json'
    auth = json.loads(auth_path.read_text())
    existing = json.loads((directory / 'destination_existing_assets.json').read_text())
    if existing['manifest_sha256'] != manifest_sha:
        raise ValueError('Destination inventory uses a different manifest')
    matched = set(existing['matches'])
    pending = [r for r in manifest['files'] if r['path'] not in matched]
    if any(Path(r['path']).suffix not in {'.pt', '.pth', '.pkl'} for r in pending):
        raise ValueError('All code and records must already be SHA256 verified')
    parsed = urllib.parse.urlparse(auth['url'])
    if parsed.scheme != 'https' or not parsed.netloc.endswith('.proxy.runpod.net'):
        raise ValueError('Expected a RunPod HTTPS destination')
    state_file = directory / 'https_state.json'
    if state_file.exists():
        previous = json.loads(state_file.read_text())
        old_pid = previous.get('pid')
        if old_pid and Path('/proc/' + str(old_pid)).exists() and previous.get('stage') not in {'failed', 'complete'}:
            raise RuntimeError('A previous uploader is still running')
    baseline_bytes = sum(r['size'] for r in manifest['files'] if r['path'] in matched)
    state = {'pid': os.getpid(), 'stage': 'uploading', 'started_utc': datetime.now(timezone.utc).isoformat(),
        'files_total': len(manifest['files']), 'bytes_total': manifest['bytes'],
        'files_complete': len(matched), 'bytes_complete': baseline_bytes, 'uploaded_bytes': 0,
        'compression': False, 'workers': args.workers, 'source_preserved': True,
        'user_authorized_ephemeral_storage': True, 'transport': 'authenticated_source_to_runpod_https',
        'manifest_sha256': manifest_sha}
    lock, started = threading.Lock(), time.monotonic()
    save(state_file, state)
    chunk_size = 16 * 1024 * 1024

    def put(name, data, chunk=None):
        body = {'type': 'file', 'format': 'base64', 'content': base64.b64encode(data).decode('ascii')}
        if chunk is not None:
            body['chunk'] = chunk
        remote = auth['destination_root'] + '/' + name
        request = urllib.request.Request(auth['url'] + '/api/contents/' + urllib.parse.quote(remote, safe='/'),
            data=json.dumps(body).encode(), method='PUT', headers={'Authorization': 'token ' + auth['token'],
                'Content-Type': 'application/json', 'User-Agent': 'curl/8.5.0'})
        with urllib.request.urlopen(request, timeout=90) as response:
            if response.status not in {200, 201}:
                raise RuntimeError('Unexpected upload status: ' + str(response.status))
            response.read()

    def upload(row):
        f = safe_file(root, row['path'])
        for attempt in range(5):
            try:
                before = f.stat()
                if (before.st_size, before.st_mtime_ns) != (row['size'], row['mtime_ns']):
                    raise ValueError('Source changed: ' + row['path'])
                digest = hashlib.sha256()
                offset, number = 0, 1
                with f.open('rb') as stream:
                    while True:
                        data = stream.read(chunk_size)
                        digest.update(data)
                        last = offset + len(data) >= row['size']
                        chunk = None if row['size'] <= chunk_size else (-1 if last else number)
                        put(row['path'], data, chunk)
                        offset += len(data)
                        number += 1
                        with lock:
                            state['uploaded_bytes'] += len(data)
                            state['average_MB_s'] = round(state['uploaded_bytes'] / 1e6 / (time.monotonic() - started), 2)
                            state['updated_utc'] = datetime.now(timezone.utc).isoformat()
                            save(state_file, state)
                        if last:
                            break
                after = f.stat()
                if (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns) or digest.hexdigest() != row['sha256']:
                    raise ValueError('Source identity changed: ' + row['path'])
                with lock:
                    state['files_complete'] += 1
                    state['bytes_complete'] += row['size']
                    save(state_file, state)
                return row['path']
            except ValueError:
                raise
            except Exception as error:
                print('RETRY_FILE', row['path'], 'attempt', attempt + 1, type(error).__name__, flush=True)
                if attempt == 4:
                    raise RuntimeError('Repeated upload failure: ' + row['path']) from None
                time.sleep(min(2 ** attempt, 10))
        raise AssertionError('Unreachable')

    priority = ('runs/cv/x4_mit_bw_r90/', 'runs/cv/x4_mit_bw_r90_s1/',
        'runs/cv/x2_mit_bw_r90/', 'runs/cv/x2_unetpp_bw_e120_r90/')
    probes = {'runs/cv/x2_mit_bw_r90/fold0/last.pt',
        'runs/cv/x2_unetpp_bw_e120_r90/fold0/last.pt',
        'runs/cv/x4_mit_bw_r90/fold0/last.pt',
        'runs/cv/x4_mit_bw_r90_s1/fold0/last.pt'}
    pending.sort(key=lambda r: (0 if r['path'] in probes else (1 if r['path'].startswith(priority) else 2), r['path']))
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(upload, row) for row in pending]
            for i, future in enumerate(as_completed(futures), 1):
                future.result()
                if i % 10 == 0 or i == len(futures):
                    print('UPLOADED_FILES', i, '/', len(futures), 'MB_s', state.get('average_MB_s'), flush=True)
        marker = {'manifest_sha256': manifest_sha, 'files': len(manifest['files']), 'bytes': manifest['bytes']}
        put('backup_metadata/https_transfer_complete.json', json.dumps(marker).encode())
        state['stage'] = 'awaiting_destination_sha256_and_checkpoint_verification'
        save(state_file, state)
        while True:
            url = auth['url'] + '/api/contents/' + auth['destination_root'] + '/backup_metadata/https_migration_result.json?content=1'
            request = urllib.request.Request(url, headers={'Authorization': 'token ' + auth['token'], 'User-Agent': 'curl/8.5.0'})
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    model = json.loads(response.read())
                value = model['content']
                if model.get('format') == 'base64':
                    value = base64.b64decode(value)
                result = json.loads(value)
                if result['manifest_sha256'] != manifest_sha or result['status'] != 'complete':
                    raise ValueError('Destination validation failed')
                save(directory / 'migration_verified.json', result)
                state['stage'] = 'complete'
                state['finished_utc'] = datetime.now(timezone.utc).isoformat()
                save(state_file, state)
                auth_path.unlink()
                print('ALL_ASSETS_SHA256_AND_CHECKPOINT_INFERENCE_VERIFIED', flush=True)
                break
            except urllib.error.HTTPError as error:
                if error.code != 404:
                    raise
            time.sleep(10)
    except Exception as error:
        state['stage'] = 'failed'
        state['error'] = str(error)
        save(state_file, state)
        raise


if __name__ == '__main__':
    main()
