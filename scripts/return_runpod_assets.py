"""Recover a frozen manifest directly over authenticated RunPod HTTPS."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import os
import time
import urllib.parse
import urllib.request

ROOT = Path('/home/j-j15a106/komap')
STATE = ROOT / 'runs/runpod_return_20261010'
AUTH = Path('/home/j-j15a106/.cache/komap_return_20261010/jupyter_auth.json')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    STATE.mkdir(parents=True, exist_ok=True)
    auth = json.loads(AUTH.read_text())
    assert auth['url'] == 'https://vbwkmskg08hbnr-8888.proxy.runpod.net'
    def get(relative):
        name = PurePosixPath(relative)
        assert not name.is_absolute() and '..' not in name.parts
        url = auth['url'] + '/files/workspace/komap/' + urllib.parse.quote(relative, safe='/')
        return urllib.request.urlopen(urllib.request.Request(url, headers={
            'Authorization': 'token ' + auth['token'], 'User-Agent': 'curl/8.5.0'}), timeout=120)
    with get('runs/runpod_return_20261010_manifest.json') as response:
        raw = response.read()
    manifest = json.loads(raw)
    (STATE / 'manifest.json').write_bytes(raw)
    state = {'pid': os.getpid(), 'stage': 'downloading', 'files': len(manifest['files']),
             'bytes': manifest['bytes'], 'verified_files': 0, 'verified_bytes': 0,
             'transport': 'direct_runpod_to_original_server_https', 'compression': False,
             'original_files_preserved': True, 'manifest_sha256': hashlib.sha256(raw).hexdigest()}
    results = []
    def save():
        state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        temp = STATE / 'state.tmp'
        temp.write_text(json.dumps(state, indent=2))
        temp.replace(STATE / 'state.json')
    def transfer(row):
        name = PurePosixPath(row['path'])
        assert not name.is_absolute() and '..' not in name.parts
        destination = ROOT.joinpath(*name.parts)
        assert destination.resolve().is_relative_to(ROOT)
        if destination.exists():
            if destination.stat().st_size == row['size'] and sha(destination) == row['sha256']:
                return {'path': row['path'], 'stored': row['path'], 'status': 'already_identical'}
            destination = STATE / 'conflicts' / Path(*name.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.stat().st_size == row['size'] and sha(destination) == row['sha256']:
            return {'path': row['path'], 'stored': str(destination.relative_to(ROOT)), 'status': 'already_recovered'}
        temporary = destination.with_name(destination.name + '.return_part')
        for attempt in range(3):
            try:
                digest = hashlib.sha256()
                size = 0
                with get(row['path']) as response, temporary.open('wb') as stream:
                    for chunk in iter(lambda: response.read(4 * 1024 * 1024), b''):
                        stream.write(chunk); digest.update(chunk); size += len(chunk)
                if size != row['size'] or digest.hexdigest() != row['sha256']:
                    raise ValueError('Downloaded content does not match manifest')
                temporary.replace(destination)
                if sha(destination) != row['sha256']:
                    raise ValueError('Stored content verification failed')
                return {'path': row['path'], 'stored': str(destination.relative_to(ROOT)), 'status': 'recovered'}
            except Exception as error:
                print('RETRY', row['path'], attempt + 1, type(error).__name__, flush=True)
                if attempt == 2:
                    raise RuntimeError('Transfer failed for ' + row['path']) from None
                time.sleep(2)
    save()
    try:
        for priority in sorted({r['priority'] for r in manifest['files']}):
            batch = [r for r in manifest['files'] if r['priority'] == priority]
            with ThreadPoolExecutor(max_workers=6) as pool:
                pending = {pool.submit(transfer, row): row for row in batch}
                for future in as_completed(pending):
                    row = pending[future]
                    results.append(future.result())
                    state['verified_files'] += 1
                    state['verified_bytes'] += row['size']
                    save()
            print('PRIORITY_VERIFIED', priority, state['verified_files'], flush=True)
        state['stage'] = 'complete'
        state['all_sha256_verified'] = True
        (STATE / 'verified.json').write_text(json.dumps({'state': state, 'files': results}, indent=2))
        save()
        AUTH.unlink()
        print('ALL_SELECTED_ASSETS_RECOVERED_AND_SHA256_VERIFIED', flush=True)
    except Exception as error:
        state['stage'] = 'failed'; state['error'] = str(error)
        save()
        raise


if __name__ == '__main__':
    main()
