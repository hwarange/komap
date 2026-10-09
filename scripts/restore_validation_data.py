"""Restore ONLY the original 90 train/valid images and their masks over HTTPS."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.parse
import urllib.request

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--root', type=Path, required=True)
p.add_argument('--state-dir', type=Path, required=True)
args = p.parse_args()
root = args.root.resolve()
sys.path.insert(0, str(root / 'scripts'))
from migration_assets import read_manifest, safe_file
manifest = read_manifest(args.state_dir / 'dataset_manifest.json')
auth_path = args.state_dir / 'jupyter_auth.json'
auth = json.loads(auth_path.read_text())
parsed = urllib.parse.urlparse(auth['url'])
assert parsed.scheme == 'https' and parsed.netloc.endswith('.proxy.runpod.net')
assert len(manifest['files']) == 180
assert all(r['path'].startswith(('data/train/', 'data/valid/')) for r in manifest['files'])

def upload(row):
    f = safe_file(root, row['path'])
    data = f.read_bytes()
    assert len(data) == row['size'] and hashlib.sha256(data).hexdigest() == row['sha256']
    body = json.dumps({'type': 'file', 'format': 'base64', 'content': base64.b64encode(data).decode('ascii')}).encode()
    name = auth['destination_root'] + '/' + row['path']
    for attempt in range(4):
        try:
            request = urllib.request.Request(auth['url'] + '/api/contents/' + urllib.parse.quote(name, safe='/'),
                data=body, method='PUT', headers={'Authorization': 'token ' + auth['token'],
                    'Content-Type': 'application/json', 'User-Agent': 'curl/8.5.0'})
            with urllib.request.urlopen(request, timeout=60) as response:
                assert response.status in {200, 201}
                response.read()
            return row['path']
        except Exception:
            if attempt == 3:
                raise RuntimeError('Dataset upload failed: ' + row['path']) from None
            time.sleep(2)

try:
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, _ in enumerate(pool.map(upload, manifest['files']), 1):
            if i % 30 == 0:
                print('DATASET_UPLOADED', i, '/180', flush=True)
finally:
    auth_path.unlink(missing_ok=True)
print('TRAIN_VALID_RESTORED_PENDING_DESTINATION_SHA256', flush=True)
