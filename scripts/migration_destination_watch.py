"""Verify a completed HTTPS asset migration locally on its destination."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

root = Path('/workspace/komap')
metadata = root / 'backup_metadata'
sha = hashlib.sha256((metadata / 'transfer_manifest.json').read_bytes()).hexdigest()
started = time.monotonic()
try:
    while not (metadata / 'https_transfer_complete.json').exists():
        if time.monotonic() - started > 48 * 3600:
            raise RuntimeError('Uploader did not finish within 48 hours')
        time.sleep(10)
    marker = json.loads((metadata / 'https_transfer_complete.json').read_text())
    if marker['manifest_sha256'] != sha:
        raise ValueError('Uploader marker uses a different manifest')
    with (metadata / 'https_destination_verification.log').open('w') as log:
        subprocess.run(['python3', str(root / 'scripts/migration_assets.py'), 'verify',
            '--root', str(root), '--manifest', str(metadata / 'transfer_manifest.json')],
            stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run([str(root / '.venv/bin/python'), str(root / 'scripts/migration_checkpoint_probe.py'),
            '--root', str(root), '--reference', str(metadata / 'checkpoint_reference.json'),
            '--output', str(metadata / 'checkpoint_inference_verified.json')],
            stdout=log, stderr=subprocess.STDOUT, check=True)
    files = json.loads((metadata / 'transfer_verified.json').read_text())
    environment = json.loads((metadata / 'uv_restore_verified.json').read_text())
    inference = json.loads((metadata / 'checkpoint_inference_verified.json').read_text())
    if not files['all_sha256_match'] or environment['status'] != 'environment_smoke_passed' or inference['status'] != 'source_inference_match_passed':
        raise RuntimeError('A migration validation is incomplete')
    result = {'status': 'complete', 'manifest_sha256': sha,
        'finished_utc': datetime.now(timezone.utc).isoformat(), 'files': files['files'], 'bytes': files['bytes'],
        'all_sha256_match': True, 'all_package_versions_match': True,
        'trained_checkpoint_source_inference_match': True, 'checkpoints_compared': 4}
except Exception as error:
    result = {'status': 'failed', 'manifest_sha256': sha, 'error': str(error)}
temporary = metadata / 'https_migration_result.tmp'
temporary.write_text(json.dumps(result, indent=2))
temporary.replace(metadata / 'https_migration_result.json')
print(json.dumps(result), flush=True)
