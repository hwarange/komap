#!/usr/bin/env python3
"""Create the final35 code-only archive using an explicit metadata/source allowlist."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[2]
PREFIX = 'delivery/final35_repro'
OUT = ROOT / 'delivery/final35_code_bundle.zip'
RECEIPT = ROOT / 'delivery/final35_code_bundle_receipt.json'


def digest(b):
    return hashlib.sha256(b).hexdigest()


def main():
    if OUT.exists() or RECEIPT.exists():
        raise RuntimeError('Refuse to overwrite an existing final35 code archive')
    m = json.loads((ROOT / PREFIX / 'manifest.json').read_text(encoding='utf-8'))
    frozen = m['source_files'] + m['provenance_files']
    payload = {}
    for v in frozen:
        data = (ROOT / v['path']).read_bytes()
        if digest(data) != v['sha256'] or len(data) != v['size_bytes']:
            raise RuntimeError('Frozen source/provenance mismatch: ' + v['path'])
        payload[v['path']] = data
    extras = [f'{PREFIX}/manifest.json', f'{PREFIX}/build_code_bundle.py', f'{PREFIX}/BUNDLE_README.md',
              'submissions/final35_train70/run_receipt.json', 'submissions/final35_train70/all.done',
              'submissions/final35_train70/mit2.complete.json', 'submissions/final35_train70/unet3.complete.json',
              'submissions/final35_train70/prediction_area_fractions.json',
              'runs/final35_submission/preflight.json', 'runs/final35_submission/completion_verification.json',
              'runs/final35_submission/local_copy_verification.json',
              'runs/review34_train70_group5fold/plan.json', 'runs/review34_train70_group5fold/per_image_rows.json',
              'runs/review34_train70_group5fold/completion_verification.json']
    for name in extras:
        payload[name] = (ROOT / name).read_bytes()
    payload['README.md'] = (ROOT / PREFIX / 'BUNDLE_README.md').read_bytes()
    forbidden = {'.pt', '.pth', '.npy', '.pkl', '.png', '.jpg', '.jpeg', '.tif', '.tiff', '.zip'}
    for name, data in payload.items():
        p = Path(name)
        if p.suffix.lower() in forbidden or 'data' in p.parts or 'test_inputs' in p.parts or 'checkpoints' in p.parts:
            raise RuntimeError('Raw image/model/array file must not enter code archive: ' + name)
        if len(data) > 2_000_000:
            raise RuntimeError('Unexpectedly large source/metadata file: ' + name)
        if p.suffix == '.py':
            ast.parse(data.decode('utf-8-sig'), filename=name)
        if p.suffix in ('.json', '.done'):
            json.loads(data)
        if p.suffix == '.jsonl':
            for line in data.decode('utf-8').splitlines():
                json.loads(line)
    receipt = json.loads(payload['submissions/final35_train70/run_receipt.json'])
    done = json.loads(payload['submissions/final35_train70/all.done'])
    if (receipt['status'] != 'complete' or done['run_receipt_sha256'] != digest(payload['submissions/final35_train70/run_receipt.json'])
            or done['manifest_sha256'] != digest(payload[f'{PREFIX}/manifest.json'])):
        raise RuntimeError('Submission completion chain mismatch')
    with zipfile.ZipFile(OUT, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name in sorted(payload):
            z.writestr(name, payload[name])
    with zipfile.ZipFile(OUT) as z:
        if z.testzip() is not None or z.namelist() != sorted(payload):
            raise RuntimeError('Archive CRC/list verification failed')
        for name, data in payload.items():
            if z.read(name) != data:
                raise RuntimeError('Archive member bytes differ: ' + name)
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), archive=OUT.relative_to(ROOT).as_posix(),
                  zip_sha256=digest(OUT.read_bytes()), zip_size_bytes=OUT.stat().st_size,
                  source_and_metadata_files=len(payload), raw_images_included=False, weights_included=False,
                  probability_arrays_included=False, models_required_separately=m['models'],
                  test_inputs_required_separately=m['test_images'],
                  files=[dict(path=name, size_bytes=len(data), sha256=digest(data)) for name, data in sorted(payload.items())])
    RECEIPT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('archive', 'zip_sha256', 'zip_size_bytes', 'source_and_metadata_files')}))


if __name__ == '__main__':
    main()
