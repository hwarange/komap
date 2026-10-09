"""Select project restore assets, rsync directly, then verify SHA256.

No compression, source deletion, or destination deletion. Run on Linux for sync.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import tempfile


CODE = {'src', 'scripts', 'experiments', 'reports', 'splits', 'backup_metadata',
        'handoff', 'eda', '.git'}
RECORDS = {'.json', '.jsonl', '.log', '.txt', '.md', '.csv', '.yaml', '.yml',
           '.png', '.jpg', '.jpeg', '.pdf', '.svg', '.done'}
ROOT_FILES = {'requirements.txt', '.gitignore', '.gitattributes'}


def selected(name):
    p = PurePosixPath(name)
    if any(x in {'__pycache__', '.venv', '.uv-cache', '.rsync-partial', 'hardmaps'} for x in p.parts):
        return False
    if p.suffix in {'.pyc', '.npy', '.npz', '.lock', '.tmp'}:
        # The uv requirements lock is a reproducibility record.
        return name == 'backup_metadata/requirements-migration.lock'
    if name in {'backup_metadata/transfer_manifest.json', 'backup_metadata/transfer_verified.json'}:
        return False
    if p.parts[0] in CODE or name in ROOT_FILES:
        return True
    if p.parts[0] == 'runs':
        return p.suffix in RECORDS | {'.pt', '.pth', '.pkl'}
    if p.parts[0] == 'submissions':
        return 'parts' not in p.parts and p.suffix in RECORDS
    if p.parts[0] == 'external':
        return p.suffix == '.json' and ('manifest' in p.name or 'audit' in p.name)
    return False


def safe_file(root, name):
    p = PurePosixPath(name)
    if not p.parts or p.is_absolute() or '..' in p.parts or '\\' in name or '\0' in name:
        raise ValueError(f'Unsafe relative path: {name!r}')
    f = root.joinpath(*p.parts)
    if not f.resolve().is_relative_to(root) or f.is_symlink():
        raise ValueError(f'Asset escapes root: {name!r}')
    return f


def digest_file(root, row, check_mtime=True):
    f = safe_file(root, row['path'])
    before = f.stat()
    if before.st_size != row['size'] or (check_mtime and before.st_mtime_ns != row['mtime_ns']):
        raise RuntimeError(f'File changed before hashing: {row["path"]}')
    h = hashlib.sha256()
    with f.open('rb') as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    after = f.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f'File changed during hashing: {row["path"]}')
    return h.hexdigest()


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')
    temp.replace(path)


def plan(args):
    root = args.root.resolve()
    output = args.manifest.resolve()
    rows = []
    for top in sorted(CODE | {'runs', 'submissions', 'external'} | ROOT_FILES):
        parent = root / top
        candidates = parent.rglob('*') if parent.is_dir() else [parent]
        for f in candidates:
            if not f.is_file() or f.is_symlink() or f.resolve() == output:
                continue
            name = f.relative_to(root).as_posix()
            if not selected(name):
                continue
            safe_file(root, name)
            s = f.stat()
            rows.append({'path': name, 'size': s.st_size, 'mtime_ns': s.st_mtime_ns})
    rows.sort(key=lambda r: r['path'])
    if args.hash:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            hashes = pool.map(lambda r: digest_file(root, r), rows)
            for i, (row, sha) in enumerate(zip(rows, hashes), 1):
                row['sha256'] = sha
                if i % 100 == 0 or i == len(rows):
                    print(f'SOURCE_HASH {i}/{len(rows)}', flush=True)
    obj = {'created_utc': datetime.now(timezone.utc).isoformat(),
           'source_root': str(root), 'compression': False, 'files': rows,
           'bytes': sum(r['size'] for r in rows), 'sha256_complete': args.hash,
           'exclusions': ['original image/label data', 'public pretrained caches',
                          'OOF npy/npz', 'hardmaps', 'virtualenv/package caches']}
    dump(output, obj)
    print(f'ASSETS {len(rows)} FILES {obj["bytes"]} BYTES HASHED={args.hash}', flush=True)


def record_data(args):
    root = args.root.resolve()
    rows = []
    extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp',
                  '.json', '.csv', '.txt', '.md', '.yaml', '.yml'}
    for top in ['data', 'external']:
        for f in (root / top).rglob('*'):
            if not f.is_file() or f.is_symlink() or f.suffix.lower() not in extensions:
                continue
            s = f.stat()
            rows.append({'path': f.relative_to(root).as_posix(),
                         'size': s.st_size, 'mtime_ns': s.st_mtime_ns})
    rows.sort(key=lambda r: r['path'])
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for row, sha in zip(rows, pool.map(lambda r: digest_file(root, r), rows)):
            row['sha256'] = sha
    dump(args.manifest, {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'purpose': 'Identify source data for re-acquisition; image/label files are not transferred',
        'fold_record': 'splits/folds5.json',
        'training_and_preprocessing_records': 'runs/**/args.json and src/data.py',
        'external_source_records': 'external manifest/audit JSON records',
        'files': rows})
    print(f'DATA_IDENTITIES_RECORDED {len(rows)} FILES', flush=True)


def read_manifest(path):
    obj = json.loads(path.read_text(encoding='utf-8-sig'))
    if not obj['sha256_complete'] or any(not re.fullmatch('[a-f0-9]{64}', r.get('sha256', '')) for r in obj['files']):
        raise ValueError('A fully hashed source manifest is required')
    names = [r['path'] for r in obj['files']]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate manifest paths')
    return obj


def verify(args):
    root = args.root.resolve()
    obj = read_manifest(args.manifest)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        hashes = pool.map(lambda r: digest_file(root, r, check_mtime=False), obj['files'])
        for i, (row, sha) in enumerate(zip(obj['files'], hashes), 1):
            if sha != row['sha256']:
                raise RuntimeError(f'SHA256 mismatch: {row["path"]}')
            if i % 100 == 0 or i == len(obj['files']):
                print(f'DESTINATION_HASH {i}/{len(obj["files"])}', flush=True)
    dump(root / 'backup_metadata/transfer_verified.json', {
        'verified_utc': datetime.now(timezone.utc).isoformat(),
        'files': len(obj['files']), 'bytes': obj['bytes'], 'all_sha256_match': True,
        'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest()})
    print('ALL_ASSET_SHA256_VERIFIED', flush=True)


def sync(args):
    root = args.root.resolve()
    obj = read_manifest(args.manifest)
    if not re.fullmatch(r'[A-Za-z0-9_-]+@[A-Za-z0-9.-]+', args.destination):
        raise ValueError('Destination must be user@hostname')
    dest = args.destination_root
    if not re.fullmatch(r'/workspace/[A-Za-z0-9_./-]+', dest) or '..' in PurePosixPath(dest).parts:
        raise ValueError('Destination must be a simple directory below /workspace')
    ssh = ['ssh', '-p', str(args.port), '-o', 'BatchMode=yes',
           '-o', 'StrictHostKeyChecking=yes', '-o', 'Compression=no']
    if args.identity:
        ssh += ['-i', str(args.identity.resolve()), '-o', 'IdentitiesOnly=yes']
    if args.host_key_alias:
        ssh += ['-o', 'HostKeyAlias=' + args.host_key_alias]
    if args.known_hosts:
        ssh += ['-o', 'UserKnownHostsFile=' + str(args.known_hosts.resolve())]
    fs = subprocess.check_output(ssh + [args.destination,
        'findmnt -T /workspace -n -o FSTYPE'], text=True).strip()
    if fs in {'overlay', 'tmpfs'} and not args.allow_ephemeral:
        raise RuntimeError('Destination /workspace is ephemeral; persistent storage is not attached')
    for row in obj['files']:
        f = safe_file(root, row['path'])
        s = f.stat()
        if (s.st_size, s.st_mtime_ns) != (row['size'], row['mtime_ns']):
            raise RuntimeError(f'Source changed since manifest: {row["path"]}')
    subprocess.run(ssh + [args.destination, f'mkdir -p {dest}/backup_metadata'], check=True)
    with tempfile.NamedTemporaryFile() as listing:
        listing.write(b''.join(r['path'].encode('utf-8') + b'\0' for r in obj['files']))
        listing.flush()
        subprocess.run(['rsync', '-rlt', '--protect-args', '--from0',
            '--files-from=' + listing.name, '--partial-dir=.rsync-partial',
            '--info=progress2', '-e', shlex.join(ssh), str(root) + '/',
            args.destination + ':' + dest + '/'], check=True)
    scp = ['scp', '-P', str(args.port), '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes']
    if args.identity:
        scp += ['-i', str(args.identity.resolve())]
    if args.host_key_alias:
        scp += ['-o', 'HostKeyAlias=' + args.host_key_alias]
    if args.known_hosts:
        scp += ['-o', 'UserKnownHostsFile=' + str(args.known_hosts.resolve())]
    subprocess.run(scp + [str(args.manifest),
        args.destination + ':' + dest + '/backup_metadata/transfer_manifest.json'], check=True)
    subprocess.run(ssh + [args.destination,
        f'python3 {dest}/scripts/migration_assets.py verify --root {dest} '
        f'--manifest {dest}/backup_metadata/transfer_manifest.json --workers {args.workers}'], check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['plan', 'verify', 'sync', 'record-data'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--hash', action='store_true')
    p.add_argument('--destination')
    p.add_argument('--port', type=int, default=22)
    p.add_argument('--identity', type=Path)
    p.add_argument('--host-key-alias', help='Pinned destination host name when using an SSH relay')
    p.add_argument('--known-hosts', type=Path)
    p.add_argument('--destination-root', default='/workspace/komap')
    p.add_argument('--allow-ephemeral', action='store_true')
    args = p.parse_args()
    if args.workers < 1:
        p.error('--workers must be positive')
    if args.command == 'sync' and not args.destination:
        p.error('sync requires --destination')
    {'plan': plan, 'verify': verify, 'sync': sync, 'record-data': record_data}[args.command](args)


if __name__ == '__main__':
    main()
