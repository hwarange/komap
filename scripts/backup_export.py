"""Read-only server side of the laptop backup protocol (metadata or PAX tar).

Only project assets are exported, never the virtualenv, user home, or symlinks.
Checkpoint files still being trained are deferred. Every tar member carries a
SHA256 digest so the laptop can reject incomplete or changed transfers.
"""
import argparse
import hashlib
import json
import os
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
TOPS = {'src', 'scripts', 'experiments', 'reports', 'splits', 'data',
        'external', 'runs', 'submissions', 'backup_metadata'}
ROOT_FILES = {'requirements.txt', '.gitattributes', '.gitignore'}


def asset_path(name):
    posix = PurePosixPath(name)
    if posix.is_absolute() or '..' in posix.parts or not posix.parts or any(':' in s for s in posix.parts):
        raise ValueError(f'Invalid relative asset path: {name}')
    if posix.parts[0] not in TOPS and str(posix) not in ROOT_FILES:
        raise ValueError(f'Not a project asset: {name}')
    path = ROOT.joinpath(*posix.parts)
    if not path.resolve().is_relative_to(ROOT) or path.is_symlink():
        raise ValueError(f'Unsafe asset: {name}')
    return path


def checkpoint_ready(path):
    if path.suffix not in {'.pt', '.pth'}:
        return True
    parent = path.parent
    if (parent / 'result.json').exists():
        return True
    last = parent / 'last.pt'
    return last.exists() and time.time() - last.stat().st_mtime >= 30


def inventory():
    files, deferred = [], []
    candidates = []
    for name in sorted(TOPS):
        parent = ROOT / name
        if parent.exists():
            candidates.extend(parent.rglob('*'))
    candidates.extend(ROOT / name for name in ROOT_FILES)
    for path in candidates:
        if not path.is_file() or path.is_symlink() or '__pycache__' in path.parts or path.suffix == '.pyc':
            continue
        rel = path.relative_to(ROOT).as_posix()
        if path.suffix == '.lock' or path.name.endswith('.tmp'):
            continue
        if not checkpoint_ready(path):
            deferred.append(rel)
            continue
        stat = path.stat()
        files.append({'path': rel, 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns})
    return {'version': 1, 'generated_utc': datetime.now(timezone.utc).isoformat(),
            'files': sorted(files, key=lambda r: r['path']),
            'deferred_checkpoints': sorted(deferred)}


def archive():
    request = json.load(sys.stdin)
    with tarfile.open(fileobj=sys.stdout.buffer, mode='w|', format=tarfile.PAX_FORMAT) as tar:
        for row in request['files']:
            path = asset_path(row['path'])
            if not path.is_file() or not checkpoint_ready(path):
                print(f"SKIP disappeared or active: {row['path']}", file=sys.stderr, flush=True)
                continue
            with open(path, 'rb') as source:
                stat = os.fstat(source.fileno())
                if stat.st_size != row['size'] or stat.st_mtime_ns != row['mtime_ns']:
                    print(f"SKIP metadata changed: {row['path']}", file=sys.stderr, flush=True)
                    continue
                digest = hashlib.sha256()
                left = row['size']
                while left:
                    buf = source.read(min(left, 4 * 1024 * 1024))
                    if not buf:
                        raise RuntimeError(f"File truncated: {row['path']}")
                    digest.update(buf)
                    left -= len(buf)
                after = os.fstat(source.fileno())
                if (after.st_size, after.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
                    print(f"SKIP changed during hashing: {row['path']}", file=sys.stderr, flush=True)
                    continue
                source.seek(0)
                info = tarfile.TarInfo(row['path'])
                info.size, info.mtime, info.mode = stat.st_size, stat.st_mtime, 0o600
                info.pax_headers = {'komap.sha256': digest.hexdigest(),
                                    'komap.mtime_ns': str(stat.st_mtime_ns)}
                tar.addfile(info, source)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['list', 'archive'])
    args = parser.parse_args()
    if args.command == 'list':
        print(json.dumps(inventory(), ensure_ascii=False))
    else:
        archive()


if __name__ == '__main__':
    main()
