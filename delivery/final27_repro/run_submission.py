#!/usr/bin/env python3
"""Reproduce the frozen review27 winner; default is read-only CPU preflight."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import struct
import subprocess
import sys
import time
import zipfile

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
FAMILIES = ('mit4', 'unet3')
COLORS = {(153, 127, 76), (25, 76, 153), (76, 178, 76), (204, 204, 204)}
CORE_PACKAGES = ('torch', 'torchvision', 'segmentation-models-pytorch', 'timm',
                 'numpy', 'opencv-python-headless', 'pillow', 'scipy')
BACKEND = dict(cudnn_benchmark=True, cuda_matmul_allow_tf32=True, cudnn_allow_tf32=True)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def checked_path(root, value):
    path = Path(value)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError(f'Reference must be project-relative: {value}')
    result = (root / path).resolve()
    if not result.is_relative_to(root):
        raise ValueError(f'Reference escapes project: {value}')
    return result


def png_size(path):
    with Path(path).open('rb') as stream:
        header = stream.read(24)
    if len(header) != 24 or header[:8] != b'\x89PNG\r\n\x1a\n' or header[12:16] != b'IHDR':
        raise ValueError(f'Invalid PNG: {path}')
    size = struct.unpack('>II', header[16:24])
    if min(size) <= 0:
        raise ValueError(f'Invalid PNG dimensions: {path}')
    return size


def load_manifest(root):
    m = read(HERE / 'manifest.json')
    if (m['schema'] != 'komap-final27-v1' or m['folds'] != 5 or m['test_count'] != 10
            or m['family_weights'] != {'mit4': 0.5, 'unet3': 0.5}
            or m['backend'] != BACKEND or m['tta'] != 'dihedral8'
            or m['probability_dtype'] != 'float32' or m['checkpoint_choice'] != 'last'):
        raise ValueError('Manifest differs from the frozen final27 design.')
    if m['mit_policy'] not in ('original', 'cap6_width675') or m['unet_policy'] not in ('original', 'interface6'):
        raise ValueError('Unknown frozen policy')
    expected = {(family, fold) for family in FAMILIES for fold in range(5)}
    if len(m['models']) != 10 or {(v['family'], v['fold']) for v in m['models']} != expected:
        raise ValueError('Exactly five models per family are required')
    paths = [v['checkpoint']['path'] for v in m['models']]
    if len(set(paths)) != 10:
        raise ValueError('Checkpoints must be distinct')
    for item in m['source_files'] + m['test_images'] + [v['checkpoint'] for v in m['models']]:
        checked_path(root, item['path'])
    for model in m['models']:
        a = model['args']
        family = model['family']
        expected_epoch = 60 if family == 'mit4' else (6 if m['unet_policy'] == 'interface6' else 120)
        expected_arch, encoder = ('Unet', 'mit_b2') if family == 'mit4' else ('UnetPlusPlus', 'resnet34')
        if (model['checkpoint_epoch'] != expected_epoch or a['epochs'] != expected_epoch
                or a['fold'] != model['fold'] or a['arch'] != expected_arch or a['encoder'] != encoder
                or a['width'] != (2560 if family == 'mit4' else 3840)
                or a['crop'] != 1536 or a['tile'] != 1536 or a['overlap'] != 128
                or a.get('scale_multiplier', 1.0) != (2.0 if family == 'mit4' else 1.0)
                or not a['amp'] or a['full'] or a.get('binary_class', -1) != -1
                or a.get('pair_classes') or a.get('feats', False)):
            raise ValueError(f'Unexpected selected checkpoint configuration: {model["id"]}')
        if family == 'unet3' and m['unet_policy'] == 'interface6':
            if (a['hard_crop'] != 0.35 or a['lr'] != 3e-5 or not a['init']
                    or model['initial_training_epochs'] != 120 or model['optimizer_state_resumed']):
                raise ValueError('Expected 120-epoch warm start plus separate six-epoch interface continuation')
    return m


def preflight(root, out, manifest):
    """Read-only stdlib checks; no torch import, GPU initialization, or output creation."""
    errors, warnings = [], []
    if out.exists() or out.is_symlink():
        errors.append(f'Output must be a new directory: {out}')
    if platform.system() != 'Linux' or sys.version_info[:2] != (3, 12):
        errors.append('Reference environment requires Linux and Python 3.12')
    if platform.python_version() != manifest['python_version']:
        warnings.append(f'Reference Python is {manifest["python_version"]}')
    versions = {}
    for package in CORE_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
        if versions[package] != manifest['packages'][package]:
            errors.append(f'Package mismatch: {package}: {versions[package]}')
    groups = dict(source_files=manifest['source_files'], checkpoints=[v['checkpoint'] for v in manifest['models']],
                  test_images=manifest['test_images'])
    verified = {name: 0 for name in groups}
    for name, values in groups.items():
        for item in values:
            path = checked_path(root, item['path'])
            if not path.is_file() or path.stat().st_size != item['size_bytes'] or sha256(path) != item['sha256']:
                errors.append(f'Missing or changed reference file: {item["path"]}')
            else:
                verified[name] += 1
    expected = {Path(item['path']).name for item in manifest['test_images']}
    actual = {p.name for p in (root / 'data/test/images').glob('*_image.png')}
    if len(expected) != 10 or actual != expected:
        errors.append('Expected exactly the recorded ten test images')
    sizes = {}
    for item in manifest['test_images']:
        path = checked_path(root, item['path'])
        if path.is_file():
            sizes[path.name.removesuffix('_image.png')] = list(png_size(path))
    if sum(size[0] == 675 for size in sizes.values()) != 3:
        errors.append('Reference test set must contain exactly three width675 images')
    return dict(ok=not errors, mode='read_only_preflight', project_root=str(root), output=str(out),
                verified_files=verified, versions=versions, image_sizes_width_height=sizes,
                errors=errors, warnings=warnings, gpu_checked=False, test_ground_truth_used=False)


def check_gpu(index):
    fields = subprocess.check_output(['nvidia-smi', f'--id={index}',
              '--query-gpu=uuid,name,memory.total,memory.used,utilization.gpu',
              '--format=csv,noheader,nounits'], text=True, timeout=15).strip().split(',')
    if len(fields) != 5:
        raise RuntimeError('Could not identify selected GPU')
    uuid, name, total, used, use = [s.strip() for s in fields]
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                    '--format=csv,noheader,nounits'], text=True, timeout=15)
    if (any(line.split(',')[0].strip() == uuid for line in apps.splitlines())
            or int(used) >= 1000 or int(use) >= 10 or int(total) - int(used) < 30000):
        raise RuntimeError(f'GPU {index} must be idle with at least30000MiB free')
    return dict(index=index, uuid=uuid, name=name, total_mib=int(total), used_mib=int(used))


def extra_scale(manifest, family, width):
    return 0.75 if family == 'mit4' and manifest['mit_policy'] == 'cap6_width675' and width == 675 else 1.0


def validate_probability(path, width, height):
    import numpy as np
    p = np.load(path, mmap_mode='r', allow_pickle=False)
    if (p.shape != (4, height, width) or p.dtype != np.float32 or not np.isfinite(p).all()
            or p.min() < 0 or p.max() > 1 or np.max(np.abs(p.sum(0) - 1)) > 0.005):
        raise ValueError(f'Invalid probability array: {path}')


def worker(root, out, manifest, shard, nshards, expected_manifest_sha):
    if sha256(HERE / 'manifest.json') != expected_manifest_sha:
        raise RuntimeError('Manifest changed before worker startup')
    launch = read(out / 'run_receipt.json')
    if launch['status'] != 'running' or launch['manifest_sha256'] != expected_manifest_sha:
        raise RuntimeError('Worker requires a matching active controller receipt')
    sys.path.insert(0, str(root / 'src'))
    import cv2
    import numpy as np
    from PIL import Image
    import torch
    from data import scale_for
    from infer import predict_probs
    from train import autocast, build_from_args
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    assigned = manifest['models'][shard::nshards]
    outputs, timings = [], []
    for selected in assigned:
        start = time.monotonic()
        record = selected['checkpoint']
        path = checked_path(root, record['path'])
        if sha256(path) != record['sha256']:
            raise RuntimeError(f'Checkpoint changed: {path}')
        ck = torch.load(path, map_location='cpu', weights_only=False)
        if ck['epoch'] != selected['checkpoint_epoch'] or ck['args'] != selected['args']:
            raise RuntimeError('Checkpoint metadata differs from frozen manifest')
        a = ck['args']
        model = build_from_args(a, weights=None)
        model.load_state_dict(ck['model'], strict=True)
        model = model.cuda().eval()
        del ck
        model_dir = out / 'parts/models' / selected['family'] / f"fold{selected['fold']}"
        model_dir.mkdir(parents=True, exist_ok=False)
        for item in manifest['test_images']:
            path = checked_path(root, item['path'])
            if sha256(path) != item['sha256']:
                raise RuntimeError(f'Test input changed: {path}')
            with Image.open(path) as image:
                gray = np.array(image.convert('L'))
            scale = extra_scale(manifest, selected['family'], gray.shape[1])
            if selected['family'] == 'mit4' and gray.shape[1] == 675:
                expected = 6 if manifest['mit_policy'] == 'cap6_width675' else 8
                if scale_for(675, a['width'], a.get('scale_multiplier', 1.0)) * scale != expected:
                    raise RuntimeError('MiT width675 scale does not match frozen policy')
            with torch.no_grad(), autocast(a['amp']):
                prob = predict_probs(model, gray, a['width'], a['tile'], a['overlap'], True, 'cuda',
                                     extra_scale=scale, scale_multiplier=a.get('scale_multiplier', 1.0))
            target = model_dir / (path.name.removesuffix('_image.png') + '.npy')
            with target.open('xb') as stream:
                np.save(stream, prob.astype(np.float32), allow_pickle=False)
            validate_probability(target, gray.shape[1], gray.shape[0])
            outputs.append(dict(path=target.relative_to(out).as_posix(), sha256=sha256(target),
                                size_bytes=target.stat().st_size, extra_scale=scale))
        del model
        torch.cuda.empty_cache()
        timings.append(dict(model=selected['id'], seconds=time.monotonic() - start))
        print(json.dumps(timings[-1]), flush=True)
    write(out / f'worker{shard}.complete.json', dict(shard=shard, models=[v['id'] for v in assigned],
          manifest_sha256=expected_manifest_sha, outputs=outputs, timings=timings, backend=BACKEND, completed_utc=now()))


def validate_parts(out, shapes):
    expected = {out / 'parts/models' / family / f'fold{k}' / f'{name}.npy'
                for family in FAMILIES for k in range(5) for name in shapes}
    if set((out / 'parts/models').rglob('*.npy')) != expected:
        raise RuntimeError('Exactly100 model/image arrays are required before combination')
    for path in sorted(expected):
        validate_probability(path, *shapes[path.stem])


def combine(root, out, manifest, shapes):
    import numpy as np
    from PIL import Image
    sys.path.insert(0, str(root / 'src'))
    from data import label_to_rgb
    from infer import postprocess
    target = out / 'masks'
    target.mkdir(exist_ok=False)
    for name in sorted(shapes):
        family_probs = []
        for family in FAMILIES:
            arrays = [np.load(out / 'parts/models' / family / f'fold{k}' / f'{name}.npy', allow_pickle=False)
                      for k in range(5)]
            family_probs.append(sum(arrays) / 5)
        prob = (family_probs[0] + family_probs[1]) / 2
        Image.fromarray(label_to_rgb(postprocess(prob))).save(target / f'{name}_mask.png')
    summary = dict(split='test', n_images=10, n_models=10, models_per_image=[10],
                   models_per_exp=dict.fromkeys(FAMILIES, 5), recipe=dict.fromkeys(FAMILIES, 1.0),
                   full_mix=0, problems=[], si_classifier=None, mit_policy=manifest['mit_policy'],
                   unet_policy=manifest['unet_policy'], test_ground_truth_used=False,
                   mixture='mean5folds per family; then 50:50 soft voting; then unchanged postprocess')
    write(out / 'summary_test.json', summary)


def validate_masks(out, shapes):
    from PIL import Image
    expected = {f'{name}_mask.png' for name in shapes}
    files = sorted((out / 'masks').iterdir())
    if len(files) != 10 or {p.name for p in files} != expected:
        raise RuntimeError('Exactly10 mask PNGs required')
    masks = []
    for path in files:
        with Image.open(path) as image:
            image.load()
            colors = image.getcolors(maxcolors=5)
            name = path.name.removesuffix('_mask.png')
            if (image.mode != 'RGB' or image.format != 'PNG' or image.size != tuple(shapes[name])
                    or colors is None or not {color for _, color in colors} <= COLORS):
                raise RuntimeError(f'Invalid mask format: {path}')
        masks.append(dict(name=path.name, sha256=sha256(path), size_bytes=path.stat().st_size))
    return masks


def run(root, out, manifest, report, gpus):
    infos = [check_gpu(gpu) for gpu in gpus]
    if len({v['uuid'] for v in infos}) != len(infos):
        raise RuntimeError('Selected GPUs must have distinct physical UUIDs')
    out.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUNBUFFERED': '1',
           'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
           'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
    manifest_sha = sha256(HERE / 'manifest.json')
    receipt = dict(status='running', started_utc=now(), manifest_sha256=manifest_sha,
                   runner_sha256=sha256(Path(__file__)), preflight=report, gpus=infos,
                   prediction_reuse=False, training_performed=False, test_ground_truth_used=False,
                   backend=BACKEND, selected_recipe=manifest['selected_recipe'])
    write(out / 'run_receipt.json', receipt)
    children, logs = [], []
    def interrupted(signum, frame):
        raise RuntimeError(f'Controller interrupted by signal{signum}')
    old_handler = signal.signal(signal.SIGTERM, interrupted)
    try:
        commands = []
        for shard, info in enumerate(infos):
            command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--project-root', str(root),
                       '--out', str(out), '--shard', str(shard), '--nshards', str(len(infos)),
                       '--manifest-sha', manifest_sha]
            commands.append(command)
            log = (out / f'predict_shard{shard}.log').open('x', encoding='utf-8')
            logs.append(log)
            children.append(subprocess.Popen(command, cwd=root, env={**env, 'CUDA_VISIBLE_DEVICES': info['uuid']},
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
        receipt.update(commands=commands, prediction_pids=[child.pid for child in children])
        write(out / 'run_receipt.json', receipt)
        started = time.monotonic()
        while any(child.poll() is None for child in children):
            failed = [child.returncode for child in children if child.poll() not in (None, 0)]
            if failed:
                raise RuntimeError(f'Prediction shard failed: {failed}')
            if time.monotonic() - started > 30 * 60:
                raise RuntimeError('Prediction exceeded30min bound')
            time.sleep(3)
        codes = [child.returncode for child in children]
        receipt['prediction_exit_codes'] = codes
        if any(codes):
            raise RuntimeError(f'Prediction shard failure: {codes}')
        records = []
        for shard in range(len(infos)):
            complete = read(out / f'worker{shard}.complete.json')
            if complete['manifest_sha256'] != manifest_sha:
                raise RuntimeError('Worker manifest mismatch')
            records.extend(complete['outputs'])
        if len(records) != 100 or len({v['path'] for v in records}) != 100:
            raise RuntimeError('Worker receipts must cover100 distinct arrays')
        for item in records:
            if sha256(checked_path(out, item['path'])) != item['sha256']:
                raise RuntimeError('Worker array changed before combination')
        shapes = report['image_sizes_width_height']
        validate_parts(out, shapes)
        combine(root, out, manifest, shapes)
        masks = validate_masks(out, shapes)
        archive_path = out / 'final27_masks.zip'
        with zipfile.ZipFile(archive_path, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for item in masks:
                archive.write(out / 'masks' / item['name'], arcname=item['name'])
        with zipfile.ZipFile(archive_path) as archive:
            if archive.testzip() is not None or archive.namelist() != [item['name'] for item in masks]:
                raise RuntimeError('Mask ZIP validation failed')
            for item in masks:
                if hashlib.sha256(archive.read(item['name'])).hexdigest() != item['sha256']:
                    raise RuntimeError('ZIP mask checksum mismatch')
        # Recheck frozen sources and selected weights after all inference completes.
        for item in manifest['source_files'] + [v['checkpoint'] for v in manifest['models']]:
            if sha256(checked_path(root, item['path'])) != item['sha256']:
                raise RuntimeError(f'Pinned source/checkpoint changed: {item["path"]}')
        if sha256(HERE / 'manifest.json') != manifest_sha:
            raise RuntimeError('Frozen manifest changed during inference')
        receipt.update(status='complete', masks=masks, probability_outputs=records,
                       zip_sha256=sha256(archive_path), finished_utc=now())
    except BaseException as error:
        receipt.update(status='failed', error=f'{type(error).__name__}: {error}', finished_utc=now())
        raise
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            if child.poll() is None:
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
        for log in logs:
            log.close()
        signal.signal(signal.SIGTERM, old_handler)
        write(out / 'run_receipt.json', receipt)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--project-root', type=Path, default=HERE.parents[1])
    p.add_argument('--out', type=Path, default=Path('submissions/final27_selected'))
    p.add_argument('--gpus', type=int, nargs='+')
    p.add_argument('--shard', type=int, help=argparse.SUPPRESS)
    p.add_argument('--nshards', type=int, help=argparse.SUPPRESS)
    p.add_argument('--manifest-sha', help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.run and (not args.gpus or min(args.gpus) < 0 or len(set(args.gpus)) != len(args.gpus) or len(args.gpus) > 10):
        p.error('--run requires1–10 distinct nonnegative --gpus indices')
    root = args.project_root.resolve()
    out = (root / args.out).absolute()
    if out.is_symlink():
        p.error('Output cannot be an existing symlink')
    out = out.resolve()
    os.chdir(root)
    manifest = load_manifest(root)
    if args.worker:
        if args.shard is None or args.nshards is None or not 0 <= args.shard < args.nshards or not args.manifest_sha:
            p.error('Invalid internal worker parameters')
        worker(root, out, manifest, args.shard, args.nshards, args.manifest_sha)
        return 0
    report = preflight(root, out, manifest)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not report['ok']:
        return 2
    if args.run:
        receipt = run(root, out, manifest, report, args.gpus)
        print(json.dumps(dict(status=receipt['status'], masks=10, zip=str(out / 'final27_masks.zip'))), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
