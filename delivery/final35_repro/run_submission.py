#!/usr/bin/env python3
"""Frozen train70 MiT2 + UNet++3 submission. Default: read-only CPU preflight."""
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
import subprocess
import sys
import time
import zipfile

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
FAMILIES = ('mit2', 'unet3')
COLORS = [(153, 127, 76), (25, 76, 153), (76, 178, 76), (204, 204, 204)]
CLASSES = ['Al', 'Al3Ni', 'Primary Si', 'Eutectic Si']
CHECKPOINTS = {
    'mit2': ('runs/review32_mit2_train70/models/mit2/base/last.pt',
             '2739b79a9e052808d318b621fb2d1cb9ae31b43473f4266e73fb275cd199aee1'),
    'unet3': ('runs/submission31_train70/models/unet3/base/last.pt',
              '1668b0ef0e25e9be093ed531d60a2425e45f9cda694f421e023d3d09e9e253a8'),
}
BACKEND = dict(cudnn_benchmark=True, cuda_matmul_allow_tf32=True, cudnn_allow_tf32=True)
PACKAGES = ('torch', 'torchvision', 'segmentation-models-pytorch', 'timm',
            'numpy', 'opencv-python-headless', 'pillow', 'scipy')


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
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def path_in(root, value):
    p = Path(value)
    if p.is_absolute() or '..' in p.parts:
        raise ValueError(f'Expected project-relative path: {value}')
    p = (root / p).resolve()
    if not p.is_relative_to(root):
        raise ValueError(f'Path escapes project: {value}')
    return p


def file_record(root, path):
    path = Path(path)
    return dict(path=path.relative_to(root).as_posix(), size_bytes=path.stat().st_size, sha256=sha256(path))


def manifest(root):
    m = read(HERE / 'manifest.json')
    if (m['schema'] != 'komap-final35-train70-v1' or m['test_count'] != 10
            or m['family_weights'] != {'mit2': 0.5, 'unet3': 0.5}
            or m['backend'] != BACKEND or m['tta'] != 'dihedral8'
            or m['probability_storage_dtype'] != 'float16' or m['combination_dtype'] != 'float32'
            or m['scale_policy'] != 'original' or m['checkpoint_choice'] != 'last'
            or len(m['models']) != 2 or {v['family'] for v in m['models']} != set(FAMILIES)):
        raise ValueError('Manifest differs from the frozen two-model design')
    if len(m['training_names']) != 70 or len(m['validation_names']) != 20:
        raise ValueError('Expected train70 and validation20 provenance')
    if set(m['training_names']) & set(m['validation_names']):
        raise ValueError('Training/validation image overlap')
    for v in m['models']:
        a, family = v['args'], v['family']
        ckpath, cksha = CHECKPOINTS[family]
        epoch, arch, enc, width = ((60, 'Unet', 'mit_b2', 2560) if family == 'mit2'
                                  else (120, 'UnetPlusPlus', 'resnet34', 3840))
        expected = dict(epochs=epoch, arch=arch, encoder=enc, width=width, scale_multiplier=1.0,
                        fold=-1, full=False, init='', train_splits=['train'], val_split='valid',
                        crop=1536, tile=1536, overlap=128, seed=0, hard_crop=0.0, train_frac=1.0,
                        binary_class=-1, pair_classes=None, feats=False, amp=True)
        if (v['checkpoint']['path'] != ckpath or v['checkpoint']['sha256'] != cksha
                or v['checkpoint_epoch'] != epoch or any(a[k] != val for k, val in expected.items())):
            raise ValueError(f'Wrong selected model or provenance: {family}')
    return m


def preflight(root, out, m):
    from PIL import Image
    errors, sizes = [], {}
    if out.exists() or out.is_symlink():
        errors.append('Output directory must not exist')
    if platform.system() != 'Linux' or platform.python_version() != m['python_version']:
        errors.append('Reference runtime requires the recorded Linux/Python version')
    versions = {p: importlib.metadata.version(p) for p in PACKAGES}
    if versions != m['packages']:
        errors.append('Installed core package versions differ from manifest')
    groups = dict(sources=m['source_files'], provenance=m['provenance_files'],
                  checkpoints=[v['checkpoint'] for v in m['models']], test_images=m['test_images'])
    for items in groups.values():
        for item in items:
            p = path_in(root, item['path'])
            if not p.is_file() or p.stat().st_size != item['size_bytes'] or sha256(p) != item['sha256']:
                errors.append('Changed/missing file: ' + item['path'])
    expected = {Path(v['path']).name for v in m['test_images']}
    actual = {p.name for p in (root / 'data/test/images').glob('*_image.png')}
    if len(expected) != 10 or expected != actual:
        errors.append('Exactly the recorded test10 images are required')
    for v in m['test_images']:
        with Image.open(path_in(root, v['path'])) as im:
            name = Path(v['path']).name.removesuffix('_image.png')
            sizes[name] = list(im.size)
            if list(im.size) != v['size_width_height']:
                errors.append('Test image dimensions differ: ' + name)
    for v in m['models']:
        audit = read(path_in(root, v['initialization_audit']['path']))
        if (audit['args'] != v['args'] or audit['training_names'] != m['training_names']
                or audit['validation_names'] != m['validation_names']
                or audit['inherited_checkpoint'] is not None or audit['previous_experiment_checkpoint_used']
                or not audit['strict_initialization'] or audit['test_images_used']):
            errors.append('Training provenance does not match: ' + v['family'])
    return dict(ok=not errors, mode='read_only_preflight', checked={k: len(v) for k, v in groups.items()},
                errors=errors, versions=versions, image_sizes_width_height=sizes,
                training_performed=False, test_ground_truth_used=False, gpu_initialized=False)


def gpu_info(index):
    vals = subprocess.check_output(['nvidia-smi', f'--id={index}', '--query-gpu=uuid,name,memory.total,memory.used,utilization.gpu',
                                   '--format=csv,noheader,nounits'], text=True, timeout=15).strip().split(',')
    uuid, name, total, used, util = [v.strip() for v in vals]
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True, timeout=15)
    if (any(line.split(',')[0].strip() == uuid for line in apps.splitlines())
            or int(used) >= 1000 or int(util) >= 10 or int(total) - int(used) < 30000):
        raise RuntimeError(f'GPU {index} must be idle with 30000MiB free')
    return dict(index=index, uuid=uuid, name=name, total_mib=int(total))


def probability(path, size):
    import numpy as np
    p = np.load(path, allow_pickle=False)
    if (p.dtype != np.float16 or p.shape != (4, size[1], size[0]) or not np.isfinite(p).all()
            or p.min() < 0 or p.max() > 1 or np.abs(p.astype(np.float32).sum(0) - 1).max() > 0.005):
        raise ValueError(f'Invalid probability array: {path}')
    return p.astype(np.float32)


def worker(root, out, m, family, expected_sha):
    if sha256(HERE / 'manifest.json') != expected_sha:
        raise RuntimeError('Manifest changed')
    receipt = read(out / 'run_receipt.json')
    if receipt['status'] != 'running' or receipt['manifest_sha256'] != expected_sha:
        raise RuntimeError('Worker requires matching active controller')
    sys.path.insert(0, str(HERE / 'source'))
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
    v = next(v for v in m['models'] if v['family'] == family)
    ckpath = path_in(root, v['checkpoint']['path'])
    if sha256(ckpath) != v['checkpoint']['sha256']:
        raise RuntimeError('Checkpoint hash changed')
    ck = torch.load(ckpath, map_location='cpu', weights_only=False)
    if ck['args'] != v['args'] or ck['epoch'] != v['checkpoint_epoch']:
        raise RuntimeError('Checkpoint configuration changed')
    if not all(torch.isfinite(t).all() for t in ck['model'].values()):
        raise RuntimeError('Checkpoint has non-finite tensors')
    a = v['args']
    model = build_from_args(a, weights=None)
    model.load_state_dict(ck['model'], strict=True)
    model = model.cuda().eval()
    del ck
    dest = out / 'parts' / family
    dest.mkdir(parents=True, exist_ok=False)
    started, records = time.monotonic(), []
    for item in m['test_images']:
        p = path_in(root, item['path'])
        if sha256(p) != item['sha256']:
            raise RuntimeError('Test image changed')
        with Image.open(p) as image:
            gray = np.array(image.convert('L'))
        with torch.no_grad(), autocast(a['amp']):
            prob = predict_probs(model, gray, a['width'], a['tile'], a['overlap'], True, 'cuda',
                                 extra_scale=1.0, scale_multiplier=1.0)
        target = dest / (p.name.removesuffix('_image.png') + '.npy')
        with target.open('xb') as stream:
            np.save(stream, prob.astype(np.float16), allow_pickle=False)
        probability(target, item['size_width_height'])
        records.append(dict(**file_record(out, target), actual_scale=scale_for(gray.shape[1], a['width'], 1.0)))
        print(json.dumps(dict(family=family, image=p.name, elapsed_seconds=time.monotonic() - started)), flush=True)
    write(out / f'{family}.complete.json', dict(family=family, manifest_sha256=expected_sha, outputs=records,
          strict_checkpoint_load=True, checkpoint=v['checkpoint'], elapsed_seconds=time.monotonic() - started, completed_utc=now()))


def combine(out, m):
    import numpy as np
    from PIL import Image
    sys.path.insert(0, str(HERE / 'source'))
    from data import label_to_rgb
    from infer import postprocess
    masks_dir = out / 'masks'
    masks_dir.mkdir(exist_ok=False)
    masks, fractions = [], []
    for item in m['test_images']:
        name = Path(item['path']).name.removesuffix('_image.png')
        p2, p3 = [probability(out / 'parts' / f / f'{name}.npy', item['size_width_height']) for f in FAMILIES]
        label = postprocess((p2 + p3) / 2)
        target = masks_dir / f'{name}_mask.png'
        Image.fromarray(label_to_rgb(label)).save(target)
        with Image.open(target) as im:
            colors = im.getcolors(maxcolors=5)
            if im.mode != 'RGB' or list(im.size) != item['size_width_height'] or colors is None or not {v for _, v in colors} <= set(COLORS):
                raise RuntimeError('Invalid output mask format')
        masks.append(file_record(out, target))
        counts = np.bincount(label.ravel(), minlength=4)
        fractions.append(dict(name=name, size_width_height=item['size_width_height'],
                              pixel_count=int(label.size), class_pixel_count=dict(zip(CLASSES, map(int, counts))),
                              area_fraction_percent=dict(zip(CLASSES, map(float, counts / label.size * 100)))))
    write(out / 'prediction_area_fractions.json', dict(split='test', ground_truth_available=False,
          interpretation='Predicted two-dimensional pixel area fractions; not measured composition or accuracy.', images=fractions))
    archive = out / 'final35_masks.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for v in masks:
            z.write(out / v['path'], arcname=Path(v['path']).name)
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None or len(z.namelist()) != 10:
            raise RuntimeError('Mask ZIP CRC/count validation failed')
        for v in masks:
            if hashlib.sha256(z.read(Path(v['path']).name)).hexdigest() != v['sha256']:
                raise RuntimeError('ZIP member hash mismatch')
    return masks, file_record(out, archive)


def run(root, out, m, report, gpus):
    infos = [gpu_info(i) for i in gpus]
    if len(infos) != 2 or len({v['uuid'] for v in infos}) != 2:
        raise ValueError('Two distinct idle GPUs required')
    out.mkdir(parents=True, exist_ok=False)
    msh = sha256(HERE / 'manifest.json')
    receipt = dict(status='running', started_utc=now(), manifest_sha256=msh, runner_sha256=sha256(__file__),
                   preflight=report, gpus=infos, training_performed=False, test_ground_truth_used=False,
                   prediction_reuse=False, family_weights=m['family_weights'], backend=BACKEND)
    write(out / 'run_receipt.json', receipt)
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '4',
           'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
    children, logs = [], []
    def interrupted(signum, frame):
        raise RuntimeError(f'Controller interrupted by signal {signum}')
    old = signal.signal(signal.SIGTERM, interrupted)
    try:
        for family, info in zip(FAMILIES, infos):
            log = (out / f'{family}.log').open('x', encoding='utf-8')
            logs.append(log)
            cmd = [sys.executable, str(Path(__file__).resolve()), '--project-root', str(root), '--out', str(out),
                   '--worker', family, '--manifest-sha', msh]
            children.append(subprocess.Popen(cmd, cwd=root, env={**env, 'CUDA_VISIBLE_DEVICES': info['uuid']},
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
        receipt['prediction_pids'] = [p.pid for p in children]
        write(out / 'run_receipt.json', receipt)
        started = time.monotonic()
        while any(p.poll() is None for p in children):
            if any(p.poll() not in (None, 0) for p in children) or time.monotonic() - started > 30 * 60:
                raise RuntimeError('Inference worker failed or exceeded 30 minutes')
            time.sleep(3)
        if any(p.returncode for p in children):
            raise RuntimeError('Inference worker failed')
        records = []
        for family in FAMILIES:
            done = read(out / f'{family}.complete.json')
            if done['manifest_sha256'] != msh or done['family'] != family:
                raise RuntimeError('Worker completion mismatch')
            records.extend(done['outputs'])
        expected = {f'parts/{family}/{Path(v["path"]).name.removesuffix("_image.png")}.npy'
                    for family in FAMILIES for v in m['test_images']}
        if len(records) != 20 or {v['path'] for v in records} != expected:
            raise RuntimeError('Expected 20 distinct model-image probability outputs')
        for v in records:
            if sha256(path_in(out, v['path'])) != v['sha256']:
                raise RuntimeError('Probability output changed')
        masks, archive = combine(out, m)
        for v in m['source_files'] + [v['checkpoint'] for v in m['models']] + m['test_images']:
            if sha256(path_in(root, v['path'])) != v['sha256']:
                raise RuntimeError('Frozen input changed during inference')
        if sha256(HERE / 'manifest.json') != msh:
            raise RuntimeError('Manifest changed during inference')
        receipt.update(status='complete', completed_utc=now(), prediction_exit_codes=[p.returncode for p in children],
                       masks=masks, probability_outputs=records, mask_zip=archive,
                       area_fractions=file_record(out, out / 'prediction_area_fractions.json'))
    except BaseException as error:
        receipt.update(status='failed', error=repr(error), failed_utc=now())
        raise
    finally:
        for p in children:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait(timeout=10)
        for log in logs:
            log.close()
        write(out / 'run_receipt.json', receipt)
        signal.signal(signal.SIGTERM, old)
    write(out / 'all.done', dict(completed_utc=now(), manifest_sha256=msh,
          run_receipt_sha256=sha256(out / 'run_receipt.json'), mask_zip=archive, masks=masks))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root', type=Path, default=HERE.parents[1])
    p.add_argument('--out', type=Path, default=Path('submissions/final35_train70'))
    p.add_argument('--execute', action='store_true')
    p.add_argument('--gpus', type=int, nargs=2, default=[0, 1])
    p.add_argument('--worker', choices=FAMILIES, help=argparse.SUPPRESS)
    p.add_argument('--manifest-sha', default='', help=argparse.SUPPRESS)
    a = p.parse_args()
    root = a.project_root.resolve()
    out = (a.out if a.out.is_absolute() else root / a.out).resolve()
    if not out.is_relative_to(root / 'submissions') or out == root / 'submissions':
        raise ValueError('Output must be a new directory beneath project/submissions')
    m = manifest(root)
    if a.worker:
        worker(root, out, m, a.worker, a.manifest_sha)
        return
    report = preflight(root, out, m)
    print(json.dumps(report, indent=2), flush=True)
    if not report['ok']:
        raise SystemExit(2)
    if a.execute:
        run(root, out, m, report, a.gpus)


if __name__ == '__main__':
    main()
