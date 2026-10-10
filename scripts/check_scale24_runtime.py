"""Disposable scale24 readiness/timing check; never writes model weights or OOF."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
ARMS = {
    'mit5': ('x4_mit_c1536_bw_r90', 'mit_b2', 2.5, 2, 2, 1536,
             '868ab6f13871dcf8c3d9f90ee4519403475b65ef'),
    'deeplab4': ('x2_deeplabv3plus_r34_bw_r90', 'resnet34', 2.0, 4, 1, 1024,
                 '7a57b34f723329ff020b3f8bc41771163c519d0c'),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def no_network(*args, **kwargs):
    raise RuntimeError('ImageNet must load from the pinned cache; network disabled')


def geometry(item, a):
    from data import scale_for
    h, w = item['gray'].shape
    scale = scale_for(w, a['width'], a['scale_multiplier'])
    sh, sw = round(h * scale), round(w * scale)
    tile, stride = a['tile'], a['tile'] - a['overlap']
    tiles = (math.ceil(max(sh - tile, 0) / stride) + 1) * (math.ceil(max(sw - tile, 0) / stride) + 1)
    return {'name': item['name'], 'original_hw': [h, w], 'scale': scale,
            'scaled_hw': [sh, sw], 'scaled_pixels': sh * sw, 'tiles': tiles,
            'inference_batches_of_four': math.ceil(tiles / 4)}


def measure(a, warmup, updates):
    import cv2
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from data import CropDataset, load_cv, worker_init_fn
    from train import SegLoss, autocast, accumulation_step, build_from_args, evaluate

    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    torch.manual_seed(a['seed'])
    np.random.seed(a['seed'])
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    train_items, valid = load_cv(a['data'], a['fold'], a['folds_file'])
    assert len(train_items) == 72 and len(valid) == 18
    assert not ({v['alloy'] for v in train_items} & {v['alloy'] for v in valid})
    ds = CropDataset(train_items, crop=a['crop'], samples_per_image=a['spi'], target_width=a['width'],
                     scale_jitter=a['scale_jitter'], rot=a['rot'], policy=a['aug_policy'],
                     hm=a['hm'], lowres=a['lowres'], no_int=a['no_int'], copypaste=a['copypaste'],
                     skel=False, int_drop=[v for v in a['int_drop'].split(',') if v],
                     elastic=a['elastic'], cutout=a['cutout'], hard=a['hard_crop'],
                     hard_maps=a['hard_maps'], feats=a['feats'], scale_multiplier=a['scale_multiplier'])
    dl = DataLoader(ds, batch_size=a['bs'], shuffle=True, num_workers=a['workers'], drop_last=True,
                    worker_init_fn=worker_init_fn, persistent_workers=True, pin_memory=True)
    assert len(ds) == 576 and math.ceil(len(dl) / a['accum_steps']) == 144
    t0 = time.perf_counter()
    with patch.object(socket.socket, 'connect', no_network), patch.object(socket, 'create_connection', no_network):
        model = build_from_args(a, weights='imagenet').cuda().train()
    init_seconds = time.perf_counter() - t0
    criterion = SegLoss(a['loss'], a['bw'], None, 0, use_ignore=False, soft_edge=0).cuda()
    enc = [p for n, p in model.named_parameters() if n.startswith('encoder.')]
    other = [p for n, p in model.named_parameters() if not n.startswith('encoder.')]
    opt = torch.optim.AdamW([{'params': enc, 'lr': a['lr'] * a['enc_lr_mult']},
                            {'params': other, 'lr': a['lr']}], weight_decay=a['wd'])
    per_epoch = math.ceil(len(dl) / a['accum_steps'])
    total, warm = a['epochs'] * per_epoch, per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: (i + 1) / warm if i < warm else
        0.5 * (1 + math.cos(math.pi * (i - warm) / max(total - warm, 1))))
    iterator = iter(dl)
    timings, losses = [], []
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    gradient_checks = []
    for update in range(warmup + updates):
        torch.cuda.synchronize()
        started = time.perf_counter()
        values = []
        for micro in range(a['accum_steps']):
            n = update * a['accum_steps'] + micro
            batch = next(iterator)
            x, y = batch[0].cuda(non_blocking=True), batch[1].cuda(non_blocking=True)
            with autocast(a['amp']):
                logits = model(x)
                loss = criterion(logits, y)
            updated = accumulation_step(loss, opt, sched, n, len(dl), a['accum_steps'])
            assert updated == (micro == a['accum_steps'] - 1)
            value = loss.detach().item()
            assert math.isfinite(value), (update, micro, value)
            values.append(value)
            del batch, x, y, logits, loss
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if update >= warmup:
            timings.append(elapsed)
            losses.extend(values)
        if update in (warmup - 1, warmup + updates - 1):
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            assert grads and all(bool(torch.isfinite(g).all()) for g in grads)
            assert all(bool(torch.isfinite(p).all()) for p in model.parameters())
            gradient_checks.append(update + 1)
        print(json.dumps({'update': update + 1, 'warmup': update < warmup,
                          'seconds': elapsed, 'microbatch_losses': values}), flush=True)
    train_allocated = torch.cuda.max_memory_allocated() / 2**30
    train_reserved = torch.cuda.max_memory_reserved() / 2**30
    opt.zero_grad(set_to_none=True)
    # Match the real trainer: optimizer state stays resident; no empty_cache here.
    torch.cuda.reset_peak_memory_stats()
    geom = [geometry(it, a) for it in valid]
    by_name = {it['name']: it for it in valid}
    small = next((g for g in geom if g['original_hw'][1] == 675), None)
    largest = max(geom, key=lambda g: g['scaled_pixels'])
    median = sorted(geom, key=lambda g: g['tiles'])[len(geom) // 2]
    chosen = []
    for g in [largest, small, median]:
        if g and g['name'] not in [x['name'] for x in chosen]:
            chosen.append(g)
    for g in sorted(geom, key=lambda g: g['scaled_pixels']):
        if len(chosen) >= 3:
            break
        if g['name'] not in [x['name'] for x in chosen]:
            chosen.append(g)
    measured = []
    for g in chosen:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        rows = evaluate(model, [by_name[g['name']]], a['width'], a['tile'], a['overlap'],
                        'cuda', tta=False, amp=a['amp'], save_dir=None,
                        scale_multiplier=a['scale_multiplier'])
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        assert len(rows) == 1
        sample = dict(g, no_tta_seconds=elapsed,
                      peak_memory_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                      peak_memory_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
        measured.append(sample)
        print(json.dumps({'validation_readiness': sample}), flush=True)
    assert largest['name'] in [v['name'] for v in measured]
    assert small is None or small['name'] in [v['name'] for v in measured]
    mean_update = statistics.mean(timings)
    mean_batch_seconds = sum(v['no_tta_seconds'] for v in measured) / sum(v['inference_batches_of_four'] for v in measured)
    val18_seconds = sum(g['inference_batches_of_four'] for g in geom) * mean_batch_seconds
    train_seconds = mean_update * per_epoch * a['epochs']
    total_seconds = train_seconds + (a['epochs'] // a['val_every'] + 8) * val18_seconds
    return {
        'cached_imagenet_init_seconds': init_seconds, 'train_images': 72, 'validation_images': 18,
        'microbatch': a['bs'], 'gradient_accumulation': a['accum_steps'], 'effective_batch': a['bs'] * a['accum_steps'],
        'microbatches_per_epoch': len(dl), 'optimizer_updates_per_epoch': per_epoch, 'crops_per_epoch': len(ds),
        'warmup_updates': warmup, 'timed_updates': updates, 'timed_update_seconds': timings,
        'seconds_per_update_mean': mean_update, 'seconds_per_update_median': statistics.median(timings),
        'estimated_train_seconds_per_epoch': mean_update * per_epoch,
        'estimated_train_minutes_60_epochs': train_seconds / 60,
        'estimated_validation18_no_tta_seconds': val18_seconds,
        'estimated_final_8tta_seconds': 8 * val18_seconds,
        'estimated_total_minutes': total_seconds / 60,
        'peak_train_memory_allocated_gib': train_allocated, 'peak_train_memory_reserved_gib': train_reserved,
        'validation_geometry_all18': geom, 'validation_measurements': measured,
        'largest_scaled_canvas_checked': largest['name'], 'width675_image_checked': small['name'] if small else None,
        'finite_gradient_and_parameter_checks_after_updates': gradient_checks,
        'loss_first_last': [losses[0], losses[-1]], 'all_losses_finite': True,
        'optimizer_state_resident_during_validation': True, 'empty_cache_called': False,
        'caveat': 'Four timed updates only; concurrent GPUs may affect speed. Validation18 extrapolated from three images by batch-of-four tile count; final 8TTA approximated. No checkpoint or OOF writes, full 8TTA, or CPU comparison timed. Readiness is not accuracy evidence.',
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', choices=sorted(ARMS), required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--warmup', type=int, default=2)
    p.add_argument('--updates', type=int, default=4)
    opts = p.parse_args()
    assert opts.gpu in (2, 3) and 2 <= opts.warmup <= 4 and 3 <= opts.updates <= 6
    os.chdir(ROOT)
    out = ROOT / 'runs/review24_single_scale' / f'runtime_{opts.arm}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    assert not out.exists(), f'Refuse overwrite: {out}'
    idle = []
    for repeat in range(2):
        csv = subprocess.check_output(['nvidia-smi', '-i', str(opts.gpu),
            '--query-gpu=uuid,name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip()
        gpu_uuid, gpu_name, used, total, util = [v.strip() for v in csv.split(',')]
        assert int(used) < 1000 and int(util) < 10, f'GPU busy: {csv}'
        idle.append(csv)
        if repeat == 0:
            time.sleep(1)
    assert idle[0].split(',')[0] == idle[1].split(',')[0]
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu_uuid, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
                      MKL_NUM_THREADS='1', HF_HUB_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
                      PYTHONDONTWRITEBYTECODE='1')
    sys.path.insert(0, str(ROOT / 'src'))
    reference_exp, encoder, multiplier, batch, accum, crop, revision = ARMS[opts.arm]
    reference_path = ROOT / 'runs/cv' / reference_exp / 'fold0/args.json'
    reference = json.loads(reference_path.read_text())
    a = dict(reference)
    a.update(scale_multiplier=multiplier, workers=4, save_oof='', out='')
    assert a['encoder'] == encoder and a['width'] == 2560 and a['crop'] == crop and a['tile'] == crop
    assert a['bs'] == batch and a['accum_steps'] == accum and a['epochs'] == 60 and a['fold'] == 0
    assert a['seed'] == 0 and a['amp'] and a['bw'] == 2 and a['spi'] == 8 and not a['max_iters']
    assert not any(a.get(k, 0) for k in ['mixup', 'cutmix', 'ema', 'hard_crop', 'soft_edge', 'cutout', 'skel', 'feats', 'init'])
    cache = Path.home() / '.cache/huggingface/hub' / f'models--smp-hub--{encoder}.imagenet'
    snapshots = list((cache / 'snapshots').iterdir())
    assert len(snapshots) == 1 and snapshots[0].name == revision, snapshots
    cached_files = {name: sha(snapshots[0] / name) for name in ['config.json', 'model.safetensors']}
    result = {'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'checking', 'arm': opts.arm,
              'purpose': 'Disposable readiness and timing only', 'writes_checkpoints': False, 'writes_oof': False,
              'weights_reused_for_experiment': False, 'imagenet_network_allowed': False,
              'imagenet_revision': revision, 'imagenet_file_sha256': cached_files,
              'source_hashes': {name: sha(ROOT / name) for name in ['src/train.py', 'src/data.py', 'src/infer.py']},
              'reference_sha256': sha(reference_path), 'script_sha256': sha(__file__),
              'reference_args': reference, 'benchmark_args': a, 'gpu_index': opts.gpu, 'gpu_uuid': gpu_uuid,
              'gpu_name': gpu_name, 'gpu_memory_total_mib': int(total), 'idle_checks': idle}
    with out.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    try:
        result['measurement'] = measure(a, opts.warmup, opts.updates)
        result.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat())
        save(out, result)
        print(json.dumps({'status': 'complete', 'arm': opts.arm, 'measurement': result['measurement']}), flush=True)
    except BaseException as error:
        result.update(status='error', error=repr(error), updated_utc=datetime.now(timezone.utc).isoformat())
        save(out, result)
        raise


if __name__ == '__main__':
    main()
