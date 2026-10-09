"""Disposable train-step timing for fixed crop1024/1536, bs2, accumulation2.

No checkpoint is written. ImageNet initialization must be available locally;
network connections are explicitly blocked during model construction.
"""
import argparse
from datetime import datetime, timezone
import gc
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


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + '.tmp')
    partial.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    partial.replace(path)


def no_network(*args, **kwargs):
    raise RuntimeError('Benchmark requires cached ImageNet files; network is disabled')


def run_case(reference, crop, warmup, updates, workers):
    import cv2
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from data import CropDataset, load_cv, worker_init_fn
    from train import SegLoss, autocast, build_from_args

    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    torch.manual_seed(reference['seed'])
    np.random.seed(reference['seed'])
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    batch_size, accumulation = 2, 2
    train_items, validation_items = load_cv(reference['data'], reference['fold'], reference['folds_file'])
    assert len(train_items) == 72 and len(validation_items) == 18
    assert not any(reference.get(k, 0) for k in ['mixup', 'cutmix', 'ema', 'hard_crop', 'soft_edge', 'cutout', 'skel'])
    assert reference.get('binary_class', -1) == -1 and not reference.get('pair_classes')
    assert reference['arch'] == 'Unet' and reference['encoder'] == 'mit_b2'
    assert reference['epochs'] == 60 and reference['spi'] == 8
    assert reference['width'] == 2560 and reference['scale_multiplier'] == 2.0 and reference['amp']
    dataset = CropDataset(
        train_items, crop=crop, samples_per_image=reference['spi'], target_width=reference['width'],
        scale_jitter=reference['scale_jitter'], rot=reference['rot'], policy=reference['aug_policy'],
        hm=reference['hm'], lowres=reference['lowres'], no_int=reference['no_int'],
        copypaste=reference['copypaste'], skel=False,
        int_drop=[v for v in reference['int_drop'].split(',') if v], elastic=reference['elastic'],
        cutout=reference['cutout'], hard=reference['hard_crop'], hard_maps=reference['hard_maps'],
        feats=reference['feats'], scale_multiplier=reference['scale_multiplier'])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
                        drop_last=True, worker_init_fn=worker_init_fn,
                        persistent_workers=workers > 0, pin_memory=True)
    assert len(loader) % accumulation == 0
    updates_per_epoch = len(loader) // accumulation
    assert updates_per_epoch == 144

    init_start = time.monotonic()
    # HF_HUB_OFFLINE plus socket guards ensures a missing cache fails promptly.
    with patch.object(socket.socket, 'connect', no_network), patch.object(socket, 'create_connection', no_network):
        model = build_from_args(reference, weights='imagenet')
    cached_init_seconds = time.monotonic() - init_start
    model = model.cuda().train()
    class_weights = [float(v) for v in reference['class_weights'].split(',')] if reference['class_weights'] else None
    criterion = SegLoss(reference['loss'], reference['bw'], class_weights, reference['skel'],
                        use_ignore=reference.get('use_ignore', False), soft_edge=reference['soft_edge']).cuda()
    enc_params = [q for name, q in model.named_parameters() if name.startswith('encoder.')]
    other_params = [q for name, q in model.named_parameters() if not name.startswith('encoder.')]
    optimizer = torch.optim.AdamW(
        [{'params': enc_params, 'lr': reference['lr'] * reference['enc_lr_mult']},
         {'params': other_params, 'lr': reference['lr']}], weight_decay=reference['wd'])
    total, warm = reference['epochs'] * updates_per_epoch, updates_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda i: (i + 1) / warm if i < warm else
        0.5 * (1 + math.cos(math.pi * (i - warm) / max(total - warm, 1))))

    iterator = iter(loader)
    timings, losses = [], []
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    for update in range(warmup + updates):
        torch.cuda.synchronize()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        micro_losses = []
        for _ in range(accumulation):
            batch = next(iterator)
            x = batch[0].cuda(non_blocking=True)
            y = batch[1].cuda(non_blocking=True)
            tube = batch[2].cuda(non_blocking=True) if len(batch) > 2 else None
            with autocast(reference['amp']):
                logits = model(x)
                loss = criterion(logits, y, tube)
                scaled_loss = loss / accumulation
            scaled_loss.backward()
            value = loss.detach().item()
            if not math.isfinite(value):
                raise RuntimeError(f'Nonfinite loss at crop={crop}, update={update}')
            micro_losses.append(value)
            del logits, scaled_loss, loss, x, y, tube, batch
        optimizer.step()
        scheduler.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if update >= warmup:
            timings.append(elapsed)
            losses.append(statistics.mean(micro_losses))
        print(json.dumps({'crop': crop, 'update': update + 1, 'warmup': update < warmup,
                          'seconds': elapsed, 'loss': statistics.mean(micro_losses)}), flush=True)

    mean = statistics.mean(timings)
    result = {
        'crop': crop, 'microbatch': batch_size, 'gradient_accumulation': accumulation,
        'effective_batch': batch_size * accumulation, 'train_images': len(train_items),
        'crops_per_epoch': len(dataset), 'microbatches_per_epoch': len(loader),
        'optimizer_updates_per_epoch': updates_per_epoch,
        'warmup_updates': warmup, 'timed_updates': updates,
        'cached_imagenet_init_seconds': cached_init_seconds,
        'seconds_per_update_mean': mean, 'seconds_per_update_median': statistics.median(timings),
        'seconds_per_update_p10_p90': np.percentile(timings, [10, 90]).tolist(),
        'estimated_train_seconds_per_epoch': mean * updates_per_epoch,
        'estimated_train_minutes_60_epochs': mean * updates_per_epoch * reference['epochs'] / 60,
        'peak_memory_allocated_gb_decimal': torch.cuda.max_memory_allocated() / 1e9,
        'peak_memory_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
        'peak_memory_reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
        'loss_first_last': [losses[0], losses[-1]], 'all_losses_finite': True,
        'timed_update_seconds': timings,
        'caveat': 'Single-GPU short benchmark; estimates exclude validation, final 8-TTA OOF, checkpoint writes, comparison, and contention from 4 concurrent jobs.'}
    del iterator, loader, dataset, optimizer, scheduler, criterion, model, enc_params, other_params
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', default='runs/cv/x4_mit_bw_r90/fold0/args.json')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--crops', type=int, nargs='+', default=[1024, 1536], choices=[1024, 1536])
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--updates', type=int, default=16)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--out', default='runs/review18_crop/benchmark.json')
    args = parser.parse_args()
    assert args.gpu >= 0 and args.warmup >= 2 and 10 <= args.updates <= 30
    assert 0 <= args.workers <= 8 and len(set(args.crops)) == len(args.crops)
    os.chdir(ROOT)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS='4',
                      OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', HF_HUB_OFFLINE='1',
                      HF_HUB_DISABLE_TELEMETRY='1')
    sys.path.insert(0, str(ROOT / 'src'))
    out = ROOT / args.out
    assert not out.exists(), f'Refuse overwrite: {out}'
    gpu_csv = subprocess.check_output(
        ['nvidia-smi', '-i', str(args.gpu), '--query-gpu=name,memory.used,memory.total,utilization.gpu',
         '--format=csv,noheader,nounits'], text=True).strip()
    gpu_name, used, total, utilization = [part.strip() for part in gpu_csv.split(',')]
    assert int(used) < 1000 and int(utilization) < 10, f'GPU busy: {gpu_csv}'
    reference_path = ROOT / args.reference
    reference = json.loads(reference_path.read_text(encoding='utf-8'))
    result = {'created_utc': now(), 'status': 'benchmarking', 'purpose': 'Disposable train-step timing only',
              'writes_checkpoints': False, 'imagenet_network_allowed': False,
              'reference_args': reference, 'reference_sha256': hashlib.sha256(reference_path.read_bytes()).hexdigest(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'gpu': args.gpu, 'gpu_name': gpu_name, 'gpu_memory_total_mib': int(total),
              'workers': args.workers, 'cases': []}
    save(out, result)
    try:
        for crop in args.crops:
            result['cases'].append(run_case(reference, crop, args.warmup, args.updates, args.workers))
            save(out, result)
        result['status'] = 'complete'
        result['completed_utc'] = now()
        if len(result['cases']) == 2:
            by_crop = {case['crop']: case for case in result['cases']}
            result['candidate_vs_control_update_time_ratio'] = (
                by_crop[1536]['seconds_per_update_mean'] / by_crop[1024]['seconds_per_update_mean'])
        save(out, result)
        print(json.dumps(result, indent=2), flush=True)
    except BaseException as error:
        result.update(status='error', error=repr(error), updated_utc=now())
        save(out, result)
        raise


if __name__ == '__main__':
    main()
