"""Disposable DeepLabV3Plus timing; no weights, OOF, or model reuse."""
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


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def no_network(*args, **kwargs):
    raise RuntimeError('Cached ImageNet initialization required; network disabled')


def measure(a, warmup, updates):
    import cv2
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from data import CropDataset, load_cv, worker_init_fn
    from train import SegLoss, autocast, build_from_args, evaluate

    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    torch.manual_seed(a['seed'])
    np.random.seed(a['seed'])
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    train, valid = load_cv(a['data'], a['fold'], a['folds_file'])
    assert len(train) == 72 and len(valid) == 18
    dataset = CropDataset(
        train, crop=a['crop'], samples_per_image=a['spi'], target_width=a['width'],
        scale_jitter=a['scale_jitter'], rot=a['rot'], policy=a['aug_policy'],
        hm=a['hm'], lowres=a['lowres'], no_int=a['no_int'], copypaste=a['copypaste'],
        skel=False, int_drop=[v for v in a['int_drop'].split(',') if v],
        elastic=a['elastic'], cutout=a['cutout'], hard=a['hard_crop'],
        hard_maps=a['hard_maps'], feats=a['feats'], scale_multiplier=a['scale_multiplier'])
    loader = DataLoader(dataset, batch_size=a['bs'], shuffle=True, num_workers=a['workers'],
                        drop_last=True, worker_init_fn=worker_init_fn,
                        persistent_workers=a['workers'] > 0, pin_memory=True)
    assert len(loader) == 144 and a['accum_steps'] == 1
    init_start = time.perf_counter()
    with patch.object(socket.socket, 'connect', no_network), patch.object(socket, 'create_connection', no_network):
        model = build_from_args(a, weights='imagenet')
    init_seconds = time.perf_counter() - init_start
    model = model.cuda().train()
    criterion = SegLoss(a['loss'], a['bw'], None, 0, use_ignore=False, soft_edge=0).cuda()
    enc = [p for n, p in model.named_parameters() if n.startswith('encoder.')]
    other = [p for n, p in model.named_parameters() if not n.startswith('encoder.')]
    optimizer = torch.optim.AdamW([{'params': enc, 'lr': a['lr'] * a['enc_lr_mult']},
                                  {'params': other, 'lr': a['lr']}], weight_decay=a['wd'])
    total, warm = a['epochs'] * len(loader), len(loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda i: (i + 1) / warm if i < warm else
        0.5 * (1 + math.cos(math.pi * (i - warm) / max(total - warm, 1))))
    iterator = iter(loader)
    timings, losses = [], []
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    for step in range(warmup + updates):
        torch.cuda.synchronize()
        started = time.perf_counter()
        batch = next(iterator)
        x, y = batch[0].cuda(non_blocking=True), batch[1].cuda(non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(a['amp']):
            logits = model(x)
            loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        value = loss.detach().item()
        assert math.isfinite(value), f'Nonfinite loss at step {step}'
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if step >= warmup:
            timings.append(elapsed)
            losses.append(value)
        print(json.dumps({'step': step + 1, 'warmup': step < warmup,
                          'seconds': elapsed, 'loss': value}), flush=True)
        del batch, x, y, logits, loss
    peak_allocated = torch.cuda.max_memory_allocated() / 2**30
    peak_reserved = torch.cuda.max_memory_reserved() / 2**30
    # Timing only: these temporary weights are discarded, and scores are not reported.
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    val_started = time.perf_counter()
    validation_rows = evaluate(model, valid, a['width'], a['tile'], a['overlap'],
                               'cuda', tta=False, amp=a['amp'], save_dir=None,
                               scale_multiplier=a['scale_multiplier'])
    torch.cuda.synchronize()
    validation_seconds = time.perf_counter() - val_started
    assert len(validation_rows) == 18
    mean = statistics.mean(timings)
    train_seconds = mean * len(loader) * a['epochs']
    # Twelve five-epoch validation passes and a rough 8x inference-time final TTA allowance.
    total_seconds = train_seconds + 12 * validation_seconds + 8 * validation_seconds
    return {
        'cached_imagenet_init_seconds': init_seconds,
        'train_images': len(train), 'validation_images': len(valid),
        'microbatch': a['bs'], 'gradient_accumulation': 1,
        'crops_per_epoch': len(dataset), 'optimizer_updates_per_epoch': len(loader),
        'warmup_updates': warmup, 'timed_updates': updates,
        'seconds_per_update_mean': mean, 'seconds_per_update_median': statistics.median(timings),
        'seconds_per_update_p10_p90': np.percentile(timings, [10, 90]).tolist(),
        'estimated_train_seconds_per_epoch': mean * len(loader),
        'estimated_train_minutes_60_epochs': train_seconds / 60,
        'validation18_no_tta_seconds': validation_seconds,
        'estimated_final_8tta_seconds': 8 * validation_seconds,
        'estimated_total_minutes_per_fold': total_seconds / 60,
        'peak_train_memory_allocated_gib': peak_allocated,
        'peak_train_memory_reserved_gib': peak_reserved,
        'peak_including_validation_memory_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
        'timed_update_seconds': timings, 'loss_first_last': [losses[0], losses[-1]],
        'all_losses_finite': True,
        'caveat': 'Short single-GPU estimate. Final TTA approximated at 8x no-TTA evaluation; excludes checkpoint writes, OOF writes, CPU comparison, loading and concurrent-job contention.',
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--updates', type=int, default=12)
    p.add_argument('--out', default='runs/review21_deeplab/benchmark.json')
    args = p.parse_args()
    assert args.gpu >= 0 and 2 <= args.warmup <= 10 and 10 <= args.updates <= 30
    os.chdir(ROOT)
    out = (ROOT / args.out).resolve()
    assert out.is_relative_to(ROOT) and not out.exists(), f'Refuse overwrite/outside path: {out}'
    csv = subprocess.check_output([
        'nvidia-smi', '-i', str(args.gpu), '--query-gpu=uuid,name,memory.used,memory.total,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True).strip()
    uuid, name, used, total, utilization = [x.strip() for x in csv.split(',')]
    assert int(used) < 1000 and int(utilization) < 10, f'GPU busy: {csv}'
    os.environ.update(CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
                      MKL_NUM_THREADS='1', HF_HUB_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
                      PYTHONDONTWRITEBYTECODE='1')
    sys.path.insert(0, str(ROOT / 'src'))
    source = ROOT / 'runs/cv/x2_mit_bw_r90/fold0/args.json'
    reference = json.loads(source.read_text(encoding='utf-8'))
    a = dict(reference)
    a.update(arch='DeepLabV3Plus', encoder='resnet34', width=2560, scale_multiplier=1.0,
             crop=1024, tile=1024, bs=4, accum_steps=1, workers=4, scale_jitter=0.15,
             epochs=60, seed=0, attention=None, hires=False, save_oof='', out='')
    assert a['bw'] == 2 and a['rot'] == 'rot90' and a['spi'] == 8 and a['amp']
    assert not any(a.get(k, 0) for k in ['mixup', 'cutmix', 'ema', 'hard_crop', 'soft_edge', 'cutout', 'skel', 'feats', 'init'])
    result = {'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'benchmarking',
              'purpose': 'Disposable train and validation timing only', 'writes_checkpoints': False,
              'writes_oof': False, 'weights_reused_for_experiment': False, 'imagenet_network_allowed': False,
              'reference_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'reference_args': reference, 'benchmark_args': a, 'gpu_index': args.gpu,
              'gpu_uuid': uuid, 'gpu_name': name, 'gpu_memory_total_mib': int(total)}
    save(out, result)
    try:
        result['measurement'] = measure(a, args.warmup, args.updates)
        result.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat())
        save(out, result)
        print(json.dumps(result, indent=2), flush=True)
    except BaseException as error:
        result.update(status='error', error=repr(error), updated_utc=datetime.now(timezone.utc).isoformat())
        save(out, result)
        raise


if __name__ == '__main__':
    main()
