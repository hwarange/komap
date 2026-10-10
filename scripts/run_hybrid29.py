"""Prepare, benchmark, and run one fixed60-epoch fold0 hybrid pilot."""
import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/review29_hybrid_pilot'
BASE = 'x4_mit_c1536_bw_r90'
EXPERIMENT = 'x4_hybrid_mit_cnn_c1536_bw_r90'
GPU_UUID = 'GPU-76da22e2-40da-ebc0-098a-fa92ba030976'
SOURCES = ['src/hybrid29.py', 'src/train_hybrid29.py', 'src/train.py', 'src/data.py', 'src/infer.py',
           'scripts/run_hybrid29.py', 'scripts/review_hybrid29.py', 'scripts/check_scale24_runtime.py',
           'scripts/run_crop18.py', 'src/review_grains15.py']
ENV = dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
           NUMEXPR_NUM_THREADS='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
           PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')


def now():
    return datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def save(name, value):
    p = OUT / name
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(p)


def verify(plan):
    for name, digest in plan['input_hashes'].items():
        assert sha(ROOT / name) == digest, f'Pinned input changed: {name}'


def idle():
    line = subprocess.check_output(['nvidia-smi', '-i', GPU_UUID,
        '--query-gpu=uuid,name,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip()
    uuid, name, used, util = map(str.strip, line.split(','))
    assert uuid == GPU_UUID and 'L40S' in name and int(used) < 1000 and int(util) < 10, line
    return line


def cpu_checks(args):
    import torch
    from hybrid29 import install
    train = install()
    torch.set_num_threads(2)
    with patch.object(socket.socket, 'connect', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('offline init required'))):
        torch.manual_seed(0)
        base = train.build_model('Unet', 'mit_b2', weights='imagenet').eval()
        base_rng = torch.get_rng_state().clone()
        torch.manual_seed(0)
        with patch.object(torch.cuda, 'manual_seed_all', side_effect=AssertionError('Hybrid initialization reseeded CUDA')):
            candidate = train.build_from_args(args, weights='imagenet').eval()
        assert torch.equal(torch.get_rng_state(), base_rng), 'Additional branch altered base/crop RNG'
    b, c = base.state_dict(), candidate.state_dict()
    for key, value in b.items():
        target = 'encoder.mit.' + key[len('encoder.'):] if key.startswith('encoder.') else key
        assert torch.equal(value, c[target]), key
    for shape in [(1, 1, 128, 160), (2, 1, 128, 128)]:
        x = torch.randn(shape)
        with torch.no_grad():
            by, cy = base(x), candidate(x)
        assert torch.equal(by, cy) and tuple(cy.shape) == (shape[0], 4, *shape[-2:])
    candidate.train()
    opt = torch.optim.SGD(candidate.parameters(), lr=0.01)
    losses = []
    for step in range(3):
        x = torch.randn(2, 1, 128, 128); y = torch.randint(0, 4, (2, 128, 128))
        opt.zero_grad(); logits = candidate(x)
        loss = torch.nn.functional.cross_entropy(logits, y); loss.backward()
        assert math.isfinite(loss.item())
        assert all(torch.isfinite(p.grad).all() for p in candidate.parameters() if p.grad is not None)
        if step == 2:
            for prefix in ['encoder.stem.', 'encoder.stages.', 'encoder.projections.', 'encoder.mit.', 'decoder.']:
                grads = [p.grad for n, p in candidate.named_parameters() if n.startswith(prefix) and p.grad is not None]
                assert grads and sum(float(g.abs().sum()) for g in grads) > 0, prefix
        opt.step(); losses.append(loss.item())
    clone = train.build_from_args(args, weights=None)
    clone.load_state_dict(candidate.state_dict(), strict=True)
    assert all(torch.isfinite(v).all() for v in clone.state_dict().values() if v.is_floating_point())
    base_n = sum(p.numel() for p in base.parameters()); cand_n = sum(p.numel() for p in candidate.parameters())
    return dict(base_parameters=base_n, hybrid_parameters=cand_n, additional_parameters=cand_n-base_n,
                initialization_logits_exactly_match_base=True, base_and_crop_rng_preserved=True,
                cuda_generators_not_reseeded_by_branch=True,
                all_branches_nonzero_finite_gradients_after3updates=True, strict_state_reload_passed=True,
                test_images_loaded=False, synthetic_step_losses=losses)


def prepare():
    import run_crop18 as common
    from hybrid29 import ARCH, architecture_spec
    from data import load_cv
    assert not OUT.exists(), 'Preparation already exists'
    reference_path = ROOT / f'runs/cv/{BASE}/fold0/args.json'
    reference = read(reference_path)
    assert all(reference[k] == v for k, v in dict(arch='Unet', encoder='mit_b2', epochs=60, fold=0,
        seed=0, crop=1536, tile=1536, width=2560, scale_multiplier=2.0, bs=2, accum_steps=2,
        spi=8, bw=2.0, init='', hires=False, full=False).items())
    args = dict(reference, arch=ARCH, out=f'runs/cv/{EXPERIMENT}/fold0', save_oof=f'runs/oof/{EXPERIMENT}')
    assert {k for k in reference if args[k] != reference[k]} == {'arch', 'out', 'save_oof'}
    for p in [args['out'], args['save_oof']]:
        assert not (ROOT / p).exists()
    # Pin the exact offline ImageNet snapshot used by the existing control.
    from huggingface_hub import try_to_load_from_cache
    from segmentation_models_pytorch.encoders import encoders
    assert encoders['mit_b2']['pretrained_settings']['imagenet'] == dict(repo_id='smp-hub/mit_b2.imagenet', revision=common.IMAGE_NET_REVISION)
    cache = []
    for filename, digest in common.IMAGE_NET_HASHES.items():
        path = try_to_load_from_cache('smp-hub/mit_b2.imagenet', filename, revision=common.IMAGE_NET_REVISION)
        assert isinstance(path, str) and sha(path) == digest
        cache.append(dict(filename=filename, path=path, sha256=digest))
    training, valid = load_cv('data', 0, 'splits/folds5.json')
    assert len(training) == 72 and len(valid) == 18
    assert {x['alloy'] for x in training}.isdisjoint({x['alloy'] for x in valid})
    paths = [ROOT / p for p in SOURCES] + [reference_path, ROOT / 'splits/folds5.json',
        ROOT / f'runs/cv/{BASE}/fold0/result.json', ROOT / f'runs/cv/{BASE}/fold0/last.pt',
        ROOT / 'runs/review27_finetune_full/results.json', ROOT / 'runs/review27_finetune_full/per_image_rows.json',
        ROOT / 'runs/review26_final_strengthening/finetune_plan/results.json',
        ROOT / 'runs/review26_final_strengthening/scale_trials/results.json']
    for item in training + valid:
        paths.extend(ROOT / f"data/{item['split']}/{kind}/{item['name']}_{suffix}.png" for kind, suffix in [('images','image'),('masks','mask')])
    for item in valid:
        n = item['name']
        paths += [ROOT / f'runs/oof/{BASE}/{n}.npy', ROOT / f'runs/review26_final_strengthening/finetune_plan/oof/interface/{n}.npy']
    scale = read(ROOT / 'runs/review26_final_strengthening/scale_trials/results.json')
    cap6 = {x['name']: x for x in scale['output_receipts'] if x['arm'] == 'cap6' and x['fold'] == 0}
    assert set(cap6) == {x['name'] for x in valid if x['gray'].shape[1] == 675}
    paths += [ROOT / x['path'] for x in cap6.values()]
    checks = cpu_checks(args)
    plan = dict(created=now(), args=args, reference_args=reference, pilot_fold=0, training_names=sorted(x['name'] for x in training),
                validation_names=sorted(x['name'] for x in valid), gpu_uuid=GPU_UUID, architecture=architecture_spec(),
                image_net_cache=cache, cpu_validation=checks, automatic_remaining_folds=[],
                fresh_imagenet_training=True, checkpoint_initialization=None, epochs=60,
                test_images_or_labels_used=False, previously_viewed_validation=True,
                current_submission_changed=False, fixed_candidate_inference_policies=['original','cap6_width675'],
                cap6_control_records=cap6, input_hashes={str(p.relative_to(ROOT)):sha(p) for p in paths},
                comparison='Only architecture differs from stored MiT fold0 control; same crop1536/scale/60epochs/seed/optimizer/augmentation/TTA',
                continuation_eligibility='Pilot only; inspect whole18-image single and equal-pair scores before extending folds. No test-specific gates.')
    OUT.mkdir(parents=True); save('plan.json', plan)
    print(json.dumps(dict(prepared=True, hashes=len(plan['input_hashes']), cpu_validation=checks), indent=2))


def benchmark():
    from hybrid29 import install
    import check_scale24_runtime as runtime
    assert not (OUT / 'runtime.json').exists() and not (OUT / 'started.json').exists()
    plan = read(OUT / 'plan.json'); verify(plan); gpu = idle()
    install()
    result = runtime.measure(plan['args'], warmup=2, updates=4)
    result.update(gpu=gpu, plan_sha256=sha(OUT / 'plan.json'), temporary_training_not_reused=True)
    save('runtime.json', result)
    print(json.dumps(result, indent=2))


def run():
    import run_crop18 as common
    plan = read(OUT / 'plan.json'); verify(plan); idle()
    runtime = read(OUT / 'runtime.json'); assert runtime['all_losses_finite'] and runtime['plan_sha256'] == sha(OUT / 'plan.json')
    lock = (OUT / 'controller.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not any((OUT / p).exists() for p in ('started.json','all.done','error.json'))
    for p in [plan['args']['out'], plan['args']['save_oof']]: assert not (ROOT / p).exists()
    command = [str(ROOT / '.venv/bin/python'), 'src/train_hybrid29.py', *common.cli_args(plan['args'])]
    bound = max(7200, runtime['estimated_total_minutes'] * 60 * 1.8 + 1800)
    child = None
    def stop(*unused):
        if child and child.poll() is None: child.terminate()
        raise RuntimeError('Controller interrupted')
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    try:
        with (OUT / 'training_controller.log').open('x') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=dict(ENV, CUDA_VISIBLE_DEVICES=GPU_UUID), stdout=log, stderr=subprocess.STDOUT)
            save('started.json', dict(time=now(), pid=os.getpid(), child_pid=child.pid, command=command,
                plan_sha256=sha(OUT / 'plan.json'), runtime_sha256=sha(OUT / 'runtime.json'), bound_seconds=bound))
            started = time.monotonic()
            while child.poll() is None:
                logs = ROOT / plan['args']['out'] / 'log.jsonl'
                rows = [json.loads(x) for x in logs.read_text().splitlines() if x] if logs.exists() else []
                save('state.json', dict(time=now(), stage='training' if len(rows)<60 else 'final_validation', pid=os.getpid(),
                    child_pid=child.pid, completed_epochs=len(rows), total_epochs=60, latest_epoch=rows[-1] if rows else None,
                    elapsed_minutes=(time.monotonic()-started)/60, further_training_scheduled=False))
                assert time.monotonic()-started < bound, 'Pilot exceeded execution bound'
                time.sleep(20)
            assert child.returncode == 0, f'Training exit{child.returncode}; see log'
        verify(plan)
        save('state.json', dict(time=now(), stage='fixed_cap6_inference_then_cpu_review', completed_epochs=60, total_epochs=60))
        with (OUT / 'review_controller.log').open('x') as log:
            command = [str(ROOT / '.venv/bin/python'), 'scripts/review_hybrid29.py']
            child = subprocess.Popen(command, cwd=ROOT, env=dict(ENV, CUDA_VISIBLE_DEVICES=GPU_UUID), stdout=log, stderr=subprocess.STDOUT)
            child.wait(timeout=1800); assert child.returncode == 0, 'Review failed'
        assert (OUT / 'all.done').exists()
        save('state.json', dict(time=now(), stage='complete_and_stop', completed_epochs=60, total_epochs=60,
            results='runs/review29_hybrid_pilot/results.json', further_training_scheduled=False))
    except BaseException as error:
        save('error.json', dict(time=now(), error=f'{type(error).__name__}: {error}'))
        raise
    finally:
        if child and child.poll() is None:
            child.terminate()
            try: child.wait(timeout=30)
            except subprocess.TimeoutExpired: child.kill(); child.wait()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['prepare','benchmark','run']); opts = p.parse_args()
    os.chdir(ROOT); sys.path.insert(0, str(ROOT / 'src'))
    os.environ.update(ENV)
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_UUID if opts.mode == 'benchmark' else ''
    {'prepare':prepare,'benchmark':benchmark,'run':run}[opts.mode]()


if __name__ == '__main__': main()
