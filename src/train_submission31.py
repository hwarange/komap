"""Guarded original train70/valid20 entry point; never reuse prior experiment weights."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
from datetime import datetime, timezone
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / 'runs/submission31_train70'
PLAN = SUBMISSION / 'plan.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def under(path, root=ROOT):
    target = (ROOT / path).resolve()
    assert target.is_relative_to(root.resolve()), f'Path outside allowed root: {path}'
    return target


def cli_args(values):
    result = []
    for key, value in values.items():
        flag = '--' + key.replace('_', '-')
        if value is None or value == '' or value is False:
            continue
        if value is True:
            result.append(flag)
        elif isinstance(value, list):
            result.extend([flag, *map(str, value)])
        else:
            result.extend([flag, str(value)])
    return result


def state_sha256(state):
    """Hash every parameter and buffer, including tensor metadata and exact bytes."""
    import torch
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        metadata = json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(',', ':'))
        digest.update(metadata.encode('utf-8') + b'\0')
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def no_network(*args, **kwargs):
    raise RuntimeError('Submission31 initialization requires the pinned local ImageNet cache')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True, choices=['mit4', 'unet3', 'context4'])
    parser.add_argument('--stage', choices=['base', 'continuation'], default='base')
    selected = parser.parse_args()
    os.chdir(ROOT)
    os.environ.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                      NUMEXPR_NUM_THREADS='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
    sys.dont_write_bytecode = True
    plan = read(PLAN)
    assert plan['input_hashes'], 'No pinned project inputs'
    for name, digest in plan['input_hashes'].items():
        assert not Path(name).is_absolute(), f'Input hash key must be project-relative: {name}'
        assert sha256(under(name)) == digest, f'Pinned project input changed: {name}'
    cache = plan['image_net_cache']
    assert cache, 'No pinned ImageNet cache'
    for item in cache:
        assert sha256(Path(item['path']).expanduser()) == item['sha256'], 'ImageNet cache hash mismatch'
    matches = [job for job in plan['jobs'] if job['id'] == selected.job]
    assert len(matches) == 1, f'Missing or duplicated job: {selected.job}'
    job = matches[0]
    base_args = job['args']
    values = base_args if selected.stage == 'base' else job['continuation']['args']
    expected = dict(fold=-1, train_splits=['train'], val_split='valid', full=False,
                    train_frac=1.0, seed=0, data='data')
    assert all(values.get(key) == value for key, value in expected.items()), 'Wrong train70/valid20 protocol'
    assert all(base_args.get(key) == value for key, value in expected.items()), 'Wrong base protocol'
    assert base_args['init'] == '' and not base_args['init_skip_head']
    output = under(values['out'], SUBMISSION)
    probability_dir = under(values['save_oof'], SUBMISSION)
    assert not output.exists() and not probability_dir.exists(), 'Refusing to overwrite a training run'
    assert output != probability_dir
    if values.get('hard_crop', 0):
        map_dir = under(values['hard_maps'], SUBMISSION)
        assert map_dir.is_dir(), 'Train-only boundary maps missing'
    training_names = sorted(plan['training_names'])
    validation_names = sorted(plan['validation_names'])
    assert len(training_names) == len(set(training_names)) == 70
    assert len(validation_names) == len(set(validation_names)) == 20
    assert set(training_names).isdisjoint(validation_names)
    if values.get('hard_crop', 0):
        assert sorted(path.stem for path in map_dir.glob('*.png')) == training_names
        assert all(str(path.relative_to(ROOT)).replace(os.sep, '/') in plan['input_hashes']
                   for path in map_dir.glob('*.png')), 'Every train boundary map must be pinned'
    expected_arch = {'mit4': ('Unet', 'mit_b2'), 'unet3': ('UnetPlusPlus', 'resnet34'),
                     'context4': ('GlobalContextMiTCNN30', 'mit_b2')}
    assert (values['arch'], values['encoder']) == expected_arch[selected.job]
    initial_checkpoint = None
    if selected.stage == 'base':
        assert values['init'] == '' and not values['init_skip_head'] and values['hard_crop'] == 0
        assert values['epochs'] == (120 if selected.job == 'unet3' else 60)
    else:
        assert selected.job == 'unet3', 'Continuation is restricted to the new UNet3 base run'
        initial_checkpoint = under(values['init'], SUBMISSION)
        assert initial_checkpoint == under(base_args['out'], SUBMISSION) / 'last.pt'
        assert base_args['epochs'] == 120 and values['epochs'] == 6
        assert not values['init_skip_head'] and values['hard_crop'] == .35 and values['lr'] == 3e-5
        assert read(initial_checkpoint.parent / 'args.json') == base_args
        base_audit = read(initial_checkpoint.parent / 'initialization_audit.json')
        assert base_audit['job'] == 'unet3' and base_audit['stage'] == 'base'
        assert base_audit['inherited_checkpoint'] is None and base_audit['args'] == base_args
        assert base_audit['plan_sha256'] == sha256(PLAN)

    import cv2
    import torch
    import train
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if values['arch'] == 'GlobalContextMiTCNN30':
        from hybrid30 import install
        train = install(values['arch'])
    original_load = train.load_split
    observed_splits = {}

    def guarded_split(root, split):
        assert under(root) == ROOT / 'data' and split in ('train', 'valid')
        items = original_load(root, split)
        names = sorted(item['name'] for item in items)
        assert names == (training_names if split == 'train' else validation_names)
        assert all(item['lab'] is not None and item['split'] == split for item in items)
        observed_splits[split] = names
        return items

    def forbidden_cv(*args, **kwargs):
        raise AssertionError('Cross-validation is forbidden in this train70/valid20 run')

    original_factory = train.build_model
    factory_calls = []

    def guarded_factory(*args, **kwargs):
        assert not factory_calls, 'Unexpected additional model initialization'
        assert observed_splits == {'train': training_names, 'valid': validation_names}
        assert kwargs.get('weights', 'imagenet') == 'imagenet'
        with patch.object(socket.socket, 'connect', no_network), patch.object(socket, 'create_connection', no_network):
            model = original_factory(*args, **kwargs)
        fresh_state = model.state_dict()
        assert all(torch.isfinite(v).all().item() for v in fresh_state.values() if v.is_floating_point())
        fresh_sha = state_sha256(fresh_state)
        inherited, initial_sha = None, fresh_sha
        if initial_checkpoint is not None:
            digest_before = sha256(initial_checkpoint)
            checkpoint = torch.load(initial_checkpoint, map_location='cpu', weights_only=False)
            assert checkpoint['epoch'] == 120 and checkpoint['args'] == base_args
            state = checkpoint['model']
            assert fresh_state.keys() == state.keys(), 'Continuation checkpoint tensor keys differ'
            assert all(fresh_state[k].shape == v.shape and fresh_state[k].dtype == v.dtype for k, v in state.items())
            assert all(torch.isfinite(v).all().item() for v in state.values() if v.is_floating_point())
            # Prove strict compatibility without changing the trainer's loading or optimizer behavior.
            model.load_state_dict(state, strict=True)
            initial_sha = state_sha256(model.state_dict())
            assert initial_sha == state_sha256(state)
            assert sha256(initial_checkpoint) == digest_before
            inherited = dict(path=str(initial_checkpoint.relative_to(ROOT)).replace(os.sep, '/'),
                             sha256=digest_before, epoch=120, source='own_fresh_train70_base')
        audit = dict(created_utc=datetime.now(timezone.utc).isoformat(), job=selected.job, stage=selected.stage,
                     plan_sha256=sha256(PLAN), args=values, training_names=training_names,
                     validation_names=validation_names, image_net_cache=cache,
                     initialization='ImageNet_encoder_and_new_decoder' if inherited is None else 'own_fresh_train70_epoch120',
                     previous_experiment_checkpoint_used=False, inherited_checkpoint=inherited,
                     fresh_imagenet_state_sha256=fresh_sha, full_initial_state_sha256=initial_sha,
                     initial_state_tensor_count=len(fresh_state), strict_initialization=True,
                     training_images=70, validation_images=20, test_images_used=False)
        audit_path = output / 'initialization_audit.json'
        assert not audit_path.exists()
        audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        factory_calls.append(audit_path)
        print(json.dumps(dict(submission31_initialization='verified', job=selected.job, stage=selected.stage,
                              initial_state_sha256=initial_sha, inherited_checkpoint=inherited)), flush=True)
        return model

    original_parse = argparse.ArgumentParser.parse_args
    parsed = []

    def guarded_parse(parser, *args, **kwargs):
        actual = original_parse(parser, *args, **kwargs)
        assert vars(actual) == values, 'Actual trainer arguments differ from pinned full argument plan'
        parsed.append(dict(vars(actual)))
        return actual

    sys.argv = [str(Path(__file__).resolve()), *cli_args(values)]
    with patch.object(train, 'load_split', guarded_split), patch.object(train, 'load_cv', forbidden_cv), \
            patch.object(train, 'build_model', guarded_factory), \
            patch.object(argparse.ArgumentParser, 'parse_args', guarded_parse):
        train.main()
    assert len(parsed) == len(factory_calls) == 1
    assert observed_splits == {'train': training_names, 'valid': validation_names}


if __name__ == '__main__':
    main()
