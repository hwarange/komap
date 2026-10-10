"""Fresh train70/valid20 MiT2 control; only scale differs from submission31 MiT4."""
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
OUT = ROOT / 'runs/review32_mit2_train70'
PLAN = OUT / 'plan.json'
REFERENCE_PLAN = 'runs/submission31_train70/plan.json'


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
    """Hash tensor names, dtypes, shapes and exact parameter/buffer bytes."""
    import torch
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        metadata = json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(',', ':'))
        digest.update(metadata.encode('utf-8') + b'\0')
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def no_network(*args, **kwargs):
    raise RuntimeError('Scale32 initialization requires the pinned local ImageNet cache')


def validate_recipe(plan):
    assert len(plan['jobs']) == 1 and plan['jobs'][0]['id'] == 'mit2'
    job = plan['jobs'][0]
    assert not job.get('continuation'), 'Scale32 never resumes a checkpoint'
    values = job['args']
    reference = plan['reference_mit4_args']
    assert REFERENCE_PLAN in plan['input_hashes'], 'Reference train70 MiT4 plan must be pinned'
    reference_plan = read(ROOT / REFERENCE_PLAN)
    reference_jobs = [entry for entry in reference_plan['jobs'] if entry['id'] == 'mit4']
    assert len(reference_jobs) == 1 and reference == reference_jobs[0]['args']
    assert set(values) == set(reference), 'Trainer argument schema differs from MiT4'
    differences = {key for key in values if values[key] != reference[key]}
    assert differences == {'scale_multiplier', 'out', 'save_oof'}, f'Unexpected recipe differences: {differences}'
    expected = dict(data='data', fold=-1, train_splits=['train'], val_split='valid', full=False,
                    train_frac=1.0, seed=0, init='', init_skip_head=False, arch='Unet', encoder='mit_b2',
                    width=2560, scale_multiplier=1.0, crop=1536, tile=1536, epochs=60,
                    bs=2, accum_steps=2, hard_crop=0.0, spi=8, max_iters=0)
    assert all(values.get(key) == value for key, value in expected.items()), 'Wrong fresh MiT2 protocol'
    assert reference['scale_multiplier'] == 2.0
    training_names, validation_names = sorted(plan['training_names']), sorted(plan['validation_names'])
    assert len(training_names) == len(set(training_names)) == 70
    assert len(validation_names) == len(set(validation_names)) == 20
    assert set(training_names).isdisjoint(validation_names)
    assert training_names == sorted(reference_plan['training_names'])
    assert validation_names == sorted(reference_plan['validation_names'])
    return values, training_names, validation_names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', choices=['mit2'], default='mit2')
    parser.add_argument('--stage', choices=['base'], default='base')
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
    values, training_names, validation_names = validate_recipe(plan)
    output, probability_dir = under(values['out'], OUT), under(values['save_oof'], OUT)
    assert not output.exists() and not probability_dir.exists(), 'Refusing to overwrite a training run'
    assert output != probability_dir

    import cv2
    import torch
    import train
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
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
        assert args[:2] == ('Unet', 'mit_b2')
        assert kwargs.get('weights', 'imagenet') == 'imagenet'
        with patch.object(socket.socket, 'connect', no_network), patch.object(socket, 'create_connection', no_network):
            model = original_factory(*args, **kwargs)
        state = model.state_dict()
        assert all(torch.isfinite(value).all().item() for value in state.values() if value.is_floating_point())
        initial_sha = state_sha256(state)
        assert initial_sha == plan['reference_initial_state_sha256'], 'Initial weights differ from the fresh MiT4 control'
        audit = dict(created_utc=datetime.now(timezone.utc).isoformat(), job=selected.job, stage=selected.stage,
                     plan_sha256=sha256(PLAN), args=values, training_names=training_names,
                     validation_names=validation_names, image_net_cache=cache,
                     reference_mit4_plan_sha256=sha256(ROOT / REFERENCE_PLAN),
                     matched_reference_initialization=True,
                     differences_from_mit4=['out', 'save_oof', 'scale_multiplier'],
                     initialization='ImageNet_encoder_and_new_decoder',
                     previous_experiment_checkpoint_used=False, inherited_checkpoint=None,
                     fresh_imagenet_state_sha256=initial_sha, full_initial_state_sha256=initial_sha,
                     initial_state_tensor_count=len(state), strict_initialization=True,
                     training_images=70, validation_images=20, test_images_used=False)
        audit_path = output / 'initialization_audit.json'
        assert not audit_path.exists()
        audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        factory_calls.append(audit_path)
        print(json.dumps(dict(scale32_initialization='verified', job=selected.job, stage=selected.stage,
                              initial_state_sha256=initial_sha, inherited_checkpoint=None)), flush=True)
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
