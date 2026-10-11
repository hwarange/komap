"""Fresh ImageNet training on alloy-group folds contained entirely in train70."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sys
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/review34_train70_group5fold'
PLAN = OUT / 'plan.json'
FAMILIES = {
    'mit2': ('Unet', 'mit_b2', 60, 2560, 1.0, 2, 2),
    'mit4': ('Unet', 'mit_b2', 60, 2560, 2.0, 2, 2),
    'unet3': ('UnetPlusPlus', 'resnet34', 120, 3840, 1.0, 4, 1),
    'context4': ('GlobalContextMiTCNN30', 'mit_b2', 60, 2560, 2.0, 2, 2),
}


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


def relative(path):
    return Path(path).relative_to(ROOT).as_posix()


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
    import torch
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        metadata = json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(',', ':'))
        digest.update(metadata.encode('utf-8') + b'\0')
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def no_network(*args, **kwargs):
    raise RuntimeError('Group34 requires the pinned local ImageNet cache')


def validate_plan(plan, identifier):
    assert plan['input_hashes'], 'No pinned inputs'
    for name, digest in plan['input_hashes'].items():
        assert not Path(name).is_absolute(), 'Project input keys must be relative'
        assert sha256(under(name)) == digest, f'Pinned input changed: {name}'
    assert plan['image_net_cache'], 'No pinned ImageNet cache'
    for record in plan['image_net_cache']:
        assert sha256(Path(record['path']).expanduser()) == record['sha256'], 'ImageNet cache changed'
    folds_path = under(plan['folds_file'], OUT)
    assert folds_path.suffix == '.json'
    assert plan['input_hashes'][relative(folds_path)] == plan['fold_assignment_sha256'] == sha256(folds_path)
    fold_of = read(folds_path)
    train70 = sorted(plan['train70_names'])
    official20 = sorted(plan['official_validation_names'])
    assert len(train70) == len(set(train70)) == 70
    assert len(official20) == len(set(official20)) == 20
    assert set(train70).isdisjoint(official20)
    assert sorted(fold_of) == train70
    assert all(type(value) is int and 0 <= value < 5 for value in fold_of.values())
    assert set(fold_of.values()) == set(range(5))
    expected_ids = {f'{family}_fold{fold}' for family in FAMILIES for fold in range(5)}
    assert len(plan['jobs']) == 20 and {job['id'] for job in plan['jobs']} == expected_ids
    assert identifier in expected_ids
    references = {}
    for family in FAMILIES:
        record = plan['initialization_reference'][family]
        path = under(record['path'])
        assert path.suffix == '.json', 'Initialization reference must be metadata, never a checkpoint'
        assert plan['input_hashes'][relative(path)] == record['sha256'] == sha256(path)
        audit = read(path)
        assert audit['inherited_checkpoint'] is None and not audit['previous_experiment_checkpoint_used']
        assert audit['strict_initialization']
        assert audit['full_initial_state_sha256'] == audit['fresh_imagenet_state_sha256'] == record['state_sha256']
        assert sorted(audit['training_names']) == train70 and sorted(audit['validation_names']) == official20
        references[family] = audit
    outputs = []
    for job in plan['jobs']:
        family, fold, args = job['family'], job['fold'], job['args']
        assert family in FAMILIES and type(fold) is int and 0 <= fold < 5
        assert job['id'] == f'{family}_fold{fold}'
        arch, encoder, epochs, width, multiplier, batch, accumulation = FAMILIES[family]
        expected = dict(data='data', fold=fold, folds_file=plan['folds_file'], train_splits=['train'],
                        val_split='valid', full=False, init='', init_skip_head=False, hard_crop=0.0,
                        train_frac=1.0, seed=0, max_iters=0, arch=arch, encoder=encoder,
                        epochs=epochs, width=width, scale_multiplier=multiplier, crop=1536,
                        tile=1536, bs=batch, accum_steps=accumulation, spi=8)
        assert all(args.get(key) == value for key, value in expected.items()), f'Wrong recipe: {job["id"]}'
        reference_args = references[family]['args']
        assert set(args) == set(reference_args), 'Incomplete trainer arguments'
        differences = {key for key in args if args[key] != reference_args[key]}
        assert differences == {'fold', 'folds_file', 'out', 'save_oof'}, f'Unexpected recipe changes: {differences}'
        training_names = sorted(name for name in train70 if fold_of[name] != fold)
        validation_names = sorted(name for name in train70 if fold_of[name] == fold)
        assert sorted(job['training_names']) == training_names
        assert sorted(job['validation_names']) == validation_names
        assert len(job['training_names']) == len(training_names) and len(job['validation_names']) == len(validation_names)
        assert training_names and validation_names and set(training_names).isdisjoint(validation_names)
        assert set(training_names + validation_names) == set(train70)
        assert set(training_names + validation_names).isdisjoint(official20)
        microbatches = len(training_names) * args['spi'] // args['bs']
        assert job['microbatches_per_epoch'] == microbatches > 0
        assert job['optimizer_updates_per_epoch'] == math.ceil(microbatches / args['accum_steps'])
        outputs.extend([under(args['out'], OUT), under(args['save_oof'], OUT)])
    assert len(outputs) == len(set(outputs)), 'Jobs must use separate output and probability directories'
    job = next(job for job in plan['jobs'] if job['id'] == identifier)
    return job, fold_of, train70, official20, references[job['family']]


def validate_items(plan, fold_of, items):
    assert sorted(item['name'] for item in items) == sorted(plan['train70_names'])
    assert len(items) == 70 and all(item['split'] == 'train' and item['lab'] is not None for item in items)
    alloy_of = {item['name']: int(item['alloy']) for item in items}
    for alloy in set(alloy_of.values()):
        assert len({fold_of[name] for name in alloy_of if alloy_of[name] == alloy}) == 1, 'Alloy spans folds'
    for job in plan['jobs']:
        training_alloys = sorted({alloy_of[name] for name in job['training_names']})
        validation_alloys = sorted({alloy_of[name] for name in job['validation_names']})
        assert sorted(job['training_alloys']) == training_alloys
        assert sorted(job['validation_alloys']) == validation_alloys
        assert len(job['training_alloys']) == len(training_alloys)
        assert len(job['validation_alloys']) == len(validation_alloys)
        assert set(training_alloys).isdisjoint(validation_alloys), 'Train/validation alloy overlap'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True)
    parser.add_argument('--check-only', action='store_true')
    selected = parser.parse_args()
    os.chdir(ROOT)
    os.environ.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                      NUMEXPR_NUM_THREADS='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
    if selected.check_only:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.dont_write_bytecode = True
    plan = read(PLAN)
    job, fold_of, train70, official20, reference = validate_plan(plan, selected.job)
    values = job['args']
    output, probability_dir = under(values['out'], OUT), under(values['save_oof'], OUT)
    assert not output.exists() and not probability_dir.exists(), 'Refusing to overwrite a fold run'

    import cv2
    import numpy as np
    import torch
    import data
    import train
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if values['arch'] == 'GlobalContextMiTCNN30':
        from hybrid30 import install
        train = install(values['arch'])
    original_load_split = data.load_split
    observed = {}

    def guarded_split(root, split):
        assert under(root) == ROOT / 'data'
        assert split == 'train', 'Official valid20 and test10 may not be loaded by group34'
        items = original_load_split(root, 'train')
        validate_items(plan, fold_of, items)
        return items

    def guarded_cv(root, fold, folds_file):
        assert fold == job['fold'] and under(folds_file, OUT) == under(plan['folds_file'], OUT)
        items = guarded_split(root, 'train')
        training = [item for item in items if fold_of[item['name']] != fold]
        validation = [item for item in items if fold_of[item['name']] == fold]
        assert sorted(item['name'] for item in training) == sorted(job['training_names'])
        assert sorted(item['name'] for item in validation) == sorted(job['validation_names'])
        observed.update(training_names=sorted(item['name'] for item in training),
                        validation_names=sorted(item['name'] for item in validation),
                        training_alloys=sorted({item['alloy'] for item in training}),
                        validation_alloys=sorted({item['alloy'] for item in validation}))
        return training, validation

    original_factory = train.build_model
    factory_calls = []

    def guarded_factory(*args, **kwargs):
        assert not factory_calls and observed, 'Unexpected model initialization'
        assert args[:2] == (values['arch'], values['encoder'])
        assert kwargs.get('weights', 'imagenet') == 'imagenet'
        with patch.object(socket.socket, 'connect', no_network), patch.object(socket, 'create_connection', no_network):
            model = original_factory(*args, **kwargs)
        state = model.state_dict()
        assert all(torch.isfinite(value).all().item() for value in state.values() if value.is_floating_point())
        initial_sha = state_sha256(state)
        assert initial_sha == plan['initialization_reference'][job['family']]['state_sha256']
        audit = dict(created_utc=datetime.now(timezone.utc).isoformat(), job=job['id'], family=job['family'],
                     fold=job['fold'], stage='base', plan_sha256=sha256(PLAN), args=values,
                     fold_assignment_sha256=plan['fold_assignment_sha256'], **observed,
                     train70_names=train70, official_validation_names=official20,
                     training_images=len(observed['training_names']), validation_images=len(observed['validation_names']),
                     official_validation_images_used=False, test_images_used=False,
                     group_disjoint=True, train70_only=True, initialization='ImageNet_encoder_and_new_decoder',
                     image_net_cache=plan['image_net_cache'], initialization_reference=plan['initialization_reference'][job['family']],
                     reference_checkpoint_loaded=False, previous_experiment_checkpoint_used=False, inherited_checkpoint=None,
                     fresh_imagenet_state_sha256=initial_sha, full_initial_state_sha256=initial_sha,
                     initial_state_tensor_count=len(state), strict_initialization=True,
                     matched_reference_initialization=True, check_only=selected.check_only)
        if selected.check_only:
            print(json.dumps(dict(preflight='passed', **audit), ensure_ascii=False, allow_nan=False), flush=True)
        else:
            path = output / 'initialization_audit.json'
            assert not path.exists()
            path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
            print(json.dumps(dict(group34_initialization='verified', job=job['id'], family=job['family'],
                                  fold=job['fold'], training_images=audit['training_images'],
                                  validation_images=audit['validation_images'], initial_state_sha256=initial_sha)), flush=True)
        factory_calls.append(initial_sha)
        return model

    original_parse = argparse.ArgumentParser.parse_args
    parsed = []

    class ParsedOnly(Exception):
        pass

    def guarded_parse(parser, *args, **kwargs):
        actual = original_parse(parser, *args, **kwargs)
        assert vars(actual) == values, 'Actual full trainer arguments differ from pinned plan'
        parsed.append(dict(vars(actual)))
        if selected.check_only:
            raise ParsedOnly()
        return actual

    sys.argv = [str(Path(__file__).resolve()), *cli_args(values)]
    with patch.object(train, 'load_split', guarded_split), patch.object(data, 'load_split', guarded_split), \
            patch.object(train, 'load_cv', guarded_cv), patch.object(data, 'load_cv', guarded_cv), \
            patch.object(train, 'build_model', guarded_factory), \
            patch.object(argparse.ArgumentParser, 'parse_args', guarded_parse):
        if selected.check_only:
            try:
                train.main()
            except ParsedOnly:
                pass
            assert len(parsed) == 1
            torch.manual_seed(values['seed'])
            np.random.seed(values['seed'])
            guarded_cv(values['data'], values['fold'], values['folds_file'])
            guarded_factory(values['arch'], values['encoder'], attention=values['attention'],
                            hires=values['hires'], n_classes=len(data.CLASSES), feats=values['feats'])
            assert not output.exists() and not probability_dir.exists(), 'Preflight must not create training outputs'
        else:
            train.main()
    assert len(parsed) == len(factory_calls) == 1
    assert observed['training_names'] == sorted(job['training_names'])
    assert observed['validation_names'] == sorted(job['validation_names'])


if __name__ == '__main__':
    main()
