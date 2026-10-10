"""Audit fresh MiT2 train70/valid20 and compare with the matched fresh MiT4 run."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/review32_mit2_train70'
REFERENCE = ROOT / 'runs/submission31_train70'
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
os.chdir(ROOT)
METRICS = ('plain', 'pp', 'tta', 'tta_pp')


def now():
    return datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def relative(path):
    return Path(path).relative_to(ROOT).as_posix()


def under(path, folder=OUT):
    result = (ROOT / path).resolve()
    assert result.is_relative_to(folder.resolve()), f'Unexpected artifact location: {path}'
    return result


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    return value


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + '\n', encoding='utf8')
    temporary.replace(path)


def receipt(path):
    return dict(path=relative(path), sha256=sha(path))


def verify(plan):
    assert len(plan['jobs']) == 1 and plan['jobs'][0]['id'] == 'mit2'
    assert plan['input_hashes']
    for path, digest in plan['input_hashes'].items():
        assert sha(ROOT / path) == digest, path
    for item in plan.get('image_net_cache', []):
        assert sha(item['path']) == item['sha256'], item['path']
    reference_path = (ROOT / plan['reference_plan']).resolve()
    assert reference_path == (REFERENCE / 'plan.json').resolve()
    assert relative(reference_path) in plan['input_hashes']
    original = next(job for job in read(reference_path)['jobs'] if job['id'] == 'mit4')
    assert original['args'] == plan['reference_mit4_args']
    args = plan['jobs'][0]['args']
    assert set(args) == set(original['args'])
    assert {key for key in args if args[key] != original['args'][key]} == {'scale_multiplier', 'out', 'save_oof'}
    assert args['scale_multiplier'] == 1. and original['args']['scale_multiplier'] == 2.


def dataset(plan):
    from data import load_split
    job = plan['jobs'][0]
    args = job['args']
    expected = dict(data='data', train_splits=['train'], val_split='valid', fold=-1, full=False,
                    train_frac=1., seed=0, init='', init_skip_head=False, epochs=60,
                    arch='Unet', encoder='mit_b2', crop=1536, tile=1536, width=2560,
                    scale_multiplier=1., bs=2, accum_steps=2, spi=8, max_iters=0, hard_crop=0.)
    assert all(args.get(key) == value for key, value in expected.items()), args
    training, valid = load_split(args['data'], 'train'), load_split(args['data'], 'valid')
    train_names = sorted(item['name'] for item in training)
    valid_names = sorted(item['name'] for item in valid)
    assert len(training) == len(set(train_names)) == 70
    assert len(valid) == len(set(valid_names)) == 20
    assert set(train_names).isdisjoint(valid_names)
    assert all(item['lab'] is not None for item in training + valid)
    for container in (plan, job):
        for key in ('training_names', 'train_names'):
            if key in container:
                assert container[key] == train_names
        if 'validation_names' in container:
            assert container['validation_names'] == valid_names
    return training, valid


def probability(path, item):
    import numpy as np
    stored = np.load(path, allow_pickle=False)
    assert stored.dtype == np.float16 and stored.shape == (4, *item['gray'].shape), str(path)
    assert np.isfinite(stored).all() and stored.min() >= 0 and stored.max() <= 1, str(path)
    result = stored.astype(np.float32)
    assert np.max(np.abs(result.sum(0) - 1)) < .005, str(path)
    return result


def row(item, ious):
    return dict(iou=ious.tolist(), alloy=int(item['alloy']), width=int(item['gray'].shape[1]),
                low=bool(item['low_contrast']))


def scores(rows):
    import numpy as np
    from data import CLASSES
    result = {}
    for label, records in rows.items():
        values = np.asarray([item['iou'] for item in records.values()], dtype=float)
        assert values.shape == (20, 4)
        assert not np.isinf(values).any()
        assert np.isfinite(values).any(axis=1).all()
        assert ((values[np.isfinite(values)] >= 0) & (values[np.isfinite(values)] <= 1)).all()
        result[label] = dict(images=20, mIoU=float(np.nanmean(values, axis=1).mean() * 100),
            class_IoU={name: float(np.nanmean(values[:, i]) * 100) for i, name in enumerate(CLASSES)},
            both_absent_one_mIoU=float(np.where(np.isnan(values), 1., values).mean() * 100),
            both_absent_class_image_pairs=int(np.isnan(values).sum()))
    return result


def review():
    import numpy as np
    import torch
    import train
    from infer import image_iou, postprocess
    from data import CLASSES, scale_for
    torch.set_num_threads(4)
    plan = read(OUT / 'plan.json')
    verify(plan)
    training, valid = dataset(plan)
    args = plan['jobs'][0]['args']
    run, probs = under(args['out']), under(args['save_oof'])
    destination = OUT / 'review'
    assert not destination.exists(), str(destination)
    assert read(run / 'args.json') == args
    trained = read(run / 'result.json')
    assert trained['args'] == args
    names = sorted(item['name'] for item in valid)
    assert sorted(item['name'] for item in trained['rows']) == names
    assert sorted(path.stem for path in probs.glob('*.npy')) == names
    logs = [json.loads(line) for line in (run / 'log.jsonl').read_text(encoding='utf8').splitlines() if line]
    assert [item['epoch'] for item in logs] == list(range(1, 61))
    assert all(math.isfinite(item['loss']) and item['microbatches'] == 280 and item['optimizer_updates'] == 140 for item in logs)
    checkpoint = run / 'last.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert state['args'] == args and state['epoch'] == 60
    assert all(torch.isfinite(value).all() for value in state['model'].values() if value.is_floating_point())
    model = train.build_from_args(args, weights=None)
    model.load_state_dict(state['model'], strict=True)
    del model, state
    audit_path = run / 'initialization_audit.json'
    audit = read(audit_path)
    assert audit['job'] == 'mit2' and audit['stage'] == 'base' and audit['args'] == args
    assert audit['plan_sha256'] == sha(OUT / 'plan.json') and audit['inherited_checkpoint'] is None
    assert not audit['previous_experiment_checkpoint_used'] and audit['strict_initialization']
    assert audit['training_names'] == sorted(item['name'] for item in training) and audit['validation_names'] == names
    assert audit['fresh_imagenet_state_sha256'] == audit['full_initial_state_sha256']
    assert audit['full_initial_state_sha256'] == plan['reference_initial_state_sha256']
    by_name = {item['name']: item for item in trained['rows']}
    rows = {key: {} for key in (*METRICS, 'stored_tta', 'stored_tta_pp')}
    scales = []
    for item in valid:
        name = item['name']
        reported = by_name[name]
        assert reported['W'] == item['gray'].shape[1] and reported['low'] == item['low_contrast']
        for key in METRICS:
            values = np.asarray(reported[key], dtype=float)
            assert values.shape == (4,)
            rows[key][name] = row(item, values)
        prob = probability(probs / f'{name}.npy', item)
        rows['stored_tta'][name] = row(item, image_iou(item['lab'], prob.argmax(0)))
        rows['stored_tta_pp'][name] = row(item, image_iou(item['lab'], postprocess(prob)))
        scales.append(dict(name=name, width=int(item['gray'].shape[1]),
            mit2=float(scale_for(item['gray'].shape[1], args['width'], 1.)),
            mit4=float(scale_for(item['gray'].shape[1], args['width'], 2.))))
    result_scores = scores(rows)
    for label, log_key in [('plain', 'val'), ('pp', 'val_pp')]:
        assert round(result_scores[label]['mIoU'], 2) == logs[-1][log_key]['mIoU']
        for name in CLASSES:
            assert round(result_scores[label]['class_IoU'][name], 2) == logs[-1][log_key][name]
    quantization = {}
    for key in ('tta', 'tta_pp'):
        differences = {name: float((np.nanmean(np.asarray(rows['stored_' + key][name]['iou'], dtype=float)) -
                                    np.nanmean(np.asarray(rows[key][name]['iou'], dtype=float))) * 100) for name in names}
        quantization[key] = dict(mean_delta_pp=float(np.mean(list(differences.values()))),
            max_absolute_image_delta_pp=float(np.max(np.abs(list(differences.values())))), per_image_delta_pp=differences)
    files = [run / name for name in ('args.json', 'result.json', 'log.jsonl', 'last.pt', 'initialization_audit.json')]
    files += [probs / f'{name}.npy' for name in names]
    result = dict(completed=now(), job='mit2', plan_sha256=sha(OUT / 'plan.json'), epochs=60,
        train_images=70, validation_images=20, args=args, scores=result_scores, checkpoint=relative(checkpoint),
        strict_cpu_reload=True, finite_checkpoint_tensors=True, same_initial_state_as_mit4=True,
        initial_state_sha256=audit['full_initial_state_sha256'], training_minutes=sum(item['train_sec'] for item in logs) / 60,
        probabilities={name: relative(probs / f'{name}.npy') for name in names},
        training_names=sorted(item['name'] for item in training), validation_names=names,
        actual_validation_scales=scales, output_receipts=[receipt(path) for path in files],
        float16_storage_effect=quantization, no_tta_scores_source='Final trainer rows cross-checked against epoch60 log',
        tta_scores_source='Final trainer float32 rows and independently recomputed stored float16 probabilities',
        checkpoint_selection='fixed epoch60 last.pt', historical_project_checkpoint_used=False,
        test_images_or_labels_used=False, previously_viewed_validation=True, independent_validation=False,
        automatic_model_selection=False, automatic_submission_change=False)
    verify(plan)
    save(destination / 'per_image_rows.json', rows)
    save(destination / 'results.json', result)
    save(destination / 'all.done', dict(time=now(), plan_sha256=sha(OUT / 'plan.json'),
        results_sha256=sha(destination / 'results.json'), rows_sha256=sha(destination / 'per_image_rows.json')))
    print(json.dumps(dict(job='mit2', scores=result_scores), indent=2), flush=True)


def completed(folder, plan_sha):
    done = read(folder / 'all.done')
    assert done['plan_sha256'] == plan_sha
    assert done['results_sha256'] == sha(folder / 'results.json')
    assert done['rows_sha256'] == sha(folder / 'per_image_rows.json')
    return read(folder / 'results.json'), read(folder / 'per_image_rows.json')


def compare():
    import numpy as np
    from infer import image_iou, postprocess
    from data import CLASSES
    plan = read(OUT / 'plan.json')
    verify(plan)
    _, valid = dataset(plan)
    assert not (OUT / 'results.json').exists() and not (OUT / 'all.done').exists()
    own, own_rows = completed(OUT / 'review', sha(OUT / 'plan.json'))
    reference_folder = REFERENCE / 'reviews' / 'mit4'
    reference, reference_rows = completed(reference_folder, sha(REFERENCE / 'plan.json'))
    assert reference['job'] == 'mit4' and reference['epochs_total'] == 60
    assert reference['training_names'] == own['training_names'] and reference['validation_names'] == own['validation_names']
    original = reference['stages']['original']
    assert original['args'] == plan['reference_mit4_args'] and original['epochs'] == 60
    assert original['strict_cpu_reload'] and original['finite_tensors']
    for artifacts, folder in [(own['output_receipts'], OUT), (original['output_receipts'], REFERENCE)]:
        for item in artifacts:
            assert sha(under(item['path'], folder)) == item['sha256'], item['path']
    ref_audit_path = under(plan['reference_mit4_args']['out'], REFERENCE) / 'initialization_audit.json'
    ref_audit = read(ref_audit_path)
    assert ref_audit['plan_sha256'] == sha(REFERENCE / 'plan.json')
    assert ref_audit['inherited_checkpoint'] is None and not ref_audit['previous_experiment_checkpoint_used']
    assert ref_audit['full_initial_state_sha256'] == own['initial_state_sha256'] == plan['reference_initial_state_sha256']
    rows = {'mit2_original': {}, 'mit4_original': {}}
    assert set(own['probabilities']) == set(original['probabilities']) == {item['name'] for item in valid}
    for item in valid:
        name = item['name']
        for label, paths, folder in [('mit2_original', own['probabilities'], OUT),
                                     ('mit4_original', original['probabilities'], REFERENCE)]:
            prob = probability(under(paths[name], folder), item)
            rows[label][name] = row(item, image_iou(item['lab'], postprocess(prob)))
        np.testing.assert_allclose(np.asarray(rows['mit2_original'][name]['iou'], dtype=float),
            np.asarray(own_rows['stored_tta_pp'][name]['iou'], dtype=float), rtol=0, atol=0, equal_nan=True)
        np.testing.assert_allclose(np.asarray(rows['mit4_original'][name]['iou'], dtype=float),
            np.asarray(reference_rows['original'][name]['tta_pp'], dtype=float), rtol=0, atol=0, equal_nan=True)
    result_scores = scores(rows)
    deltas = {item['name']: float((np.nanmean(np.asarray(rows['mit4_original'][item['name']]['iou'], dtype=float)) -
                                  np.nanmean(np.asarray(rows['mit2_original'][item['name']]['iou'], dtype=float))) * 100)
              for item in valid}
    values = np.asarray(list(deltas.values()), dtype=float)
    contrast = dict(direction='MiT4 minus MiT2; positive favors MiT4', images=20,
        mean_delta_pp=float(values.mean()), median_delta_pp=float(np.median(values)),
        minimum_image_delta_pp=float(values.min()), maximum_image_delta_pp=float(values.max()),
        improved=int((values > 0).sum()), worsened=int((values < 0).sum()), unchanged=int((values == 0).sum()),
        class_delta_pp={name: result_scores['mit4_original']['class_IoU'][name] -
                             result_scores['mit2_original']['class_IoU'][name] for name in CLASSES},
        per_image_delta_pp=deltas)
    result = dict(completed=now(), plan_sha256=sha(OUT / 'plan.json'), train_images=70, validation_images=20,
        scores=result_scores, comparison=contrast, recipe='TTA8 + postprocess, original scale policies, no cap6',
        metric='Mean of per-image class IoU; exclude classes absent in both ground truth and prediction',
        probability_storage='Both stored float16 probabilities, converted to float32 for scoring',
        only_training_argument_changes=['scale_multiplier', 'out', 'save_oof'],
        same_initial_state_sha256=own['initial_state_sha256'], same_seed=0, same_epochs=60,
        own_review_completion=receipt(OUT / 'review' / 'all.done'),
        reference_review_completion=receipt(reference_folder / 'all.done'),
        reference_initialization_receipt=receipt(ref_audit_path),
        previously_viewed_validation=True, independent_validation=False, historical_oof_probabilities_used=False,
        historical_project_checkpoints_used=False, test_images_or_labels_used=False,
        automatic_model_selection=False, automatic_submission_change=False,
        notes=['Only fresh train70 runs are compared on the same provided validation20.',
               'Changing scale with a fixed1536 crop also changes effective field of view, not just resolution.',
               'Nominal MiT2/MiT4 policies have image-width-dependent actual scale; see the per-image scale audit.',
               'Single-seed comparison on previously inspected validation; not a guarantee of generalization.',
               'No ensembles or per-image policy selection are evaluated.'])
    verify(plan)
    save(OUT / 'per_image_rows.json', rows)
    save(OUT / 'results.json', result)
    save(OUT / 'all.done', dict(time=now(), outcome='matched_fresh_train70_mit2_vs_mit4_complete',
        plan_sha256=sha(OUT / 'plan.json'), results_sha256=sha(OUT / 'results.json'),
        rows_sha256=sha(OUT / 'per_image_rows.json'), own_review_completion=result['own_review_completion'],
        reference_review_completion=result['reference_review_completion'], automatic_submission_change=False))
    print(json.dumps(dict(scores=result_scores, comparison=contrast), indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compare', action='store_true')
    args = parser.parse_args()
    compare() if args.compare else review()
