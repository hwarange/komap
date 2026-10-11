"""CPU audit and same-fold OOF scoring within the provided training70 only."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/review34_train70_group5fold'
FAMILIES = ('mit2', 'mit4', 'unet3', 'context4')
PAIRS = {'mit2_plus_unet3': ('mit2', 'unet3'), 'mit4_plus_unet3': ('mit4', 'unet3'),
         'context4_plus_unet3': ('context4', 'unet3')}
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
                  MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
os.chdir(ROOT)


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


def under(value):
    path = (ROOT / value).resolve()
    assert path.is_relative_to(OUT.resolve()), value
    return path


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
    assert len(plan['jobs']) == 20
    assert {(job['family'], job['fold']) for job in plan['jobs']} == {
        (family, fold) for family in FAMILIES for fold in range(5)}
    assert {job['id'] for job in plan['jobs']} == {
        f'{family}_fold{fold}' for family in FAMILIES for fold in range(5)}
    assert len(plan['train70_names']) == len(set(plan['train70_names'])) == 70
    assert len(plan['official_validation_names']) == len(set(plan['official_validation_names'])) == 20
    assert set(plan['train70_names']).isdisjoint(plan['official_validation_names'])
    for name, digest in plan['input_hashes'].items():
        assert sha(ROOT / name) == digest, name
    for item in plan['image_net_cache']:
        assert sha(item['path']) == item['sha256'], item['path']
    folds_path = ROOT / plan['folds_file']
    assert sha(folds_path) == plan['fold_assignment_sha256']
    assert plan['input_hashes'][relative(folds_path)] == plan['fold_assignment_sha256']
    assignment = read(folds_path)
    assert set(assignment) == set(plan['train70_names']) and set(assignment.values()) == set(range(5))
    for family in FAMILIES:
        reference = plan['initialization_reference'][family]
        assert sha(ROOT / reference['path']) == reference['sha256']
        audit = read(ROOT / reference['path'])
        assert audit['full_initial_state_sha256'] == reference['state_sha256']
        assert audit['fresh_imagenet_state_sha256'] == reference['state_sha256']
        assert audit['inherited_checkpoint'] is None and not audit['previous_experiment_checkpoint_used']
    return assignment


def load_training70(plan, assignment):
    from data import load_split
    # Never call load_split for the official validation20 or test10.
    items = load_split('data', 'train')
    assert sorted(item['name'] for item in items) == sorted(plan['train70_names'])
    assert all(item['split'] == 'train' and item['lab'] is not None for item in items)
    assert len(items) == 70
    groups = {}
    for item in items:
        groups.setdefault(item['alloy'], set()).add(assignment[item['name']])
    assert all(len(folds) == 1 for folds in groups.values()), 'Alloy crosses folds'
    return items


def split_job(job, plan, items, assignment):
    family, fold, args = job['family'], job['fold'], job['args']
    assert job['id'] == f'{family}_fold{fold}'
    reference = read(ROOT / plan['initialization_reference'][family]['path'])['args']
    assert set(args) == set(reference)
    assert {key for key in args if args[key] != reference[key]} == {'fold', 'folds_file', 'out', 'save_oof'}
    expected = dict(data='data', train_splits=['train'], val_split='valid', fold=fold,
                    folds_file=plan['folds_file'], full=False, train_frac=1., seed=0,
                    init='', init_skip_head=False, max_iters=0, hard_crop=0., crop=1536, tile=1536,
                    spi=8, epochs=120 if family == 'unet3' else 60)
    assert all(args.get(key) == value for key, value in expected.items()), args
    architecture = ('UnetPlusPlus', 'resnet34') if family == 'unet3' else (
        'GlobalContextMiTCNN30' if family == 'context4' else 'Unet', 'mit_b2')
    assert (args['arch'], args['encoder']) == architecture
    assert args['width'] == (3840 if family == 'unet3' else 2560)
    assert args['scale_multiplier'] == (2. if family in ('mit4', 'context4') else 1.)
    assert under(args['out']) == OUT / 'models' / family / f'fold{fold}'
    assert under(args['save_oof']) == OUT / 'probs' / family / f'fold{fold}'
    training = [item for item in items if assignment[item['name']] != fold]
    valid = [item for item in items if assignment[item['name']] == fold]
    assert training and valid and len(training) + len(valid) == 70
    assert sorted(item['name'] for item in training) == job['training_names']
    assert sorted(item['name'] for item in valid) == job['validation_names']
    assert sorted({item['alloy'] for item in training}) == job['training_alloys']
    assert sorted({item['alloy'] for item in valid}) == job['validation_alloys']
    assert set(job['training_alloys']).isdisjoint(job['validation_alloys'])
    assert set(job['validation_names']).isdisjoint(plan['official_validation_names'])
    microbatches = len(training) * args['spi'] // args['bs']
    assert job['microbatches_per_epoch'] == microbatches
    assert job['optimizer_updates_per_epoch'] == math.ceil(microbatches / args['accum_steps'])
    return training, valid


def probability(path, item):
    import numpy as np
    stored = np.load(path, allow_pickle=False)
    assert stored.dtype == np.float16 and stored.shape == (4, *item['gray'].shape), str(path)
    assert np.isfinite(stored).all() and stored.min() >= 0 and stored.max() <= 1, str(path)
    value = stored.astype(np.float32)
    assert np.max(np.abs(value.sum(axis=0) - 1)) < .005, str(path)
    return value


def image_row(item, prediction, fold):
    from infer import image_iou
    return dict(tta_pp=image_iou(item['lab'], prediction).tolist(), fold=fold,
                alloy=int(item['alloy']), low=bool(item['low_contrast']), width=int(item['gray'].shape[1]))


def summary(records):
    import numpy as np
    from data import CLASSES
    values = np.asarray([record['tta_pp'] for record in records.values()], dtype=float)
    assert values.shape == (len(records), 4) and len(records) > 0
    assert not np.isinf(values).any() and np.isfinite(values).any(axis=1).all()
    finite = values[np.isfinite(values)]
    assert (finite >= 0).all() and (finite <= 1).all()
    classes = {name: float(np.nanmean(values[:, index]) * 100) if np.isfinite(values[:, index]).any() else None
               for index, name in enumerate(CLASSES)}
    return dict(images=len(records), mIoU=float(np.nanmean(values, axis=1).mean() * 100),
                class_IoU=classes, both_absent_one_mIoU=float(np.where(np.isnan(values), 1., values).mean() * 100),
                both_absent_class_image_pairs=int(np.isnan(values).sum()))


def fold_statistics(records):
    import numpy as np
    grouped = {fold: {name: row for name, row in records.items() if row['fold'] == fold} for fold in range(5)}
    assert all(grouped.values())
    individual = {str(fold): summary(group) for fold, group in grouped.items()}
    values = np.asarray([individual[str(fold)]['mIoU'] for fold in range(5)], dtype=float)
    return dict(by_fold=individual, unweighted_fold_mIoU_mean=float(values.mean()),
                fold_mIoU_std_ddof0=float(values.std(ddof=0)), fold_mIoU_min=float(values.min()),
                fold_mIoU_max=float(values.max()),
                note='Fold summaries are secondary; primary OOF score gives every image equal weight.')


def paired(candidate, reference):
    import numpy as np
    from data import CLASSES
    names = sorted(candidate)
    assert set(names) == set(reference) and len(names) == 70
    assert all(candidate[name]['fold'] == reference[name]['fold'] and
               candidate[name]['alloy'] == reference[name]['alloy'] for name in names)
    a = np.asarray([candidate[name]['tta_pp'] for name in names], dtype=float)
    b = np.asarray([reference[name]['tta_pp'] for name in names], dtype=float)
    delta = (np.nanmean(a, axis=1) - np.nanmean(b, axis=1)) * 100
    alloys = np.asarray([candidate[name]['alloy'] for name in names])
    unique = np.unique(alloys)
    sums = np.asarray([delta[alloys == group].sum() for group in unique])
    counts = np.asarray([(alloys == group).sum() for group in unique])
    draws = np.random.default_rng(0).integers(0, len(unique), size=(10000, len(unique)))
    bootstrap = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return dict(direction='candidate minus reference', mean_delta_pp=float(delta.mean()),
        improved=int((delta > 0).sum()), worsened=int((delta < 0).sum()), unchanged=int((delta == 0).sum()),
        class_delta_pp={name: float((np.nanmean(a[:, index]) - np.nanmean(b[:, index])) * 100)
                        if np.isfinite(a[:, index]).any() and np.isfinite(b[:, index]).any() else None
                        for index, name in enumerate(CLASSES)},
        by_fold_delta_pp={str(fold): float(delta[np.asarray([candidate[name]['fold'] == fold for name in names])].mean())
                          for fold in range(5)}, per_image_delta_pp={name: float(delta[i]) for i, name in enumerate(names)},
        alloy_bootstrap95_pp=np.percentile(bootstrap, [2.5, 97.5]).tolist(), alloy_groups=len(unique),
        bootstrap_repeats=10000, bootstrap_seed=0,
        bootstrap_weighting='Resample alloys; divide sampled image-delta sums by sampled image counts.',
        uncertainty_scope='Does not include repeated model selection, training seed variation, or overlapping training-set dependence.')


def review(identifier):
    import numpy as np
    import torch
    from hybrid30 import install
    from infer import postprocess
    train = install()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    torch.set_num_threads(4)
    plan = read(OUT / 'plan.json')
    assignment = verify(plan)
    items = load_training70(plan, assignment)
    job = next(job for job in plan['jobs'] if job['id'] == identifier)
    _, valid = split_job(job, plan, items, assignment)
    args = job['args']
    folder = OUT / 'reviews' / identifier
    assert not folder.exists(), str(folder)
    run, probs = under(args['out']), under(args['save_oof'])
    assert read(run / 'args.json') == args
    trained = read(run / 'result.json')
    assert trained['args'] == args
    assert sorted(row['name'] for row in trained['rows']) == job['validation_names']
    assert sorted(path.stem for path in probs.glob('*.npy')) == job['validation_names']
    logs = [json.loads(line) for line in (run / 'log.jsonl').read_text(encoding='utf8').splitlines() if line]
    assert [row['epoch'] for row in logs] == list(range(1, args['epochs'] + 1))
    assert all(math.isfinite(row['loss']) and row['microbatches'] == job['microbatches_per_epoch']
               and row['optimizer_updates'] == job['optimizer_updates_per_epoch'] for row in logs)
    audit = read(run / 'initialization_audit.json')
    for key in ('family', 'fold', 'args', 'training_names', 'validation_names', 'training_alloys', 'validation_alloys'):
        assert audit[key] == job[key], key
    assert audit['job'] == identifier and audit['plan_sha256'] == sha(OUT / 'plan.json')
    assert audit['stage'] == 'base' and not audit['check_only']
    assert audit['fold_assignment_sha256'] == plan['fold_assignment_sha256']
    assert audit['train70_names'] == plan['train70_names']
    assert audit['official_validation_names'] == plan['official_validation_names']
    assert audit['group_disjoint'] and audit['train70_only']
    assert not audit['official_validation_images_used'] and not audit['test_images_used']
    assert not audit['reference_checkpoint_loaded']
    assert audit['strict_initialization'] and not audit['previous_experiment_checkpoint_used']
    assert audit['inherited_checkpoint'] is None
    initial = plan['initialization_reference'][job['family']]['state_sha256']
    assert audit['full_initial_state_sha256'] == audit['fresh_imagenet_state_sha256'] == initial
    checkpoint = run / 'last.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert state['args'] == args and state['epoch'] == args['epochs']
    assert all(torch.isfinite(value).all() for value in state['model'].values() if value.is_floating_point())
    model = train.build_from_args(args, weights=None)
    model.load_state_dict(state['model'], strict=True)
    del model, state
    original, raw_stored = {}, {}
    trainer_rows = {row['name']: row for row in trained['rows']}
    reported = {key: {} for key in ('plain', 'pp', 'tta', 'tta_pp')}
    for item in valid:
        name = item['name']
        prob = probability(probs / f'{name}.npy', item)
        original[name] = image_row(item, postprocess(prob), job['fold'])
        raw_stored[name] = image_row(item, prob.argmax(axis=0), job['fold'])
        assert trainer_rows[name]['W'] == item['gray'].shape[1]
        for metric in reported:
            reported[metric][name] = dict(tta_pp=trainer_rows[name][metric])
    reported_scores = {key: summary(records) for key, records in reported.items()}
    for metric, log_key in (('plain', 'val'), ('pp', 'val_pp')):
        assert round(reported_scores[metric]['mIoU'], 2) == logs[-1][log_key]['mIoU']
    quantization = {}
    for metric, records in (('tta', raw_stored), ('tta_pp', original)):
        differences = {name: float((np.nanmean(np.asarray(records[name]['tta_pp'], dtype=float)) -
                                   np.nanmean(np.asarray(reported[metric][name]['tta_pp'], dtype=float))) * 100)
                       for name in job['validation_names']}
        quantization[metric] = dict(mean_delta_pp=float(np.mean(list(differences.values()))),
            maximum_absolute_image_delta_pp=float(np.max(np.abs(list(differences.values())))), per_image_delta_pp=differences)
    files = [run / name for name in ('args.json', 'result.json', 'log.jsonl', 'last.pt', 'initialization_audit.json')]
    files += [probs / f'{name}.npy' for name in job['validation_names']]
    result = dict(completed=now(), job=identifier, family=job['family'], fold=job['fold'], args=args,
        plan_sha256=sha(OUT / 'plan.json'), train_images=len(job['training_names']), validation_images=len(valid),
        training_names=job['training_names'], validation_names=job['validation_names'],
        training_alloys=job['training_alloys'], validation_alloys=job['validation_alloys'],
        epochs=args['epochs'], scores={'original': summary(original)}, trainer_float32_scores=reported_scores,
        float16_storage_effect=quantization, strict_cpu_reload=True, finite_checkpoint_tensors=True,
        initialization_state_sha256=initial, checkpoint=relative(checkpoint), checkpoint_sha256=sha(checkpoint),
        probabilities={name: relative(probs / f'{name}.npy') for name in job['validation_names']},
        output_receipts=[receipt(path) for path in files],
        source_receipts=[dict(path=name, sha256=digest) for name, digest in plan['input_hashes'].items()
                         if name.startswith(('src/', 'scripts/'))],
        training_minutes=sum(row['train_sec'] for row in logs) / 60,
        official_validation_images_used=False, test_images_or_labels_used=False,
        historical_project_checkpoint_used=False, checkpoint_selection='Fixed last epoch; no best.pt',
        original_policy_only=True, cpu_only_review=True, automatic_model_selection=False)
    verify(plan)
    save(folder / 'per_image_rows.json', {'original': original})
    save(folder / 'results.json', result)
    save(folder / 'all.done', dict(time=now(), job=identifier, plan_sha256=sha(OUT / 'plan.json'),
        results_sha256=sha(folder / 'results.json'), rows_sha256=sha(folder / 'per_image_rows.json')))
    print(json.dumps(dict(job=identifier, scores=result['scores']), indent=2), flush=True)


def aggregate():
    import numpy as np
    from infer import postprocess
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    plan = read(OUT / 'plan.json')
    assignment = verify(plan)
    items = load_training70(plan, assignment)
    assert not (OUT / 'results.json').exists() and not (OUT / 'all.done').exists()
    manifests, stored, completions = {}, {}, {}
    for job in plan['jobs']:
        split_job(job, plan, items, assignment)
        identifier = job['id']
        folder = OUT / 'reviews' / identifier
        done = read(folder / 'all.done')
        assert done['job'] == identifier and done['plan_sha256'] == sha(OUT / 'plan.json')
        assert done['results_sha256'] == sha(folder / 'results.json')
        assert done['rows_sha256'] == sha(folder / 'per_image_rows.json')
        result = read(folder / 'results.json')
        assert result['job'] == identifier and result['args'] == job['args']
        assert result['validation_names'] == job['validation_names'] and result['training_names'] == job['training_names']
        assert result['strict_cpu_reload'] and result['finite_checkpoint_tensors']
        assert result['initialization_state_sha256'] == plan['initialization_reference'][job['family']]['state_sha256']
        for entry in result['output_receipts']:
            assert sha(under(entry['path'])) == entry['sha256'], entry['path']
        assert set(result['probabilities']) == set(job['validation_names'])
        for path in result['probabilities'].values():
            assert (ROOT / path).parent == under(job['args']['save_oof'])
            assert any(entry['path'] == path for entry in result['output_receipts'])
        manifests[(job['family'], job['fold'])] = result
        stored[(job['family'], job['fold'])] = read(folder / 'per_image_rows.json')['original']
        assert set(stored[(job['family'], job['fold'])]) == set(job['validation_names'])
        completions[identifier] = receipt(folder / 'all.done')
    rows = {recipe: {} for recipe in (*FAMILIES, *PAIRS)}
    provenance = {}
    for item in items:
        name, fold = item['name'], assignment[item['name']]
        probabilities = {}
        for family in FAMILIES:
            manifest = manifests[(family, fold)]
            assert name in manifest['validation_names'] and name not in manifest['training_names']
            path = under(manifest['probabilities'][name])
            probabilities[family] = probability(path, item)
            rows[family][name] = image_row(item, postprocess(probabilities[family]), fold)
            np.testing.assert_allclose(np.asarray(rows[family][name]['tta_pp'], dtype=float),
                np.asarray(stored[(family, fold)][name]['tta_pp'], dtype=float), rtol=0, atol=0, equal_nan=True)
        for recipe, (first, second) in PAIRS.items():
            assert name not in rows[recipe]
            rows[recipe][name] = image_row(item, postprocess((probabilities[first] + probabilities[second]) / 2), fold)
        provenance[name] = dict(fold=fold, jobs={family: f'{family}_fold{fold}' for family in FAMILIES})
    assert all(set(records) == set(plan['train70_names']) and len(records) == 70 for records in rows.values())
    calculated = {recipe: summary(records) for recipe, records in rows.items()}
    by_fold = {recipe: fold_statistics(records) for recipe, records in rows.items()}
    for family in FAMILIES:
        for fold in range(5):
            assert abs(by_fold[family]['by_fold'][str(fold)]['mIoU'] -
                       manifests[(family, fold)]['scores']['original']['mIoU']) < 1e-12
    contrasts = [('mit4', 'mit2'), ('context4', 'mit4'),
                 ('mit2_plus_unet3', 'mit4_plus_unet3'), ('context4_plus_unet3', 'mit4_plus_unet3'),
                 ('mit2_plus_unet3', 'context4_plus_unet3')]
    contrasts += [(recipe, 'unet3') for recipe in PAIRS]
    comparisons = {a + '_vs_' + b: paired(rows[a], rows[b]) for a, b in contrasts}
    result = dict(completed=now(), plan_sha256=sha(OUT / 'plan.json'), images=70, folds=5,
        scores=calculated, fold_statistics=by_fold, comparisons=comparisons,
        recipes={**{family: dict(models=[family], weights=[1.]) for family in FAMILIES},
                 **{recipe: dict(models=list(models), weights=[.5, .5]) for recipe, models in PAIRS.items()}},
        job_completion_receipts=completions, image_prediction_provenance=provenance,
        metric='Per-image class IoU mean, then equal-weight mean across70 images; both-absent classes excluded.',
        sensitivity='both_absent_one_mIoU uses IoU1 for both-absent classes; not an alternative selected metric.',
        probability_storage='float16 converted to float32 before fixed soft voting and postprocessing',
        inference_policy='Original scale policies, TTA8; no cap6. Only each image own held-out-fold models.',
        official_validation_images_used=False, test_images_or_labels_used=False,
        historical_project_checkpoints_used=False, additional_seed_evaluation=False,
        previously_inspected_training_data=True, independent_test=False,
        automatic_model_selection=False, automatic_submission_change=False,
        notes=['The provided training70 alone is split by alloy; original validation20 and test10 are not loaded.',
               'Each image receives one prediction per family from the model that did not train on its fold.',
               'Pairs combine exactly two models of that same fold; predictions from other folds are never averaged.',
               'Primary result is the70-image mean; unweighted fold mean can differ when fold sizes differ.',
               'MiT/Context use60 epochs and UNet++120; their comparison is between full recipes, not equal budgets.',
               'Architecture selection predates this CV; repeated selection and shared training data limit inference.'])
    verify(plan)
    save(OUT / 'per_image_rows.json', rows)
    save(OUT / 'results.json', result)
    save(OUT / 'all.done', dict(time=now(), outcome='four_families_train70_group5fold_complete',
        plan_sha256=sha(OUT / 'plan.json'), results_sha256=sha(OUT / 'results.json'),
        rows_sha256=sha(OUT / 'per_image_rows.json'), job_completion_receipts=completions,
        automatic_model_selection=False, automatic_submission_change=False))
    print(json.dumps(dict(scores=calculated, selected_recipe=None), indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--job', choices=[f'{family}_fold{fold}' for family in FAMILIES for fold in range(5)])
    mode.add_argument('--aggregate', action='store_true')
    arguments = parser.parse_args()
    aggregate() if arguments.aggregate else review(arguments.job)
