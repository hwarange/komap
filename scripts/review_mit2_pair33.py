"""CPU-only fixed MiT2/UNet++ probability pairs on the original validation20."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/review33_mit2_pair'
P31 = ROOT / 'runs/submission31_train70'
P32 = ROOT / 'runs/review32_mit2_train70'
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
                  MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
os.chdir(ROOT)
from review_submission31 import read, sha, save, relative, now, receipt, probability, image_row, scores


def under(path, directory):
    target = (ROOT / path).resolve()
    assert target.is_relative_to(directory.resolve()), path
    return target


def complete(folder, plan_sha, pins):
    done = read(folder / 'all.done')
    assert done['plan_sha256'] == plan_sha
    assert done['results_sha256'] == sha(folder / 'results.json')
    assert done['rows_sha256'] == sha(folder / 'per_image_rows.json')
    for name in ('all.done', 'results.json', 'per_image_rows.json'):
        pins[relative(folder / name)] = sha(folder / name)
    return read(folder / 'results.json'), read(folder / 'per_image_rows.json')


def add_receipts(records, directory, pins):
    for record in records:
        path = under(record['path'], directory)
        assert sha(path) == record['sha256'], record['path']
        if record['path'] in pins:
            assert pins[record['path']] == record['sha256']
        pins[record['path']] = record['sha256']


def init_audit(args, job, stage, plan_path, training_names, validation_names, pins, base_args=None):
    directory = P32 if job == 'mit2' else P31
    path = under(args['out'], directory) / 'initialization_audit.json'
    audit = read(path)
    assert audit['job'] == job and audit['stage'] == stage and audit['args'] == args
    assert audit['plan_sha256'] == sha(plan_path)
    assert audit['training_names'] == training_names and audit['validation_names'] == validation_names
    assert audit['strict_initialization'] and not audit['previous_experiment_checkpoint_used']
    expected = dict(data='data', train_splits=['train'], val_split='valid', fold=-1,
                    full=False, train_frac=1., seed=0, max_iters=0, init_skip_head=False)
    assert all(args.get(key) == value for key, value in expected.items())
    inherited = audit['inherited_checkpoint']
    if base_args is None:
        assert args['init'] == '' and inherited is None
        assert audit['fresh_imagenet_state_sha256'] == audit['full_initial_state_sha256']
    else:
        initial = under(base_args['out'], P31) / 'last.pt'
        assert under(args['init'], P31) == initial
        assert inherited['path'] == relative(initial) and inherited['sha256'] == sha(initial)
        assert inherited['epoch'] == 120 and inherited['source'] == 'own_fresh_train70_base'
    pins[relative(path)] = sha(path)
    return audit


def compare(rows, candidate, reference):
    import numpy as np
    from data import CLASSES
    names = sorted(rows[candidate])
    assert set(names) == set(rows[reference])
    a = np.asarray([rows[candidate][name]['tta_pp'] for name in names], dtype=float)
    b = np.asarray([rows[reference][name]['tta_pp'] for name in names], dtype=float)
    delta = (np.nanmean(a, axis=1) - np.nanmean(b, axis=1)) * 100
    alloys = np.asarray([rows[candidate][name]['alloy'] for name in names])
    groups = np.unique(alloys)
    assert len(groups) == 19
    sums = np.asarray([delta[alloys == group].sum() for group in groups])
    counts = np.asarray([(alloys == group).sum() for group in groups])
    draw = np.random.default_rng(0).integers(0, len(groups), size=(10000, len(groups)))
    bootstrap = sums[draw].sum(axis=1) / counts[draw].sum(axis=1)
    return dict(candidate=candidate, reference=reference, direction='candidate minus reference',
        mean_delta_pp=float(delta.mean()), improved=int((delta > 0).sum()),
        worsened=int((delta < 0).sum()), unchanged=int((delta == 0).sum()),
        class_delta_pp={name: float((np.nanmean(a[:, i]) - np.nanmean(b[:, i])) * 100)
                        for i, name in enumerate(CLASSES)},
        per_image_delta_pp={name: float(delta[i]) for i, name in enumerate(names)},
        alloy_bootstrap95_pp=np.percentile(bootstrap, [2.5, 97.5]).tolist(),
        bootstrap_repeats=10000, bootstrap_seed=0, alloy_groups=19,
        uncertainty_scope='Image-group resampling only; excludes repeated selection and training seed variation')


def main():
    import json
    import cv2
    import numpy as np
    import torch
    from data import load_split
    assert not OUT.exists(), 'Refusing to overwrite review33'
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    plans = [read(P31 / 'plan.json'), read(P32 / 'plan.json')]
    pins = {}
    for plan, folder in zip(plans, (P31, P32)):
        for name, digest in plan['input_hashes'].items():
            if name in pins:
                assert pins[name] == digest
            assert sha(ROOT / name) == digest, name
            pins[name] = digest
        for record in plan['image_net_cache']:
            assert sha(record['path']) == record['sha256']
        pins[relative(folder / 'plan.json')] = sha(folder / 'plan.json')
    pins[relative(Path(__file__).resolve())] = sha(Path(__file__).resolve())
    result31, rows31 = complete(P31, sha(P31 / 'plan.json'), pins)
    result32, _ = complete(P32, sha(P32 / 'plan.json'), pins)
    mit2, rows2 = complete(P32 / 'review', sha(P32 / 'plan.json'), pins)
    job_results = {}
    job_rows = {}
    for job in ('mit4', 'unet3', 'context4'):
        job_results[job], job_rows[job] = complete(P31 / 'reviews' / job, sha(P31 / 'plan.json'), pins)
        assert result31['job_completion_receipts'][job]['sha256'] == sha(P31 / 'reviews' / job / 'all.done')
    training, valid = load_split('data', 'train'), load_split('data', 'valid')
    training_names = sorted(item['name'] for item in training)
    names = sorted(item['name'] for item in valid)
    assert len(training_names) == len(set(training_names)) == 70
    assert len(names) == len(set(names)) == 20 and set(names).isdisjoint(training_names)
    assert all(item['lab'] is not None for item in training + valid)
    for plan in plans:
        assert plan['training_names'] == training_names and plan['validation_names'] == names
    assert mit2['training_names'] == training_names and mit2['validation_names'] == names
    assert mit2['strict_cpu_reload'] and mit2['finite_checkpoint_tensors'] and mit2['epochs'] == 60
    add_receipts(mit2['output_receipts'], P32, pins)
    sources = {'mit2': mit2['probabilities']}
    audit2 = init_audit(mit2['args'], 'mit2', 'base', P32 / 'plan.json', training_names, names, pins)
    for job in ('mit4', 'unet3', 'context4'):
        result = job_results[job]
        assert result['training_names'] == training_names and result['validation_names'] == names
        keys = ('base120_original', 'final126_original') if job == 'unet3' else (('original', 'cap6') if job == 'mit4' else ('original',))
        for key in keys:
            stage = result['stages'][key]
            add_receipts(stage['output_receipts'], P31, pins)
            label = {'base120_original': 'unet120', 'final126_original': 'unet126'}.get(key, job + '_' + key)
            sources[label] = stage['probabilities']
            if key == 'cap6':
                continue
            assert stage['strict_cpu_reload'] and stage['finite_tensors']
            base_args = result['stages']['base120_original']['args'] if key == 'final126_original' else None
            audit = init_audit(stage['args'], job, 'continuation' if base_args else 'base',
                P31 / 'plan.json', training_names, names, pins, base_args)
            if job == 'mit4':
                assert audit['full_initial_state_sha256'] == audit2['full_initial_state_sha256']
    for label, paths in sources.items():
        assert set(paths) == set(names)
        allowed = P32 if label == 'mit2' else P31
        for path in paths.values():
            under(path, allowed)
            assert path in pins, f'Unreceipted probability: {path}'
    pairs = {'mit2_plus_unet120': ('mit2', 'unet120'), 'mit2_plus_unet126': ('mit2', 'unet126')}
    references = {'context4_original_plus_unet3_base120_original': ('context4_original', 'unet120'),
                  'mit4_cap6_plus_unet3_base120_original': ('mit4_cap6', 'unet120')}
    plan = dict(created=now(), input_hashes=pins, training_names=training_names, validation_names=names,
        recipes={label: dict(models=list(models), weights=[.5, .5]) for label, models in pairs.items()},
        reference_recipes={label: list(models) for label, models in references.items()},
        probability_sources=sources, cpu_only=True, new_training=False, new_inference=False,
        automatic_adoption=False, automatic_test_prediction=False, image_net_cache=plans[0]['image_net_cache'])
    save(OUT / 'plan.json', plan)
    rows = {label: {} for label in (*sources, *pairs, *references)}
    for item in valid:
        name = item['name']
        probabilities = {label: probability(ROOT / paths[name], item) for label, paths in sources.items()}
        for label, value in probabilities.items():
            rows[label][name] = image_row(item, value)
        for label, (first, second) in {**pairs, **references}.items():
            rows[label][name] = image_row(item, (probabilities[first] + probabilities[second]) / 2)
        for label, expected in [('mit2', rows2['stored_tta_pp'][name]['iou']),
                                ('unet120', job_rows['unet3']['base120_original'][name]['tta_pp']),
                                ('unet126', job_rows['unet3']['final126_original'][name]['tta_pp'])]:
            np.testing.assert_allclose(np.asarray(rows[label][name]['tta_pp'], dtype=float),
                np.asarray(expected, dtype=float), rtol=0, atol=0, equal_nan=True)
        for label in references:
            np.testing.assert_allclose(np.asarray(rows[label][name]['tta_pp'], dtype=float),
                np.asarray(rows31[label][name]['tta_pp'], dtype=float), rtol=0, atol=0, equal_nan=True)
    calculated = scores(rows)
    assert abs(calculated['mit2']['mIoU'] - result32['scores']['mit2_original']['mIoU']) < 1e-12
    assert abs(calculated['mit2']['mIoU'] - 80.0486641719) < 1e-8
    assert abs(calculated['unet120']['mIoU'] - 79.5618261925) < 1e-8
    for label in references:
        assert abs(calculated[label]['mIoU'] - result31['scores'][label]['mIoU']) < 1e-12
    contrasts = {candidate + '_vs_' + reference: compare(rows, candidate, reference)
                 for candidate in pairs for reference in references}
    contrasts['continuation126_vs_base120_pair'] = compare(rows, 'mit2_plus_unet126', 'mit2_plus_unet120')
    for path, digest in pins.items():
        assert sha(ROOT / path) == digest, path
    result = dict(completed=now(), plan_sha256=sha(OUT / 'plan.json'), train_images=70, validation_images=20,
        scores=calculated, comparisons=contrasts, newly_evaluated_pairs=list(pairs),
        reference_pairs_exactly_reproduced=True, source_singles_exactly_reproduced=True,
        metric='Per-image class IoU mean then image mean; both-absent class excluded',
        sensitivity='both_absent_one_mIoU treats both-absent classes as1; official convention unconfirmed',
        probability_storage='float16 sources converted tofloat32; 50:50 arithmetic probability average then fixed postprocess',
        cpu_only=True, new_training=False, new_inference=False, historical_oof_used=False,
        test_images_or_labels_used=False, previously_viewed_validation=True, independent_validation=False,
        automatic_model_selection=False, automatic_submission_change=False,
        caveat='Comparisons are on repeatedly inspected20images; confidence intervals omit selection and seed uncertainty')
    save(OUT / 'per_image_rows.json', rows)
    save(OUT / 'results.json', result)
    save(OUT / 'all.done', dict(time=now(), outcome='two_fixed_mit2_unet_pairs_complete',
        plan_sha256=sha(OUT / 'plan.json'), results_sha256=sha(OUT / 'results.json'),
        rows_sha256=sha(OUT / 'per_image_rows.json'), cpu_only=True,
        automatic_model_selection=False, automatic_submission_change=False))
    print(json.dumps(dict(completed=result['completed'], scores=calculated, comparisons=contrasts), indent=2), flush=True)


if __name__ == '__main__':
    main()
