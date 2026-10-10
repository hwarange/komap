"""CPU-only, fixed 2-by-2 review of the completed five-fold continuation.

Ranks the four declared ensembles by image-mean full90 OOF mIoU. Confidence
intervals describe uncertainty; they are not an adoption gate. This evaluator
never trains, creates test masks, edits the submission, or chooses per-image
models. The already explored fold0 is reported apart from the remaining72.
"""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for _variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_variable] = '2'

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/review27_finetune_full'
PRIOR = ROOT / 'runs/review26_final_strengthening/finetune_plan'
SCALE = ROOT / 'runs/review26_final_strengthening/scale_trials'
MIT = 'x4_mit_c1536_bw_r90'
UNET = 'x3_unetpp_bw_e120_r90'
CONFIGS = ('baseline_original', 'baseline_cap6', 'interface_original', 'interface_cap6')
EXPECTED = {'baseline_original': 80.900653378248, 'baseline_cap6': 80.95664213120858}
EXPECTED_FOLD0 = 80.2879402942288
sys.path.insert(0, str(ROOT / 'src'))


def now():
    return datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def relative(path):
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def under_root(name):
    path = (ROOT / name).resolve()
    assert path.is_relative_to(ROOT.resolve()), f'Path outside project: {name}'
    return path


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    return value


def write_new(name, value):
    path = OUT / name
    assert path.resolve().is_relative_to(OUT.resolve())
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def verify_hashes(records):
    for name, digest in records.items():
        assert sha(under_root(name)) == digest, f'Pinned file changed: {name}'


def verify_receipt(record):
    path = under_root(record['path'])
    assert sha(path) == record['sha256'], f'Receipt changed: {record["path"]}'
    size = record.get('bytes', record.get('size_bytes'))
    assert size is None or path.stat().st_size == size


def data_items():
    import numpy as np
    from data import load_split
    items = load_split('data', 'train') + load_split('data', 'valid')
    folds = read(ROOT / 'splits/folds5.json')
    assert len(items) == len(folds) == 90
    assert len({item['name'] for item in items}) == 90
    assert {item['name'] for item in items} == set(folds)
    assert all(sum(f == fold for f in folds.values()) == 18 for fold in range(5))
    for item in items:
        assert item['lab'].shape == item['gray'].shape
        assert set(np.unique(item['lab'])) <= {0, 1, 2, 3}
    return items, folds


def probability(path, shape):
    import numpy as np
    stored = np.load(path, allow_pickle=False)
    assert stored.dtype == np.float16, f'OOF precision changed: {path}: {stored.dtype}'
    assert stored.shape == (4, *shape) and np.isfinite(stored).all(), path
    value = stored.astype(np.float32)
    assert value.min() >= 0 and value.max() <= 1
    assert np.max(np.abs(value.sum(0) - 1)) < 0.005, path
    return value


def cap6_records(items, folds):
    done, results, plan = [read(SCALE / filename) for filename in ('all.done', 'results.json', 'plan.json')]
    assert done['results_sha256'] == sha(SCALE / 'results.json')
    assert done['plan_sha256'] == sha(SCALE / 'plan.json') == results['plan_sha256']
    assert done['rows_sha256'] == sha(SCALE / 'per_image_rows.json')
    assert done['nonce'] == plan['nonce'] == results['nonce']
    assert done['images'] == 90 and done['affected_images'] == 26
    affected = {item['name'] for item in items if item['gray'].shape[1] == 675}
    assert len(affected) == 26 and set(plan['affected_names']) == affected
    selected = [record for record in results['output_receipts'] if record['arm'] == 'cap6']
    assert len(selected) == 26 and {record['name'] for record in selected} == affected
    for record in selected:
        assert record['fold'] == folds[record['name']] and record['actual_scale'] == 6
        verify_receipt(record)
    return {record['name']: record for record in selected}


def prior_fold0():
    done, results, plan = [read(PRIOR / filename) for filename in ('all.done', 'results.json', 'plan.json')]
    assert done['results_sha256'] == sha(PRIOR / 'results.json')
    assert done['plan_sha256'] == sha(PRIOR / 'plan.json') == results['plan_sha256']
    assert results['images'] == 18 and results['fold'] == 0
    assert abs(results['scores']['interface_pair']['mIoU'] - EXPECTED_FOLD0) < 1e-10
    job = next(job for job in plan['jobs'] if job['arm'] == 'interface')
    receipt = next(receipt for receipt in read(PRIOR / 'completed_jobs.json') if receipt['id'] == 'interface')
    assert receipt in results['completion_receipts']
    verify_hashes(receipt['hashes'])
    return job, receipt


def validate_job(job, receipt, folds, reference_args):
    import torch
    args, fold = job['args'], job['fold']
    valid = sorted(name for name in folds if folds[name] == fold)
    train = sorted(set(folds) - set(valid))
    assert job['validation_names'] == valid
    if fold:
        assert job['train_names'] == train
    else:
        assert job.get('train_names', train) == train
    assert sorted(path.stem for path in under_root(args['hard_maps']).glob('*.png')) == train
    assert {name.split('_')[0] for name in train}.isdisjoint(name.split('_')[0] for name in valid)
    expected = dict(reference_args, fold=fold,
                    init=f'runs/cv/{UNET}/fold{fold}/last.pt')
    for key in ('out', 'save_oof', 'hard_maps'):
        expected[key] = args[key]
    assert args == expected, f'Undeclared recipe difference, fold {fold}'
    if fold:
        assert args['out'] == f'runs/review27_finetune_full/interface/fold{fold}'
        assert args['save_oof'] == f'runs/review27_finetune_full/oof/interface/fold{fold}'
        assert args['hard_maps'] == f'runs/review27_finetune_full/train_interface_maps/fold{fold}'
    target = under_root(args['out'])
    assert read(target / 'args.json') == args
    logs = [json.loads(line) for line in (target / 'log.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    assert [row['epoch'] for row in logs] == list(range(1, 7))
    assert all(math.isfinite(row['loss']) and row['microbatches'] == 144
               and row['optimizer_updates'] == 144 and row['effective_batch'] == 4
               and row['accum_steps'] == 1 for row in logs)
    result = read(target / 'result.json')
    assert result['args'] == args and sorted(row['name'] for row in result['rows']) == valid
    checkpoint = torch.load(target / 'last.pt', map_location='cpu', weights_only=False)
    assert checkpoint['epoch'] == 6 and checkpoint['args'] == args
    assert all(torch.isfinite(value).all().item() for value in checkpoint['model'].values() if value.is_floating_point())
    del checkpoint
    required = {relative(target / filename) for filename in ('args.json', 'log.jsonl', 'last.pt', 'result.json')}
    required |= {relative(under_root(args['save_oof']) / f'{name}.npy') for name in valid}
    assert required <= set(receipt['hashes']) and receipt['id'] == job['id']
    verify_hashes(receipt['hashes'])
    return dict(fold=fold, training_images=72, validation_images=18, epochs=6,
                initialization=args['init'], checkpoint=relative(target / 'last.pt'),
                checkpoint_sha256=receipt['hashes'][relative(target / 'last.pt')],
                oof_directory=args['save_oof'], train_seconds=sum(row['train_sec'] for row in logs))


def row_for(item, prob, fold):
    from infer import image_iou, postprocess
    mask = postprocess(prob)
    assert mask.shape == item['lab'].shape
    return dict(tta_pp=image_iou(item['lab'], mask).tolist(), low=bool(item['low_contrast']),
                fold=fold, width=int(item['gray'].shape[1]),
                prediction_sha256=hashlib.sha256(mask.tobytes()).hexdigest())


def score(rows):
    import numpy as np
    return float(np.nanmean(np.asarray([rows[name]['tta_pp'] for name in sorted(rows)], float) * 100, axis=1).mean())


def summary(rows, folds):
    import numpy as np
    from data import CLASSES
    values = np.asarray([rows[name]['tta_pp'] for name in sorted(rows)], float) * 100
    return dict(images=len(rows), mIoU=score(rows),
                class_IoU=dict(zip(CLASSES, np.nanmean(values, axis=0).tolist())),
                class_valid_images=dict(zip(CLASSES, np.isfinite(values).sum(0).tolist())),
                per_fold={str(fold): score({name: row for name, row in rows.items() if folds[name] == fold})
                          for fold in sorted({folds[name] for name in rows})})


def compare(rows, baseline, folds):
    import numpy as np
    from review_grains15 import metrics
    result = metrics(rows, baseline, folds, repeats=10000)
    deltas = {name: float(100 * (np.nanmean(row['tta_pp']) - np.nanmean(baseline[name]['tta_pp'])))
              for name, row in rows.items()}
    result.update(worsened_images=sum(value < 0 for value in deltas.values()),
                  tied_images=sum(value == 0 for value in deltas.values()),
                  worst10=[dict(name=name, delta_pp=value) for name, value in sorted(deltas.items(), key=lambda x: (x[1], x[0]))[:10]],
                  best10=[dict(name=name, delta_pp=value) for name, value in sorted(deltas.items(), key=lambda x: (-x[1], x[0]))[:10]],
                  per_image_delta_pp=deltas)
    return result


def calculate_rows(items, folds, cap6, jobs, include_all_new):
    import numpy as np
    labels = CONFIGS + ('baseline_UNet3', 'interface_UNet3')
    rows = {label: {} for label in labels}
    by_fold = {job['fold']: job for job in jobs}
    for item in items:
        name, fold = item['name'], folds[item['name']]
        shape = item['gray'].shape
        old_unet = probability(ROOT / f'runs/oof/{UNET}/{name}.npy', shape)
        old_mit = probability(ROOT / f'runs/oof/{MIT}/{name}.npy', shape)
        capped_mit = probability(under_root(cap6[name]['path']), shape) if name in cap6 else old_mit
        for label, value in [('baseline_original', (old_unet + old_mit) / 2),
                             ('baseline_cap6', (old_unet + capped_mit) / 2),
                             ('baseline_UNet3', old_unet)]:
            rows[label][name] = row_for(item, value, fold)
        if fold in by_fold:
            new_unet = probability(under_root(by_fold[fold]['args']['save_oof']) / f'{name}.npy', shape)
            for label, value in [('interface_original', (new_unet + old_mit) / 2),
                                 ('interface_cap6', (new_unet + capped_mit) / 2),
                                 ('interface_UNet3', new_unet)]:
                rows[label][name] = row_for(item, value, fold)
        for family in ('baseline', 'interface'):
            if name not in cap6 and name in rows[f'{family}_original']:
                assert rows[f'{family}_original'][name] == rows[f'{family}_cap6'][name]
    for label, expected in EXPECTED.items():
        assert abs(score(rows[label]) - expected) < 1e-9, (label, score(rows[label]), expected)
    fold0 = {name: row for name, row in rows['interface_original'].items() if folds[name] == 0}
    assert abs(score(fold0) - EXPECTED_FOLD0) < 1e-9
    prior_scale = read(SCALE / 'per_image_rows.json')
    prior_train = read(PRIOR / 'per_image_rows.json')
    for label, prior_label in [('baseline_original', 'saved_baseline'), ('baseline_cap6', 'cap6')]:
        for name, row in rows[label].items():
            np.testing.assert_allclose(row['tta_pp'], np.asarray(prior_scale[prior_label][name]['tta_pp'], float), rtol=0, atol=0, equal_nan=True)
    for name, row in fold0.items():
        np.testing.assert_allclose(row['tta_pp'], np.asarray(prior_train['interface_pair'][name]['tta_pp'], float), rtol=0, atol=0, equal_nan=True)
    if include_all_new:
        assert all(set(values) == set(folds) for values in rows.values())
    return rows


def validate_plan_pins(plan, items, cap6):
    required = {'splits/folds5.json', 'src/train.py', 'src/data.py', 'src/infer.py',
                'src/review_grains15.py', 'scripts/review_finetune27.py'}
    for item in items:
        name, split = item['name'], item['split']
        required |= {f'data/{split}/images/{name}_image.png', f'data/{split}/masks/{name}_mask.png'}
        required |= {f'runs/oof/{experiment}/{name}.npy' for experiment in (UNET, MIT)}
    required |= {record['path'] for record in cap6.values()}
    required |= {relative(SCALE / filename) for filename in ('plan.json', 'results.json', 'all.done')}
    assert required <= set(plan['input_hashes']), sorted(required - set(plan['input_hashes']))
    verify_hashes(plan['input_hashes'])
    verify_hashes(plan['source_hashes'])


def preflight():
    items, folds = data_items()
    cap6 = cap6_records(items, folds)
    job, receipt = prior_fold0()
    validate_job(job, receipt, folds, job['args'])
    rows = calculate_rows(items, folds, cap6, [job], include_all_new=False)
    result = dict(time=now(), cpu_only=True, images=90, affected_images=26,
                  reproduced={label: score(rows[label]) for label in EXPECTED},
                  reproduced_fold0_interface_pair=score(rows['interface_original']),
                  new_training_started=False, files_written=False)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


def evaluate():
    for filename in ('results.json', 'per_image_rows.json', 'all.done'):
        assert not (OUT / filename).exists(), f'Refusing to overwrite {filename}'
    plan = read(OUT / 'plan.json')
    assert plan['comparison_recipes'] == list(CONFIGS)
    start = read(OUT / 'started.json')
    assert start['plan_sha256'] == sha(OUT / 'plan.json')
    items, folds = data_items()
    cap6 = cap6_records(items, folds)
    validate_plan_pins(plan, items, cap6)
    old_job, old_receipt = prior_fold0()
    assert plan['reused_fold0']['job'] == old_job
    assert plan['reused_fold0']['completion_receipt'] == old_receipt
    jobs = plan['jobs']
    assert len(jobs) == 4 and {job['fold'] for job in jobs} == {1, 2, 3, 4}
    receipts = read(OUT / 'completed_jobs.json')
    assert len(receipts) == 4 and {row['id'] for row in receipts} == {job['id'] for job in jobs}
    receipt_by_id = {row['id']: row for row in receipts}
    validations = [validate_job(old_job, old_receipt, folds, old_job['args'])]
    validations += [validate_job(job, receipt_by_id[job['id']], folds, old_job['args']) for job in jobs]
    rows = calculate_rows(items, folds, cap6, [old_job] + jobs, include_all_new=True)
    scopes = {}
    named_sets = [('full90', set(folds)),
                  ('remaining72', {name for name in folds if folds[name] != 0}),
                  ('reused_fold0_18', {name for name in folds if folds[name] == 0})]
    pairs = [('baseline_cap6', 'baseline_original'), ('interface_original', 'baseline_original'),
             ('interface_cap6', 'baseline_original'), ('interface_cap6', 'baseline_cap6'),
             ('interface_cap6', 'interface_original'), ('interface_UNet3', 'baseline_UNet3')]
    for scope, names in named_sets:
        selected = {label: {name: row for name, row in values.items() if name in names} for label, values in rows.items()}
        scopes[scope] = dict(scores={label: summary(values, folds) for label, values in selected.items()},
                             comparisons={f'{a}_vs_{b}': compare(selected[a], selected[b], folds) for a, b in pairs})
    for scope, names in [('affected26', set(cap6)), ('unaffected64', set(folds) - set(cap6))]:
        scopes[scope] = dict(scores={label: summary({name: row for name, row in values.items() if name in names}, folds)
                                   for label, values in rows.items()})
    ranking = sorted([dict(config=label, mIoU=scopes['full90']['scores'][label]['mIoU'],
                           delta_vs_original_pp=scopes['full90']['scores'][label]['mIoU'] - EXPECTED['baseline_original'])
                      for label in CONFIGS], key=lambda row: (-row['mIoU'], CONFIGS.index(row['config'])))
    verify_hashes(plan['input_hashes'])
    for receipt in [old_receipt] + receipts:
        verify_hashes(receipt['hashes'])
    result = dict(time=now(), images=90, folds=5, scopes=scopes, ranking=ranking,
                  highest_observed_full90_config=ranking[0]['config'],
                  selection_rule='Highest full90 image-mean OOF mIoU among four fixed configurations; retain original on exact tie.',
                  confidence_intervals_are_adoption_gate=False,
                  configs={label: dict(unet='interface_6ep_all5folds' if label.startswith('interface') else 'original_120ep_all5folds',
                                      mit='width675_actual_scale6_else_original' if label.endswith('cap6') else 'original',
                                      family_weights=[0.5, 0.5], tta=8, unchanged_postprocess=True) for label in CONFIGS},
                  validations=validations, completion_receipts=[old_receipt] + receipts,
                  plan_sha256=sha(OUT / 'plan.json'), completed_jobs_sha256=sha(OUT / 'completed_jobs.json'),
                  cap6_receipts=list(cap6.values()), source_hashes=plan['source_hashes'],
                  caveats=['OOF probabilities are saved float16 and loaded float32 for every compared recipe.',
                           'Fold0 was explored before this follow-up and is reused without rerunning.',
                           'Remaining72 tests recipe extension, but it is not an independent new dataset.',
                           'All90 and cap6 were already inspected during development; ranking is a model-selection result.',
                           'Bootstrap resamples alloy groups within folds, 10000 times, seed0; intervals are not selection-adjusted.',
                           'Only the targeted continuation is extended to folds1-4; no full90 matched continuation control exists.',
                           'Therefore full90 gains cannot isolate interface sampling from six more epochs and a reset optimizer.',
                           'No test labels are available; the highest OOF score does not guarantee the highest hidden-test score.'],
                  submission_changed=False, test_inference=False, evaluator_started_training=False,
                  additional_training_completed=True,
                  automatic_adoption=False)
    write_new('per_image_rows.json', rows)
    write_new('results.json', result)
    write_new('all.done', dict(time=now(), outcome='five_fold_continuation_fixed_2x2_review_complete', images=90,
                               results_sha256=sha(OUT / 'results.json'), rows_sha256=sha(OUT / 'per_image_rows.json'),
                               plan_sha256=sha(OUT / 'plan.json'), completed_jobs_sha256=sha(OUT / 'completed_jobs.json'),
                               highest_observed_full90_config=ranking[0]['config'], submission_changed=False,
                               test_inference=False, automatic_adoption=False))
    print(json.dumps(dict(completed=True, ranking=ranking, results=relative(OUT / 'results.json')), ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--preflight', action='store_true', help='Read-only CPU reproduction of the two baselines and reused fold0')
    action.add_argument('--evaluate', action='store_true', help='Validate all five completed folds and write fixed 2x2 results once')
    args = parser.parse_args()
    os.chdir(ROOT)
    import cv2
    import torch
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if args.preflight:
        preflight()
    else:
        evaluate()


if __name__ == '__main__':
    main()
