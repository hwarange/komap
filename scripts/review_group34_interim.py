"""CPU-only partial group34 review of completed folds0/1; no training changes."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'runs/review34_train70_group5fold'
OUT = BASE / 'interim_20261011_1215'
FOLDS = (0, 1)
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
                  MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
os.chdir(ROOT)
from review_group34 import (read, verify, load_training70, split_job, probability, image_row,
                            summary, sha, save, relative, receipt, now, FAMILIES, PAIRS)


def main():
    import json
    import cv2
    import numpy as np
    import torch
    from infer import postprocess
    assert not OUT.exists(), 'Refusing to overwrite interim output'
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    frozen = read(BASE / 'plan.json')
    assignment = verify(frozen)
    all_items = load_training70(frozen, assignment)
    items = [item for item in all_items if assignment[item['name']] in FOLDS]
    assert len(items) == 28 and len({item['name'] for item in items}) == 28
    selected_jobs = [job for job in frozen['jobs'] if job['fold'] in FOLDS]
    assert len(selected_jobs) == 8
    pins = dict(frozen['input_hashes'])
    pins[relative(BASE / 'plan.json')] = sha(BASE / 'plan.json')
    pins[relative(Path(__file__).resolve())] = sha(Path(__file__).resolve())
    manifests, original, completions = {}, {}, {}
    for job in selected_jobs:
        split_job(job, frozen, all_items, assignment)
        folder = BASE / 'reviews' / job['id']
        done = read(folder / 'all.done')
        assert done['job'] == job['id'] and done['plan_sha256'] == sha(BASE / 'plan.json')
        assert done['results_sha256'] == sha(folder / 'results.json')
        assert done['rows_sha256'] == sha(folder / 'per_image_rows.json')
        result = read(folder / 'results.json')
        assert result['job'] == job['id'] and result['args'] == job['args']
        assert result['plan_sha256'] == sha(BASE / 'plan.json')
        for key in ('training_names', 'validation_names', 'training_alloys', 'validation_alloys'):
            assert result[key] == job[key]
        assert result['strict_cpu_reload'] and result['finite_checkpoint_tensors']
        assert result['initialization_state_sha256'] == frozen['initialization_reference'][job['family']]['state_sha256']
        assert not result['official_validation_images_used'] and not result['test_images_or_labels_used']
        assert result['original_policy_only']
        for record in result['output_receipts']:
            path = (ROOT / record['path']).resolve()
            assert path.is_relative_to(BASE.resolve())
            assert sha(path) == record['sha256'], record['path']
            pins[record['path']] = record['sha256']
        assert set(result['probabilities']) == set(job['validation_names'])
        for name, path in result['probabilities'].items():
            assert path in pins
            assert (ROOT / path).parent == ROOT / job['args']['save_oof']
        for name in ('all.done', 'results.json', 'per_image_rows.json'):
            pins[relative(folder / name)] = sha(folder / name)
        manifests[(job['family'], job['fold'])] = result
        original[(job['family'], job['fold'])] = read(folder / 'per_image_rows.json')['original']
        assert set(original[(job['family'], job['fold'])]) == set(job['validation_names'])
        completions[job['id']] = receipt(folder / 'all.done')
    plan = dict(created=now(), purpose='Partial train70 group34 OOF review; completedfolds0/1 only',
        frozen_plan=receipt(BASE / 'plan.json'), folds=list(FOLDS), images=28,
        names=sorted(item['name'] for item in items), input_hashes=pins,
        recipes={**{family: dict(models=[family], weights=[1.]) for family in FAMILIES},
                 **{label: dict(models=list(models), weights=[.5, .5]) for label, models in PAIRS.items()}},
        job_completion_receipts=completions, partial=True, remaining_folds=[2, 3, 4],
        cpu_only=True, automatic_model_selection=False, automatic_submission_change=False)
    save(OUT / 'plan.json', plan)
    rows = {label: {} for label in (*FAMILIES, *PAIRS)}
    provenance = {}
    for item in items:
        name, fold = item['name'], assignment[item['name']]
        probabilities = {}
        for family in FAMILIES:
            manifest = manifests[(family, fold)]
            assert name in manifest['validation_names'] and name not in manifest['training_names']
            probabilities[family] = probability(ROOT / manifest['probabilities'][name], item)
            rows[family][name] = image_row(item, postprocess(probabilities[family]), fold)
            np.testing.assert_allclose(np.asarray(rows[family][name]['tta_pp'], dtype=float),
                np.asarray(original[(family, fold)][name]['tta_pp'], dtype=float), rtol=0, atol=0, equal_nan=True)
        for label, (first, second) in PAIRS.items():
            rows[label][name] = image_row(item, postprocess((probabilities[first] + probabilities[second]) / 2), fold)
        provenance[name] = dict(fold=fold, jobs={family: f'{family}_fold{fold}' for family in FAMILIES})
    assert all(set(records) == set(plan['names']) for records in rows.values())
    calculated = {label: summary(records) for label, records in rows.items()}
    by_fold = {str(fold): {label: summary({name: record for name, record in records.items() if record['fold'] == fold})
                          for label, records in rows.items()} for fold in FOLDS}
    for family in FAMILIES:
        for fold in FOLDS:
            assert abs(by_fold[str(fold)][family]['mIoU'] -
                       manifests[(family, fold)]['scores']['original']['mIoU']) < 1e-12
    comparisons = {}
    for candidate, baseline in [('mit2', 'mit4'), ('context4', 'mit4'),
        ('mit2_plus_unet3', 'mit4_plus_unet3'), ('context4_plus_unet3', 'mit4_plus_unet3'),
        ('mit2_plus_unet3', 'context4_plus_unet3')]:
        names = sorted(plan['names'])
        delta = {name: float((np.nanmean(np.asarray(rows[candidate][name]['tta_pp'], dtype=float)) -
                             np.nanmean(np.asarray(rows[baseline][name]['tta_pp'], dtype=float))) * 100) for name in names}
        values = np.asarray(list(delta.values()))
        comparisons[candidate + '_vs_' + baseline] = dict(direction='candidate minus baseline',
            mean_delta_pp=float(values.mean()), improved=int((values > 0).sum()),
            worsened=int((values < 0).sum()), unchanged=int((values == 0).sum()),
            class_delta_pp={name: (calculated[candidate]['class_IoU'][name] - calculated[baseline]['class_IoU'][name])
                            if calculated[candidate]['class_IoU'][name] is not None and calculated[baseline]['class_IoU'][name] is not None else None
                            for name in calculated[candidate]['class_IoU']}, per_image_delta_pp=delta)
    for name, digest in pins.items():
        assert sha(ROOT / name) == digest, name
    verify(frozen)
    result = dict(completed=now(), plan_sha256=sha(OUT / 'plan.json'), partial=True,
        completed_folds=list(FOLDS), remaining_folds=[2, 3, 4], images=28, total_planned_images=70,
        scores=calculated, fold_scores=by_fold, comparisons=comparisons,
        prediction_provenance=provenance, source_singles_exactly_reproduced=True,
        metric='Equal image mean of per-image class IoU; both-absent class excluded',
        inference='Stored TTA8, original scale policies; same-fold50:50 probabilities then fixed postprocessing',
        float16_probabilities_converted_to_float32=True, cpu_only=True, new_training=False,
        new_inference=False, official_validation_images_used=False, test_images_or_labels_used=False,
        independent_validation=False, automatic_model_selection=False, automatic_submission_change=False,
        caveats=['Only the28 held-out images of completedfolds0/1; not the final70-image OOF result.',
                 'Never averages predictions across differentfold models for the same image.',
                 'No comparison to the differently composed officialvalidation20 score is implied.',
                 'Three folds remain, and no new training-seed stability test is provided.'])
    save(OUT / 'per_image_rows.json', rows)
    save(OUT / 'results.json', result)
    save(OUT / 'all.done', dict(time=now(), partial=True, plan_sha256=sha(OUT / 'plan.json'),
        results_sha256=sha(OUT / 'results.json'), rows_sha256=sha(OUT / 'per_image_rows.json'),
        completed_folds=list(FOLDS), images=28, automatic_submission_change=False))
    print(json.dumps(dict(completed=result['completed'], scores=calculated, fold_scores=by_fold,
                         comparisons={key: {k: v for k, v in value.items() if k != 'per_image_delta_pp'}
                                      for key, value in comparisons.items()}), indent=2), flush=True)


if __name__ == '__main__':
    main()
