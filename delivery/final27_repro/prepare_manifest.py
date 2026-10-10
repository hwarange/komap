#!/usr/bin/env python3
"""Freeze the completed review27 90-image ranking winner; CPU only, never train/infer."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import run_submission as runner

REVIEW = ROOT / 'runs/review27_finetune_full'
CONFIGS = {'baseline_original', 'baseline_cap6', 'interface_original', 'interface_cap6'}
SCORE_KEYS = CONFIGS | {'baseline_UNet3', 'interface_UNet3'}


def record(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return dict(path=path.relative_to(ROOT).as_posix(), size_bytes=path.stat().st_size,
                sha256=runner.sha256(path))


def completed_review():
    """Bind the ranking to its fixed plan, evaluator, rows, and evaluated weights."""
    results_path = REVIEW / 'results.json'
    done_path = REVIEW / 'all.done'
    completed = runner.read(done_path)
    results = runner.read(results_path)
    plan = runner.read(REVIEW / 'plan.json')
    receipts = runner.read(REVIEW / 'completed_jobs.json')
    for field, filename in [('results_sha256', 'results.json'), ('rows_sha256', 'per_image_rows.json'),
                            ('plan_sha256', 'plan.json'), ('completed_jobs_sha256', 'completed_jobs.json')]:
        if completed[field] != runner.sha256(REVIEW / filename):
            raise ValueError(f'review27 completion checksum mismatch: {filename}')
    if (results['plan_sha256'] != completed['plan_sha256']
            or results['completed_jobs_sha256'] != completed['completed_jobs_sha256']
            or results['source_hashes'] != plan['source_hashes']
            or 'scripts/review_finetune27.py' not in plan['source_hashes']):
        raise ValueError('Results do not match the fixed plan and evaluator')
    for name, digest in plan['source_hashes'].items():
        if runner.sha256(runner.checked_path(ROOT, name)) != digest:
            raise ValueError(f'Evaluation source changed: {name}')
    old_receipt = plan['reused_fold0']['completion_receipt']
    if results['completion_receipts'] != [old_receipt] + receipts:
        raise ValueError('Results checkpoint receipts differ from the completed jobs')
    if len(receipts) != 4 or {v['id'] for v in receipts} != {f'interface_fold{k}' for k in range(1, 5)}:
        raise ValueError('Expected four new completed fold receipts')
    by_id = {v['id']: v for v in [old_receipt] + receipts}
    validations = results['validations']
    if len(validations) != 5 or {v['fold'] for v in validations} != set(range(5)):
        raise ValueError('Expected validated checkpoints for all five folds')
    checkpoint_hashes = {}
    for value in validations:
        fold = value['fold']
        path = ('runs/review26_final_strengthening/finetune_plan/interface/last.pt' if fold == 0
                else f'runs/review27_finetune_full/interface/fold{fold}/last.pt')
        receipt_id = 'interface' if fold == 0 else f'interface_fold{fold}'
        digest = by_id[receipt_id]['hashes'].get(path)
        if (value['checkpoint'] != path or value['checkpoint_sha256'] != digest
                or value['epochs'] != 6 or value['training_images'] != 72 or value['validation_images'] != 18
                or value['initialization'] != f'runs/cv/x3_unetpp_bw_e120_r90/fold{fold}/last.pt'):
            raise ValueError(f'Evaluated checkpoint provenance mismatch for fold{fold}')
        checkpoint_hashes[path] = digest
    scores = results['scopes']['full90']['scores']
    if set(scores) != SCORE_KEYS or any(scores[key]['images'] != 90 or not math.isfinite(scores[key]['mIoU']) for key in CONFIGS):
        raise ValueError('Expected four completed90-image recipe scores plus two standalone model diagnostics')
    ranking = results['ranking']
    if len(ranking) != 4 or {v['config'] for v in ranking} != CONFIGS:
        raise ValueError('Invalid completed ranking')
    for index, row in enumerate(ranking):
        if row['mIoU'] != scores[row['config']]['mIoU']:
            raise ValueError('Ranking/score mismatch')
        if index and ranking[index - 1]['mIoU'] < row['mIoU']:
            raise ValueError('Ranking is not descending')
    winner = results['highest_observed_full90_config']
    if winner != ranking[0]['config'] or completed['highest_observed_full90_config'] != winner:
        raise ValueError('Recorded winner must match highest observed90-image score')
    if winner == 'baseline_original':
        raise ValueError('Original recipe remains best: keep existing artifacts; regeneration is unnecessary')
    if abs(scores['baseline_original']['mIoU'] - 80.900653378248) > 1e-6:
        raise ValueError('Original90-image baseline failed reproduction')
    return results, plan, checkpoint_hashes


def verify_checkpoint_reference(reference, digest):
    if not digest or reference['sha256'] != digest:
        raise ValueError(f'Selected checkpoint differs from the evaluated weights: {reference["path"]}')


def prepared_manifest():
    import torch
    results, plan, evaluated_checkpoint_hashes = completed_review()
    results_path, done_path = REVIEW / 'results.json', REVIEW / 'all.done'
    scores = results['scopes']['full90']['scores']
    winner = results['highest_observed_full90_config']
    old = runner.read(ROOT / 'delivery/final_pair_repro/manifest.json')
    old_ck = {Path(v['path']).as_posix(): v for v in old['checkpoints']}
    new_unet = winner.startswith('interface_')
    cap6 = winner.endswith('_cap6')
    models = []
    for family, exp in [('mit4', 'x4_mit_c1536_bw_r90'), ('unet3', 'x3_unetpp_bw_e120_r90')]:
        for fold in range(5):
            original = f'runs/cv/{exp}/fold{fold}/last.pt'
            path = ROOT / original
            if family == 'unet3' and new_unet:
                path = (ROOT / 'runs/review26_final_strengthening/finetune_plan/interface/last.pt' if fold == 0
                        else REVIEW / f'interface/fold{fold}/last.pt')
            reference = record(path)
            expected_sha = (evaluated_checkpoint_hashes[reference['path']] if family == 'unet3' and new_unet
                            else old_ck[original]['sha256'])
            verify_checkpoint_reference(reference, expected_sha)
            ck = torch.load(path, map_location='cpu', weights_only=False)
            args = ck['args']
            expected_epoch = 60 if family == 'mit4' else (6 if new_unet else 120)
            if ck['epoch'] != expected_epoch or args['epochs'] != expected_epoch or args['fold'] != fold:
                raise ValueError(f'Wrong checkpoint epoch/fold: {path}')
            if family == 'mit4' or not new_unet:
                if reference['sha256'] != old_ck[original]['sha256']:
                    raise ValueError(f'Original checkpoint changed: {original}')
            else:
                if args['init'] != original or args['hard_crop'] != 0.35 or args['lr'] != 3e-5:
                    raise ValueError(f'Wrong continuation source/recipe: {path}')
                job = (plan['reused_fold0']['job'] if fold == 0
                       else next(value for value in plan['jobs'] if value['fold'] == fold))
                if args != job['args']:
                    raise ValueError(f'Selected continuation args differ from evaluated plan: {path}')
            item = dict(id=f'{family}_fold{fold}', family=family, fold=fold, checkpoint=reference,
                        checkpoint_epoch=ck['epoch'], args=args,
                        initial_training_epochs=120 if family == 'unet3' and new_unet else 0,
                        optimizer_state_resumed=False)
            if family == 'unet3' and new_unet:
                item['warmstart_reference'] = old_ck[original]
                item['training_schedule'] = '120 epochs, then separate6-epoch continuation with new AdamW/cosine schedule'
            models.append(item)
            del ck
    sources = [ROOT / name for name in ('src/train.py', 'src/data.py', 'src/infer.py',
                                        'delivery/final27_repro/run_submission.py',
                                        'delivery/final27_repro/prepare_manifest.py')]
    for item in old['source_files']:
        if item['path'] in ('src/train.py', 'src/data.py', 'src/infer.py'):
            if runner.sha256(ROOT / item['path']) != item['sha256']:
                raise ValueError(f'Pinned shared source changed: {item["path"]}')
    images = []
    for item in old['test_images']:
        current = record(ROOT / item['path'])
        if current != item:
            raise ValueError(f'Test input changed: {item["path"]}')
        images.append(current)
    for package in runner.CORE_PACKAGES:
        if importlib.metadata.version(package) != old['packages'][package]:
            raise ValueError(f'Package changed: {package}')
    result = dict(schema='komap-final27-v1', created_utc=datetime.now(timezone.utc).isoformat(),
                  selected_recipe=winner, selected_mIoU=scores[winner]['mIoU'],
                  selected_delta_vs_original_pp=scores[winner]['mIoU'] - scores['baseline_original']['mIoU'],
                  selection_rule='Highest observed full90 OOF mIoU among the four fixed recipes; confidence interval is reported, not an adoption gate',
                  selection_sources=[record(REVIEW / name) for name in
                                     ('results.json', 'all.done', 'plan.json', 'per_image_rows.json', 'completed_jobs.json')],
                  evaluation_source_hashes=plan['source_hashes'],
                  independent_validation=False, official_test_score_available=False,
                  mit_policy='cap6_width675' if cap6 else 'original',
                  unet_policy='interface6' if new_unet else 'original',
                  folds=5, test_count=10, family_weights={'mit4': 0.5, 'unet3': 0.5},
                  checkpoint_choice='last', tta='dihedral8', probability_dtype='float32',
                  backend=runner.BACKEND, source_files=[record(p) for p in sources],
                  models=models, test_images=images, packages=old['packages'],
                  python_version=platform.python_version(), platform=platform.platform(),
                  fresh_environment_installation_verified=False,
                  prediction_reuse=False, test_ground_truth_used=False,
                  notes=['MiT model architecture is U-Net with MiT-b2 encoder.',
                         'For the cap6 policy only MiT inputs whose original width is exactly675 use extra_scale0.75.',
                         'Five probabilities per family are averaged, then equal family soft voting and original postprocessing.',
                         'Existing final_pair artifacts are preserved. No official submission is performed.'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true', help='Write new frozen manifest after completed90-image ranking')
    args = parser.parse_args()
    if not args.prepare:
        print('No action. After review27 finishes, --prepare freezes the highest observed90-image recipe; no GPU is used.')
        return 0
    target = HERE / 'manifest.json'
    if target.exists() or target.is_symlink():
        raise FileExistsError(f'Refusing to overwrite frozen manifest: {target}')
    manifest = prepared_manifest()
    runner.write(target, manifest)
    runner.load_manifest(ROOT)
    print(json.dumps(dict(path=str(target), sha256=runner.sha256(target),
                         selected_recipe=manifest['selected_recipe'], mIoU=manifest['selected_mIoU'],
                         training_performed=False, gpu_inference_performed=False), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
