"""Package the completed review27 winner and verified test run; default is read-only.

The explicit allowlist extends the prior strengthened bundle without changing it.
Run --build only after review27 and selected test inference have both completed.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import json
import math
from pathlib import Path, PurePosixPath
import zipfile

import build_final_strengthened_bundle as history

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'delivery/final27_code_bundle.zip'
RECEIPT = ROOT / 'delivery/final27_code_bundle_receipt.json'
PREFIX = 'runs/review27_finetune_full'
PACKAGE = 'delivery/final27_repro'
EXECUTION = 'submissions/final27_selected'
README = 'delivery/final27_bundle_README.md'
CONFIGS = ('baseline_original', 'baseline_cap6', 'interface_original', 'interface_cap6')
FAMILIES = ('mit4', 'unet3')
EXTRA = (
    'scripts/run_finetune27.py', 'scripts/review_finetune27.py',
    'delivery/build_final27_code_bundle.py',
    'delivery/build_final_strengthened_bundle.py', 'delivery/build_final_pair_code_bundle.py',
    'delivery/final_strengthened_code_bundle_receipt.json', 'delivery/final_pair_code_bundle_receipt.json',
    *(f'{PACKAGE}/{name}' for name in ('run_submission.py', 'prepare_manifest.py', 'README.md',
                                      'environment.yml', 'manifest.json', 'cpu_validation.json')),
    *(f'{PREFIX}/{name}' for name in ('plan.json', 'started.json', 'results.json', 'per_image_rows.json',
                                     'completed_jobs.json', 'all.done')),
    *(f'{PREFIX}/interface/fold{fold}/{name}' for fold in range(1, 5)
      for name in ('args.json', 'log.jsonl', 'result.json')),
    f'{EXECUTION}/summary_test.json', f'{EXECUTION}/run_receipt.json',
)
OPTIONAL = (
    f'{PREFIX}/local_completion_verification.json',
    f'{PACKAGE}/preflight_reference_server.json',
    f'{PACKAGE}/local_validation.json',
)
digest, read, record = history.digest, history.read, history.record


def prepare():
    payload, prior_verification = history.prepare()
    prior_receipt = read(history.RECEIPT)
    assert digest(history.OUTPUT.read_bytes()) == prior_receipt['zip_sha256'], 'Prior strengthened ZIP changed'
    prior_files = {v['path']: v for v in prior_receipt['files']}
    assert set(payload) == set(prior_files), 'Prior strengthened allowlist no longer matches its frozen archive'
    for name, data in payload.items():
        assert digest(data) == prior_files[name]['sha256'], f'Historical evidence changed: {name}'
    payload['historical/final_strengthened_README.md'] = payload['README.md']
    payload[README] = (ROOT / README).read_bytes()
    payload['README.md'] = payload[README]
    for name in EXTRA:
        assert name not in payload, name
        payload[name] = (ROOT / name).read_bytes()
    for name in OPTIONAL:
        if (ROOT / name).is_file():
            assert name not in payload, name
            payload[name] = (ROOT / name).read_bytes()
    parsed = {name: json.loads(data) for name, data in payload.items()
              if name.endswith('.json') or name.endswith('/all.done')}
    plan, result, rows, done = [parsed[f'{PREFIX}/{name}'] for name in
                                ('plan.json', 'results.json', 'per_image_rows.json', 'all.done')]
    assert done['outcome'] == 'five_fold_continuation_fixed_2x2_review_complete'
    assert done['images'] == result['images'] == 90 and result['folds'] == 5
    for key, filename in [('results_sha256', 'results.json'), ('rows_sha256', 'per_image_rows.json'),
                          ('plan_sha256', 'plan.json'), ('completed_jobs_sha256', 'completed_jobs.json')]:
        record(payload, f'{PREFIX}/{filename}', done[key])
    assert result['plan_sha256'] == done['plan_sha256']
    assert result['completed_jobs_sha256'] == done['completed_jobs_sha256']
    assert parsed[f'{PREFIX}/started.json']['plan_sha256'] == done['plan_sha256']
    assert result['source_hashes'] == plan['source_hashes']
    for name, sha in plan['source_hashes'].items():
        record(payload, name, sha)
    for name, sha in plan['input_hashes'].items():
        if name in payload:
            record(payload, name, sha)
    assert plan['comparison_recipes'] == list(CONFIGS)
    assert plan['fixed_family_weights'] == [0.5, 0.5] and plan['no_per_fold_recipe_selection']
    assert plan['new_training_folds'] == [1, 2, 3, 4] and plan['reused_completed_folds'] == [0]
    assert plan['continuation_epochs'] == 6 and plan['initial_training_epochs'] == 120
    assert not plan['map_uses_validation_labels'] and not plan['map_uses_prediction_errors']
    assert plan['focus_probability'] == 0.35 and not plan['optimizer_state_resumed']

    expected_keys = set(CONFIGS) | {'baseline_UNet3', 'interface_UNet3'}
    scores = result['scopes']['full90']['scores']
    assert set(scores) == set(rows) == expected_keys
    for config, values in rows.items():
        assert len(values) == scores[config]['images'] == 90
        assert abs(history.image_macro(values) - scores[config]['mIoU']) < 1e-8
    ranking = sorted([dict(config=name, mIoU=scores[name]['mIoU'],
                           delta_vs_original_pp=scores[name]['mIoU'] - scores['baseline_original']['mIoU'])
                      for name in CONFIGS], key=lambda row: (-row['mIoU'], CONFIGS.index(row['config'])))
    assert len(result['ranking']) == 4 and {v['config'] for v in result['ranking']} == set(CONFIGS)
    for actual, expected in zip(result['ranking'], ranking):
        assert actual['config'] == expected['config'] and actual['mIoU'] == expected['mIoU']
        assert abs(actual['delta_vs_original_pp'] - expected['delta_vs_original_pp']) < 1e-8
    winner = ranking[0]['config']
    assert winner == result['highest_observed_full90_config'] == done['highest_observed_full90_config']
    assert winner != 'baseline_original', 'Keep prior artifacts if original recipe wins'
    assert abs(scores['baseline_original']['mIoU'] - 80.900653378248) < 1e-8

    receipts = parsed[f'{PREFIX}/completed_jobs.json']
    original_receipt = plan['reused_fold0']['completion_receipt']
    assert len(receipts) == 4 and {v['id'] for v in receipts} == {f'interface_fold{k}' for k in range(1, 5)}
    assert result['completion_receipts'] == [original_receipt] + receipts
    receipts_by_id = {v['id']: v for v in result['completion_receipts']}
    for receipt in result['completion_receipts']:
        for name, sha in receipt['hashes'].items():
            if name in payload:
                record(payload, name, sha)
    validations = {v['fold']: v for v in result['validations']}
    assert len(result['validations']) == len(validations) == 5 and set(validations) == set(range(5))
    for job in [plan['reused_fold0']['job']] + plan['jobs']:
        args, fold = job['args'], job['fold']
        prefix = args['out']
        assert parsed[f'{prefix}/args.json'] == args
        assert parsed[f'{prefix}/result.json']['args'] == args
        log = [json.loads(line) for line in payload[f'{prefix}/log.jsonl'].decode('utf-8').splitlines()]
        assert [v['epoch'] for v in log] == list(range(1, 7))
        assert all(math.isfinite(v['loss']) and v['optimizer_updates'] == 144 for v in log)
        validation = validations[fold]
        assert validation['checkpoint'] == f'{prefix}/last.pt'
        assert validation['checkpoint_sha256'] == receipts_by_id[job['id']]['hashes'][validation['checkpoint']]
        assert validation['initialization'] == args['init'] and validation['epochs'] == 6

    manifest_bytes = payload[f'{PACKAGE}/manifest.json']
    manifest = parsed[f'{PACKAGE}/manifest.json']
    assert manifest['schema'] == 'komap-final27-v1' and manifest['selected_recipe'] == winner
    assert manifest['selected_mIoU'] == scores[winner]['mIoU']
    assert abs(manifest['selected_delta_vs_original_pp'] - ranking[0]['delta_vs_original_pp']) < 1e-8
    assert manifest['family_weights'] == dict.fromkeys(FAMILIES, 0.5) and manifest['folds'] == 5
    assert manifest['tta'] == 'dihedral8' and manifest['probability_dtype'] == 'float32'
    assert manifest['checkpoint_choice'] == 'last' and not manifest['fresh_environment_installation_verified']
    assert manifest['evaluation_source_hashes'] == plan['source_hashes']
    for item in manifest['source_files'] + manifest['selection_sources']:
        record(payload, item['path'], item['sha256'], item['size_bytes'])
    models = manifest['models']
    assert len(models) == 10 and {(m['family'], m['fold']) for m in models} == {
        (family, fold) for family in FAMILIES for fold in range(5)}
    old_manifest = parsed['delivery/final_pair_repro/manifest.json']
    old_checkpoints = {item['path']: item for item in old_manifest['checkpoints']}
    new_unet, cap6 = winner.startswith('interface_'), winner.endswith('_cap6')
    assert manifest['unet_policy'] == ('interface6' if new_unet else 'original')
    assert manifest['mit_policy'] == ('cap6_width675' if cap6 else 'original')
    for model in models:
        family, fold = model['family'], model['fold']
        ck = model['checkpoint']
        if family == 'unet3' and new_unet:
            assert ck['path'] == validations[fold]['checkpoint']
            assert ck['sha256'] == validations[fold]['checkpoint_sha256']
            assert model['checkpoint_epoch'] == model['args']['epochs'] == 6
            assert model['initial_training_epochs'] == 120 and not model['optimizer_state_resumed']
            selected_job = plan['reused_fold0']['job'] if fold == 0 else next(v for v in plan['jobs'] if v['fold'] == fold)
            assert model['args'] == selected_job['args']
        else:
            exp = 'x4_mit_c1536_bw_r90' if family == 'mit4' else 'x3_unetpp_bw_e120_r90'
            assert ck == old_checkpoints[f'runs/cv/{exp}/fold{fold}/last.pt']
            assert model['checkpoint_epoch'] == (60 if family == 'mit4' else 120)
    assert manifest['test_images'] == old_manifest['test_images']

    run, summary = [parsed[f'{EXECUTION}/{name}'] for name in ('run_receipt.json', 'summary_test.json')]
    assert run['status'] == 'complete' and run['selected_recipe'] == winner
    assert run['manifest_sha256'] == digest(manifest_bytes)
    assert run['runner_sha256'] == digest(payload[f'{PACKAGE}/run_submission.py'])
    assert run['backend'] == manifest['backend']
    assert not run['prediction_reuse'] and not run['training_performed'] and not run['test_ground_truth_used']
    assert run['prediction_exit_codes'] and not any(run['prediction_exit_codes'])
    assert run['preflight']['ok'] and run['preflight']['verified_files'] == {'source_files': 5, 'checkpoints': 10, 'test_images': 10}
    assert summary['n_images'] == summary['n_models'] == 10 and summary['models_per_image'] == [10]
    assert summary['models_per_exp'] == dict.fromkeys(FAMILIES, 5)
    assert summary['recipe'] == dict.fromkeys(FAMILIES, 1.0)
    assert summary['mit_policy'] == manifest['mit_policy'] and summary['unet_policy'] == manifest['unet_policy']
    assert summary['full_mix'] == 0 and not summary['problems'] and summary['si_classifier'] is None
    assert not summary['test_ground_truth_used']
    shapes = run['preflight']['image_sizes_width_height']
    assert len(shapes) == 10
    expected_arrays = {f'parts/models/{family}/fold{fold}/{name}.npy'
                       for family in FAMILIES for fold in range(5) for name in shapes}
    assert len(run['probability_outputs']) == 100
    assert {v['path'] for v in run['probability_outputs']} == expected_arrays
    masks = run['masks']
    assert len(masks) == 10 and {m['name'] for m in masks} == {f'{name}_mask.png' for name in shapes}
    mask_zip = ROOT / EXECUTION / 'final27_masks.zip'
    assert digest(mask_zip.read_bytes()) == run['zip_sha256'], 'Selected mask ZIP differs from verified inference'
    with zipfile.ZipFile(mask_zip) as archive:
        assert archive.testzip() is None and len(archive.namelist()) == 10
        assert set(archive.namelist()) == {v['name'] for v in masks}
        for item in masks:
            data = archive.read(item['name'])
            assert len(data) == item['size_bytes'] and digest(data) == item['sha256']
    proof = parsed[f'{PACKAGE}/cpu_validation.json']
    assert proof['source_sha256'] == digest(payload[f'{PACKAGE}/run_submission.py'])
    assert proof['prepare_sha256'] == digest(payload[f'{PACKAGE}/prepare_manifest.py'])
    assert not proof['gpu_inference_performed']

    for name, data in payload.items():
        path = PurePosixPath(name)
        assert not path.is_absolute() and '..' not in path.parts and '\\' not in name
        assert not name.startswith('data/') and '/train_interface_maps/' not in name
        assert path.suffix.lower() not in {'.pt', '.pth', '.npy', '.pkl', '.png', '.jpg', '.zip', '.pem', '.key'}
        assert not any(p in {'.ssh', '.env', '__pycache__'} for p in path.parts)
        # Builder source contains marker literals; reject a real multiline key header.
        for kind in (b'OPENSSH', b'RSA', b'EC', b'DSA', b'ENCRYPTED'):
            header = b'-----BEGIN ' + kind + b' PRIVATE KEY-----'
            assert header + b'\n' not in data and header + b'\r\n' not in data
        if name.endswith('.py'):
            ast.parse(data.decode('utf-8'), filename=name)
    verification = dict(prior_strengthened_zip_sha256=prior_receipt['zip_sha256'],
                        original_zip_sha256=prior_verification['original_zip_sha256'],
                        prior_artifacts_preserved=True, full90_all_six_score_rows_recomputed=True,
                        selected_recipe=winner, selected_mIoU=manifest['selected_mIoU'],
                        selected_delta_vs_original_pp=manifest['selected_delta_vs_original_pp'],
                        selected_test_mask_zip_sha256=run['zip_sha256'],
                        results_sha256=done['results_sha256'], manifest_sha256=digest(manifest_bytes),
                        selected_checkpoints_match_evaluated_receipts=True,
                        external_weights_and_probabilities_re_read_locally=False,
                        external_verification='Hashes are bound to completed server evaluation/inference receipts; excluded checkpoint and probability binaries are not reread by local packaging.',
                        raw_data_weights_probabilities_or_masks_included=False,
                        fresh_environment_install_tested=False, official_submission_performed=False,
                        independent_validation=False)
    return payload, verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', action='store_true', help='Write a new ZIP only after completed review and selected test inference')
    args = parser.parse_args()
    payload, verification = prepare()
    if not args.build:
        print(json.dumps(dict(ok=True, read_only=True, files=len(payload), verification=verification), indent=2))
        return
    if OUTPUT.exists() or RECEIPT.exists():
        raise FileExistsError('Refusing to overwrite final27 code ZIP or receipt')
    records = [dict(path=name, size_bytes=len(data), sha256=digest(data)) for name, data in sorted(payload.items())]
    with zipfile.ZipFile(OUTPUT, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, data in sorted(payload.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 10, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)
    with zipfile.ZipFile(OUTPUT) as archive:
        assert archive.testzip() is None and archive.namelist() == [v['path'] for v in records]
        for item in records:
            data = archive.read(item['path'])
            assert len(data) == item['size_bytes'] and digest(data) == item['sha256']
    assert digest(history.OUTPUT.read_bytes()) == verification['prior_strengthened_zip_sha256']
    result = dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
                  status='selected_after_completed_full90_review_and_verified_test_inference',
                  zip_path=OUTPUT.relative_to(ROOT).as_posix(), zip_size_bytes=OUTPUT.stat().st_size,
                  zip_sha256=digest(OUTPUT.read_bytes()), file_count=len(records),
                  source_py_count=sum(v['path'].endswith('.py') for v in records),
                  zip_crc_passed=True, archive_file_hashes_verified=True,
                  builder_sha256=digest(Path(__file__).read_bytes()), files=records, verification=verification)
    with RECEIPT.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('zip_path', 'zip_size_bytes', 'zip_sha256', 'file_count')}, indent=2))


if __name__ == '__main__':
    main()
