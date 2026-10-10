"""Package completed final checks without altering the frozen final-pair core.

Default is read-only. --build additionally requires an explicit final decision.
Only the fixed allowlist below is archived; no directory sweep is performed.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import zipfile

import build_final_pair_code_bundle as base

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'delivery/final_strengthened_code_bundle.zip'
RECEIPT = ROOT / 'delivery/final_strengthened_code_bundle_receipt.json'
PREFIX = 'runs/review26_final_strengthening'
README = 'delivery/final_strengthened_bundle_README.md'
EXTRA_SOURCES = (
    'scripts/analyze_final_confusion26.py', 'scripts/refine_scale26.py',
    'scripts/finetune_final26.py', 'src/review_grains15.py', 'scripts/run_crop18.py',
)
EXTRA_FILES = (
    'runs/review9/upp_hard.txt',
    f'{PREFIX}/diagnostics/results.json', f'{PREFIX}/diagnostics/per_image.json',
    f'{PREFIX}/diagnostics/all.done',
    f'{PREFIX}/scale_trials/plan.json', f'{PREFIX}/scale_trials/baseline_rows.json',
    f'{PREFIX}/scale_trials/started.json', f'{PREFIX}/scale_trials/results.json',
    f'{PREFIX}/scale_trials/per_image_rows.json', f'{PREFIX}/scale_trials/all.done',
    *(f'{PREFIX}/scale_trials/workers/gpu{i}/complete.json' for i in range(4)),
    f'{PREFIX}/finetune_plan/plan.json', f'{PREFIX}/finetune_plan/started.json',
    f'{PREFIX}/finetune_plan/results.json', f'{PREFIX}/finetune_plan/per_image_rows.json',
    f'{PREFIX}/finetune_plan/completed_jobs.json', f'{PREFIX}/finetune_plan/all.done',
    *(f'{PREFIX}/finetune_plan/{arm}/{name}' for arm in ('control', 'interface')
      for name in ('args.json', 'log.jsonl', 'result.json')),
)
OPTIONAL_METADATA = (
    f'{PREFIX}/scale_trials/cap6_sensitivity_diagnostic.json',
    f'{PREFIX}/scale_trials/local_metadata_verification.json',
    f'{PREFIX}/finetune_plan/local_completion_verification.json',
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8-sig'))


def record(payload: dict[str, bytes], path: str, sha: str, size: int | None = None) -> None:
    assert path in payload, f'Required packaged evidence missing: {path}'
    assert digest(payload[path]) == sha, f'Pinned evidence changed: {path}'
    assert size is None or len(payload[path]) == size, path


def verify_if_included(payload: dict[str, bytes], item: dict) -> bool:
    if item['path'] not in payload:
        return False
    record(payload, item['path'], item['sha256'], item.get('bytes', item.get('size_bytes')))
    return True


def image_macro(rows: dict) -> float:
    means = []
    for row in rows.values():
        values = [float(v) for v in row['tta_pp'] if v is not None and math.isfinite(float(v))]
        assert values
        means.append(sum(values) / len(values))
    assert means
    return sum(means) / len(means) * 100


def prepare() -> tuple[dict[str, bytes], dict]:
    payload, original_run = base.prepare()
    old_receipt = read(base.RECEIPT)
    old_zip = base.OUTPUT.read_bytes()
    assert digest(old_zip) == old_receipt['zip_sha256'], 'Original code ZIP changed'
    # The validated original payload must still match its own completed bundle.
    old_records = {item['path']: item for item in old_receipt['files']}
    assert set(payload) == set(old_records)
    for name, data in payload.items():
        assert digest(data) == old_records[name]['sha256'], name
    payload['delivery/final_pair_bundle_README.md'] = payload['README.md']
    payload[README] = (ROOT / README).read_bytes()
    payload['README.md'] = payload[README]
    for name in (*EXTRA_SOURCES, *EXTRA_FILES):
        assert name not in payload, name
        payload[name] = (ROOT / name).read_bytes()
    for name in OPTIONAL_METADATA:
        if (ROOT / name).is_file():
            payload[name] = (ROOT / name).read_bytes()

    parsed = {name: json.loads(data) for name, data in payload.items() if name.endswith('.json') or name.endswith('/all.done')}
    diag_done = parsed[f'{PREFIX}/diagnostics/all.done']
    assert diag_done['outcome'] == 'frozen_final_pair_full90_diagnostics_complete'
    assert diag_done['gpu_used'] is False and diag_done['submission_changed'] is False
    for item in diag_done['artifacts']:
        if item['path'].endswith('.json'):
            assert verify_if_included(payload, item)
    assert verify_if_included(payload, diag_done['script'])
    diag = parsed[f'{PREFIX}/diagnostics/results.json']
    assert diag['images'] == 90 and diag['submission_changed'] is False
    verified_diagnostic_pins = sum(verify_if_included(payload, item) for item in diag['verification']['receipts'])

    scale_prefix = f'{PREFIX}/scale_trials'
    scale_plan, scale_result, scale_done = (parsed[f'{scale_prefix}/{name}'] for name in ('plan.json', 'results.json', 'all.done'))
    assert scale_done['outcome'] == 'fixed_low_resolution_inference_scale_review_complete'
    assert scale_done['images'] == 90 and scale_done['affected_images'] == 26
    assert scale_done['submission_changed'] is False and scale_done['training_started'] is False
    for name, key in [('plan.json', 'plan_sha256'), ('results.json', 'results_sha256'), ('per_image_rows.json', 'rows_sha256')]:
        record(payload, f'{scale_prefix}/{name}', scale_done[key])
    assert scale_result['plan_sha256'] == scale_done['plan_sha256']
    assert scale_result['unchanged64_verified'] is True and scale_result['submission_changed'] is False
    assert scale_plan['arms'] == {'fresh_scale8': 8, 'cap6': 6, 'cap4': 4}
    for item in scale_plan['source_receipts']:
        assert verify_if_included(payload, item)
    for item in scale_plan['prerequisite_receipts']:
        assert verify_if_included(payload, item)
    for item in scale_result['worker_receipts']:
        assert verify_if_included(payload, item)
    scale_rows = parsed[f'{scale_prefix}/per_image_rows.json']
    for arm, rows in scale_rows.items():
        assert len(rows) == 90
        assert abs(image_macro(rows) - scale_result['scopes']['full90']['arms'][arm]['mIoU']) < 1e-8
    sensitivity_path = f'{scale_prefix}/cap6_sensitivity_diagnostic.json'
    if sensitivity_path in parsed:
        sensitivity = parsed[sensitivity_path]
        changes = scale_result['scopes']['affected26']['comparisons_vs_saved']['cap6']['per_image_delta_pp']
        values = list(changes.values())
        assert len(values) == 26
        assert sensitivity['improved'] == sum(v > 0 for v in values)
        assert sensitivity['worsened'] == sum(v < 0 for v in values)
        assert abs(sensitivity['median_delta_pp'] - statistics.median(values)) < 1e-10
        assert abs(sensitivity['mean_pp'] - statistics.mean(values)) < 1e-10
    local_receipt_path = f'{scale_prefix}/local_metadata_verification.json'
    if local_receipt_path in parsed:
        assert parsed[local_receipt_path]['verified'] is True
        for item in parsed[local_receipt_path]['files']:
            full_path = f"{scale_prefix}/{item['path']}"
            if full_path in payload:
                record(payload, full_path, item['sha256'], item['bytes'])

    train_prefix = f'{PREFIX}/finetune_plan'
    plan, result, done = (parsed[f'{train_prefix}/{name}'] for name in ('plan.json', 'results.json', 'all.done'))
    assert done['outcome'] == 'paired_six_epoch_continuation_complete'
    assert result['images'] == 18 and result['fold'] == 0
    assert done['submission_changed'] is False and result['submission_changed'] is False
    assert done['further_training_scheduled'] is False
    record(payload, f'{train_prefix}/plan.json', done['plan_sha256'])
    record(payload, f'{train_prefix}/results.json', done['results_sha256'])
    assert result['plan_sha256'] == done['plan_sha256']
    assert plan['continuation_epochs'] == 6 and plan['initial_training_epochs'] == 120
    assert plan['map_uses_validation_labels'] is False and plan['map_uses_prediction_errors'] is False
    assert plan['focus_probability'] == 0.35 and plan['radius_original_pixels'] == 2
    assert plan['automatic_remaining_folds'] == []
    assert plan['strict_initialization_tensors'] == 350
    for name, sha in plan['source_hashes'].items():
        record(payload, name, sha)
    for name, sha in plan['input_hashes'].items():
        if name in payload:
            record(payload, name, sha)
    assert parsed[f'{train_prefix}/completed_jobs.json'] == result['completion_receipts']
    local_train_receipt = f'{train_prefix}/local_completion_verification.json'
    if local_train_receipt in parsed:
        verification = parsed[local_train_receipt]
        assert verification['results_and_plan_verified'] and verification['all_six_epochs_verified']
        for item in verification['downloaded_files']:
            verify_if_included(payload, item)
    for receipt in result['completion_receipts']:
        for name, sha in receipt['hashes'].items():
            if name in payload:
                record(payload, name, sha)
    train_rows = parsed[f'{train_prefix}/per_image_rows.json']
    assert set(train_rows) == set(plan['recipes'])
    for arm, rows in train_rows.items():
        assert sorted(rows) == plan['validation_names']
        assert abs(image_macro(rows) - result['scores'][arm]['mIoU']) < 1e-8
    for job in plan['jobs']:
        relative = job['args']['out']
        assert parsed[f'{relative}/args.json'] == job['args']
        assert parsed[f'{relative}/result.json']['args'] == job['args']
        logs = [json.loads(line) for line in payload[f'{relative}/log.jsonl'].decode('utf-8').splitlines()]
        assert [row['epoch'] for row in logs] == list(range(1, 7))
        assert all(math.isfinite(row['loss']) and row['optimizer_updates'] == 144 for row in logs)

    for name, data in payload.items():
        path = PurePosixPath(name)
        assert not path.is_absolute() and '..' not in path.parts and '\\' not in name
        assert not name.startswith('data/') and '/train_interface_maps/' not in name
        assert path.suffix.lower() not in {'.pt', '.pth', '.npy', '.pkl', '.png', '.jpg', '.zip', '.pem', '.key'}
        assert not any(part in {'.ssh', '.env', '__pycache__'} for part in path.parts)
        if name.endswith('.py'):
            ast.parse(data.decode('utf-8'), filename=name)
        if name.endswith(('.md', '.py', '.txt', '.json', '.yml', '.jsonl')):
            assert b'-----BEGIN OPENSSH PRIVATE KEY-----' not in data
            assert b'-----BEGIN RSA PRIVATE KEY-----' not in data
    verification = dict(
        original_zip_sha256=old_receipt['zip_sha256'], original_zip_path=old_receipt['zip_path'],
        frozen_final_core_unchanged=True, original_test_mask_zip_sha256=original_run['zip_sha256'],
        diagnostics_included_pins_verified=verified_diagnostic_pins,
        scale_results_sha256=scale_done['results_sha256'], finetune_results_sha256=done['results_sha256'],
        diagnostic_results_sha256=digest(payload[f'{PREFIX}/diagnostics/results.json']),
        full90_scale_rows_recomputed=True, holdout18_finetune_rows_recomputed=True,
        raw_data_weights_probabilities_included=False, raw_images_or_gt_focus_maps_included=False,
        original_external_artifacts_reverified_locally=False,
        original_external_artifact_verification='Completed server receipts included; excluded data, weights and OOF arrays are not re-read by this local packaging step',
        fresh_environment_install_tested=False, official_submission_performed=False,
        new_trials_independent_validation=False,
    )
    return payload, verification


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--final-decision', choices=['retain-current-pair'],
                        help='Required for writing; only supports an explicitly retained original final pair')
    args = parser.parse_args()
    if args.build and args.final_decision != 'retain-current-pair':
        parser.error('--build requires --final-decision retain-current-pair after final review')
    payload, verification = prepare()
    if not args.build:
        print(json.dumps(dict(ok=True, read_only=True, files=len(payload), verification=verification), indent=2))
        return
    assert not OUTPUT.exists() and not RECEIPT.exists(), 'Refuse to overwrite strengthened bundle or receipt'
    records = [dict(path=name, size_bytes=len(data), sha256=digest(data)) for name, data in sorted(payload.items())]
    with zipfile.ZipFile(OUTPUT, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, data in sorted(payload.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 10, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)
    with zipfile.ZipFile(OUTPUT) as archive:
        assert archive.testzip() is None
        assert archive.namelist() == [item['path'] for item in records]
        for item in records:
            data = archive.read(item['path'])
            assert len(data) == item['size_bytes'] and digest(data) == item['sha256']
    assert digest(base.OUTPUT.read_bytes()) == verification['original_zip_sha256']
    result = dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
                  status='final_pair_retained_with_completed_strengthening_evidence',
                  final_decision=args.final_decision, zip_path=OUTPUT.relative_to(ROOT).as_posix(),
                  zip_size_bytes=OUTPUT.stat().st_size, zip_sha256=digest(OUTPUT.read_bytes()),
                  file_count=len(records), source_py_count=sum(item['path'].endswith('.py') for item in records),
                  zip_crc_passed=True, archive_file_hashes_verified=True,
                  builder_sha256=digest(Path(__file__).read_bytes()), files=records, verification=verification)
    with RECEIPT.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ['zip_path', 'zip_size_bytes', 'zip_sha256', 'file_count']}, indent=2))


if __name__ == '__main__':
    main()
