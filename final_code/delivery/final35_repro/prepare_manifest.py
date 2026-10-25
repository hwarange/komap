#!/usr/bin/env python3
"""Freeze the explicitly selected train70 checkpoints and their test10 inference protocol."""
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

sys.dont_write_bytecode = True
from run_submission import BACKEND, CHECKPOINTS, PACKAGES, file_record, now, read, sha256, write

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PINNED = {
    'src/train.py': '8be5a7dcadc5809a4c07757a86d909c2407272c0e9d1184a342a1e60ff363db9',
    'src/data.py': '8e1ee2cfc7f296d001777f60c57d74c0030e28788e85949341700a55343a2fd9',
    'src/infer.py': '66ab7875b796b203db9dae40851eee25841f61435e409f7bbb043015febf1bea',
    'runs/review32_mit2_train70/plan.json': 'aaa05c3b8a7b3d3752dee4a747f6030be12e2ad3b99be43e8e4a47cf3dd40008',
    'runs/submission31_train70/plan.json': '2c78389e34e365a45c77131893972eee5a06e4db6a101693c52b3dc44c7da5ed',
    'runs/review33_mit2_pair/plan.json': '9e947e5e70edfc4daacc8e911be9dbae2a265f8eff4ad1885e9c0fefc375214c',
    'runs/review33_mit2_pair/results.json': '1367a8cb04573e4880b4857a8c5dd4ee24417faedeb51967027bde6d5ebb0de0',
    'runs/review33_mit2_pair/per_image_rows.json': '2950865a48784918d4892b9f53f5bd23ee09b9eda12a15745116a30440e437bc',
    'runs/review34_train70_group5fold/results.json': 'e6ab947336f92410477ea637be904e331c199dd71bd8eb7df9113765017ae563',
}


def main():
    from PIL import Image
    if (HERE / 'manifest.json').exists() or (HERE / 'source').exists():
        raise RuntimeError('Refuse to replace an existing frozen manifest or source snapshot')
    for rel, expected in PINNED.items():
        if sha256(ROOT / rel) != expected:
            raise RuntimeError('Reference pin mismatch: ' + rel)
    score = read(ROOT / 'runs/review33_mit2_pair/results.json')['scores']['mit2_plus_unet120']
    if abs(score['mIoU'] - 80.62075310543737) > 1e-10:
        raise RuntimeError('Unexpected selected validation score')
    cv = read(ROOT / 'runs/review34_train70_group5fold/results.json')['scores']['mit2_plus_unet3']
    models, provenance = [], [ROOT / p for p in PINNED if not p.startswith('src/')]
    for family, (rel, expected) in CHECKPOINTS.items():
        ck = ROOT / rel
        if sha256(ck) != expected:
            raise RuntimeError('Selected checkpoint hash mismatch: ' + family)
        args = read(ck.parent / 'args.json')
        audit = read(ck.parent / 'initialization_audit.json')
        if (audit['args'] != args or audit['inherited_checkpoint'] is not None
                or audit['previous_experiment_checkpoint_used'] or not audit['strict_initialization']
                or len(audit['training_names']) != 70 or len(audit['validation_names']) != 20):
            raise RuntimeError('Unexpected selected fresh train70 provenance')
        models.append(dict(id=family + '_train70_last', family=family, checkpoint=file_record(ROOT, ck),
                           checkpoint_epoch=args['epochs'], args=args,
                           initialization_audit=file_record(ROOT, ck.parent / 'initialization_audit.json'),
                           initial_state_sha256=audit['full_initial_state_sha256']))
        provenance.extend(ck.parent / f for f in ('args.json', 'initialization_audit.json', 'result.json', 'log.jsonl'))
    audits = [read(ROOT / v['initialization_audit']['path']) for v in models]
    if any(a['training_names'] != audits[0]['training_names'] or a['validation_names'] != audits[0]['validation_names'] for a in audits):
        raise RuntimeError('Selected models did not use identical image splits')
    provenance.extend(ROOT / f'runs/{r}/all.done' for r in ('review32_mit2_train70', 'review33_mit2_pair', 'review34_train70_group5fold'))
    tests = []
    for p in sorted((ROOT / 'data/test/images').glob('*_image.png')):
        with Image.open(p) as im:
            tests.append(dict(**file_record(ROOT, p), size_width_height=list(im.size)))
    if len(tests) != 10:
        raise RuntimeError('Expected exactly ten test images')
    (HERE / 'source').mkdir()
    for name in ('train.py', 'data.py', 'infer.py'):
        (HERE / 'source' / name).write_bytes((ROOT / 'src' / name).read_bytes())
    packages = {p: importlib.metadata.version(p) for p in PACKAGES}
    distributions = sorted({d.metadata['Name'].lower().replace('_', '-'): d.version for d in importlib.metadata.distributions()}.items())
    lines = ['# Reference environment snapshot; clean installation has not been independently tested.',
             'name: komap-final35-train70', 'channels:', '  - conda-forge', 'dependencies:',
             f'  - python={platform.python_version()}', '  - pip', '  - pip:',
             '      - "--extra-index-url https://download.pytorch.org/whl/cu126"']
    lines += [f'      - "{name}=={version}"' for name, version in distributions]
    (HERE / 'environment.yml').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    source = [HERE / p for p in ('run_submission.py', 'prepare_manifest.py', 'environment.yml', 'README.md')]
    source += [HERE / 'source' / name for name in ('train.py', 'data.py', 'infer.py')]
    value = dict(schema='komap-final35-train70-v1', created_utc=now(), selection='User-approved MiT2 original + UNet++3 base120',
                 training_images=70, validation_images=20, training_names=audits[0]['training_names'],
                 validation_names=audits[0]['validation_names'], validation_used_for_training=False,
                 validation_reused_for_model_selection=True, independent_test_score_available=False,
                 selected_validation20=score, train70_internal_group5fold_oof=cv,
                 cv_note='Separate supporting experiment with 56 train/14 heldout images per fold. Its weights are not used here.',
                 models=models, checkpoint_choice='last', family_weights={'mit2': 0.5, 'unet3': 0.5},
                 scale_policy='original', extra_scale=1.0, tta='dihedral8', amp_dtype='bfloat16',
                 probability_storage_dtype='float16', combination_dtype='float32',
                 postprocess=dict(source='source/infer.py:postprocess', min_comp=1e-4, min_total=0.005),
                 preprocessing='Per-image grayscale p1/p99 normalization followed by (x-0.5)/0.25',
                 inference_sequence='Per-model D4 TTA8 -> original-size probability -> float16 cache -> float32 50:50 mean -> unchanged postprocess -> RGB mask',
                 test_count=10, test_images=tests, test_ground_truth_used=False, new_training=False,
                 old90_checkpoints_used=False, continuation126_used=False, cap6_used=False,
                 python_version=platform.python_version(), packages=packages, backend=BACKEND,
                 source_files=[file_record(ROOT, p) for p in source],
                 provenance_files=[file_record(ROOT, p) for p in sorted(set(provenance))])
    write(HERE / 'manifest.json', value)
    print(json.dumps(dict(manifest_sha256=sha256(HERE / 'manifest.json'), models=2, test_images=10)))


if __name__ == '__main__':
    main()
