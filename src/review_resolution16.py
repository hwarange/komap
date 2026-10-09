"""Predeclared resolution screening and fixed-sub6 comparisons."""
import argparse
import json
from pathlib import Path

import numpy as np

from review_grains15 import BASE, metrics, read_rows

EXP = 'x4_mit_bw_r90'
UNET = 'x3_unetpp_bw_e120_r90'
RULES = {'mean_delta_min': 0.10, 'each_fold_delta_min': -0.30,
         'group_ci_lower_min': -0.15, 'low_contrast_delta_min': -0.15,
         'eutectic_delta_min': -0.15, 'other_classes_delta_min': -0.30}


def gate(m):
    return {
        'mean_delta': m['delta'] >= RULES['mean_delta_min'],
        'folds_no_large_drop': all(x >= RULES['each_fold_delta_min'] for x in m['delta_per_fold'].values()),
        'group_ci_lower': m['group_bootstrap_ci95'][0] >= RULES['group_ci_lower_min'],
        'low_contrast': m['delta_low_contrast'] >= RULES['low_contrast_delta_min'],
        'eutectic': m['delta_per_class']['Eutectic Si'] >= RULES['eutectic_delta_min'],
        'other_classes': all(m['delta_per_class'][c] >= RULES['other_classes_delta_min']
                             for c in ['Al', 'Al3Ni', 'Primary Si']),
    }


def review(folds, exp=EXP, expected_seed=0):
    from data import load_split
    from infer import image_iou, postprocess
    fold_of = json.loads(Path('splits/folds5.json').read_text())
    a, b = read_rows(exp, folds), read_rows(BASE, folds)
    expected = {n for n, f in fold_of.items() if f in folds}
    assert set(a) == set(b) == expected
    for f in folds:
        cfg = json.loads(Path(f'runs/cv/{exp}/fold{f}/args.json').read_text())
        assert cfg['width'] == 2560 and cfg['scale_multiplier'] == 2 and not cfg['init']
        assert cfg['seed'] == expected_seed
    single = metrics(a, b, fold_of)
    base_rows, replace_rows, add_rows = {}, {}, {}
    for it in load_split('data', 'train') + load_split('data', 'valid'):
        n = it['name']
        if n not in expected:
            continue
        probs = []
        for e in [BASE, UNET, exp]:
            p = np.load(Path('runs/oof') / e / f'{n}.npy').astype(np.float32)
            assert p.shape == (4, *it['gray'].shape) and np.isfinite(p).all()
            assert p.min() >= 0 and p.max() <= 1 and np.max(np.abs(p.sum(0) - 1)) < 0.005
            probs.append(p)
        pb, pu, pn = probs
        for target, prob in [(base_rows, (pb + pu) / 2), (replace_rows, (pn + pu) / 2),
                             (add_rows, (pb + pu + pn) / 3)]:
            target[n] = {'tta_pp': image_iou(it['lab'], postprocess(prob)).tolist(), 'low': it['low_contrast']}
    recipes = {}
    for k, rows in [('replace_MiT_equal_pair', replace_rows), ('add_x4_equal_three', add_rows)]:
        m = metrics(rows, base_rows, fold_of)
        checks = gate(m)
        recipes[k] = {'metrics': m, 'pilot_checks': checks, 'pilot_pass': all(checks.values())}
        if folds == [0, 1, 2, 3, 4]:
            held = metrics({n: r for n, r in rows.items() if fold_of[n] in [2, 3, 4]}, base_rows, fold_of)
            recipes[k]['nonpilot'] = held
            full_checks = {
                'full_delta_min': m['delta'] >= 0.05,
                'full_ci_positive': m['group_bootstrap_ci95'][0] > 0,
                'nonpilot_ci_positive': held['group_bootstrap_ci95'][0] > 0,
                'nonpilot_folds_positive': all(x > 0 for x in held['delta_per_fold'].values()),
                'low_contrast_not_worse': m['delta_low_contrast'] >= 0,
                'eutectic_not_worse': m['delta_per_class']['Eutectic Si'] >= 0,
                'other_class_limit': all(m['delta_per_class'][c] >= -0.1
                                        for c in ['Al', 'Al3Ni', 'Primary Si']),
            }
            recipes[k]['submission_review_checks'] = full_checks
            recipes[k]['qualified_for_submission_review'] = all(full_checks.values())
    single_checks = gate(single)
    return {'exp': exp, 'folds': folds, 'single': single, 'single_pilot_checks': single_checks,
            'fixed_recipes': recipes, 'pilot_rules': RULES,
            'continue_full_cv': all(single_checks.values()) or any(v['pilot_pass'] for v in recipes.values()),
            'purpose': 'Pilot screens compute; full CV after screening is exploratory.',
            'submission_changed': False}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--folds', nargs='+', type=int, default=[0, 1])
    p.add_argument('--out', required=True)
    p.add_argument('--exp', default=EXP)
    p.add_argument('--expected-seed', type=int, default=0)
    args = p.parse_args()
    assert args.folds in ([0, 1], [0, 1, 2, 3, 4])
    result = review(args.folds, args.exp, args.expected_seed)
    text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
    Path(args.out).write_text(text, encoding='utf-8')
    print(text, flush=True)


if __name__ == '__main__':
    main()
