"""Predeclared screening gate for the external-grain transfer experiment.

The pilot decides whether to spend compute on the remaining folds. It does not
select a competition submission. Bootstrap resamples alloy groups within folds.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

BASE = 'x2_mit_bw_r90'
EXP = 'x2_mit_grains20_r90'
RULES = {
    'mean_delta_min': 0.15,
    'each_fold_delta_min_exclusive': 0.0,
    'group_ci95_lower_min': -0.15,
    'low_contrast_delta_min': -0.15,
    'eutectic_si_delta_min': -0.15,
    'other_class_delta_min': -0.30,
}


def metrics(rows_a, rows_b, fold_of, repeats=10000):
    names = sorted(rows_a)
    assert names and set(names).issubset(rows_b)
    a = np.asarray([rows_a[n]['tta_pp'] for n in names], float) * 100
    b = np.asarray([rows_b[n]['tta_pp'] for n in names], float) * 100
    d = np.nanmean(a, axis=1) - np.nanmean(b, axis=1)
    assert np.isfinite(d).all()
    folds = np.asarray([fold_of[n] for n in names])
    alloy = np.asarray([int(re.match(r'A(\d+)_', n).group(1)) for n in names])
    low = np.asarray([rows_b[n]['low'] for n in names], bool)
    blocks = [[np.flatnonzero((folds == f) & (alloy == g))
               for g in sorted(set(alloy[folds == f]))] for f in sorted(set(folds))]
    rng = np.random.default_rng(0)
    boot = np.empty(repeats)
    for i in range(repeats):
        idx = np.concatenate([np.concatenate([groups[j] for j in
                              rng.integers(0, len(groups), len(groups))]) for groups in blocks])
        boot[i] = d[idx].mean()
    class_delta = np.nanmean(a - b, axis=0)
    return {
        'images': len(names), 'alloy_groups': len(set(alloy)),
        'cv_mIoU': float(np.nanmean(a, axis=1).mean()),
        'base_cv_mIoU': float(np.nanmean(b, axis=1).mean()),
        'delta': float(d.mean()),
        'group_bootstrap_ci95': np.percentile(boot, [2.5, 97.5]).tolist(),
        'delta_per_fold': {str(f): float(d[folds == f].mean()) for f in sorted(set(folds))},
        'delta_per_class': dict(zip(['Al', 'Al3Ni', 'Primary Si', 'Eutectic Si'], class_delta.tolist())),
        'delta_low_contrast': float(d[low].mean()) if low.any() else None,
        'improved_images': int((d > 0).sum()),
        'names': names,
    }


def pilot_decision(m):
    c = m['delta_per_class']
    checks = {
        'mean_delta': m['delta'] >= RULES['mean_delta_min'],
        'both_folds_improve': all(d > 0 for d in m['delta_per_fold'].values()),
        'group_ci_lower': m['group_bootstrap_ci95'][0] >= RULES['group_ci95_lower_min'],
        'low_contrast': m['delta_low_contrast'] is not None and m['delta_low_contrast'] >= RULES['low_contrast_delta_min'],
        'eutectic_si': c['Eutectic Si'] >= RULES['eutectic_si_delta_min'],
        'other_classes': all(c[k] >= RULES['other_class_delta_min'] for k in ['Al', 'Al3Ni', 'Primary Si']),
    }
    return {'continue_full_cv': all(checks.values()), 'checks': checks, 'rules': RULES}


def read_rows(exp, folds):
    rows = {}
    for f in folds:
        root = Path('runs/cv') / exp / f'fold{f}'
        result = json.loads((root / 'result.json').read_text())
        logs = [json.loads(line) for line in (root / 'log.jsonl').read_text().splitlines()]
        assert len(logs) == 60 and logs[-1]['epoch'] == 60
        assert all(np.isfinite(r['loss']) for r in logs)
        for row in result['rows']:
            assert row['name'] not in rows
            rows[row['name']] = row
    return rows


def ensemble_review(exp, base, fold_of):
    from data import load_split
    from infer import image_iou, postprocess
    items = load_split('data', 'train') + load_split('data', 'valid')
    base_rows, replace_rows, add_rows = {}, {}, {}
    for it in items:
        name = it['name']
        probs = []
        for e in [base, 'x3_unetpp_bw_e120_r90', exp]:
            prob = np.load(Path('runs/oof') / e / f'{name}.npy').astype(np.float32)
            assert prob.shape == (4, *it['gray'].shape) and np.isfinite(prob).all()
            assert prob.min() >= 0 and prob.max() <= 1
            assert np.max(np.abs(prob.sum(0) - 1)) < 0.005
            probs.append(prob)
        pb, pu, pn = probs
        for target, prob in [(base_rows, (pb + pu) / 2),
                             (replace_rows, (pn + pu) / 2),
                             (add_rows, (pb + pu + pn) / 3)]:
            target[name] = {'tta_pp': image_iou(it['lab'], postprocess(prob)).tolist(),
                            'low': it['low_contrast']}
    result = {}
    for label, rows in [('replace_MiT_equal_pair', replace_rows),
                        ('add_external_equal_three', add_rows)]:
        full = metrics(rows, base_rows, fold_of)
        held = metrics({n: r for n, r in rows.items() if fold_of[n] in [2, 3, 4]}, base_rows, fold_of)
        # No automatic submission replacement. A positive pilot is selection:
        # the untouched three folds must also support any proposed candidate.
        checks = {
            'full_delta_at_least_0_05pp': full['delta'] >= 0.05,
            'full_group_ci_positive': full['group_bootstrap_ci95'][0] > 0,
            'nonpilot_group_ci_positive': held['group_bootstrap_ci95'][0] > 0,
            'all_nonpilot_folds_improve': all(x > 0 for x in held['delta_per_fold'].values()),
            'low_contrast_not_worse': full['delta_low_contrast'] >= 0,
            'eutectic_si_not_worse': full['delta_per_class']['Eutectic Si'] >= 0,
            'other_classes_drop_at_most_0_1pp': all(full['delta_per_class'][c] >= -0.1
                                                     for c in ['Al', 'Al3Ni', 'Primary Si']),
        }
        result[label] = {'full': full, 'nonpilot': held, 'checks': checks,
                         'qualified_for_submission_review': all(checks.values())}
    return {'fixed_recipes': result,
            'reference': f'{base} + x3_unetpp_bw_e120_r90, equal pair',
            'selection_note': 'Pilot-screened full CV is exploratory; nonpilot folds reported separately.',
            'submission_changed': False}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--exp', default=EXP)
    p.add_argument('--base', default=BASE)
    p.add_argument('--folds', type=int, nargs='+', default=[0, 1])
    p.add_argument('--ensemble', action='store_true')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    fold_of = json.loads(Path('splits/folds5.json').read_text())
    a, b = read_rows(args.exp, args.folds), read_rows(args.base, args.folds)
    expected = {n for n, f in fold_of.items() if f in args.folds}
    assert set(a) == expected == set(b)
    for n in expected:
        assert (Path('runs/oof') / args.exp / f'{n}.npy').exists()
    m = metrics(a, b, fold_of)
    out = {'exp': args.exp, 'base': args.base, 'folds': args.folds, **m}
    if args.folds == [0, 1]:
        out.update(pilot_decision(m))
        out['purpose'] = 'Screen for remaining-three-fold compute, not submission selection.'
    else:
        heldout = {n: r for n, r in a.items() if fold_of[n] in [2, 3, 4]}
        out['nonpilot_folds'] = metrics(heldout, b, fold_of)
        out['purpose'] = 'Exploratory full CV after pilot screening; submission unchanged.'
        if args.ensemble:
            out['ensemble'] = ensemble_review(args.exp, args.base, fold_of)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False, allow_nan=False))
    print(json.dumps(out, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
