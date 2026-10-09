"""Fixed two-seed 5-fold validation, reporting nonpilot folds separately."""
import json
from pathlib import Path

import numpy as np

from data import load_split
from infer import image_iou, postprocess
from review_grains15 import BASE, metrics, read_rows
from review_resolution16 import EXP, UNET

OUT = Path('runs/review16_additional')


def main():
    fold_of = json.loads(Path('splits/folds5.json').read_text())
    expected = set(fold_of)
    seed1 = EXP + '_s1'
    single0, single1, base = (read_rows(e, list(range(5))) for e in [EXP, seed1, BASE])
    assert set(single0) == set(single1) == set(base) == expected
    for e, seed in [(EXP, 0), (seed1, 1)]:
        for fold in range(5):
            a = json.loads(Path(f'runs/cv/{e}/fold{fold}/args.json').read_text())
            assert a['seed'] == seed and a['fold'] == fold and a['epochs'] == 60
            assert a['width'] == 2560 and a['scale_multiplier'] == 2.0 and not a['init']
    sub6, average, addition, replacement = {}, {}, {}, {}
    items = load_split('data', 'train') + load_split('data', 'valid')
    assert len(items) == len(expected) == 90
    for it in items:
        name = it['name']
        probabilities = []
        for e in [BASE, UNET, EXP, seed1]:
            p = np.load(Path('runs/oof') / e / f'{name}.npy').astype(np.float32)
            assert p.shape == (4, *it['gray'].shape) and np.isfinite(p).all()
            assert p.min() >= 0 and p.max() <= 1 and np.max(np.abs(p.sum(0) - 1)) < 0.005
            probabilities.append(p)
        pb, pu, p0, p1 = probabilities
        ps = (p0 + p1) / 2
        for rows, p in [(sub6, (pb + pu) / 2), (average, ps),
                        (addition, (pb + pu + ps) / 3), (replacement, (ps + pu) / 2)]:
            rows[name] = {'tta_pp': image_iou(it['lab'], postprocess(p)).tolist(), 'low': it['low_contrast']}

    def compare(a, b):
        return {'full': metrics(a, b, fold_of),
            'nonpilot': metrics({n: r for n, r in a.items() if fold_of[n] in [2, 3, 4]}, b, fold_of)}

    result = {'seed0_vs_MiT': compare(single0, base), 'seed1_vs_MiT': compare(single1, base),
        'seed1_vs_seed0': compare(single1, single0), 'seed_average_vs_MiT': compare(average, base),
        'add_seed_average_equal_three': compare(addition, sub6),
        'replace_MiT_seed_average': compare(replacement, sub6),
        'recipe': 'x2 MiT 1/3 + x3 UNet++ 1/3 + x4 seed0 1/6 + x4 seed1 1/6',
        'checked_folds': list(range(5)), 'new_training_folds': [2, 3, 4],
        'scope': 'Pilot-screened candidate; nonpilot folds reported separately. No weight search.',
        'submission_changed': False}
    candidate = result['add_seed_average_equal_three']
    full, held = candidate['full'], candidate['nonpilot']
    candidate['predeclared_checks'] = {
        'full_delta_at_least_0_05pp': full['delta'] >= 0.05,
        'full_group_ci_positive': full['group_bootstrap_ci95'][0] > 0,
        'nonpilot_group_ci_positive': held['group_bootstrap_ci95'][0] > 0,
        'all_nonpilot_folds_improve': all(x > 0 for x in held['delta_per_fold'].values()),
        'low_contrast_not_worse': full['delta_low_contrast'] is not None and full['delta_low_contrast'] >= 0,
        'eutectic_si_not_worse': full['delta_per_class']['Eutectic Si'] >= 0,
        'other_class_drop_at_most_0_1pp': all(full['delta_per_class'][c] >= -0.1 for c in ['Al', 'Al3Ni', 'Primary Si'])}
    candidate['qualified_for_submission_review'] = all(candidate['predeclared_checks'].values())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'full_results.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    lines = ['# 4배 MiT 두 시드 전체 검증', '',
        '기존 fold 0·1은 재사용했고, fold 2·3·4를 두 시드로 추가 학습했다.',
        '주 후보는 사전에 정한 추가 앙상블이며 가중치를 탐색하지 않았다.', '',
        '| 비교 | 전체 mIoU (%) | 전체 변화 (%p) | 비pilot 변화 (%p) | 비pilot 95% CI |',
        '|---|---:|---:|---:|---|']
    for key in ['seed0_vs_MiT', 'seed1_vs_MiT', 'seed1_vs_seed0', 'seed_average_vs_MiT',
                'add_seed_average_equal_three', 'replace_MiT_seed_average']:
        value = result[key]
        m, h = value['full'], value['nonpilot']
        lo, hi = h['group_bootstrap_ci95']
        lines.append(f"| {key} | {m['cv_mIoU']:.4f} | {m['delta']:+.4f} | {h['delta']:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
    lines += ['', '상별·저대비·fold별 변화와 사전 판정 기준은 full_results.json에 기록했다.',
        '제출 파일을 자동 변경하거나 공식 제출하지 않는다.']
    (OUT / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
