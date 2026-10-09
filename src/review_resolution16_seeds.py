"""Fixed two-seed pilot comparisons, with no weight selection or new folds."""
import json
from pathlib import Path

import numpy as np

from data import load_split
from infer import image_iou, postprocess
from review_grains15 import BASE, metrics, read_rows
from review_resolution16 import EXP, UNET


def main():
    exp1 = 'x4_mit_bw_r90_s1'
    folds = json.loads(Path('splits/folds5.json').read_text())
    s0, s1, base = (read_rows(e, [0, 1]) for e in [EXP, exp1, BASE])
    expected = {n for n, f in folds.items() if f in [0, 1]}
    assert set(s0) == set(s1) == set(base) == expected
    sub6, seed_average, replace_pair, add_three = {}, {}, {}, {}
    for it in load_split('data', 'train') + load_split('data', 'valid'):
        n = it['name']
        if n not in expected:
            continue
        probs = []
        for e in [BASE, UNET, EXP, exp1]:
            p = np.load(Path('runs/oof') / e / f'{n}.npy').astype(np.float32)
            assert p.shape == (4, *it['gray'].shape) and np.isfinite(p).all()
            assert p.min() >= 0 and p.max() <= 1 and np.max(np.abs(p.sum(0) - 1)) < 0.005
            probs.append(p)
        pb, pu, p0, p1 = probs
        ps = (p0 + p1) / 2
        for rows, p in [(sub6, (pb + pu) / 2), (seed_average, ps),
                        (replace_pair, (ps + pu) / 2), (add_three, (pb + pu + ps) / 3)]:
            rows[n] = {'tta_pp': image_iou(it['lab'], postprocess(p)).tolist(), 'low': it['low_contrast']}
    result = {'checked_folds': [0, 1], 'seed0': metrics(s0, base, folds),
              'seed1': metrics(s1, base, folds), 'seed1_vs_seed0': metrics(s1, s0, folds),
              'seed_average_vs_MiT': metrics(seed_average, base, folds),
              'replace_MiT_seed_average': metrics(replace_pair, sub6, folds),
              'add_seed_average_equal_three': metrics(add_three, sub6, folds),
              'scope': 'Two-fold exploratory pilot only, not full CV or submission selection',
              'submission_changed': False, 'auto_remaining_folds': False}
    out = Path('runs/review16_closeout')
    out.mkdir(exist_ok=True)
    (out / 'results.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    lines = ['# 4배 MiT 두 시드 예비 결과', '',
             'fold 0·1, 36장/28합금의 탐색 비교다. 5-fold 점수나 공식 평가 점수가 아니다.', '',
             '| 비교 | mIoU (%) | 기준 대비 (%p) | 합금 bootstrap 95% 구간 |',
             '|---|---:|---:|---|']
    labels = {'seed0': '4배 seed0 / 기존 2배 MiT', 'seed1': '4배 seed1 / 기존 2배 MiT',
              'seed_average_vs_MiT': '4배 시드 평균 / 기존 2배 MiT',
              'replace_MiT_seed_average': 'sub6의 MiT를 4배 시드 평균으로 교체 / sub6',
              'add_seed_average_equal_three': 'sub6에 4배 시드 평균 추가 / sub6'}
    for k, label in labels.items():
        m = result[k]
        lo, hi = m['group_bootstrap_ci95']
        lines.append(f"| {label} | {m['cv_mIoU']:.4f} | {m['delta']:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
    lines += ['', '추가 fold 실행은 사용자 요청에 따라 중단한다. 현재 sub6과 제출 파일은 유지한다.',
              '시드별/fold별/상별/저대비 변화는 results.json에 기록했다.',
              '백업 검증이 완료되기 전에는 서버 이동 준비 완료로 판단하지 않는다.']
    text = '\n'.join(lines) + '\n'
    (out / 'summary.md').write_text(text, encoding='utf-8')
    print(text, flush=True)


if __name__ == '__main__':
    main()
