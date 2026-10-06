"""교차검증 결과 집계: 실험마다 90장 전체의 mIoU(대회 방식), 상별·스타일별·해상도별 점수, 기준 실험과의 차이.

예) python src/aggregate.py                  (runs/cv 아래 모든 실험)
    python src/aggregate.py --base baseline  (기준 실험과의 차이 기준)

열 설명
- plain/pp/tta/tta_pp : 그대로 / Primary Si 후처리 / 8방향 TTA / TTA+후처리 의 90장 평균 mIoU
- est_test            : 평가 세트 구성(저대비 60%)에 맞춘 예상 점수 = 0.6×저대비 + 0.4×고대비
- d_mIoU, CI95        : 같은 사진끼리 비교한 기준 대비 차이와, 사진을 다시 뽑는 부트스트랩 95% 구간
- improved            : 기준보다 점수가 오른 사진 수
- d_<상>              : 상별 IoU 차이
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ['Al', 'Al3Ni', 'Primary Si', 'Eutectic Si']
SHORT = ['Al', 'Al3Ni', 'SiP', 'SiE']


def load(exp, key):
    rows = []
    files = sorted(exp.glob('fold*/result.json'))
    for f in files:
        rows += json.loads(f.read_text())['rows']
    return files, rows


def boot_ci(d, n=5000, seed=0):
    idx = np.random.default_rng(seed).integers(0, len(d), (n, len(d)))
    return np.percentile(d[idx].mean(1), [2.5, 97.5])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='runs/cv')
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--base', default='baseline')
    p.add_argument('--key', default='tta_pp', help='기준 점수 (plain, pp, tta, tta_pp)')
    args = p.parse_args()

    table, cls_img, img = [], {}, {}
    for exp in sorted(Path(args.root).iterdir()):
        files, rows = load(exp, args.key)
        if not files:
            continue
        s = {'exp': exp.name, 'folds': f'{len(files)}/{args.folds}'}
        for k in ['plain', 'pp', 'tta', 'tta_pp']:
            s[k] = np.mean([np.nanmean(r[k]) for r in rows]) * 100
        ious = np.array([r[args.key] for r in rows]) * 100
        sc = np.nanmean(ious, 1)
        for i, c in enumerate(SHORT):
            s[c] = np.nanmean(ious[:, i])
        low = np.array([r['low'] for r in rows])
        small = np.array([r['W'] == 675 for r in rows])
        s['low'], s['high'] = sc[low].mean(), sc[~low].mean()
        s['est_test'] = 0.6 * s['low'] + 0.4 * s['high']
        s['675px'], s['large'] = sc[small].mean(), sc[~small].mean()
        s['fold_std'] = np.std([np.mean([np.nanmean(r[args.key]) for r in json.loads(f.read_text())['rows']]) * 100 for f in files])
        table.append(s)
        names = [r['name'] for r in rows]
        img[exp.name] = dict(zip(names, sc))
        cls_img[exp.name] = dict(zip(names, ious))

    df = pd.DataFrame(table).set_index('exp')
    if args.base in img:
        base, base_c = img[args.base], cls_img[args.base]
        extra = []
        for e in df.index:
            common = [n for n in img[e] if n in base]
            d = np.array([img[e][n] - base[n] for n in common])
            dc = np.array([cls_img[e][n] - base_c[n] for n in common])
            lo, hi = boot_ci(d) if len(d) else (np.nan, np.nan)
            row = {'d_mIoU': d.mean(), 'CI95': f'[{lo:+.2f}, {hi:+.2f}]', 'improved': f'{(d > 0).sum()}/{len(d)}'}
            row.update({f'd_{c}': np.nanmean(dc[:, i]) for i, c in enumerate(SHORT)})
            extra.append(row)
        ex = pd.DataFrame(extra, index=df.index)
        df = pd.concat([df.iloc[:, :1], ex[['d_mIoU', 'CI95', 'improved']], df.iloc[:, 1:], ex[[f'd_{c}' for c in SHORT]]], axis=1)
    df = df.sort_values(args.key, ascending=False)
    pd.set_option('display.width', 300)
    pd.set_option('display.max_columns', 40)
    print(df.round(2).to_string())
    df.round(3).to_csv(Path(args.root) / 'summary.csv', encoding='utf-8-sig')


if __name__ == '__main__':
    main()
