"""7단계 ③ 상한 계산: 예측된 덩어리를 '정답을 보고 완벽하게 다시 판정'하면 점수가 얼마나 오르나 (학습 없음).
덩어리 재판정 분류기를 만들 가치가 있는지 먼저 확인한다 (Si 덩어리 분류기 때: 상한 +2.6 → 실제 +0.1).

방식 (덩어리 = 예측 마스크의 8-연결 성분, 덩어리 안 정답 픽셀의 다수결로 다시 칠함)
- al3ni_sie_sep  : Al3Ni 덩어리와 Eutectic Si 덩어리를 각각 따로 묶어, Al3Ni/Eutectic Si 중 다수인 쪽으로
- al3ni_sie_union: 두 상을 합쳐서 덩어리로 묶은 뒤 다수결 (공정 조직에서 붙어 있으면 큰 덩어리로 합쳐져 손해)
- si_union (참고): Primary Si/Eutectic Si (지금 Si 분류기가 하는 일의 상한)
예) python src/oracle_components.py --ens runs/oof/_ens_A_r6
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from data import load_split, scale_for
from infer import image_iou, postprocess

G = {}
MODES = {'al3ni_sie_sep': ((1,), (3,)), 'al3ni_sie_union': ((1, 3),), 'si_union': ((2, 3),)}
MIN_AREAS = [20, 100, 500, 2000]   # 크기별 상한: 이 넓이(1280px 폭 기준 픽셀) 이상인 덩어리만 다시 칠할 때


def relabel(pred, gt, groups, min_area=0):
    """groups: 덩어리로 묶을 예측 상들의 묶음. 각 덩어리를 그 묶음에 속한 상 중 정답 다수인 상으로 칠한다.
    min_area보다 작은 덩어리는 그대로 둔다."""
    out = pred.copy()
    stats = []
    cand = sorted(groups_classes(groups))   # 다시 칠할 수 있는 상 (예: Al3Ni, Eutectic Si)
    for g in groups:
        k, cc = cv2.connectedComponents(np.isin(pred, g).astype(np.uint8), connectivity=8)
        if k <= 1:
            continue
        flat = cc.ravel()
        counts = np.stack([np.bincount(flat, weights=(gt.ravel() == c).astype(float), minlength=k) for c in cand])[:, 1:]
        sizes = np.bincount(flat, minlength=k)[1:]
        has = counts.sum(0) > 0
        choose = np.zeros(k, np.int64)
        choose[1:] = np.where(has & (sizes >= min_area), np.array(cand)[counts.argmax(0)], -1)
        m = (cc > 0) & (choose[cc] >= 0)
        out[m] = choose[cc[m]]
        stats.append({'n_comp': int(k - 1), 'largest_comp_share': float(sizes.max() / sizes.sum())})
    return out, stats


def groups_classes(groups):
    return [c for g in groups for c in g]


def _one(n):
    P = np.load(Path(G['ens']) / f'{n}.npy').astype(np.float32)
    gt = G['labs'][n]
    pred = postprocess(P)
    res = {'base': image_iou(gt, pred) * 100}
    st = {}
    for mode, groups in MODES.items():
        new, s = relabel(pred, gt, groups)
        res[mode] = image_iou(gt, new) * 100
        st[mode] = s
    for a in MIN_AREAS:   # 큰 덩어리만 (675px 사진은 2배 처리하므로 넓이를 배율로 환산)
        for mode in ['al3ni_sie_sep', 'si_union']:
            new, _ = relabel(pred, gt, MODES[mode], a / G['scale'][n] ** 2)
            res[f'{mode}_min{a}'] = image_iou(gt, new) * 100
    return n, res, st


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ens', required=True)
    p.add_argument('--procs', type=int, default=40)
    p.add_argument('--out', default='runs/rules/oracle_components.json')
    args = p.parse_args()
    items = load_split('data', 'train') + load_split('data', 'valid')
    G.update(ens=args.ens, labs={it['name']: it['lab'] for it in items},
             scale={it['name']: scale_for(it['gray'].shape[1], 1280) for it in items})
    with Pool(args.procs) as pool:
        rows = list(pool.imap_unordered(_one, [it['name'] for it in items]))
    out = {}
    for mode in ['base', *MODES, *[f'{m}_min{a}' for m in ['al3ni_sie_sep', 'si_union'] for a in MIN_AREAS]]:
        A = np.stack([r[1][mode] for r in rows])
        out[mode] = {'mIoU': round(float(np.nanmean(A, 1).mean()), 3),
                     'class_IoU': dict(zip(['Al', 'Al3Ni', 'SiP', 'SiE'], np.round(np.nanmean(A, 0), 2).tolist())),
                     'upper_bound_gain': round(float(np.nanmean(A, 1).mean()) - float(np.nanmean(np.stack([r[1]['base'] for r in rows]), 1).mean()), 3)}
    for mode in MODES:
        s = [x for r in rows for x in r[2][mode]]
        out[mode]['components_per_image'] = round(float(np.mean([x['n_comp'] for x in s])), 1)
        out[mode]['largest_component_share_mean'] = round(float(np.mean([x['largest_comp_share'] for x in s])), 3)
    print(json.dumps(out, indent=1, ensure_ascii=False))
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
