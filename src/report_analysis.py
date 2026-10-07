"""보고서용 분석 (학습 없음): 상 분율 오차, 탄성률(Hashin–Shtrikman) 영향, step별 조직 크기 지표.

예측은 OOF(그 사진을 학습하지 않은 모델) 확률 캐시를 쓴다. 기본은 지금 최선 조합
(27개 앙상블 + Eutectic Si 전용 2종 + Al3Ni 전용, Si 분류기 전) = runs/oof/_ens_A27_sie_al.
예) python src/report_analysis.py --ens runs/oof/_ens_A27_sie_al --out runs/report
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import spearmanr
from skimage.morphology import skeletonize

from data import load_split
from infer import image_iou, postprocess

CLS = ['Al', 'Al3Ni', 'SiP', 'SiE']
W_REF = 1280   # 같은 시야라 픽셀 크기 ∝ 1/폭. 길이는 폭 1280 px 기준 픽셀로 환산한다
# 상별 탄성 상수 (E GPa, 푸아송비). Al3Ni의 E는 측정값마다 달라(140~215) 세 가지로 본다. ν 0.30은 가정
PHASES = {'Al': (70.0, 0.33), 'Si': (162.0, 0.22)}
AL3NI_E = [140.0, 160.0, 185.0]
G = {}


def kg(E, nu):
    return E / (3 * (1 - 2 * nu)), E / (2 * (1 + nu))


def hs_bounds(f, K, Gs):
    """다상 등방 복합재의 Hashin–Shtrikman(Walpole) 범위 → (E 하한, E 상한) GPa."""
    f, K, Gs = map(np.asarray, (f, K, Gs))
    kb = lambda g: 1 / np.sum(f / (K + 4 / 3 * g)) - 4 / 3 * g
    gb = lambda k, g: (lambda z: 1 / np.sum(f / (Gs + z)) - z)(g / 6 * (9 * k + 8 * g) / (k + 2 * g))
    E = lambda k, g: 9 * k * g / (3 * k + g)
    return E(kb(Gs.min()), gb(K.min(), Gs.min())), E(kb(Gs.max()), gb(K.max(), Gs.max()))


def modulus(fr, e_al3ni):
    """fr: [Al, Al3Ni, SiP, SiE] 면적 분율 → HS 범위의 중간값 (GPa)."""
    props = [kg(*PHASES['Al']), kg(e_al3ni, 0.30), kg(*PHASES['Si'])]
    f = [fr[0], fr[1], fr[2] + fr[3]]
    lo, hi = hs_bounds(f, [p[0] for p in props], [p[1] for p in props])
    return lo, hi


def thickness(mask, s):
    """구조 두께: 뼈대 위 거리변환 ×2, 폭 1280 기준 픽셀로 환산 (중앙값)."""
    if mask.sum() < 20:
        return np.nan
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    sk = skeletonize(mask)
    return float(np.median(2 * dt[sk]) * s) if sk.any() else np.nan


def morph(lab, W):
    s = W_REF / W
    out = {f'frac_{c}': float((lab == i).mean()) for i, c in enumerate(CLS)}
    sie, ni, sip = lab == 3, lab == 1, lab == 2
    out['SiE_thick'] = thickness(sie, s)
    out['Al3Ni_thick'] = thickness(ni, s)
    # 공정 Si의 세밀함: 경계 길이 / 면적 (1280 기준 1/px). 클수록 가늘고 촘촘하다
    per = cv2.Canny(sie.astype(np.uint8) * 255, 50, 150) > 0
    out['SiE_fineness'] = float(per.sum() / max(sie.sum(), 1) / s)
    n, cc, st, _ = cv2.connectedComponentsWithStats(sip.astype(np.uint8), connectivity=8)
    areas = st[1:, 4][st[1:, 4] >= 50 / s ** 2] * s ** 2
    out['SiP_eqdiam'] = float(np.median(np.sqrt(4 * areas / np.pi))) if len(areas) else np.nan
    out['SiP_count_norm'] = float(len(areas))
    return out


def _one(n):
    it = G['items'][n]
    gt = it['lab']
    pred = postprocess(np.load(Path(G['ens']) / f'{n}.npy').astype(np.float32))
    W = it['gray'].shape[1]
    iou = image_iou(gt, pred) * 100
    return n, {'alloy': it['alloy'], 'step': it['step'], 'W': W, 'low': it['low_contrast'], 'iou': iou.tolist(),
               'gt': morph(gt, W), 'pred': morph(pred, W)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ens', default='runs/oof/_ens_A27_sie_al')
    p.add_argument('--out', default='runs/report')
    p.add_argument('--procs', type=int, default=24)
    args = p.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    items = load_split('data', 'train') + load_split('data', 'valid')
    G.update(ens=args.ens, items={it['name']: it for it in items})
    with Pool(args.procs) as pool:
        R = dict(pool.map(_one, [it['name'] for it in items]))
    names = sorted(R)
    res = {}

    # 1) 상 분율 오차 (Si는 Primary + Eutectic 합도 본다)
    fr = {}
    for c in CLS + ['Si']:
        key = lambda d: d['frac_SiP'] + d['frac_SiE'] if c == 'Si' else d[f'frac_{c}']
        g = np.array([key(R[n]['gt']) for n in names]) * 100
        q = np.array([key(R[n]['pred']) for n in names]) * 100
        ok = g > 0.5   # 거의 없는 상(Primary Si 없는 사진)은 상대 오차에서 뺀다
        rel = np.abs(q - g)[ok] / g[ok] * 100
        fr[c] = {'gt_mean_vol%': round(float(g.mean()), 2), 'pred_mean_vol%': round(float(q.mean()), 2),
                 'abs_err_mean_%p': round(float(np.abs(q - g).mean()), 2), 'abs_err_max_%p': round(float(np.abs(q - g).max()), 2),
                 'bias_mean_%p': round(float((q - g).mean()), 2), 'rel_err_mean_%': round(float(rel.mean()), 1),
                 'rel_err_median_%': round(float(np.median(rel)), 1), 'images_within_10%RA': f'{int((rel <= 10).sum())}/{int(ok.sum())}',
                 'mean_IoU': round(float(np.nanmean([R[n]['iou'][CLS.index(c)] for n in names])), 2) if c in CLS else None}
    res['1_phase_fraction'] = fr

    # 2) 탄성률: 정답 분율 vs 예측 분율로 계산한 HS 범위 중간값
    mod = {}
    for e in AL3NI_E:
        gm = np.array([np.mean(modulus([R[n]['gt'][f'frac_{c}'] for c in CLS], e)) for n in names])
        pm = np.array([np.mean(modulus([R[n]['pred'][f'frac_{c}'] for c in CLS], e)) for n in names])
        lo, hi = np.mean([modulus([R[n]['gt'][f'frac_{c}'] for c in CLS], e) for n in names], 0)
        mod[f'E_Al3Ni={e:.0f}'] = {'gt_HS_mean_range_GPa': [round(float(lo), 1), round(float(hi), 1)],
                                   'gt_mid_mean_GPa': round(float(gm.mean()), 1), 'gt_mid_sd_between_images': round(float(gm.std()), 2),
                                   'pred_minus_gt_mean_GPa': round(float((pm - gm).mean()), 2),
                                   'pred_minus_gt_abs_mean_GPa': round(float(np.abs(pm - gm).mean()), 2),
                                   'pred_minus_gt_abs_max_GPa': round(float(np.abs(pm - gm).max()), 2)}
    res['2_modulus_HS'] = mod

    # 3) step별 크기 지표: 같은 합금 안에서 step 순서와의 순위 상관 (정답·예측 각각)
    keys = ['frac_Al3Ni', 'frac_SiE', 'frac_SiP', 'SiE_thick', 'Al3Ni_thick', 'SiE_fineness', 'SiP_eqdiam']
    alloys = sorted({R[n]['alloy'] for n in names})
    steps = {}
    for src in ('gt', 'pred'):
        per_key = {}
        for k in keys:
            rhos, signs = [], []
            for a in alloys:
                rows = [(R[n]['step'], R[n][src][k]) for n in names if R[n]['alloy'] == a and not np.isnan(R[n][src][k])]
                st = sorted({s for s, _ in rows})
                if len(st) < 2:
                    continue
                means = [np.mean([v for s2, v in rows if s2 == s]) for s in st]
                if len(st) >= 3:
                    rhos.append(spearmanr(st, means).correlation)
                signs.append(np.sign(means[-1] - means[0]))
            per_key[k] = {'alloys_with_2+steps': len(signs), 'up_from_first_to_last_step': f'{int((np.array(signs) > 0).sum())}/{len(signs)}',
                          'spearman_mean(alloys with 3+ steps)': round(float(np.nanmean(rhos)), 2) if rhos else None}
        steps[src] = per_key
    by_step = {}
    for s in sorted({R[n]['step'] for n in names}):
        ns = [n for n in names if R[n]['step'] == s]
        by_step[f'step{s}'] = {'n': len(ns), **{k: round(float(np.nanmean([R[n]['gt'][k] for n in ns])), 4) for k in keys}}
    steps['gt_mean_by_step'] = by_step
    res['3_steps'] = steps
    Path(args.out, 'report_analysis.json').write_text(json.dumps(res, indent=1, ensure_ascii=False))
    Path(args.out, 'per_image.json').write_text(json.dumps(R, ensure_ascii=False))
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
