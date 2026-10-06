"""Si 덩어리 분류기 (2단계): 앙상블이 예측한 Si(Primary + Eutectic) 영역을 연결 덩어리로 묶고,
덩어리마다 '모양·주변·확률' 특징으로 Primary Si인지 Eutectic Si인지 다시 판정한다.

동기 (OOF 진단, 2026-10-06)
- 경계에서 떨어진 Eutectic Si 픽셀의 15.8%를 Primary Si로 예측. 두 상은 같은 Si 결정이라 밝기로 구분 불가.
- 덩어리마다 정답 다수결로 배정하는 상한: 79.73 → 82.33.
금속학 근거: Primary Si는 크고 각진 다면체이며 α-Al 할로에 둘러싸이고, Eutectic Si는 가늘고 불규칙하다.

평가 (cv): fold k의 사진은 나머지 fold의 덩어리로 학습한 분류기로 판정 (합금 단위 fold, 누수 없음)
예) python src/si_classifier.py cv --exps A B C
    python src/si_classifier.py fit --exps A B C --out runs/si_clf/model.pkl   (90장 전체로 학습, 제출용)
"""
import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from data import load_split, normalize, scale_for
from infer import image_iou, postprocess

FEATS = ['log_area', 'max_thick', 'mean_thick', 'compact', 'solidity', 'extent', 'aspect', 'n_vertices',
         'frac_pred_sip', 'mean_p2', 'mean_p3', 'mean_margin', 'max_p2', 'gray_mean', 'gray_std',
         'ring_al', 'ring_al3ni', 'ring_si', 'ring_gray', 'img_sip_frac', 'img_sie_frac', 'img_low', 'img_small']


def components(P, pred, img, W, low):
    """예측된 Si 덩어리마다 특징 벡터와 픽셀 마스크(라벨 번호)를 돌려준다."""
    s = scale_for(W, 1280)                     # 675px은 2배로 처리하므로 크기를 같은 배율로 맞춘다
    si = np.isin(pred, (2, 3)).astype(np.uint8)
    k, cc, st, _ = cv2.connectedComponentsWithStats(si, connectivity=8)
    dt = cv2.distanceTransform(si, cv2.DIST_L2, 3)
    img_sip, img_sie = (pred == 2).mean(), (pred == 3).mean()
    ring_kernel = np.ones((7, 7), np.uint8)
    feats = []
    for j in range(1, k):
        x, y, w, h, area = st[j]
        x0, y0, x1, y1 = max(x - 4, 0), max(y - 4, 0), min(x + w + 4, pred.shape[1]), min(y + h + 4, pred.shape[0])
        m = cc[y0:y1, x0:x1] == j
        mu8 = m.astype(np.uint8)
        cnts, _ = cv2.findContours(mu8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cnts, key=cv2.contourArea)
        perim = max(cv2.arcLength(c, True), 1.0)
        hull = max(cv2.contourArea(cv2.convexHull(c)), 1.0)
        (_, (rw, rh), _) = cv2.minAreaRect(c)
        approx = cv2.approxPolyDP(c, 0.02 * perim, True)
        ring = (cv2.dilate(mu8, ring_kernel) > 0) & ~m
        pr_ring = pred[y0:y1, x0:x1][ring]
        Pj = P[:, y0:y1, x0:x1][:, m]
        g = img[y0:y1, x0:x1]
        d = dt[y0:y1, x0:x1][m]
        feats.append([
            np.log1p(area * s * s), 2 * d.max() * s, 2 * d.mean() * s, 4 * np.pi * area / perim ** 2,
            area / hull, area / max(w * h, 1), max(rw, rh) / max(min(rw, rh), 1.0), len(approx),
            (pred[y0:y1, x0:x1][m] == 2).mean(), Pj[2].mean(), Pj[3].mean(), (Pj[2] - Pj[3]).mean(), Pj[2].max(),
            g[m].mean(), g[m].std(),
            (pr_ring == 0).mean() if pr_ring.size else 0, (pr_ring == 1).mean() if pr_ring.size else 0,
            np.isin(pr_ring, (2, 3)).mean() if pr_ring.size else 0, g[ring].mean() if ring.any() else 0,
            img_sip, img_sie, float(low), float(W == 675)])
    return np.array(feats, np.float32), cc, k


def build_table(exps, oof_root, folds_file):
    items = load_split('data', 'train') + load_split('data', 'valid')
    fold_of = json.loads(Path(folds_file).read_text())
    table = []
    for it in items:
        n, gt = it['name'], it['lab']
        P = np.mean([np.load(Path(oof_root) / e / f'{n}.npy').astype(np.float32) for e in exps], 0)
        pred = postprocess(P)
        X, cc, k = components(P, pred, normalize(it['gray']), it['gray'].shape[1], it['low_contrast'])
        flat = cc.ravel()
        g2 = np.bincount(flat, weights=(gt == 2).ravel().astype(float), minlength=k)[1:]
        g3 = np.bincount(flat, weights=(gt == 3).ravel().astype(float), minlength=k)[1:]
        table.append(dict(name=n, fold=fold_of[n], gt=gt, pred=pred, cc=cc, X=X,
                          y=(g2 > g3).astype(int), w=g2 + g3))
    return table


def make_clf():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                          min_samples_leaf=40, l2_regularization=1.0, random_state=0)


def apply(pred, cc, prob_sip, thr=0.5):
    out = pred.copy()
    choose = np.concatenate([[0], np.where(prob_sip > thr, 2, 3)])
    m = cc > 0
    out[m] = choose[cc[m]]
    return out


def apply_selective(pred, cc, prob_sip, thr=0.8):
    """분류기가 확신할 때만 덩어리 전체를 그 상으로 바꾼다 (p>=thr → Primary Si, p<=1-thr → Eutectic Si).
    모든 덩어리에 적용하면 애매한 덩어리에서 손해가 나서 이득이 상쇄된다 (OOF: +0.02 vs 확신 80% 기준 +0.09)."""
    choose = np.zeros(len(prob_sip) + 1, np.uint8)
    choose[1:][prob_sip >= thr] = 2
    choose[1:][prob_sip <= 1 - thr] = 3
    out = pred.copy()
    m = (cc > 0) & (choose[cc] > 0)
    out[m] = choose[cc[m]]
    return out


def cmd_cv(args):
    tab = build_table(args.exps, args.oof, args.folds_file)
    base, new = [], []
    cls_b, cls_n = [], []
    for k in range(5):
        tr = [t for t in tab if t['fold'] != k]
        X = np.concatenate([t['X'] for t in tr])
        y = np.concatenate([t['y'] for t in tr])
        w = np.concatenate([t['w'] for t in tr])
        keep = w > 0                              # 정답에 Si가 없는 덩어리는 학습에서 제외
        clf = make_clf().fit(X[keep], y[keep], sample_weight=w[keep])
        for t in tab:
            if t['fold'] != k:
                continue
            p = clf.predict_proba(t['X'])[:, 1] if len(t['X']) else np.zeros(0)
            nb, nn = image_iou(t['gt'], t['pred']), image_iou(t['gt'], apply_selective(t['pred'], t['cc'], p, args.thr))
            base.append(np.nanmean(nb) * 100); new.append(np.nanmean(nn) * 100)
            cls_b.append(nb * 100); cls_n.append(nn * 100)
    base, new = np.array(base), np.array(new)
    cb, cn = np.nanmean(np.array(cls_b), 0), np.nanmean(np.array(cls_n), 0)
    d = new - base
    idx = np.random.default_rng(0).integers(0, len(d), (5000, len(d)))
    lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
    print(f'ensemble        mIoU {base.mean():.3f} | SiP {cb[2]:.2f} SiE {cb[3]:.2f}')
    print(f'(threshold {args.thr})')
    print(f'+ Si classifier mIoU {new.mean():.3f} | SiP {cn[2]:.2f} SiE {cn[3]:.2f} | delta {d.mean():+.3f} '
          f'[{lo:+.3f}, {hi:+.3f}] better {(d > 0).sum()}/90 worse {(d < 0).sum()}/90')
    clf = make_clf().fit(np.concatenate([t['X'] for t in tab]), np.concatenate([t['y'] for t in tab]),
                         sample_weight=np.concatenate([t['w'] for t in tab]))
    n_comp = sum(len(t['X']) for t in tab)
    print(f'components: {n_comp} | class balance (area-weighted SiP share): '
          f'{np.concatenate([t["w"] * t["y"] for t in tab]).sum() / np.concatenate([t["w"] for t in tab]).sum():.3f}')


def cmd_fit(args):
    tab = build_table(args.exps, args.oof, args.folds_file)
    X = np.concatenate([t['X'] for t in tab]); y = np.concatenate([t['y'] for t in tab])
    w = np.concatenate([t['w'] for t in tab]); keep = w > 0
    clf = make_clf().fit(X[keep], y[keep], sample_weight=w[keep])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_bytes(pickle.dumps({'clf': clf, 'feats': FEATS, 'exps': args.exps}))
    print('saved', args.out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['cv', 'fit'])
    p.add_argument('--exps', nargs='+', required=True)
    p.add_argument('--oof', default='runs/oof')
    p.add_argument('--folds-file', default='splits/folds5.json')
    p.add_argument('--out', default='runs/si_clf/model.pkl')
    p.add_argument('--thr', type=float, default=0.8, help='확신 기준 (0.5면 모든 덩어리에 적용)')
    args = p.parse_args()
    cmd_cv(args) if args.cmd == 'cv' else cmd_fit(args)


if __name__ == '__main__':
    main()
