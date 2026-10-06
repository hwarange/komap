"""검증 세트 오답 분석: 사진별 점수, 혼동 행렬, 경계 오차 비율, 오답 지도.

예) python src/analyze.py runs/baseline/best.pt
결과는 체크포인트 폴더의 analysis/ 에 저장된다.
"""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from data import CLASSES, label_to_rgb, load_split
from infer import image_iou, postprocess, predict_probs
from train import autocast, build_from_args


def boundary_dist(lab):
    """각 픽셀에서 가장 가까운 정답 경계까지의 거리(px)."""
    b = np.zeros(lab.shape, bool)
    b[:, 1:] |= lab[:, 1:] != lab[:, :-1]
    b[1:, :] |= lab[1:, :] != lab[:-1, :]
    return cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2, 3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('ckpt')
    p.add_argument('--data', default='data')
    p.add_argument('--split', default='valid')
    p.add_argument('--worst', type=int, default=3)
    args = p.parse_args()

    out = Path(args.ckpt).parent / 'analysis'
    out.mkdir(exist_ok=True)
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    a = ck['args']
    model = build_from_args(a)
    model.load_state_dict(ck['model'])
    model = model.cuda().eval()

    items = load_split(args.data, args.split)
    rows, conf, preds = [], np.zeros((4, 4), np.int64), {}
    err_d = []
    for it in items:
        gt = it['lab']
        with autocast(a.get('amp', False)):
            prob_plain = predict_probs(model, it['gray'], a['width'], a['tile'], a['overlap'], False, 'cuda')
            prob_tta = predict_probs(model, it['gray'], a['width'], a['tile'], a['overlap'], True, 'cuda')
        variants = {'plain': prob_plain.argmax(0), 'pp': postprocess(prob_plain),
                    'tta': prob_tta.argmax(0), 'tta+pp': postprocess(prob_tta)}
        r = dict(name=it['name'], step=it['step'], W=it['gray'].shape[1], low=it['low_contrast'])
        for k, v in variants.items():
            r[k] = np.nanmean(image_iou(gt, v)) * 100
        best = variants['tta+pp']
        r.update({c: v * 100 for c, v in zip(CLASSES, image_iou(gt, best))})
        rows.append(r)
        preds[it['name']] = best
        conf += np.bincount(gt.ravel().astype(np.int64) * 4 + best.ravel(), minlength=16).reshape(4, 4)
        wrong = best != gt
        err_d.append(boundary_dist(gt)[wrong])
        print(f"{it['name']:24s} plain {r['plain']:5.1f}  tta+pp {r['tta+pp']:5.1f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / 'per_image.csv', index=False, encoding='utf-8-sig')
    summary = {k: round(df[k].mean(), 2) for k in ['plain', 'pp', 'tta', 'tta+pp']}
    summary.update({f'class_{c}': round(np.nanmean(df[c]), 2) for c in CLASSES})
    for flag, name in [(True, 'low_contrast'), (False, 'high_contrast')]:
        summary[name] = round(df.loc[df.low == flag, 'tta+pp'].mean(), 2)
    d = np.concatenate(err_d)
    summary['error_px_within_1px_of_boundary_%'] = round(float((d <= 1).mean() * 100), 1)
    summary['error_px_within_2px_of_boundary_%'] = round(float((d <= 2).mean() * 100), 1)
    summary['error_px_farther_than_3px_%'] = round(float((d > 3).mean() * 100), 1)
    cm = pd.DataFrame(conf / conf.sum(1, keepdims=True) * 100, index=[f'GT {c}' for c in CLASSES], columns=[f'pred {c}' for c in CLASSES])
    cm.round(2).to_csv(out / 'confusion_row_pct.csv', encoding='utf-8-sig')
    (out / 'summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    print(cm.round(2).to_string())

    # 가장 못 맞힌 사진의 오답 지도: 원본 | 정답 | 예측 | 틀린 픽셀(빨강)
    by_name = {it['name']: it for it in items}
    for name in df.nsmallest(args.worst, 'tta+pp')['name']:
        it, pr = by_name[name], preds[name]
        err = np.stack([it['gray']] * 3, -1).copy()
        err[pr != it['lab']] = (220, 40, 40)
        fig, axes = plt.subplots(1, 4, figsize=(20, 4.2))
        for ax, im, t in zip(axes, [it['gray'], label_to_rgb(it['lab']), label_to_rgb(pr), err], ['image', 'ground truth', 'prediction', 'errors (red)']):
            ax.imshow(im, cmap='gray' if im.ndim == 2 else None)
            ax.set_title(t)
            ax.axis('off')
        fig.suptitle(f"{name}  mIoU {df.set_index('name').loc[name, 'tta+pp']:.1f}")
        fig.tight_layout()
        fig.savefig(out / f'worst_{name}.png', dpi=110)
        plt.close(fig)
    print(f'saved -> {out}')


if __name__ == '__main__':
    main()
