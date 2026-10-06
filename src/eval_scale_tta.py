"""6단계 A2: 크기 키운 TTA 평가 (학습 없음, OOF).
각 fold 모델로 그 fold의 검증 사진을 원래 배율과 1.25배(각 8방향 TTA)로 예측해
'원래만', '1.25배만', '둘의 평균'을 비교한다. 축소는 1~2px 구조를 지우므로 시험하지 않는다.

예) CUDA_VISIBLE_DEVICES=0 python src/eval_scale_tta.py unetpp_bw_e120_r90 cnx_bw --out runs/scale_tta.json
"""
import argparse
import json

import numpy as np
import torch

from data import load_cv
from infer import image_iou, postprocess, predict_probs
from train import autocast, build_from_args


def main():
    p = argparse.ArgumentParser()
    p.add_argument('exps', nargs='+')
    p.add_argument('--scale', type=float, default=1.25)
    p.add_argument('--out', default='runs/scale_tta.json')
    args = p.parse_args()
    res = {}
    for exp in args.exps:
        s1, s2, both, low = [], [], [], []
        for k in range(5):
            ck = torch.load(f'runs/cv/{exp}/fold{k}/last.pt', map_location='cpu', weights_only=False)
            a = ck['args']
            model = build_from_args(a)
            model.load_state_dict(ck['model'])
            model = model.cuda().eval()
            _, val = load_cv(a['data'], k, a['folds_file'])
            for it in val:
                with torch.no_grad(), autocast(a.get('amp', False)):
                    p1 = predict_probs(model, it['gray'], a['width'], a['tile'], a['overlap'], True, 'cuda')
                    p2 = predict_probs(model, it['gray'], a['width'], a['tile'], a['overlap'], True, 'cuda', extra_scale=args.scale)
                for lst, pr in [(s1, p1), (s2, p2), (both, (p1 + p2) / 2)]:
                    lst.append(np.nanmean(image_iou(it['lab'], postprocess(pr))) * 100)
                low.append(it['low_contrast'])
            print(f'{exp} fold{k} done', flush=True)
        s1, s2, both, low = map(np.array, (s1, s2, both, low))
        d = both - s1
        idx = np.random.default_rng(0).integers(0, len(d), (5000, len(d)))
        lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
        res[exp] = {'orig': round(float(s1.mean()), 3), f'x{args.scale}': round(float(s2.mean()), 3),
                    'avg': round(float(both.mean()), 3), 'delta_avg_vs_orig': round(float(d.mean()), 3),
                    'ci95': [round(float(lo), 3), round(float(hi), 3)], 'improved': f'{int((d > 0).sum())}/{len(d)}',
                    'delta_low_contrast': round(float(d[low].mean()), 3)}
        print(exp, json.dumps(res[exp]), flush=True)
    with open(args.out, 'w') as f:
        json.dump(res, f, indent=1)


if __name__ == '__main__':
    main()
