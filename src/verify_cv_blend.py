"""제출 코드 검증 (90장): 제출 코드(submit.py predict --split cv)로 만든 예측을 섞어 채점한다.
각 사진은 그 사진을 학습하지 않은 fold 모델만 쓴다. 기대값은 같은 구성의 OOF 평가(ensemble.py, decision_rules.py)와 같아야 한다.
Si 분류기는 90장 전체로 학습한 것이라 여기 점수는 부풀려진 참고값이다 (정직한 효과는 si_classifier.py cv).

--ens 폴더가 여러 개면 모델 수로 가중해 합친다 (예: 25개 실험 폴더 + 새 실험 2개 폴더 = 27개 평균).
--sie 폴더들은 전용 모델끼리 같은 비중으로 평균 (decision_rules.spec_prob와 같음).
예) python src/verify_cv_blend.py --ens submissions/sub3_cv submissions/sub5_cv_ens \
      --sie submissions/sub4_cv_sie submissions/sub5_cv_sie2 --alpha 0.4 \
      --al3ni submissions/sub5_cv_al3ni --alpha-al3ni 0.3 --si-clf runs/si_clf/sub5_model.pkl
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from data import load_split, normalize
from infer import image_iou, postprocess
from si_classifier import apply_selective, components
from submit import blend_cls


def shard_sum(root, n):
    total, c = 0, 0
    for s in sorted(Path(root, 'parts').glob('shard*')):
        f = s / f'{n}.npy'
        if f.exists():
            total = total + np.load(f)
            c += json.loads((s / 'counts.json').read_text())[n]
    return total, c


def pooled_mean(roots, n):
    """여러 폴더의 모델 합을 모아 전체 모델 수로 나눈다."""
    parts = [shard_sum(r, n) for r in roots]
    c = sum(p[1] for p in parts)
    return sum(p[0] for p in parts) / c, c


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ens', nargs='+', required=True)
    p.add_argument('--sie', nargs='+', required=True, help='Eutectic Si 전용 모델 폴더 (전용 모델별로 하나씩)')
    p.add_argument('--alpha', type=float, default=0.3)
    p.add_argument('--al3ni', nargs='*', default=[], help='Al3Ni 전용 모델 폴더')
    p.add_argument('--alpha-al3ni', type=float, default=0.3)
    p.add_argument('--si-clf', default='')
    p.add_argument('--si-thr', type=float, default=0.8)
    args = p.parse_args()
    si = pickle.loads(Path(args.si_clf).read_bytes())['clf'] if args.si_clf else None
    items = load_split('data', 'train') + load_split('data', 'valid')
    s_ens, s_sie, s_al, s_si, counts = [], [], [], [], set()
    for it in items:
        n = it['name']
        P, ca = pooled_mean(args.ens, n)
        qs = [pooled_mean([r], n) for r in args.sie]
        counts.add((ca, tuple(c for _, c in qs)))
        s_ens.append(np.nanmean(image_iou(it['lab'], postprocess(P))) * 100)
        Q = blend_cls(P, np.mean([q[1] for q, _ in qs], 0), args.alpha, 3)
        s_sie.append(np.nanmean(image_iou(it['lab'], postprocess(Q))) * 100)
        if args.al3ni:
            qa = np.mean([pooled_mean([r], n)[0][1] for r in args.al3ni], 0)
            Q = blend_cls(Q, qa, args.alpha_al3ni, 1)
            s_al.append(np.nanmean(image_iou(it['lab'], postprocess(Q))) * 100)
        pred = postprocess(Q)
        if si is not None:
            X, cc, _ = components(Q, pred, normalize(it['gray']), it['gray'].shape[1], it['low_contrast'])
            if len(X):
                pred = apply_selective(pred, cc, si.predict_proba(X)[:, 1], args.si_thr)
            s_si.append(np.nanmean(image_iou(it['lab'], pred)) * 100)
    out = {'models_per_image (ensemble, specialists)': sorted(counts), 'ensemble_only': round(float(np.mean(s_ens)), 3),
           f'+ sie blend {args.alpha}': round(float(np.mean(s_sie)), 3)}
    if s_al:
        out[f'+ al3ni blend {args.alpha_al3ni}'] = round(float(np.mean(s_al)), 3)
    if s_si:
        out['+ Si clf (in-sample, optimistic)'] = round(float(np.mean(s_si)), 3)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
