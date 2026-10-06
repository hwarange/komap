"""sub4 제출 코드 검증 (90장): 제출 코드(submit.py predict --split cv)로 만든 예측을 섞어 채점한다.
각 사진은 그 사진을 학습하지 않은 fold 모델만 쓴다. 기대값: 앙상블만 79.936 (sub3 검증), 섞은 뒤 80.068 (decision_rules, α=0.3).
Si 분류기는 90장 전체로 학습한 것이라 여기 점수는 부풀려진 참고값이다 (정직한 효과는 si_classifier.py cv).

예) python src/verify_cv_blend.py --ens submissions/sub3_cv --sie submissions/sub4_cv_sie --alpha 0.3 --si-clf runs/si_clf/sub4_model.pkl
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from data import load_split, normalize
from infer import image_iou, postprocess
from si_classifier import apply_selective, components
from submit import blend_sie


def shard_mean(root, n):
    total, c = 0, 0
    for s in sorted(Path(root, 'parts').glob('shard*')):
        f = s / f'{n}.npy'
        if f.exists():
            total = total + np.load(f)
            c += json.loads((s / 'counts.json').read_text())[n]
    return (total / c if c else None), c


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ens', required=True)
    p.add_argument('--sie', required=True)
    p.add_argument('--alpha', type=float, default=0.3)
    p.add_argument('--si-clf', default='')
    p.add_argument('--si-thr', type=float, default=0.8)
    args = p.parse_args()
    si = pickle.loads(Path(args.si_clf).read_bytes())['clf'] if args.si_clf else None
    items = load_split('data', 'train') + load_split('data', 'valid')
    s_ens, s_blend, s_si, counts = [], [], [], set()
    for it in items:
        n = it['name']
        P, ca = shard_mean(args.ens, n)
        S, cs = shard_mean(args.sie, n)
        counts.add((ca, cs))
        s_ens.append(np.nanmean(image_iou(it['lab'], postprocess(P))) * 100)
        Q = blend_sie(P, S[1], args.alpha)
        pred = postprocess(Q)
        s_blend.append(np.nanmean(image_iou(it['lab'], pred)) * 100)
        if si is not None:
            X, cc, _ = components(Q, pred, normalize(it['gray']), it['gray'].shape[1], it['low_contrast'])
            if len(X):
                pred = apply_selective(pred, cc, si.predict_proba(X)[:, 1], args.si_thr)
            s_si.append(np.nanmean(image_iou(it['lab'], pred)) * 100)
    out = {'models_per_image (ensemble, specialist)': sorted(counts), 'ensemble_only': round(float(np.mean(s_ens)), 3),
           f'blend_alpha{args.alpha}': round(float(np.mean(s_blend)), 3)}
    if s_si:
        out['blend + Si clf (in-sample, optimistic)'] = round(float(np.mean(s_si)), 3)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
