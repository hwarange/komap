"""교차검증으로 학습된 모델(fold별 last.pt)로 각 fold의 검증 사진을 다시 예측해 확률을 저장한다.
이렇게 모은 '학습에 쓰이지 않은 사진에 대한 예측(OOF)'으로 앙상블과 후처리를 공정하게 평가할 수 있다.

예) CUDA_VISIBLE_DEVICES=0 python src/oof.py convnext_t unetpp
저장: runs/oof/<실험>/<사진이름>.npy  (float16, [4, H, W], TTA 적용)
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from data import load_cv
from train import autocast, build_from_args
from infer import predict_probs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('exps', nargs='+')
    p.add_argument('--root', default='runs/cv')
    p.add_argument('--out', default='runs/oof')
    p.add_argument('--folds', type=int, default=5)
    args = p.parse_args()

    for exp in args.exps:
        out = Path(args.out) / exp
        out.mkdir(parents=True, exist_ok=True)
        for k in range(args.folds):
            ck = torch.load(Path(args.root) / exp / f'fold{k}' / 'last.pt', map_location='cpu', weights_only=False)
            a = ck['args']
            model = build_from_args(a)
            model.load_state_dict(ck['model'])
            model = model.cuda().eval()
            _, val = load_cv(a['data'], k, a['folds_file'])
            for it in val:
                with torch.no_grad(), autocast(a.get('amp', False)):
                    prob = predict_probs(model, it['gray'], a['width'], a['tile'], a['overlap'], True, 'cuda', scale_multiplier=a.get('scale_multiplier', 1.0))
                np.save(out / f"{it['name']}.npy", prob.astype(np.float16))
            print(f'{exp} fold{k}: {len(val)} images', flush=True)


if __name__ == '__main__':
    main()
