"""학습된 모델로 예측·채점하고 마스크를 대회 형식(4색 RGB PNG)으로 저장.

예) python src/predict.py runs/baseline/best.pt --split valid --tta
    python src/predict.py runs/baseline/best.pt --split test --tta --save preds/test
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data import label_to_rgb, load_split
from infer import image_iou, postprocess, predict_probs
from train import autocast, build_from_args, summarize


def main():
    p = argparse.ArgumentParser()
    p.add_argument('ckpt')
    p.add_argument('--data', default='data')
    p.add_argument('--split', default='valid')
    p.add_argument('--tta', action='store_true')
    p.add_argument('--no-pp', action='store_true', help='Primary Si 후처리 끄기')
    p.add_argument('--save', default='', help='예측 마스크를 저장할 폴더')
    args = p.parse_args()

    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    a = ck['args']
    model = build_from_args(a)
    model.load_state_dict(ck['model'])
    model = model.cuda().eval()

    items = load_split(args.data, args.split)
    rows = []
    save = Path(args.save) if args.save else None
    if save:
        save.mkdir(parents=True, exist_ok=True)
    for it in items:
        with autocast(a.get('amp', False)):
            prob = predict_probs(model, it['gray'], a['width'], a['tile'], a['overlap'], args.tta, 'cuda')
        pred = prob.argmax(0).astype(np.uint8) if args.no_pp else postprocess(prob)
        if save:
            Image.fromarray(label_to_rgb(pred)).save(save / f"{it['name']}_mask.png")
        if it['lab'] is not None:
            rows.append(dict(name=it['name'], step=it['step'], low=it['low_contrast'],
                             iou=image_iou(it['lab'], pred)))
            print(f"{it['name']:24s} mIoU {np.nanmean(rows[-1]['iou']) * 100:5.1f}", flush=True)

    if rows:
        print(json.dumps(summarize(rows), ensure_ascii=False, indent=1))
    if save:
        print(f'saved {len(items)} masks -> {save}')


if __name__ == '__main__':
    main()
