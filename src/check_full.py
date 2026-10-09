"""90장 전체 학습 모델 안전 점검 (6단계 C, 검증 사진이 없으므로 검증 대신).

1) train-score: 학습한 사진 10장(90장 중 9장 간격)을 예측해 점수를 낸다. 학습한 사진이므로
   같은 사진의 OOF 점수(그 사진을 안 본 fold 모델)보다 높아야 정상이다. 낮으면 학습이 깨진 것.
   CUDA_VISIBLE_DEVICES=0 python src/check_full.py train-score --exps full_A full_B --out runs/full/check_train.json
2) agree: 평가 사진에서 모델끼리 예측이 다른 픽셀 비율. 시드 3개끼리의 불일치가
   같은 구성의 fold 모델끼리의 불일치와 비슷하면 정상이다.
   python src/check_full.py agree --parts submissions/sub3_pred --exps full_A A
3) diff: 두 제출본의 마스크가 사진마다 몇 % 다른지 (sub2와 2% 이상 다르면 원인 확인).
   python src/check_full.py diff submissions/sub2 submissions/sub3a_full
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from PIL import Image


def cmd_train_score(args):
    import torch
    from data import load_split
    from infer import image_iou, postprocess, predict_probs
    from train import autocast, build_from_args
    items = (load_split('data', 'train') + load_split('data', 'valid'))[::9]
    res = {}
    for e in args.exps:
        base = e.removeprefix('full_')
        oof = [np.nanmean(image_iou(it['lab'], postprocess(np.load(Path(args.oof) / base / f"{it['name']}.npy").astype(np.float32)))) * 100
               for it in items]
        seeds = {}
        for ck_path in sorted(Path(args.root, e).glob('fold*/last.pt')):
            ck = torch.load(ck_path, map_location='cpu', weights_only=False)
            a = ck['args']
            model = build_from_args(a)
            model.load_state_dict(ck['model'])
            model = model.cuda().eval()
            sc = []
            for it in items:
                with torch.no_grad(), autocast(a.get('amp', False)):
                    p = predict_probs(model, it['gray'], a['width'], a['tile'], a['overlap'], True, 'cuda', scale_multiplier=a.get('scale_multiplier', 1.0))
                sc.append(np.nanmean(image_iou(it['lab'], postprocess(p))) * 100)
            seeds[ck_path.parent.name] = round(float(np.mean(sc)), 2)
            print(e, ck_path.parent.name, seeds[ck_path.parent.name], flush=True)
            del model
            torch.cuda.empty_cache()
        res[e] = {'train_imgs_score_per_seed': seeds, 'oof_score_same_imgs': round(float(np.mean(oof)), 2),
                  'ok': bool(seeds) and bool(min(seeds.values()) > np.mean(oof))}
        print(e, json.dumps(res[e]), flush=True)
    Path(args.out).write_text(json.dumps({'images': [it['name'] for it in items], 'results': res}, indent=1))


def cmd_agree(args):
    mroot = Path(args.parts) / 'parts' / 'models'
    names = sorted(f.stem for f in (mroot / args.exps[0]).glob('fold0/*.npy'))
    res = {}
    for e in args.exps:
        folds = sorted((mroot / e).glob('fold*'))
        per_img = []
        for n in names:
            preds = [np.load(d / f'{n}.npy').argmax(0) for d in folds]
            per_img.append(np.mean([(a != b).mean() * 100 for a, b in itertools.combinations(preds, 2)]))
        res[e] = {'n_models': len(folds), 'pairwise_disagree_pct_mean': round(float(np.mean(per_img)), 3),
                  'max_image': round(float(np.max(per_img)), 3)}
        print(e, json.dumps(res[e]), flush=True)


def cmd_diff(args):
    a_dir, b_dir = Path(args.a) / 'masks', Path(args.b) / 'masks'
    res = {}
    for f in sorted(a_dir.glob('*_mask.png')):
        a, b = np.array(Image.open(f)), np.array(Image.open(b_dir / f.name))
        res[f.name.removesuffix('_mask.png')] = round(float((a != b).any(-1).mean() * 100), 4)
    v = np.array(list(res.values()))
    print(json.dumps({'per_image_pct': res, 'mean': round(float(v.mean()), 4), 'max': round(float(v.max()), 4),
                      'images_over_2pct': int((v >= 2).sum())}, indent=1))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('train-score')
    t.add_argument('--exps', nargs='+', required=True)
    t.add_argument('--root', default='runs/full')
    t.add_argument('--oof', default='runs/oof')
    t.add_argument('--out', default='runs/full/check_train.json')
    g = sub.add_parser('agree')
    g.add_argument('--parts', required=True)
    g.add_argument('--exps', nargs='+', required=True)
    d = sub.add_parser('diff')
    d.add_argument('a')
    d.add_argument('b')
    args = p.parse_args()
    {'train-score': cmd_train_score, 'agree': cmd_agree, 'diff': cmd_diff}[args.cmd](args)


if __name__ == '__main__':
    main()
