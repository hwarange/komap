"""9단계: 사진별 '안쪽 오답' 지도 만들기 (오답 위치 중심 학습 `train.py --hard-crop`용).

앙상블 OOF 예측이 틀린 픽셀 중 정답 경계 1px 띠 밖에 있는 것만 남긴다.
경계 띠 안의 오답은 대부분 라벨 애매함이라(경계 가중 5배·경계 부드러운 정답 모두 실패) 빼고,
조각째 다른 상으로 칠한 '실제 혼동'만 학습 위치로 쓴다.
주의: OOF 예측은 다른 fold 모델이 만든 것이라, 그 모델이 본 사진의 정보가 '어디를 자를지'에 약하게 섞인다
(정답 라벨은 그대로이므로 영향은 작다).

예) python src/make_hardmaps.py --ens runs/oof/_ens_A_r6 --out runs/hardmaps
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from data import load_split
from infer import postprocess

G = {}
K = np.ones((3, 3), np.uint8)


def _one(n):
    gt = G['labs'][n]
    pred = postprocess(np.load(Path(G['ens']) / f'{n}.npy').astype(np.float32))
    edge = cv2.dilate(gt, K) != cv2.erode(gt, K)
    hard = (pred != gt) & ~edge
    Image.fromarray((hard * 255).astype(np.uint8)).save(Path(G['out']) / f'{n}.png')
    pairs = {f'{a}->{b}': int(((gt == a) & (pred == b) & hard).sum()) for a in range(4) for b in range(4) if a != b}
    return n, float(hard.mean() * 100), float(((pred != gt) & edge).mean() * 100), pairs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ens', required=True)
    p.add_argument('--out', default='runs/hardmaps')
    p.add_argument('--procs', type=int, default=24)
    args = p.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    items = load_split('data', 'train') + load_split('data', 'valid')
    G.update(ens=args.ens, out=args.out, labs={it['name']: it['lab'] for it in items})
    with Pool(args.procs) as pool:
        rows = pool.map(_one, [it['name'] for it in items])
    tot = {}
    for _, _, _, pr in rows:
        for k, v in pr.items():
            tot[k] = tot.get(k, 0) + v
    s = sum(tot.values())
    top = sorted(tot.items(), key=lambda kv: -kv[1])[:6]
    cls = ['Al', 'Al3Ni', 'SiP', 'SiE']
    out = {'images': len(rows), 'interior_error_pct_mean': round(float(np.mean([r[1] for r in rows])), 3),
           'edge_error_pct_mean': round(float(np.mean([r[2] for r in rows])), 3),
           'images_without_interior_error': sum(r[1] == 0 for r in rows),
           'top_pairs (정답->예측, 안쪽 오답 중 비율 %)': {f'{cls[int(k[0])]}->{cls[int(k[-1])]}': round(v / s * 100, 1) for k, v in top}}
    print(json.dumps(out, indent=1, ensure_ascii=False))
    (Path(args.out) / 'summary.json').write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
