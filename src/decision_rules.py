"""7단계: 상별 판정 규칙 실험 (학습 없음, 90장 OOF + 중첩 검증).

① bias: 상마다 확률에 가중치를 곱한 뒤 가장 높은 상을 고른다 (예: Eutectic Si ×1.1).
   지금 앙상블은 Eutectic Si를 정답보다 조금 적게 칠한다 (면적 비율 0.96~0.99).
② sie : Eutectic Si 전용 모델(그 상인가 아닌가만 구분) 확률과 합친다.
   (가) blend   α: Eutectic Si 확률을 (1-α)·앙상블 + α·전용 모델로 바꾸고 나머지 상은 비율대로 줄인다
   (나) cascade t: 전용 모델 확률이 t를 넘는 픽셀을 먼저 Eutectic Si로 정하고, 나머지는 앙상블이 다른 3상 중에서 고른다
가중치·α·t는 '그 사진이 속한 fold를 뺀 72장'에서 고르고 그 fold에 적용한다 (중첩 검증 = 정직한 추정).
점수는 Primary Si 후처리까지 포함하고, Si 덩어리 분류기는 빼고 잰다 (기준: 앙상블 + 후처리).

예) python src/decision_rules.py cache --exps A B ... --out runs/oof/_ens_A
    python src/decision_rules.py bias --ens runs/oof/_ens_A --out runs/rules/bias_A.json
    python src/decision_rules.py sie --ens runs/oof/_ens_A --spec sie_x2_unetpp --out runs/rules/sie_A.json
"""
import argparse
import itertools
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from data import load_split
from infer import image_iou, postprocess

CLS = ['Al', 'Al3Ni', 'SiP', 'SiE']
G = {}   # 작업 프로세스와 공유하는 값 (Linux fork로 복사 없이 공유)


def _cache_one(n):
    P = np.mean([np.load(Path(G['oof']) / e / f'{n}.npy').astype(np.float32) for e in G['exps']], 0)
    np.save(Path(G['out']) / f'{n}.npy', P.astype(np.float16))
    return n


def cmd_cache(args):
    Path(args.out).mkdir(parents=True, exist_ok=True)
    names = [it['name'] for it in load_split('data', 'train') + load_split('data', 'valid')]
    G.update(oof=args.oof, exps=args.exps, out=args.out)
    with Pool(args.procs) as pool:
        done = list(pool.imap_unordered(_cache_one, names))
    (Path(args.out) / 'exps.json').write_text(json.dumps(args.exps))
    print(f'cached {len(done)} images -> {args.out}', flush=True)


def blend(P, q, alpha, c=3):
    """c번 상의 확률을 (1-α)·앙상블 + α·전용 모델로 바꾸고, 나머지 상은 합이 1이 되도록 비율대로 줄인다."""
    pc = (1 - alpha) * P[c] + alpha * q
    Q = P * ((1 - pc) / np.maximum(1 - P[c], 1e-6))[None]
    Q[c] = pc
    return Q


def decide(P, q, cand, c=3):
    kind, v = cand
    if kind == 'bias':
        return postprocess(P * np.exp(np.array([0.0, *v], np.float32))[:, None, None])
    if kind == 'blend':
        return postprocess(blend(P, q, v, c))
    if kind == 'cascade':
        Q = P.copy()
        Q[c] = 0
        pred = postprocess(Q)
        pred[q > v] = c
        return pred
    raise ValueError(kind)


def spec_prob(specs, n, c):
    """전용 모델 [2,H,W]는 1번 채널, 4상 모델 [4,H,W]는 c번 채널 (대조 시험: 4상 모델로 같은 섞기). 여러 개면 평균."""
    return np.mean([(lambda a: a[1] if a.shape[0] == 2 else a[c])(np.load(Path(G['oof']) / s / f'{n}.npy').astype(np.float32))
                    for s in specs], 0)


def _score_one(n):
    P = np.load(Path(G['ens']) / f'{n}.npy').astype(np.float32)
    lab = G['labs'][n]
    c = G['cls']
    q = spec_prob(G['spec'], n, c) if G['spec'] else None
    S = np.array([image_iou(lab, decide(P, q, cand, c)) * 100 for cand in G['cands']])   # [후보, 4상]
    extra = {}
    if q is not None:   # 전용 모델 단독으로 그 상의 IoU (0.5 기준)
        g, p = lab == c, q > 0.5
        extra['spec_sie_iou'] = float((g & p).sum() / max((g | p).sum(), 1) * 100)
    return n, S, extra


def nested(M, folds):
    """M: [사진, 후보] 사진별 mIoU. fold마다 나머지 72장에서 고른 후보를 그 fold에 적용."""
    honest, picks = np.zeros(M.shape[0]), []
    for k in range(5):
        tr, te = folds != k, folds == k
        j = int(np.argmax(M[tr].mean(0)))
        honest[te] = M[te, j]
        picks.append(j)
    return honest, picks


def run(args, cands):
    items = load_split('data', 'train') + load_split('data', 'valid')
    fold_of = json.loads(Path(args.folds_file).read_text())
    G.update(ens=args.ens, oof=args.oof, spec=getattr(args, 'spec', []), cands=cands, cls=getattr(args, 'cls', 3),
             labs={it['name']: it['lab'] for it in items})
    with Pool(args.procs) as pool:
        res = {n: (S, ex) for n, S, ex in pool.imap_unordered(_score_one, [it['name'] for it in items])}
    names = [it['name'] for it in items]
    A = np.stack([res[n][0] for n in names])            # [사진, 후보, 4상]
    M = np.nanmean(A, 2)                                 # [사진, 후보]
    folds = np.array([fold_of[n] for n in names])
    low = np.array([it['low_contrast'] for it in items])
    b = cands.index(args.base_cand)
    base = M[:, b]
    honest, picks = nested(M, folds)
    d = honest - base
    idx = np.random.default_rng(0).integers(0, len(d), (5000, len(d)))
    lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
    sel = np.array([picks[f] for f in folds])
    cls_h = np.nanmean(A[np.arange(len(names)), sel], 0)
    best = int(np.argmax(M.mean(0)))
    out = {'n_candidates': len(cands),
           'base': {'cand': cands[b], 'mIoU': round(float(base.mean()), 3),
                    'class_IoU': dict(zip(CLS, np.round(np.nanmean(A[:, b], 0), 2).tolist()))},
           'in_sample_best (optimistic)': {'cand': cands[best], 'mIoU': round(float(M[:, best].mean()), 3)},
           'nested (honest)': {'mIoU': round(float(honest.mean()), 3), 'delta': round(float(d.mean()), 3),
                               'ci95': [round(float(lo), 3), round(float(hi), 3)],
                               'better/worse': f'{int((d > 0).sum())}/{int((d < 0).sum())}',
                               'est_test': round(float(0.6 * honest[low].mean() + 0.4 * honest[~low].mean()), 3),
                               'class_IoU': dict(zip(CLS, np.round(cls_h, 2).tolist())),
                               'picks_per_fold': [cands[j] for j in picks]}}
    if G['spec']:
        out['cls'] = G['cls']
        out['spec_alone_SiE_IoU (q>0.5)'] = round(float(np.mean([res[n][1]['spec_sie_iou'] for n in names
                                                                 if (G['labs'][n] == G['cls']).any()])), 2)
    # 후보별 표 (같은 종류 안에서 값에 따른 변화)
    out['by_candidate'] = {str(c): round(float(M[:, i].mean()), 3) for i, c in enumerate(cands)
                           if c[0] != 'bias' or sum(abs(x) > 1e-9 for x in c[1]) <= 1}
    print(json.dumps(out, indent=1, ensure_ascii=False), flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))


def _blend_one(n):
    P = np.load(Path(G['ens']) / f'{n}.npy').astype(np.float32)
    Q = blend(P, spec_prob(G['spec'], n, G['cls']), G['alpha'], G['cls'])
    np.save(Path(G['out']) / f'{n}.npy', Q.astype(np.float16))
    return n


def cmd_blendcache(args):
    """앙상블 확률에 전용 모델을 섞은 확률을 저장 (Si 분류기 평가·학습에 '실험 하나'처럼 넘기기 위해)."""
    Path(args.out).mkdir(parents=True, exist_ok=True)
    names = [it['name'] for it in load_split('data', 'train') + load_split('data', 'valid')]
    G.update(ens=args.ens, oof=args.oof, spec=args.spec, alpha=args.alpha, out=args.out, cls=args.cls)
    with Pool(args.procs) as pool:
        done = list(pool.imap_unordered(_blend_one, names))
    print(f'blended {len(done)} images (alpha {args.alpha}, spec {args.spec}) -> {args.out}', flush=True)


def cmd_bias(args):
    grid = [round(float(x), 2) for x in np.arange(-0.3, 0.31, 0.1)]
    cands = [('bias', tuple(v)) for v in itertools.product(grid, grid, grid)]   # Al3Ni, SiP, SiE 로그 가중 (Al 기준 0)
    args.base_cand = ('bias', (0.0, 0.0, 0.0))
    run(args, cands)


def cmd_sie(args):
    cands = [('blend', round(float(a), 2)) for a in np.arange(0, 1.01, 0.1)] + \
            [('cascade', round(float(t), 2)) for t in np.arange(0.3, 0.91, 0.1)]
    args.base_cand = ('blend', 0.0)
    run(args, cands)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('cache')
    c.add_argument('--exps', nargs='+', required=True)
    c.add_argument('--out', required=True)
    for name in ['bias', 'sie']:
        s = sub.add_parser(name)
        s.add_argument('--ens', required=True, help='cache로 만든 앙상블 확률 폴더')
        s.add_argument('--out', default='')
        s.add_argument('--folds-file', default='splits/folds5.json')
        if name == 'sie':
            s.add_argument('--spec', nargs='+', required=True, help='전용 모델 실험 이름 (OOF [2,H,W])')
            s.add_argument('--cls', type=int, default=3, help='전용 모델이 맡은 상 (3 = Eutectic Si, 1 = Al3Ni)')
    b = sub.add_parser('blendcache')
    b.add_argument('--ens', required=True)
    b.add_argument('--spec', nargs='+', required=True)
    b.add_argument('--alpha', type=float, required=True)
    b.add_argument('--out', required=True)
    b.add_argument('--cls', type=int, default=3)
    for s in sub.choices.values():
        s.add_argument('--oof', default='runs/oof')
        s.add_argument('--procs', type=int, default=40)
    args = p.parse_args()
    {'cache': cmd_cache, 'bias': cmd_bias, 'sie': cmd_sie, 'blendcache': cmd_blendcache}[args.cmd](args)


if __name__ == '__main__':
    main()
