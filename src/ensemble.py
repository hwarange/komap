"""OOF 예측 확률로 앙상블과 후처리 규칙을 평가한다.

예) python src/ensemble.py --exps baseline baseline_s1 convnext_t unetpp unet_mit effb4 x2_c1024 c768

평가 항목
1. 단일 모델 점수 (result.json의 TTA 점수와 같아야 함: 재현 확인)
2. 미리 정한 조합의 앙상블 (확률 평균)
3. 탐욕적 선택 앙상블: 매번 점수를 가장 많이 올리는 모델을 추가.
   같은 사진으로 고르고 채점하면 점수가 부풀므로, 'fold k를 뺀 나머지로 고르고 fold k로 채점'하는 정직한 추정도 함께 낸다.
4. 후처리: 기존 Primary Si 후처리 / + Primary Si–Eutectic Si 비인접 규칙 (EDA 8절)
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from data import load_split
from infer import image_iou, postprocess

SIP, SIE = 2, 3


def adjacency_rule(prob, pred, d=1):
    """정답에서는 Primary Si와 Eutectic Si가 거의 맞닿지 않는다 (경계 1,260만 px 중 462px).
    예측된 Primary Si에서 d px 이내의 Eutectic Si 예측은, Eutectic Si를 뺀 나머지 상 중 확률이 가장 높은 상으로 바꾼다."""
    sip = (pred == SIP).astype(np.uint8)
    if not sip.any():
        return pred
    near = cv2.dilate(sip, np.ones((2 * d + 1, 2 * d + 1), np.uint8)).astype(bool)
    fix = near & (pred == SIE)
    if fix.any():
        alt = prob.copy()
        alt[SIE] = -1
        pred = pred.copy()
        pred[fix] = alt.argmax(0)[fix]
    return pred


def contacts(pred):
    a, b = pred[:, 1:], pred[:, :-1]
    c, e = pred[1:, :], pred[:-1, :]
    return int((((a == SIP) & (b == SIE)) | ((a == SIE) & (b == SIP))).sum() + (((c == SIP) & (e == SIE)) | ((c == SIE) & (e == SIP))).sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--exps', nargs='+', required=True)
    p.add_argument('--oof', default='runs/oof')
    p.add_argument('--folds-file', default='splits/folds5.json')
    p.add_argument('--max-size', type=int, default=6)
    p.add_argument('--out', default='runs/oof/ensemble_summary.json')
    args = p.parse_args()

    items = load_split('data', 'train') + load_split('data', 'valid')
    fold_of = json.loads(Path(args.folds_file).read_text())
    names = [it['name'] for it in items]
    labs = {it['name']: it['lab'] for it in items}
    low = np.array([it['low_contrast'] for it in items])
    fold = np.array([fold_of[n] for n in names])
    exps = [e for e in args.exps if all((Path(args.oof) / e / f'{n}.npy').exists() for n in names)]
    print('experiments with complete OOF:', exps)

    # 사진별로 모든 실험의 확률을 읽어 두고, 조합 점수는 사진별 mIoU로 계산
    probs = {n: {e: np.load(Path(args.oof) / e / f'{n}.npy').astype(np.float32) for e in exps} for n in names}

    def per_image(members, post='pp'):
        sc = []
        for n in names:
            pr = np.mean([probs[n][e] for e in members], 0)
            pred = postprocess(pr) if post in ('pp', 'pp+adj') else pr.argmax(0).astype(np.uint8)
            if post == 'pp+adj':
                pred = adjacency_rule(pr, pred)
            sc.append(np.nanmean(image_iou(labs[n], pred)) * 100)
        return np.array(sc)

    def summary(sc):
        return {'cv': round(float(sc.mean()), 3), 'est_test': round(float(0.6 * sc[low].mean() + 0.4 * sc[~low].mean()), 3)}

    res = {'single': {}, 'combos': {}, 'greedy': {}, 'adjacency': {}}
    cache = {}
    print('\n== single models (TTA + pp)')
    for e in exps:
        cache[(e,)] = per_image([e])
        res['single'][e] = summary(cache[(e,)])
        print(f"  {e:12s} {res['single'][e]}")

    print('\n== predefined ensembles (probability average, TTA + pp)')
    ranked = sorted(exps, key=lambda e: -cache[(e,)].mean())
    combos = {}
    if 'baseline' in exps and 'baseline_s1' in exps:
        combos['seed pair (baseline + baseline_s1)'] = ['baseline', 'baseline_s1']
    for k in (2, 3, 4, 5):
        if len(ranked) >= k:
            combos[f'top{k} by single score'] = ranked[:k]
    combos['all'] = exps
    for label, mem in combos.items():
        sc = per_image(mem)
        res['combos'][label] = {'members': mem, **summary(sc)}
        print(f"  {label:38s} {summary(sc)}  {mem}")

    print('\n== greedy forward selection')
    def greedy(mask):
        chosen, best = [], -1
        for _ in range(args.max_size):
            cand = []
            for e in exps:
                key = tuple(sorted(chosen + [e]))
                if key not in cache:
                    cache[key] = per_image(list(key))
                cand.append((cache[key][mask].mean(), e))
            s, e = max(cand)
            if s <= best:
                break
            chosen.append(e)
            best = s
        return chosen, best
    chosen, best = greedy(np.ones(len(names), bool))
    res['greedy']['in_sample'] = {'members': chosen, 'cv': round(float(best), 3), 'note': '같은 90장으로 고르고 채점 -> 낙관적'}
    print(f'  in-sample (optimistic): {best:.3f}  {chosen}')
    honest = np.zeros(len(names))
    picks = []
    for k in range(5):
        mem, _ = greedy(fold != k)
        honest[fold == k] = cache[tuple(sorted(mem))][fold == k]
        picks.append(mem)
    res['greedy']['nested'] = {**summary(honest), 'picks_per_fold': picks}
    print(f"  nested (honest): {summary(honest)}")
    for m in picks:
        print('    picked:', m)

    print('\n== Primary Si - Eutectic Si adjacency rule (on best predefined ensemble and best single)')
    best_combo = max(res['combos'].values(), key=lambda v: v['cv'])['members']
    for label, mem in [('best single', [ranked[0]]), ('best predefined ensemble', best_combo)]:
        base = per_image(mem, 'pp')
        adj = per_image(mem, 'pp+adj')
        n_contact = sum(contacts(postprocess(np.mean([probs[n][e] for e in mem], 0))) for n in names)
        res['adjacency'][label] = {'members': mem, 'pp': summary(base), 'pp+adj': summary(adj),
                                   'delta': round(float((adj - base).mean()), 3), 'improved_images': int((adj > base).sum()),
                                   'worse_images': int((adj < base).sum()), 'pred_contacts': n_contact}
        print(f"  {label:26s} pp {summary(base)} -> +adj {summary(adj)}  delta {(adj - base).mean():+.3f}  "
              f"better {int((adj > base).sum())} / worse {int((adj < base).sum())}  predicted SiP-SiE contacts {n_contact} (GT 462)")

    Path(args.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f'\nsaved -> {args.out}')


if __name__ == '__main__':
    main()
