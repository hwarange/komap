"""제출용 앙상블 예측: 여러 실험 × 5-fold 모델의 TTA 확률을 평균해 대회 형식(4색 RGB PNG)으로 저장한다.

1) 예측 (GPU 여러 장이면 모델을 나눠서 부분합을 저장)
   CUDA_VISIBLE_DEVICES=0 python src/submit.py predict --exps A B C --split test --out submissions/sub1 --shard 0 --nshards 4
2) 합치기 + 저장 + 형식 검사 (정답이 있으면 채점)
   python src/submit.py combine --split test --out submissions/sub1

검증 모드: --split cv 로 주면 학습+검증 90장을 예측하되, 각 사진을 '학습하지 않은 fold 모델'만 쓴다.
이 점수가 OOF 앙상블 평가(ensemble.py)와 같아야 제출 코드가 맞는 것이다.

6단계 (sub3): 평가 사진 예측은 모델별로 따로 저장한다 (parts/models/<실험>/fold<k>/<사진>.npy).
예측은 한 번만 하고 combine에서 섞는 방식만 바꿔 여러 제출본을 만들 수 있다.
   python src/submit.py predict --exps A B --split test --out submissions/sub3_pred --shard 0 --nshards 4
   python src/submit.py predict --exps full_A --root runs/full --folds 3 --split test --out submissions/sub3_pred ...
   python src/submit.py combine --split test --parts submissions/sub3_pred --out submissions/sub3a \
       --recipe A:2 B:1 --full-mix 0.5
   --recipe 실험:가중치 (가중치를 빼면 1). 실험마다 fold 모델 평균을 낸 뒤 가중 평균한다.
   --full-mix m: 'full_<실험>' 예측(90장 전체 학습 모델)이 있으면 그 실험은 (1-m)·fold 평균 + m·90장 모델 평균.
   --recipe를 주지 않으면 저장된 모델 전부를 같은 비중으로 평균한다 (이전 방식과 같음).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data import MASK_RGB, label_to_rgb, load_split, normalize
from infer import image_iou, postprocess, predict_probs
from train import autocast, build_from_args


def load_items(split):
    if split == 'cv':
        return load_split('data', 'train') + load_split('data', 'valid')
    return load_split('data', split)


def model_dir(parts, e, k):
    return Path(parts) / 'parts' / 'models' / e / f'fold{k}'


def cmd_predict(args):
    items = load_items(args.split)
    fold_of = json.loads(Path(args.folds_file).read_text()) if args.split == 'cv' else None
    per_model = args.split != 'cv'   # 평가 사진은 모델별로 저장 (combine에서 비중을 바꿔 섞을 수 있게)
    models = [(e, k) for e in args.exps for k in range(args.folds)][args.shard::args.nshards]
    part = Path(args.out) / 'parts' / f'shard{args.shard}'
    if not per_model:
        part.mkdir(parents=True, exist_ok=True)
    acc, cnt = {}, {}
    for e, k in models:
        t0 = time.time()
        md = model_dir(args.out, e, k)
        if per_model and all((md / f"{it['name']}.npy").exists() for it in items):
            print(f'{e}/fold{k}: already saved, skip', flush=True)
            continue
        ck = torch.load(Path(args.root) / e / f'fold{k}' / 'last.pt', map_location='cpu', weights_only=False)
        a = ck['args']
        model = build_from_args(a)
        model.load_state_dict(ck['model'])
        model = model.cuda().eval()
        for it in items:
            if fold_of is not None and fold_of[it['name']] != k:
                continue  # 검증 모드: 이 사진을 학습한 모델은 쓰지 않음
            with torch.no_grad(), autocast(a.get('amp', False)):
                prob = predict_probs(model, it['gray'], a['width'], a['tile'], a['overlap'], True, 'cuda', scale_multiplier=a.get('scale_multiplier', 1.0))
            if per_model:
                md.mkdir(parents=True, exist_ok=True)
                np.save(md / f"{it['name']}.npy", prob.astype(np.float32))
                continue
            acc[it['name']] = acc.get(it['name'], 0) + prob
            cnt[it['name']] = cnt.get(it['name'], 0) + 1
        print(f'{e}/fold{k}: {time.time() - t0:.0f}s', flush=True)
        del model
        torch.cuda.empty_cache()
    if per_model:
        print(f'shard {args.shard}: {len(models)} models saved per model under {Path(args.out) / "parts" / "models"}', flush=True)
        return
    for n, s in acc.items():
        np.save(part / f'{n}.npy', s.astype(np.float32))
    (part / 'counts.json').write_text(json.dumps(cnt))
    (part / 'models.json').write_text(json.dumps(models))
    print(f'shard {args.shard}: {len(models)} models, {len(acc)} images saved', flush=True)


def parse_recipe(recipe):
    """['A:2', 'B'] -> {'A': 2.0, 'B': 1.0}. 같은 실험이 여러 번 나오면 가중치를 더한다."""
    w = {}
    for r in recipe:
        e, _, x = r.partition(':')
        w[e] = w.get(e, 0.0) + (float(x) if x else 1.0)
    return w


def mean_of(dirs, n):
    fs = [d / f'{n}.npy' for d in dirs if (d / f'{n}.npy').exists()]
    return (sum(np.load(f) for f in fs) / len(fs), len(fs)) if fs else (None, 0)


def mixed_prob(mroot, n, weights, full_mix):
    """모델별 예측을 recipe대로 섞은 확률과, 실험마다 쓴 모델 수를 돌려준다.
    weights가 None이면 저장된 모델 전부를 같은 비중으로 평균한다 (이전 방식)."""
    if weights is None:
        P, c = mean_of(sorted(mroot.glob('*/fold*')), n)
        return P, {'all_models': c}
    total, wsum, used = 0, 0.0, {}
    for e, w in weights.items():
        P, c = mean_of(sorted((mroot / e).glob('fold*')), n)
        if c == 0:
            raise SystemExit(f'{e}: {n} 예측 없음')
        used[e] = c
        if full_mix > 0:
            Pf, cf = mean_of(sorted((mroot / f'full_{e}').glob('fold*')), n)
            if cf:
                P = (1 - full_mix) * P + full_mix * Pf
                used[f'full_{e}'] = cf
        total, wsum = total + w * P, wsum + w
    return total / wsum, used


def blend_cls(P, q, alpha, c):
    """한 상 전용 모델 섞기 (decision_rules.py의 blend와 같은 계산):
    c번 상의 확률을 (1-α)·앙상블 + α·전용 모델로 바꾸고, 나머지 상은 합이 1이 되도록 비율대로 줄인다."""
    pc = (1 - alpha) * P[c] + alpha * q
    Q = P * ((1 - pc) / np.maximum(1 - P[c], 1e-6))[None]
    Q[c] = pc
    return Q


def blend_sie(P, q, alpha):
    """Eutectic Si 전용 모델 섞기 (7단계)."""
    return blend_cls(P, q, alpha, 3)


def spec_mean(mroot, specs, n):
    """전용 모델([2,H,W], 1번 채널)들의 fold 평균을 실험끼리 다시 평균. 실험마다 쓴 모델 수도 돌려준다."""
    qs = [mean_of(sorted((mroot / s).glob('fold*')), n) for s in specs]
    if any(c == 0 for _, c in qs):
        raise SystemExit(f'{n}: 전용 모델 예측 없음 {specs}')
    return np.mean([q[1] for q, _ in qs], 0), {s: c for s, (_, c) in zip(specs, qs)}


def cmd_combine(args):
    items = load_items(args.split)
    si = None
    if args.si_clf:
        import pickle
        from si_classifier import apply_selective, components
        si = pickle.loads(Path(args.si_clf).read_bytes())['clf']
    si_log = {}
    out = Path(args.out)
    per_model = args.split != 'cv'
    if per_model:
        mroot = Path(args.parts or args.out) / 'parts' / 'models'
        weights = parse_recipe(args.recipe) if args.recipe else None
        models, used_first = [], None
    else:
        shards = sorted((out / 'parts').glob('shard*'))
        models = [m for s in shards for m in json.loads((s / 'models.json').read_text())]
        shard_counts = {s: json.loads((s / 'counts.json').read_text()) for s in shards}
    mask_dir = out / 'masks'
    if args.split != 'cv':
        mask_dir.mkdir(parents=True, exist_ok=True)
    scores, counts, problems = [], {}, []
    for it in items:
        n = it['name']
        if per_model:
            P, used = mixed_prob(mroot, n, weights, args.full_mix)
            if P is None:
                problems.append(f'{n}: no predictions')
                continue
            if args.sie_spec:
                # Eutectic Si 전용 모델들을 섞는다 (먼저)
                q, cnt = spec_mean(mroot, args.sie_spec, n)
                P = blend_sie(P, q, args.sie_alpha)
                used = {**used, **{f'sie:{s}': c for s, c in cnt.items()}}
            if args.al3ni_spec:
                # Al3Ni 전용 모델을 섞는다 (Eutectic Si 다음. 평가(eval_final_combo.sh)와 같은 순서)
                q, cnt = spec_mean(mroot, args.al3ni_spec, n)
                P = blend_cls(P, q, args.al3ni_alpha, 1)
                used = {**used, **{f'al3ni:{s}': c for s, c in cnt.items()}}
            used_first = used_first or used
            if used != used_first:
                problems.append(f'{n}: 사진마다 쓴 모델 수가 다름 {used}')
            counts[n] = sum(used.values())
        else:
            total, c = 0, 0
            for s in shards:
                f = s / f'{n}.npy'
                if f.exists():
                    total = total + np.load(f)
                    c += shard_counts[s][n]
            if c == 0:
                problems.append(f'{n}: no predictions')
                continue
            counts[n] = c
            P = total / c
        pred = postprocess(P)
        if si is not None:
            # Si 덩어리 분류기: 확신할 때만 덩어리 단위로 Primary/Eutectic 재판정
            X, cc, k = components(P, pred, normalize(it['gray']), it['gray'].shape[1], it['low_contrast'])
            if len(X):
                prob = si.predict_proba(X)[:, 1]
                new = apply_selective(pred, cc, prob, args.si_thr)
                si_log[n] = {'components': int(len(X)), 'decided': int(((prob >= args.si_thr) | (prob <= 1 - args.si_thr)).sum()),
                             'pixels_changed_pct': round(float((new != pred).mean() * 100), 3)}
                pred = new
        if it['lab'] is not None:
            scores.append(np.nanmean(image_iou(it['lab'], pred)) * 100)
        if args.split != 'cv':
            path = mask_dir / f'{n}_mask.png'
            Image.fromarray(label_to_rgb(pred)).save(path)
            # 형식 검사: 원본과 같은 크기, RGB, 대회 4색만
            chk = np.array(Image.open(path))
            colors = {tuple(c) for c in chk.reshape(-1, 3)}
            if Image.open(path).mode != 'RGB' or chk.shape[:2] != it['gray'].shape or not colors <= {tuple(c) for c in MASK_RGB}:
                problems.append(f'{n}: format problem mode={Image.open(path).mode} shape={chk.shape} colors={len(colors)}')
    if per_model:
        models_info = {'parts': str(mroot), 'recipe': weights, 'full_mix': args.full_mix, 'models_per_exp': used_first,
                       'sie_spec': args.sie_spec, 'sie_alpha': args.sie_alpha if args.sie_spec else None,
                       'al3ni_spec': args.al3ni_spec, 'al3ni_alpha': args.al3ni_alpha if args.al3ni_spec else None}
        n_models = sum((used_first or {}).values())
        exps = sorted(used_first or {})
    else:
        models_info, n_models, exps = {}, len(models), sorted({e for e, _ in models})
    summary = {'split': args.split, 'n_images': len(counts), 'n_models': n_models,
               'models_per_image': sorted(set(counts.values())), 'experiments': exps, **models_info,
               'problems': problems, 'created': time.strftime('%Y-%m-%d %H:%M'),
               'si_classifier': {'path': args.si_clf, 'thr': args.si_thr, 'per_image': si_log} if si is not None else None}
    if scores:
        low = np.array([it['low_contrast'] for it in items if it['name'] in counts])
        sc = np.array(scores)
        summary['mIoU'] = round(float(sc.mean()), 3)
        summary['est_test'] = round(float(0.6 * sc[low].mean() + 0.4 * sc[~low].mean()), 3)
    (out / f'summary_{args.split}.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['predict', 'combine'])
    p.add_argument('--exps', nargs='+', default=[])
    p.add_argument('--split', default='test', help='test / valid / train / cv(90장 검증 모드)')
    p.add_argument('--out', default='submissions/sub1')
    p.add_argument('--root', default='runs/cv')
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--folds-file', default='splits/folds5.json')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshards', type=int, default=1)
    p.add_argument('--si-clf', default='', help='Si 덩어리 분류기 pickle (si_classifier.py fit 결과)')
    p.add_argument('--si-thr', type=float, default=0.8)
    p.add_argument('--parts', default='', help='combine: 모델별 예측이 저장된 폴더 (기본: --out)')
    p.add_argument('--recipe', nargs='+', default=[], help='combine: 실험:가중치 목록 (평가 사진 모드)')
    p.add_argument('--full-mix', type=float, default=0.0, help='combine: full_<실험> 예측을 섞는 비율 (0이면 안 섞음)')
    p.add_argument('--sie-spec', nargs='+', default=[], help='combine: Eutectic Si 전용 모델 실험 이름 (7단계)')
    p.add_argument('--sie-alpha', type=float, default=0.3, help='combine: 전용 모델을 섞는 비율 (중첩 검증에서 0.2~0.4)')
    p.add_argument('--al3ni-spec', nargs='+', default=[], help='combine: Al3Ni 전용 모델 실험 이름 (9단계, Eutectic Si 섞기 뒤)')
    p.add_argument('--al3ni-alpha', type=float, default=0.3, help='combine: Al3Ni 전용 모델을 섞는 비율')
    args = p.parse_args()
    cmd_predict(args) if args.cmd == 'predict' else cmd_combine(args)


if __name__ == '__main__':
    main()
