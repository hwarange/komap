"""끝난 교차검증 실험 하나를 기준 실험과 비교하고, 버그·이상 징후를 점검한다.

예) python src/review.py a_hm --base cnx_bw

점검 항목
- 완결성: fold 5개의 result.json, OOF 확률 90장, 로그의 오류·경고·NaN
- 학습 상태: 마지막 손실(기준과 비교), 학습 시간, GPU 메모리
- 성능: 같은 사진끼리 비교한 Δ와 부트스트랩 95% 구간, 개선된 사진 수, 상별·저대비·675px Δ
- 이상 징후: 상별 예측 면적 / 정답 면적 (과다·과소 예측), Primary Si가 없는 사진 2장의 결과
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

from data import load_split
from infer import postprocess

CLS = ['Al', 'Al3Ni', 'SiP', 'SiE']
NO_SIP = ['A57_step1_sample3', 'A71_step2_sample1']


def rows_of(exp, root):
    rows = []
    for f in sorted(Path(root, exp).glob('fold*/result.json')):
        rows += json.loads(f.read_text())['rows']
    return {r['name']: r for r in rows}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('exp')
    p.add_argument('--base', default='cnx_bw')
    p.add_argument('--root', default='runs/cv')
    p.add_argument('--oof', default='runs/oof')
    args = p.parse_args()
    out = {'exp': args.exp, 'base': args.base}

    # 1) 완결성과 로그
    folds = sorted(Path(args.root, args.exp).glob('fold*'))
    out['result_files'] = sum((f / 'result.json').exists() for f in folds)
    oof_dir = Path(args.oof, args.exp)
    out['oof_files'] = len(list(oof_dir.glob('*.npy'))) if oof_dir.exists() else 0
    issues = []
    last_loss, secs, mem = [], [], []
    for f in folds:
        txt = (f / 'stdout.txt').read_text(errors='ignore') if (f / 'stdout.txt').exists() else ''
        for pat in ['Traceback', 'Error', 'nan', 'inf ', 'RuntimeWarning']:
            if re.search(pat, txt):
                issues.append(f'{f.name}: "{pat}" in stdout')
        logs = [json.loads(l) for l in (f / 'log.jsonl').read_text().splitlines()]
        last_loss.append(logs[-1]['loss'])
        secs.append(sum(l['train_sec'] for l in logs))
        mem.append(max(l['gpu_mem_gb'] for l in logs))
    base_loss = [json.loads((f / 'log.jsonl').read_text().splitlines()[-1])['loss'] for f in sorted(Path(args.root, args.base).glob('fold*'))]
    out['final_loss'] = round(float(np.mean(last_loss)), 4)
    out['base_final_loss'] = round(float(np.mean(base_loss)), 4)
    out['train_min_per_fold'] = round(float(np.mean(secs)) / 60, 1)
    out['gpu_mem_gb'] = max(mem)
    out['log_issues'] = issues or 'none'

    # 2) 성능: 같은 사진끼리 비교
    a, b = rows_of(args.exp, args.root), rows_of(args.base, args.root)
    names = [n for n in b if n in a]

    def img(r, k='tta_pp'):
        return np.array([np.nanmean(r[n][k]) * 100 for n in names])

    def cls(r):
        return np.array([r[n]['tta_pp'] for n in names], float) * 100

    sa, sb = img(a), img(b)
    d = sa - sb
    idx = np.random.default_rng(0).integers(0, len(d), (5000, len(d)))
    lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
    low = np.array([b[n]['low'] for n in names])
    small = np.array([b[n]['W'] == 675 for n in names])
    out['cv_mIoU'] = round(float(sa.mean()), 2)
    out['base_cv_mIoU'] = round(float(sb.mean()), 2)
    out['delta'] = round(float(d.mean()), 2)
    out['ci95'] = [round(float(lo), 2), round(float(hi), 2)]
    out['improved_images'] = f'{int((d > 0).sum())}/{len(d)}'
    out['delta_plain(no TTA/pp)'] = round(float((img(a, 'plain') - img(b, 'plain')).mean()), 2)
    out['delta_per_class'] = dict(zip(CLS, np.round(np.nanmean(cls(a) - cls(b), 0), 2).tolist()))
    out['class_IoU'] = dict(zip(CLS, np.round(np.nanmean(cls(a), 0), 2).tolist()))
    out['delta_low_contrast'] = round(float(d[low].mean()), 2)
    out['delta_high_contrast'] = round(float(d[~low].mean()), 2)
    out['delta_675px'] = round(float(d[small].mean()), 2)
    out['delta_large'] = round(float(d[~small].mean()), 2)
    out['est_test'] = round(float(0.6 * sa[low].mean() + 0.4 * sa[~low].mean()), 2)
    out['no_SiP_images'] = {n: {'mIoU': round(float(np.nanmean(a[n]['tta_pp']) * 100), 1),
                                'base': round(float(np.nanmean(b[n]['tta_pp']) * 100), 1),
                                'false_SiP': not np.isnan(a[n]['tta_pp'][2])} for n in NO_SIP if n in a}

    # 3) 예측 면적 / 정답 면적 (OOF 확률이 있으면)
    if out['oof_files'] == 90:
        items = {it['name']: it['lab'] for it in load_split('data', 'train') + load_split('data', 'valid')}
        pa, ga, pb = np.zeros(4), np.zeros(4), np.zeros(4)
        for n, lab in items.items():
            pa += np.bincount(postprocess(np.load(oof_dir / f'{n}.npy').astype(np.float32)).ravel(), minlength=4)
            ga += np.bincount(lab.ravel(), minlength=4)
            fb = Path(args.oof, args.base, f'{n}.npy')
            if fb.exists():
                pb += np.bincount(postprocess(np.load(fb).astype(np.float32)).ravel(), minlength=4)
        out['area_ratio_pred/true'] = dict(zip(CLS, np.round(pa / ga, 3).tolist()))
        if pb.sum():
            out['base_area_ratio'] = dict(zip(CLS, np.round(pb / ga, 3).tolist()))
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
