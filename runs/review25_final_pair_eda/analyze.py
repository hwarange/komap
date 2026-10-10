"""CPU-only exploratory analysis of frozen full90 pair OOF rows; no fitting.

Run from any directory. --capture-images reads only train/valid images over SSH
and stores image-level descriptive statistics locally. No test data is loaded.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CLASSES = ['Al', 'Al3Ni', 'Primary Si', 'Eutectic Si']


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def clean(x):
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    return x


def dump(p, x):
    p.write_text(json.dumps(clean(x), indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def receipt(p):
    return {'path': str(p.relative_to(ROOT)).replace('\\', '/'),
            'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size}


def spearman(x, y):
    def rank(a):
        order = np.argsort(a)
        ranks = np.empty(len(a), float)
        start = 0
        while start < len(a):
            end = start + 1
            while end < len(a) and a[order[end]] == a[order[start]]:
                end += 1
            ranks[order[start:end]] = (start + end - 1) / 2
            start = end
        return ranks
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


def capture():
    code = r'''
from pathlib import Path
import hashlib,json
from PIL import Image
import numpy as np
root=Path('/home/j-j15a106/komap')
rows={}
for split in ('train','valid'):
 for p in sorted((root/'data'/split/'images').glob('*_image.png')):
  g=np.array(Image.open(p).convert('L'))
  q=np.percentile(g,[1,50,99]); lo,median,hi=map(float,q)
  rows[p.name[:-10]]={'split':split,'W':g.shape[1],'H':g.shape[0],
   'p01':lo,'median':median,'p99':hi,'range_p99_p01':hi-lo,'std':float(g.std()),
   'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
print(json.dumps(rows))
'''
    r = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'komap',
                        'cd /home/j-j15a106/komap && .venv/bin/python -'],
                       input=code.encode(), capture_output=True, check=True, timeout=120)
    rows = json.loads(r.stdout)
    assert len(rows) == 90
    dump(OUT/'image_statistics.json', rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--capture-images', action='store_true')
    a = p.parse_args()
    started = time.monotonic()
    if a.capture_images:
        capture()
    rowpath = ROOT/'runs/review22_pair/per_image_rows.json'
    metapath = ROOT/'runs/report/per_image.json'
    data, meta = read(rowpath), read(metapath)
    names = sorted(data['rows']['sub6'])
    assert len(names) == 90 and set(names) == set(meta)
    pair = np.array([data['rows']['x4_mit_x3_unetpp_pair'][n]['tta_pp'] for n in names], float)*100
    old = np.array([data['rows']['sub6'][n]['tta_pp'] for n in names], float)*100
    score, prev = np.nanmean(pair, axis=1), np.nanmean(old, axis=1)
    delta = score-prev
    features = {}
    for k in ('W', 'step', 'alloy'):
        features[k] = np.array([meta[n][k] for n in names], float)
    for k in meta[names[0]]['gt']:
        features[k] = np.array([meta[n]['gt'][k] for n in names], float)
    low = np.array([data['rows']['sub6'][n]['low'] for n in names])
    assert all(bool(meta[n]['low']) == low[i] for i,n in enumerate(names))
    receipts = [receipt(rowpath), receipt(metapath), receipt(ROOT/'src/report_analysis.py')]
    imagepath = OUT/'image_statistics.json'
    if imagepath.exists():
        stats = read(imagepath)
        assert set(stats) == set(names)
        assert all(stats[n]['W'] == meta[n]['W'] for n in names)
        assert all((stats[n]['p01'] > 60) == low[i] for i,n in enumerate(names))
        for k in ('p01','median','p99','range_p99_p01','std'):
            features[k] = np.array([stats[n][k] for n in names])
        receipts.append(receipt(imagepath))

    def group(mask):
        return {'n': int(mask.sum()), 'alloys': len(set(features['alloy'][mask])),
                'pair_mIoU_pct': float(score[mask].mean()),
                'sub6_mIoU_pct': float(prev[mask].mean()),
                'delta_pp': float(delta[mask].mean()),
                'pair_class_IoU_pct': dict(zip(CLASSES, np.nanmean(pair[mask], axis=0).tolist())),
                'class_delta_pp': dict(zip(CLASSES, np.nanmean(pair[mask]-old[mask], axis=0).tolist()))}

    groups = {'all90': group(np.ones(90,bool)), 'low_style_p01_gt60': group(low),
              'other_style_p01_le60': group(~low)}
    for k in ('W','step'):
        for val in sorted(set(features[k])):
            groups[f'{k}={int(val)}'] = group(features[k] == val)
    correlations, quantiles = {}, {}
    for k, vals in features.items():
        if k in ('step','alloy'):
            continue
        correlations[k] = {}
        for target, y in [('pair_mIoU',score), ('Eutectic_IoU',pair[:,3]), ('delta_vs_sub6',delta)]:
            finite = np.isfinite(vals)&np.isfinite(y)
            correlations[k][target] = {'n':int(finite.sum()), 'spearman_rho':spearman(vals[finite],y[finite])}
        q1,q2=np.nanquantile(vals,[1/3,2/3])
        bins=[vals<=q1,(vals>q1)&(vals<=q2),vals>q2]
        quantiles[k]={'cutpoints':[float(q1),float(q2)],'groups':[group(b) for b in bins if b.any()]}

    def case(i):
        return {'name':names[i], 'fold':data['fold_of'][names[i]],
                'pair_mIoU_pct':float(score[i]),'delta_pp':float(delta[i]),
                'pair_class_IoU_pct':dict(zip(CLASSES,pair[i].tolist())),
                'class_delta_pp':dict(zip(CLASSES,(pair[i]-old[i]).tolist())),
                'features':{k:float(v[i]) for k,v in features.items()}}
    result={'sources':receipts,'scope':'Exploratory associations on repeatedly examined full90 out-of-fold images, not independent validation or causal inference.',
            'metric':'Mean of per-image nanmean class IoU; absent classes are excluded within image.',
            'notes':[
                'Ground-truth morphology only is reused from old report. Old report predictions and IoUs are not used.',
                'Thickness and equivalent diameter use 1280/image_width pixel scaling, not micrometers; matched physical field of view is unverified.',
                'Existing low-style flag is p01 > 60; this is a dark-level style proxy, not a direct contrast measurement.',
                'No thresholds or model weights are tuned; quantile groups summarize observed distributions.',
                'Correlations are unadjusted exploratory image-level associations; alloy grouping, resolution, contrast and phase composition may confound them.',
                'No test labels or test images read; no new inference, training, threshold or mask changes.'
            ],'groups':groups,'quantiles':quantiles,'correlations':correlations,
            'worst_mIoU':[case(i) for i in np.argsort(score)[:10]],
            'worst_Eutectic':[case(i) for i in np.argsort(pair[:,3])[:10]],
            'worst_regressions':[case(i) for i in np.argsort(delta)[:10]],
            'best_improvements':[case(i) for i in np.argsort(delta)[-10:][::-1]],
            'images_improved':int((delta>0).sum()),'duration_seconds':time.monotonic()-started}
    dump(OUT/'results.json',result)
    print(json.dumps({'duration_seconds':result['duration_seconds'],'groups':groups,
                      'Eutectic_correlations':{k:v['Eutectic_IoU'] for k,v in correlations.items()},
                      'worst_regressions':[{k:x[k] for k in ['name','pair_mIoU_pct','delta_pp']} for x in result['worst_regressions'][:3]]},indent=2))


if __name__ == '__main__':
    main()
