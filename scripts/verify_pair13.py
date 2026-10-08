"""새 제출 코드의 90장 예측을 기존 OOF와 비교한다 (모델 재학습 없음)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from data import load_split
from infer import image_iou, postprocess

EXPS = ['x2_mit_bw_r90', 'x3_unetpp_bw_e120_r90']
ROOT = Path('submissions/sub6_cv_pair/parts')
items = load_split('data', 'train') + load_split('data', 'valid')
expected = {it['name'] for it in items}
shards = sorted(ROOT.glob('shard*'))
assert len(shards) == 4, 'expected four prediction shards'
counts = {p: json.loads((p / 'counts.json').read_text()) for p in shards}
assert set().union(*(set(c) for c in counts.values())) == expected
models = [m for p in shards for m in json.loads((p / 'models.json').read_text())]
assert sorted(tuple(m) for m in models) == sorted((e, k) for e in EXPS for k in range(5))
rows = []
for it in items:
    n = it['name']
    total, count = 0, 0
    for p in shards:
        if n not in counts[p]:
            continue
        total = total + np.load(p / f'{n}.npy')
        count += counts[p][n]
    assert count == 2, (n, count)
    P = total / count
    assert P.shape == (4, *it['lab'].shape), (n, P.shape)
    assert np.isfinite(P).all() and P.min() >= 0 and P.max() <= 1, n
    assert np.max(np.abs(P.sum(0) - 1)) < 1e-3, n
    old = np.mean([np.load(Path('runs/oof', e, f'{n}.npy')).astype(np.float32) for e in EXPS], 0)
    cv = image_iou(it['lab'], postprocess(P)) * 100
    oof = image_iou(it['lab'], postprocess(old)) * 100
    rows.append({'name': n, 'cv': float(np.nanmean(cv)), 'oof': float(np.nanmean(oof)),
                 'delta': float(np.nanmean(cv) - np.nanmean(oof)),
                 'max_probability_diff': float(np.max(np.abs(P - old)))})
out = {'n_images': len(rows), 'models_per_image': 2,
       'cv_mIoU': float(np.mean([r['cv'] for r in rows])),
       'oof_mIoU': float(np.mean([r['oof'] for r in rows])),
       'mean_score_delta': float(np.mean([r['delta'] for r in rows])),
       'max_probability_diff': max(r['max_probability_diff'] for r in rows), 'rows': rows}
assert abs(out['mean_score_delta']) < 0.01, out['mean_score_delta']
Path('runs/pair13/verify_pair.json').write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != 'rows'}, indent=1), flush=True)
print('PASS: independent prediction, complete folds, probability validity, OOF reproduction', flush=True)
