"""Evaluate seed-1 and distance-field (SDF) models on validation20 from saved probabilities (CPU only).

Families found on disk are used; missing ones are skipped. Reports singles, the final pair, pair + seed models,
pair + SDF models, everything, and for SDF models three decision rules: probability head, distance head,
and a 50:50 mix of the two.
"""
import os, sys, json, itertools
from pathlib import Path
ROOT = Path('/home/j-j15a106/komap'); OUT = ROOT / 'reports/final36_evidence'
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]; os.chdir(ROOT)
import numpy as np, cv2
cv2.setNumThreads(4)
from data import load_split, CLASSES
from infer import image_iou, postprocess
from review_submission31 import probability

valid = load_split('data', 'valid')
DIRS = {'mit2': 'runs/review32_mit2_train70/valid_probs/mit2/base', 'unet3': 'runs/submission31_train70/valid_probs/unet3/base',
        'context4': 'runs/submission31_train70/valid_probs/context4/base', 'mit4': 'runs/submission31_train70/valid_probs/mit4/base',
        'mit2_s1': 'runs/seed36/valid_probs/mit2_s1', 'unet3_s1': 'runs/seed36/valid_probs/unet3_s1',
        'mit2_sdf': 'runs/sdf36/valid_probs/mit2_sdf', 'unet3_sdf': 'runs/sdf36/valid_probs/unet3_sdf'}
AVAIL = {k: ROOT / v for k, v in DIRS.items() if (ROOT / v).is_dir() and len(list((ROOT / v).glob('*.npy'))) == 20}
print('available:', sorted(AVAIL))
K3 = np.ones((3, 3), np.uint8)


def softmax(d, beta=4.0):
    e = np.exp(beta * (d - d.max(0, keepdims=True))); return e / e.sum(0, keepdims=True)


def load_sdf(fam, it):
    p = AVAIL[fam].parent / (AVAIL[fam].name + '_sdf') / f"{it['name']}.npy"
    s = np.load(p, allow_pickle=False).astype(np.float32); assert s.shape == (4, *it['gray'].shape); return s


recipes = {}
base = [f for f in ('mit2', 'unet3') if f in AVAIL]
for f in AVAIL:
    recipes[f] = [f]
if len(base) == 2:
    recipes['pair'] = base
    seeds = [f for f in ('mit2_s1', 'unet3_s1') if f in AVAIL]; sdfs = [f for f in ('mit2_sdf', 'unet3_sdf') if f in AVAIL]
    if seeds: recipes['pair+seed1'] = base + seeds
    if sdfs: recipes['pair+sdf'] = base + sdfs
    if seeds and sdfs: recipes['pair+seed1+sdf'] = base + seeds + sdfs
    if 'mit2_s1' in AVAIL: recipes['mit2x2+unet3'] = ['mit2', 'mit2_s1', 'unet3']
    if 'unet3_s1' in AVAIL: recipes['mit2+unet3x2'] = ['mit2', 'unet3', 'unet3_s1']
    if 'context4' in AVAIL and seeds: recipes['pair+seed1+ctx'] = base + seeds + ['context4']
rows = {k: [] for k in recipes}; sdf_rows = {}; band = {k: [0, 0] for k in recipes}
for it in valid:
    lab = it['lab']; bm = cv2.dilate(lab, K3) != cv2.erode(lab, K3)
    probs = {f: probability(AVAIL[f] / f"{it['name']}.npy", it) for f in AVAIL}
    for key, fams in recipes.items():
        p = sum(probs[f] for f in fams) / len(fams); pred = postprocess(p)
        rows[key].append(image_iou(lab, pred)); err = pred != lab
        band[key][0] += int((err & bm).sum()); band[key][1] += int(err.sum())
    for f in AVAIL:
        if f.endswith('_sdf'):
            d = load_sdf(f, it); q = softmax(d)
            for name, pr in ((f + ':dist', q), (f + ':mix', (probs[f] + q) / 2)):
                sdf_rows.setdefault(name, []).append(image_iou(lab, postprocess(pr)))
            if len(base) == 2:
                mixed = (probs['mit2'] + probs['unet3'] + probs[f] + q) / 4
                sdf_rows.setdefault('pair+' + f + ':prob+dist', []).append(image_iou(lab, postprocess(mixed)))


def summary(v):
    v = np.array(v); return dict(mIoU=float(np.nanmean(v, 1).mean() * 100), class_IoU={c: float(np.nanmean(v[:, i]) * 100) for i, c in enumerate(CLASSES)})


res = {k: dict(summary(v), band_share_percent=band[k][0] / max(band[k][1], 1) * 100, members=recipes[k]) for k, v in rows.items()}
res.update({k: summary(v) for k, v in sdf_rows.items()})
if 'pair' in rows:
    ref = np.nanmean(np.array(rows['pair']), 1) * 100; alloys = [it['alloy'] for it in valid]
    for k, v in list(rows.items()) + list(sdf_rows.items()):
        d = np.nanmean(np.array(v), 1) * 100 - ref; rng = np.random.default_rng(0); uniq = sorted(set(alloys)); boots = []
        for _ in range(10000):
            pick = rng.choice(uniq, len(uniq), replace=True); vals = np.concatenate([d[[i for i, a in enumerate(alloys) if a == u]] for u in pick]); boots.append(vals.mean())
        res[k]['vs_pair'] = dict(mean_delta=float(d.mean()), improved=int((d > 0).sum()), worsened=int((d < 0).sum()), ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))])
(OUT / 'combo_check36c.json').write_text(json.dumps(res, indent=2), encoding='utf8')
print(f"{'recipe':28s} {'mIoU':>7s} {'EutSi':>6s} {'Al3Ni':>6s}  vs pair")
for k, v in res.items():
    c = v.get('vs_pair'); extra = f"{c['mean_delta']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] up {c['improved']}/down {c['worsened']}" if c else ''
    print(f"{k:28s} {v['mIoU']:7.3f} {v['class_IoU']['Eutectic Si']:6.2f} {v['class_IoU']['Al3Ni']:6.2f}  {extra}")
