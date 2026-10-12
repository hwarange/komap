"""Zero-training checks from saved probabilities (CPU only).

1. Equal-mean combinations of the four train70 families on validation20 and on the train70 5-fold OOF.
2. Boundary-band sensitivity of the final pair: does shifting the Eutectic Si / Al3Ni boundary by one pixel
   or biasing their probabilities help? (If both directions hurt, the boundary error is symmetric noise.)
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

valid = load_split('data', 'valid'); train = load_split('data', 'train')
VP = {'mit2': ROOT / 'runs/review32_mit2_train70/valid_probs/mit2/base',
      'mit4': ROOT / 'runs/submission31_train70/valid_probs/mit4/base',
      'context4': ROOT / 'runs/submission31_train70/valid_probs/context4/base',
      'unet3': ROOT / 'runs/submission31_train70/valid_probs/unet3/base'}
FAM = list(VP)
COMBOS = [c for r in (1, 2, 3, 4) for c in itertools.combinations(FAM, r)]
K3 = np.ones((3, 3), np.uint8)
P34 = ROOT / 'runs/review34_train70_group5fold/probs'


def oof_loader(f, it):
    for fold in range(5):
        p = P34 / f / f'fold{fold}' / f"{it['name']}.npy"
        if p.exists():
            return probability(p, it)
    raise FileNotFoundError(it['name'])


def score_set(items, loader):
    res = {'+'.join(c): [] for c in COMBOS}; band = {'+'.join(c): [0, 0] for c in COMBOS}
    for it in items:
        probs = {f: loader(f, it) for f in FAM}
        lab = it['lab']; bm = cv2.dilate(lab, K3) != cv2.erode(lab, K3)
        for c in COMBOS:
            key = '+'.join(c); p = sum(probs[f] for f in c) / len(c); pred = postprocess(p)
            res[key].append(image_iou(lab, pred))
            err = pred != lab; band[key][0] += int((err & bm).sum()); band[key][1] += int(err.sum())
    out = {}
    for k, v in res.items():
        v = np.array(v)
        out[k] = dict(mIoU=float(np.nanmean(v, 1).mean() * 100),
                      class_IoU={c: float(np.nanmean(v[:, i]) * 100) for i, c in enumerate(CLASSES)},
                      error_pixels=band[k][1], band_share_percent=band[k][0] / max(band[k][1], 1) * 100)
    return out


val = score_set(valid, lambda f, it: probability(VP[f] / f"{it['name']}.npy", it))
oof = score_set(train, oof_loader)
assert abs(val['mit2+unet3']['mIoU'] - 80.62075310543737) < 1e-9
assert abs(oof['mit2+unet3']['mIoU'] - 80.77829756479382) < 1e-9

# ---- boundary sensitivity of the final pair on validation20 ----
def shifted(pred, prob, cls, mode):
    m = (pred == cls).astype(np.uint8)
    if mode == 'dilate':
        grown = cv2.dilate(m, K3).astype(bool) & ~m.astype(bool)
        out = pred.copy(); out[grown] = cls; return out
    eroded = m.astype(bool) & ~cv2.erode(m, K3).astype(bool)
    alt = prob.copy(); alt[cls] = -1
    out = pred.copy(); out[eroded] = alt.argmax(0)[eroded]; return out


def biased(prob, cls, gain):
    q = prob.copy(); q[cls] *= gain; return postprocess(q / q.sum(0, keepdims=True))


variants = {'baseline': []}
for cls, name in ((3, 'EutecticSi'), (1, 'Al3Ni')):
    for mode in ('dilate', 'erode'):
        variants[f'{name}_{mode}_1px'] = []
    for g in (0.7, 0.85, 1.15, 1.4):
        variants[f'{name}_prob_x{g}'] = []
for it in valid:
    prob = (probability(VP['mit2'] / f"{it['name']}.npy", it) + probability(VP['unet3'] / f"{it['name']}.npy", it)) / 2
    pred = postprocess(prob); lab = it['lab']
    variants['baseline'].append(image_iou(lab, pred))
    for cls, name in ((3, 'EutecticSi'), (1, 'Al3Ni')):
        for mode in ('dilate', 'erode'):
            variants[f'{name}_{mode}_1px'].append(image_iou(lab, shifted(pred, prob, cls, mode)))
        for g in (0.7, 0.85, 1.15, 1.4):
            variants[f'{name}_prob_x{g}'].append(image_iou(lab, biased(prob, cls, g)))
sens = {k: dict(mIoU=float(np.nanmean(np.array(v), 1).mean() * 100),
                EutecticSi_IoU=float(np.nanmean(np.array(v)[:, 3]) * 100), Al3Ni_IoU=float(np.nanmean(np.array(v)[:, 1]) * 100)) for k, v in variants.items()}
result = dict(validation20=val, train70_oof=oof, boundary_sensitivity_validation20=sens,
              note='Equal-weight probability means; same postprocess; validation20 used for scoring only.')
(OUT / 'combo_check36.json').write_text(json.dumps(result, indent=2), encoding='utf8')
print('COMBOS  (val20 mIoU | OOF70 mIoU | val band share %)')
for k in val:
    print(f"{k:28s} {val[k]['mIoU']:7.3f} | {oof[k]['mIoU']:7.3f} | {val[k]['band_share_percent']:5.1f}")
print('SENSITIVITY (val20, final pair)')
for k, v in sens.items():
    print(f"{k:24s} mIoU {v['mIoU']:7.3f}  EutSi {v['EutecticSi_IoU']:6.2f}  Al3Ni {v['Al3Ni_IoU']:6.2f}")
