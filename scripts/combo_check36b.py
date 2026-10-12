"""Follow-up zero-training checks on the train70 5-fold OOF (CPU only).

1. Does the Al3Ni probability gain that looked good on validation20 (x1.15) also help on the 70-image OOF?
2. Alloy-bootstrap 95% intervals for the triple / quad means versus the final pair, on OOF and on validation20.
"""
import os, sys, json
from pathlib import Path
ROOT = Path('/home/j-j15a106/komap'); OUT = ROOT / 'reports/final36_evidence'
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]; os.chdir(ROOT)
import numpy as np
from data import load_split, CLASSES
from infer import image_iou, postprocess
from review_submission31 import probability

valid = load_split('data', 'valid'); train = load_split('data', 'train')
VP = {'mit2': ROOT / 'runs/review32_mit2_train70/valid_probs/mit2/base',
      'mit4': ROOT / 'runs/submission31_train70/valid_probs/mit4/base',
      'context4': ROOT / 'runs/submission31_train70/valid_probs/context4/base',
      'unet3': ROOT / 'runs/submission31_train70/valid_probs/unet3/base'}
P34 = ROOT / 'runs/review34_train70_group5fold/probs'


def oof_loader(f, it):
    for fold in range(5):
        p = P34 / f / f'fold{fold}' / f"{it['name']}.npy"
        if p.exists():
            return probability(p, it)
    raise FileNotFoundError(it['name'])


def biased(prob, cls, gain):
    q = prob.copy(); q[cls] *= gain; return postprocess(q / q.sum(0, keepdims=True))


RECIPES = {'pair': ('mit2', 'unet3'), 'triple_ctx': ('mit2', 'context4', 'unet3'), 'triple_mit4': ('mit2', 'mit4', 'unet3'),
           'quad': ('mit2', 'mit4', 'context4', 'unet3')}
GAINS = (1.0, 1.1, 1.15, 1.3)


def run(items, loader):
    rows = []
    for it in items:
        probs = {f: loader(f, it) for f in VP}; lab = it['lab']; r = {'name': it['name'], 'alloy': it['alloy']}
        for key, fams in RECIPES.items():
            p = sum(probs[f] for f in fams) / len(fams)
            r[key] = float(np.nanmean(image_iou(lab, postprocess(p))) * 100)
            if key == 'pair':
                for g in GAINS[1:]:
                    r[f'pair_al3ni_x{g}'] = float(np.nanmean(image_iou(lab, biased(p, 1, g))) * 100)
        rows.append(r)
    return rows


def ci(rows, a, b, reps=10000, seed=0):
    alloys = sorted({r['alloy'] for r in rows}); by = {al: [r[a] - r[b] for r in rows if r['alloy'] == al] for al in alloys}
    rng = np.random.default_rng(seed); d = np.array([r[a] - r[b] for r in rows])
    boots = []
    for _ in range(reps):
        pick = rng.choice(alloys, len(alloys), replace=True)
        vals = np.concatenate([by[al] for al in pick]); boots.append(vals.mean())
    return dict(mean_delta=float(d.mean()), improved=int((d > 0).sum()), worsened=int((d < 0).sum()),
                ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))])


out = {}
for name, items, loader in (('train70_oof', train, oof_loader), ('validation20', valid, lambda f, it: probability(VP[f] / f"{it['name']}.npy", it))):
    rows = run(items, loader); keys = [k for k in rows[0] if k not in ('name', 'alloy')]
    out[name] = {'mean': {k: float(np.mean([r[k] for r in rows])) for k in keys},
                 'vs_pair': {k: ci(rows, k, 'pair') for k in keys if k != 'pair'}}
    print(f'== {name} ==')
    for k in keys:
        line = f"{k:18s} {out[name]['mean'][k]:7.3f}"
        if k != 'pair':
            c = out[name]['vs_pair'][k]; line += f"  vs pair {c['mean_delta']:+.3f}  [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]  up {c['improved']} / down {c['worsened']}"
        print(line)
(OUT / 'combo_check36b.json').write_text(json.dumps(out, indent=2), encoding='utf8')
