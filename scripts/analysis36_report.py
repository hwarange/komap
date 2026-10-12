"""Extra train70 EDA and final-model validation20 analysis for the revised report (CPU only, no training)."""
import os, sys, json
from pathlib import Path
ROOT = Path('/home/j-j15a106/komap'); OUT = ROOT / 'reports/final36_evidence'
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]; os.chdir(ROOT)
import numpy as np, cv2
cv2.setNumThreads(4)
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from data import load_split, CLASSES, MASK_RGB
from infer import image_iou, postprocess
from review_submission31 import probability
OUT.mkdir(exist_ok=True)
K3 = np.ones((3, 3), np.uint8)
COLORS = MASK_RGB / 255.


def jdump(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=float), encoding='utf8')


train = load_split('data', 'train'); valid = load_split('data', 'valid'); test = load_split('data', 'test')
assert len(train) == 70 and len(valid) == 20 and len(test) == 10
R = {'scope': 'train70 ground truth for EDA; validation20 only for final-model error analysis; test10 images only for contrast counts'}
# ---------- A. EDA extras on train70 ----------
R['low_contrast_p1_gt_60'] = {s: int(sum(it['low_contrast'] for it in items)) for s, items in [('train70', train), ('validation20', valid), ('test10', test)]}
thick = {}
for it in train:
    lab = it['lab']; w = lab.shape[1]
    for c in (1, 3):
        m = (lab == c).astype(np.uint8)
        if m.sum() == 0:
            continue
        dt = cv2.distanceTransform(m, cv2.DIST_L2, 3); sk = skeletonize(m.astype(bool))
        if sk.sum() == 0:
            continue
        thick.setdefault(w, {}).setdefault(CLASSES[c], []).append(float(np.median(2 * dt[sk])))
R['thickness_px_median_by_width'] = {str(w): {c: float(np.median(v)) for c, v in d.items()} for w, d in sorted(thick.items())}
ceil = {}
for f in (1.5, 2, 3, 4):
    ious = []
    for it in train:
        lab = it['lab']; h, w = lab.shape
        small = cv2.resize(lab, (max(1, round(w / f)), max(1, round(h / f))), interpolation=cv2.INTER_NEAREST)
        back = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        ious.append(image_iou(lab, back))
    ious = np.array(ious)
    ceil[str(f)] = dict(mIoU=float(np.nanmean(ious, 1).mean() * 100), class_IoU={c: float(np.nanmean(ious[:, i]) * 100) for i, c in enumerate(CLASSES)})
R['resolution_ceiling_train70'] = ceil
pairs = 0; sisi = 0
for it in train:
    lab = it['lab']
    for a, b in ((lab[:, :-1], lab[:, 1:]), (lab[:-1, :], lab[1:, :])):
        diff = a != b; pairs += int(diff.sum())
        sisi += int((diff & (((a == 2) & (b == 3)) | ((a == 3) & (b == 2)))).sum())
R['si_si_boundary_pairs'] = dict(total_class_boundary_pairs=pairs, primary_eutectic_pairs=sisi, share_percent=sisi / pairs * 100)
by_alloy = {}
for it in train:
    frac = np.bincount(it['lab'].ravel(), minlength=4) / it['lab'].size * 100
    by_alloy.setdefault(it['alloy'], {}).setdefault(it['step'], []).append(frac)
plist = []
for alloy, steps in by_alloy.items():
    if len(steps) < 2:
        continue
    lo, hi = min(steps), max(steps)
    plist.append(dict(alloy=alloy, step_low=lo, step_high=hi, delta_pp=(np.mean(steps[hi], 0) - np.mean(steps[lo], 0)).tolist()))
R['step_pairs_train70'] = dict(n_alloys_with_two_steps=len(plist), al3ni_increase=int(sum(p['delta_pp'][1] > 0 for p in plist)),
                               eutectic_decrease=int(sum(p['delta_pp'][3] < 0 for p in plist)),
                               mean_delta_pp={c: float(np.mean([p['delta_pp'][i] for p in plist])) for i, c in enumerate(CLASSES)}, pairs=plist)
sf = {}
for it in train:
    sf.setdefault(it['step'], []).append(np.bincount(it['lab'].ravel(), minlength=4) / it['lab'].size * 100)
R['step_mean_fraction_train70'] = {str(s): dict(images=len(v), **{c: float(np.mean([x[i] for x in v])) for i, c in enumerate(CLASSES)}) for s, v in sorted(sf.items())}
psi = []; num = 0; den = 0
for it in train:
    lab = it['lab']; m = (lab == 2).astype(np.uint8)
    if m.sum():
        psi.append(m.mean() * 100)
        n, cc, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8); areas = st[1:, 4]
        num += int(areas[areas < 1e-4 * lab.size].sum()); den += int(areas.sum())
R['primary_si_rule_support'] = dict(images_with_primary_si=len(psi), min_fraction_percent_when_present=float(min(psi)),
                                    pixels_in_components_below_0p01pct_of_image_percent=num / den * 100)
# ---------- B. final model on validation20 ----------
P1 = ROOT / 'runs/review32_mit2_train70/valid_probs/mit2/base'; P2 = ROOT / 'runs/submission31_train70/valid_probs/unet3/base'
conf = np.zeros((4, 4), np.int64); band_err = 0; err_total = 0; pix_total = 0; ious_all = []; ious_out = []
eut = {'fn': {}, 'fp': {}}; eut_band = {'fn': {}, 'fp': {}}; frows = []
for it in valid:
    name = it['name']; lab = it['lab']
    prob = (probability(P1 / f'{name}.npy', it) + probability(P2 / f'{name}.npy', it)) / 2
    pred = postprocess(prob); iou = image_iou(lab, pred); ious_all.append(iou)
    conf += np.bincount((lab.astype(np.int64) * 4 + pred).ravel(), minlength=16).reshape(4, 4)
    band = cv2.dilate(lab, K3) != cv2.erode(lab, K3); err = pred != lab
    err_total += int(err.sum()); band_err += int((err & band).sum()); pix_total += int(lab.size)
    out = np.full(4, np.nan)
    for c in range(4):
        g = (lab == c) & ~band; p = (pred == c) & ~band; u = (g | p).sum()
        if u:
            out[c] = (g & p).sum() / u
    ious_out.append(out)
    for kind, mask in (('fn', (lab == 3) & (pred != 3)), ('fp', (pred == 3) & (lab != 3))):
        other = pred if kind == 'fn' else lab
        for c in (0, 1, 2):
            sel = mask & (other == c)
            eut[kind][CLASSES[c]] = eut[kind].get(CLASSES[c], 0) + int(sel.sum())
            eut_band[kind][CLASSES[c]] = eut_band[kind].get(CLASSES[c], 0) + int((sel & band).sum())
    g = np.bincount(lab.ravel(), minlength=4) / lab.size * 100; p = np.bincount(pred.ravel(), minlength=4) / pred.size * 100
    frows.append(dict(name=name, width=int(lab.shape[1]), low=bool(it['low_contrast']), mIoU=float(np.nanmean(iou) * 100),
                      class_IoU=(iou * 100).tolist(), gt=g.tolist(), pred=p.tolist()))
ious_all = np.array(ious_all); ious_out = np.array(ious_out)
mIoU = float(np.nanmean(ious_all, 1).mean() * 100); assert abs(mIoU - 80.62075310543737) < 1e-9, mIoU
V = dict(mIoU=mIoU, class_IoU={c: float(np.nanmean(ious_all[:, i]) * 100) for i, c in enumerate(CLASSES)},
         mIoU_excluding_1px_boundary_band=float(np.nanmean(ious_out, 1).mean() * 100),
         class_IoU_excluding_band={c: float(np.nanmean(ious_out[:, i]) * 100) for i, c in enumerate(CLASSES)},
         error_pixels_within_1px_of_gt_boundary_percent=band_err / err_total * 100, error_pixel_percent_of_all=err_total / pix_total * 100,
         confusion_pixels={CLASSES[i]: {CLASSES[j]: int(conf[i, j]) for j in range(4)} for i in range(4)},
         confusion_row_percent={CLASSES[i]: {CLASSES[j]: float(conf[i, j] / conf[i].sum() * 100) for j in range(4)} for i in range(4)})
tot = sum(eut['fn'].values()) + sum(eut['fp'].values())
V['eutectic_error_breakdown'] = {c: dict(share_of_eutectic_errors_percent=(eut['fn'][c] + eut['fp'][c]) / tot * 100,
                                        within_band_percent=(eut_band['fn'][c] + eut_band['fp'][c]) / max(eut['fn'][c] + eut['fp'][c], 1) * 100,
                                        missed=eut['fn'][c], false=eut['fp'][c]) for c in ['Al3Ni', 'Al', 'Primary Si']}
G = np.array([r['gt'] for r in frows]); Pp = np.array([r['pred'] for r in frows])
fr = {}
for i, c in enumerate(CLASSES + ['Si total']):
    g = G[:, i] if i < 4 else G[:, 2] + G[:, 3]; p = Pp[:, i] if i < 4 else Pp[:, 2] + Pp[:, 3]; d = p - g; ok = g >= 0.5
    fr[c] = dict(gt_mean_percent=float(g.mean()), mae_pp=float(np.abs(d).mean()), bias_pp=float(d.mean()), max_abs_pp=float(np.abs(d).max()),
                 relative_error_median_percent=float(np.median(np.abs(d[ok]) / g[ok] * 100)),
                 images_within_10pct_relative=int((np.abs(d[ok]) / g[ok] <= .1).sum()), images_evaluated=int(ok.sum()))
V['area_fraction_error'] = fr
sub = {}
for key, sel in [('width675', [r['width'] == 675 for r in frows]), ('width_other', [r['width'] != 675 for r in frows]),
                 ('low_contrast', [r['low'] for r in frows]), ('normal_contrast', [not r['low'] for r in frows])]:
    sel = np.array(sel); sub[key] = dict(images=int(sel.sum()), mIoU=float(np.mean([r['mIoU'] for r, s in zip(frows, sel) if s])))
V['subgroups'] = sub


def kg(E, nu):
    return E / (3 * (1 - 2 * nu)), E / (2 * (1 + nu))


PH = {'Al': kg(70, .33), 'Al3Ni': kg(160, .30), 'Si': kg(162, .22)}


def hs(f):
    Ks = np.array([PH[p][0] for p in PH]); Gs = np.array([PH[p][1] for p in PH]); f = np.array([f[p] for p in PH]); f = f / f.sum()
    out = []
    for Km, Gm in ((Ks.min(), Gs.min()), (Ks.max(), Gs.max())):
        K = 1 / np.sum(f / (Ks + 4 / 3 * Gm)) - 4 / 3 * Gm
        zz = Gm / 6 * (9 * Km + 8 * Gm) / (Km + 2 * Gm)
        Gc = 1 / np.sum(f / (Gs + zz)) - zz
        out.append(9 * K * Gc / (3 * K + Gc))
    return out


check = hs({'Al': .797, 'Al3Ni': 0., 'Si': .203})
Eg = np.array([np.mean(hs({'Al': r['gt'][0], 'Al3Ni': r['gt'][1], 'Si': r['gt'][2] + r['gt'][3]})) for r in frows])
Ep = np.array([np.mean(hs({'Al': r['pred'][0], 'Al3Ni': r['pred'][1], 'Si': r['pred'][2] + r['pred'][3]})) for r in frows])
V['modulus_hs_midpoint_gpa'] = dict(constants='Al E70 nu0.33; Al3Ni E160 nu0.30 (assumed); Si E162 nu0.22; 2D area fraction taken as volume fraction',
                                    check_Al18Si_20p3volSi_lower_upper=check, gt_mean=float(Eg.mean()), gt_std_across_images=float(Eg.std()),
                                    gt_min=float(Eg.min()), gt_max=float(Eg.max()), pred_minus_gt_mean=float((Ep - Eg).mean()),
                                    pred_minus_gt_mean_abs=float(np.abs(Ep - Eg).mean()), pred_minus_gt_max_abs=float(np.abs(Ep - Eg).max()))
V['per_image'] = frows
R['validation20_final_model'] = V
# ---------- test10 predicted fractions + modulus ----------
area = json.loads((ROOT / 'submissions/final35_train70/prediction_area_fractions.json').read_text(encoding='utf8'))
timgs = sorted(area['images'], key=lambda x: x['name'])
tE = [np.mean(hs({'Al': x['area_fraction_percent']['Al'], 'Al3Ni': x['area_fraction_percent']['Al3Ni'],
                  'Si': x['area_fraction_percent']['Primary Si'] + x['area_fraction_percent']['Eutectic Si']})) for x in timgs]
R['test10_predicted'] = [dict(name=x['name'], fractions=x['area_fraction_percent'], hs_mid_gpa=float(e)) for x, e in zip(timgs, tE)]
jdump('analysis36.json', R)
# ---------- figures ----------
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
disp = ['Al', 'Al3Ni', 'Primary\nSi', 'Eutectic\nSi']
eda = json.loads((ROOT / 'reports/final35_evidence/train70_eda.json').read_text(encoding='utf8'))
fig, ax = plt.subplots(2, 2, figsize=(8.2, 5.0), layout='constrained')
a = ax[0, 0]; v = [eda['pooled_class_pixel_percent'][c] for c in CLASSES]; b = [eda['pooled_class_boundary_percent'][c] for c in CLASSES]
a.bar(disp, v, color=COLORS, edgecolor='#67717c', linewidth=.5)
for x, (y, bb) in enumerate(zip(v, b)):
    a.text(x, y + 1, f'{y:.1f}%\n(edge {bb:.0f}%)', ha='center', fontsize=7.5)
a.set(title='A  Class area in train70 (edge: share within 1 px of a boundary)', ylabel='Pixel share (%)', ylim=(0, 75))
a = ax[0, 1]; ws = sorted(thick); x = np.arange(len(ws))
for k, (c, col) in enumerate([('Al3Ni', COLORS[1]), ('Eutectic Si', COLORS[3])]):
    vals = [R['thickness_px_median_by_width'][str(w)][c] for w in ws]
    a.bar(x + (k - .5) * .38, vals, .38, color=col, edgecolor='#67717c', linewidth=.5, label=c)
    for xi, yv in zip(x, vals):
        a.text(xi + (k - .5) * .38, yv + .05, f'{yv:.1f}', ha='center', fontsize=7.5)
a.set(title='B  Median thickness of thin phases by image width', xticks=x, xticklabels=[f'{w} px' for w in ws], ylabel='Thickness (px)')
a.legend(fontsize=8, frameon=False)
a = ax[1, 0]; rows = sorted(eda['per_image'], key=lambda r: (r['p1'], r['name']))
for i, r in enumerate(rows):
    a.plot([r['p1'], r['p99']], [i, i], color='#537da4', alpha=.6, linewidth=1)
a.axvline(60, color='#bb7143', linestyle='--', linewidth=1); a.text(62, 2, 'p1 = 60', color='#bb7143', fontsize=8)
a.set(title='C  Raw gray range p1 to p99 per image', xlabel='Gray value (0-255)', ylabel='Images sorted by p1', xlim=(0, 255))
a = ax[1, 1]; fs = ['1', '1.5', '2', '3', '4']; xs = list(range(len(fs)))
for i, c in enumerate(CLASSES):
    a.plot(xs, [100] + [ceil[f]['class_IoU'][c] for f in fs[1:]], 'o-', color=COLORS[i], markeredgecolor='#555', label=c, linewidth=1.2, markersize=4)
a.plot(xs, [100] + [ceil[f]['mIoU'] for f in fs[1:]], 'o-', color='black', label='mIoU', linewidth=2, markersize=4)
a.set(title='D  Ceiling: perfect mask down/up-sampled', xticks=xs, xticklabels=['1/1', '1/1.5', '1/2', '1/3', '1/4'],
      xlabel='Prediction resolution', ylabel='IoU vs original (%)', ylim=(25, 102))
a.legend(fontsize=7, frameon=False, ncol=2)
fig.savefig(OUT / 'report36_eda.png', dpi=220, bbox_inches='tight', facecolor='white'); plt.close(fig)
fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.3), gridspec_kw={'width_ratios': [1, 1.15]}, layout='constrained')
a = ax[0]; M = np.array([[V['confusion_row_percent'][CLASSES[i]][CLASSES[j]] for j in range(4)] for i in range(4)])
a.imshow(M, cmap='Blues', vmin=0, vmax=100)
for i in range(4):
    for j in range(4):
        a.text(j, i, f'{M[i, j]:.1f}', ha='center', va='center', fontsize=8, color='white' if M[i, j] > 60 else 'black')
a.set(xticks=range(4), yticks=range(4), xticklabels=['Al', 'Al3Ni', 'Pri. Si', 'Eut. Si'], yticklabels=['Al', 'Al3Ni', 'Pri. Si', 'Eut. Si'],
      xlabel='Predicted', ylabel='Ground truth', title='A  Row-normalised confusion (%) on validation20')
for s in a.spines.values():
    s.set_visible(True)
a = ax[1]
for i, c in enumerate(CLASSES):
    a.scatter(G[:, i], Pp[:, i], s=22, color=COLORS[i], edgecolor='#555', linewidth=.4, label=c, zorder=3)
lim = max(G.max(), Pp.max()) * 1.05; a.plot([0, lim], [0, lim], color='#999', linewidth=1, zorder=1)
a.set(xlabel='Ground-truth area fraction (%)', ylabel='Predicted area fraction (%)', title='B  Phase area fraction per image (20 images)', xlim=(0, lim), ylim=(0, lim))
a.legend(fontsize=8, frameon=False)
fig.savefig(OUT / 'report36_errors.png', dpi=220, bbox_inches='tight', facecolor='white'); plt.close(fig)
fig, a = plt.subplots(figsize=(8.0, 3.0), layout='constrained')
names = [x['name'] for x in timgs]; left = np.zeros(len(names))
for i, c in enumerate(CLASSES):
    vals = np.array([x['area_fraction_percent'][c] for x in timgs])
    a.barh(names, vals, left=left, color=COLORS[i], edgecolor='#67717c', linewidth=.4, label=c)
    for y, (l, v) in enumerate(zip(left, vals)):
        if v > 6:
            a.text(l + v / 2, y, f'{v:.1f}', ha='center', va='center', fontsize=7, color='white' if i in (1, 2) else 'black')
    left += vals
for y, e in enumerate(tE):
    a.text(101, y, f'{e:.1f} GPa', va='center', fontsize=7.5)
a.set(xlim=(0, 118), xlabel='Predicted 2-D area fraction (%)   |   right: Hashin-Shtrikman midpoint modulus estimate',
      title='Test10 predicted phase fractions from the submitted masks')
a.invert_yaxis(); a.legend(fontsize=7.5, frameon=False, loc='lower right', ncol=4, bbox_to_anchor=(1, 1.0))
fig.savefig(OUT / 'report36_test_fractions.png', dpi=220, bbox_inches='tight', facecolor='white'); plt.close(fig)
print(json.dumps({k: v for k, v in R.items() if k not in ('step_pairs_train70', 'validation20_final_model', 'test10_predicted')}, indent=1))
print(json.dumps({k: v for k, v in V.items() if k != 'per_image'}, indent=1))
