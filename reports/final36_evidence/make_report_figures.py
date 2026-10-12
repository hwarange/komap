"""Regenerate the EDA, test-fraction and pipeline figures locally from the saved JSON evidence (layout fixes)."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

E = 'reports/final36_evidence/'
R = json.load(open(E + 'analysis36.json', encoding='utf-8'))
eda = json.load(open('reports/final35_evidence/train70_eda.json', encoding='utf-8'))
CLASSES = ['Al', 'Al3Ni', 'Primary Si', 'Eutectic Si']
COLORS = np.array([(153, 127, 76), (25, 76, 153), (76, 178, 76), (204, 204, 204)]) / 255.
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
disp = ['Al', 'Al3Ni', 'Primary\nSi', 'Eutectic\nSi']

# ---- EDA 4-panel ----
fig, ax = plt.subplots(2, 2, figsize=(8.2, 5.0), layout='constrained')
a = ax[0, 0]; v = [eda['pooled_class_pixel_percent'][c] for c in CLASSES]; b = [eda['pooled_class_boundary_percent'][c] for c in CLASSES]
a.bar(disp, v, color=COLORS, edgecolor='#67717c', linewidth=.5)
for x, (y, bb) in enumerate(zip(v, b)):
    a.text(x, y + 1, f'{y:.1f}%\n(edge {bb:.0f}%)', ha='center', fontsize=7.5)
a.set(title='A  Class area (edge = share within 1 px of a boundary)', ylabel='Pixel share (%)', ylim=(0, 75))
a.title.set_fontsize(8.5)
a = ax[0, 1]; th = R['thickness_px_median_by_width']; ws = sorted(th, key=int); x = np.arange(len(ws))
for k, (c, col) in enumerate([('Al3Ni', COLORS[1]), ('Eutectic Si', COLORS[3])]):
    vals = [th[w][c] for w in ws]
    a.bar(x + (k - .5) * .38, vals, .38, color=col, edgecolor='#67717c', linewidth=.5, label=c)
    for xi, yv in zip(x, vals):
        a.text(xi + (k - .5) * .38, yv + .08, f'{yv:.1f}', ha='center', fontsize=7.5)
a.set(title='B  Median thickness of thin phases by image width', xticks=x, xticklabels=[f'{w} px' for w in ws], ylabel='Thickness (px)')
a.title.set_fontsize(8.5); a.legend(fontsize=8, frameon=False)
a = ax[1, 0]; rows = sorted(eda['per_image'], key=lambda r: (r['p1'], r['name']))
for i, r in enumerate(rows):
    a.plot([r['p1'], r['p99']], [i, i], color='#537da4', alpha=.6, linewidth=1)
a.axvline(60, color='#bb7143', linestyle='--', linewidth=1); a.text(63, 66, 'p1 = 60', color='#bb7143', fontsize=8)
a.set(title='C  Raw gray range p1 to p99 per image', xlabel='Gray value (0-255)', ylabel='Images sorted by p1', xlim=(0, 255)); a.title.set_fontsize(8.5)
a = ax[1, 1]; ceil = R['resolution_ceiling_train70']; fs = ['1.5', '2', '3', '4']; xs = list(range(5))
for i, c in enumerate(CLASSES):
    a.plot(xs, [100] + [ceil[f]['class_IoU'][c] for f in fs], 'o-', color=COLORS[i], markeredgecolor='#555', label=c, linewidth=1.2, markersize=4)
a.plot(xs, [100] + [ceil[f]['mIoU'] for f in fs], 'o-', color='black', label='mIoU', linewidth=2, markersize=4)
a.set(title='D  Ceiling: perfect mask down/up-sampled', xticks=xs, xticklabels=['1/1', '1/1.5', '1/2', '1/3', '1/4'],
      xlabel='Prediction resolution', ylabel='IoU vs original (%)', ylim=(25, 102)); a.title.set_fontsize(8.5)
a.legend(fontsize=7, frameon=False, ncol=2)
fig.savefig(E + 'report36_eda.png', dpi=220, bbox_inches='tight', facecolor='white'); plt.close(fig)

# ---- test10 fractions ----
t = R['test10_predicted']
fig, a = plt.subplots(figsize=(8.0, 3.1), layout='constrained')
names = [x['name'] for x in t]; left = np.zeros(len(names))
for i, c in enumerate(CLASSES):
    vals = np.array([x['fractions'][c] for x in t])
    a.barh(names, vals, left=left, color=COLORS[i], edgecolor='#67717c', linewidth=.4, label=c)
    for y, (l, v) in enumerate(zip(left, vals)):
        if v > 6:
            a.text(l + v / 2, y, f'{v:.1f}', ha='center', va='center', fontsize=7, color='white' if i in (1, 2) else 'black')
    left += vals
for y, x in enumerate(t):
    a.text(101.5, y, f"{x['hs_mid_gpa']:.1f} GPa", va='center', fontsize=7.5)
a.set(xlim=(0, 118), xlabel='Predicted 2-D area fraction (%)        right: Hashin-Shtrikman midpoint modulus estimate')
a.invert_yaxis(); a.tick_params(axis='y', labelsize=8)
a.legend(fontsize=7.5, frameon=False, loc='upper center', bbox_to_anchor=(0.45, -0.22), ncol=4)
fig.savefig(E + 'report36_test_fractions.png', dpi=220, bbox_inches='tight', facecolor='white'); plt.close(fig)

# ---- pipeline ----
plt.rcParams.update({'font.size': 8})
fig, ax = plt.subplots(figsize=(6.7, 2.5))
ax.set_xlim(0, 100); ax.set_ylim(0, 33); ax.axis('off')

def box(x, y, w, h, text, fc='#f2f4f7', ec='#67717c', fs=6.9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.3,rounding_size=1.2', fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, fontweight='bold' if bold else 'normal', linespacing=1.25)

def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=9, lw=0.9, color='#4a5560'))

box(1, 12.5, 15, 10, 'Input image\n(gray, 675-1440 px)', fc='#ffffff')
box(19.5, 12.5, 16, 10, 'Per-image p1-p99\nnormalisation')
arrow(16.3, 17.5, 19.2, 17.5)
box(40, 21, 25, 10, 'Scale x round(2560 / W)\n(2x typical) -> 1536 px tiles\nU-Net / MiT-B2  (60 ep)', fc='#e3eef7', ec='#537da4')
box(40, 4, 25, 10, 'Scale x round(3840 / W)\n(3x typical) -> 1536 px tiles\nUNet++ / ResNet34  (120 ep)', fc='#e7f1ec', ec='#3f8f6b')
arrow(35.8, 18.5, 39.7, 25); arrow(35.8, 16.5, 39.7, 10)
box(68, 21, 14, 10, '8-way TTA\nGaussian tile blend\nresize to original', fc='#e3eef7', ec='#537da4')
box(68, 4, 14, 10, '8-way TTA\nGaussian tile blend\nresize to original', fc='#e7f1ec', ec='#3f8f6b')
arrow(65.3, 26, 67.7, 26); arrow(65.3, 9, 67.7, 9)
box(85, 12.5, 14.5, 10, '50:50 soft voting\nPrimary Si min-area rule\nRGB mask + fractions', fc='#fff4e6', ec='#bb7143', bold=True, fs=6.6)
arrow(82.3, 25, 84.7, 19.5); arrow(82.3, 10, 84.7, 15.5)
ax.text(50, 0.8, 'Training: train 70 images only; ImageNet-initialised encoders; CE + Dice loss with 3x boundary weight; 8 random crops per image; rot90 / flip; photometric augmentation',
        ha='center', va='bottom', fontsize=6.3, color="#4a5560", style="italic")
fig.savefig(E + 'report36_pipeline.png', dpi=220, bbox_inches='tight', facecolor='white'); plt.close(fig)
print('figures ok')
