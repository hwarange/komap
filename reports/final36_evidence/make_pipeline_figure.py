"""Pipeline diagram for the revised report (English labels; Korean caption lives in the document)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8})
fig, ax = plt.subplots(figsize=(6.7, 2.5))
ax.set_xlim(0, 100); ax.set_ylim(0, 33); ax.axis('off')


def box(x, y, w, h, text, fc='#f2f4f7', ec='#67717c', fs=6.2, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.3,rounding_size=1.2', fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, fontweight='bold' if bold else 'normal', linespacing=1.3)


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=9, lw=0.9, color='#4a5560'))


box(1, 12.5, 15, 10, 'Input image\n(gray, 675-1440 px)', fc='#ffffff')
box(18.5, 12.5, 16, 10, 'Per-image\np1-p99 normalisation')
arrow(16.3, 17.5, 18.2, 17.5)
box(38.5, 21, 25, 10, 'x round(2560 / W), about 2x\n1536 px tiles\nU-Net / MiT-B2  (60 ep)', fc='#e3eef7', ec='#537da4')
box(38.5, 4, 25, 10, 'x round(3840 / W), about 3x\n1536 px tiles\nUNet++ / ResNet34  (120 ep)', fc='#e7f1ec', ec='#3f8f6b')
arrow(34.8, 18.5, 38.2, 25); arrow(34.8, 16.5, 38.2, 10)
box(66, 21, 15, 10, '8-way TTA\nGaussian blend\nresize to original', fc='#e3eef7', ec='#537da4')
box(66, 4, 15, 10, '8-way TTA\nGaussian blend\nresize to original', fc='#e7f1ec', ec='#3f8f6b')
arrow(63.8, 26, 65.7, 26); arrow(63.8, 9, 65.7, 9)
box(83.5, 12.5, 16, 10, '50:50 soft voting\nPrimary Si rule\nRGB mask + fractions', fc='#fff4e6', ec='#bb7143', bold=True, fs=6.1)
arrow(81.3, 25, 83.2, 19.5); arrow(81.3, 10, 83.2, 15.5)
ax.text(50, 0.8, 'Training: train 70 images only; ImageNet-initialised encoders; CE + Dice with 3x boundary weight; 8 crops per image; rot90 / flip; photometric augmentation',
        ha='center', va='bottom', fontsize=6.0, color='#4a5560', style='italic')
fig.savefig('reports/final36_evidence/report36_pipeline.png', dpi=220, bbox_inches='tight', facecolor='white')
print('pipeline ok')
