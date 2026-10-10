"""Frozen final-pair full90 CPU diagnostics; no training, inference or tuning.

Run on the data server using the existing project environment. Only aggregate
diagnostics and image identifiers are written; no source images are exported.
"""
import os

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/review26_final_strengthening/diagnostics'
PRIOR = ROOT / 'runs/review22_pair'
TRUSTED = {
    'plan.json': '97a72ef3158aa84d91cf30025d4b778e73e8cd0affb878baef15f9454239a592',
    'results.json': '3f51b84982c992e8abb28ff1be10328ef58a87e60ba39bcd32b7ddb261d216c3',
    'per_image_rows.json': '84469027fe8869d266b5764409e8e2a6022e1356d911b631c66f5141dccdb19d',
    'all.done': '21cb6f69da41737f2466331997ba3527471f92d4864f5e36c4fdadac62d62030',
}
sys.path.insert(0, str(ROOT / 'src'))
import cv2
import numpy as np
import torch
from data import CLASSES, load_split
from infer import image_iou, postprocess


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def receipt(path):
    return {'path': path.relative_to(ROOT).as_posix(),
            'sha256': sha(path), 'bytes': path.stat().st_size}


def verify(record):
    path = (ROOT / record['path']).resolve()
    assert path.is_relative_to(ROOT.resolve())
    assert sha(path) == record['sha256'], f'Changed source: {path}'
    if 'bytes' in record:
        assert path.stat().st_size == record['bytes'], path
    return receipt(path)


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(item) for item in value]
    return value


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(clean(value), ensure_ascii=False, indent=2,
                                allow_nan=False) + '\n')


def ratio(num, den):
    return float(num / den) if den else None


def matrix(gt, pred, region=None):
    if region is not None:
        gt, pred = gt[region], pred[region]
    return np.bincount(gt.ravel().astype(np.int64) * 4 + pred.ravel(),
                       minlength=16).reshape(4, 4)


def describe(c):
    n = int(c.sum())
    out = {'matrix_gt_rows_prediction_columns': c.tolist(), 'pixels': n,
           'pixel_error_fraction': ratio(n - int(c.trace()), n), 'classes': {}}
    for k, label in enumerate(CLASSES):
        tp = int(c[k, k]); fn = int(c[k].sum()) - tp; fp = int(c[:, k].sum()) - tp
        out['classes'][label] = {
            'tp': tp, 'fn': fn, 'fp': fp, 'iou': ratio(tp, tp + fn + fp),
            'precision': ratio(tp, tp + fp), 'recall': ratio(tp, tp + fn),
            'gt_area_fraction': ratio(tp + fn, n),
            'prediction_area_fraction': ratio(tp + fp, n),
            'area_bias_pp': 100 * (fp - fn) / n if n else None,
            'FN_predicted_as': {CLASSES[j]: int(c[k, j]) for j in range(4) if j != k},
            'FP_true_class': {CLASSES[j]: int(c[j, k]) for j in range(4) if j != k},
        }
    return out


def avg(values):
    values = [value for value in values if value is not None and math.isfinite(value)]
    return float(np.mean(values)) if values else None


def aggregate(rows, names):
    selected = [rows[name] for name in names]
    out = {'images': len(names), 'image_names': names,
           'mIoU_pct_image_macro': avg([r['mIoU_pct'] for r in selected]),
           'regions': {}, 'image_macro_classes': {}}
    for region in ('all', 'boundary_2px', 'interior'):
        total = np.sum([r['regions'][region]['matrix_gt_rows_prediction_columns']
                        for r in selected], axis=0, dtype=np.int64)
        out['regions'][region] = {
            'pixel_pooled': describe(total),
            'pixel_error_fraction_image_macro': avg([
                r['regions'][region]['pixel_error_fraction'] for r in selected]),
        }
    for label in CLASSES:
        parts = [r['regions']['all']['classes'][label] for r in selected]
        out['image_macro_classes'][label] = {
            'IoU_pct': 100 * avg([p['iou'] for p in parts]),
            'IoU_valid_images': sum(p['iou'] is not None for p in parts),
            'precision_pct': 100 * avg([p['precision'] for p in parts]),
            'precision_valid_images': sum(p['precision'] is not None for p in parts),
            'recall_pct': 100 * avg([p['recall'] for p in parts]),
            'recall_valid_images': sum(p['recall'] is not None for p in parts),
            'gt_area_fraction_mean': avg([p['gt_area_fraction'] for p in parts]),
            'prediction_area_fraction_mean': avg([p['prediction_area_fraction'] for p in parts]),
            'area_MAE_pp': avg([abs(p['area_bias_pp']) for p in parts]),
            'area_bias_pp': avg([p['area_bias_pp'] for p in parts]),
        }
    all_pixel = out['regions']['all']['pixel_pooled']
    boundary = out['regions']['boundary_2px']['pixel_pooled']
    all_c = np.asarray(all_pixel['matrix_gt_rows_prediction_columns'])
    boundary_c = np.asarray(boundary['matrix_gt_rows_prediction_columns'])
    out['boundary_pixel_fraction'] = boundary['pixels'] / all_pixel['pixels']
    out['fraction_of_all_errors_in_boundary'] = ratio(
        int(boundary_c.sum() - boundary_c.trace()), int(all_c.sum() - all_c.trace()))
    return out


def plot(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    full = result['groups']['all90']
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    c = np.asarray(full['regions']['all']['pixel_pooled']['matrix_gt_rows_prediction_columns'])
    percent = 100 * c / c.sum(1, keepdims=True)
    ax = axes[0, 0]
    ax.imshow(percent, vmin=0, vmax=100, cmap='Blues')
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f'{percent[i, j]:.1f}%', ha='center', va='center',
                    color='white' if percent[i, j] > 50 else '#111111')
    labels = ['Al', 'Al3Ni', 'Primary Si', 'Eutectic Si']
    ax.set(xticks=range(4), yticks=range(4), xticklabels=labels, yticklabels=labels,
           xlabel='Predicted class', ylabel='Ground truth class',
           title='Pixel confusion: each GT row totals 100%')
    ax = axes[0, 1]
    regions = ['boundary_2px', 'interior']
    values = [100 * full['regions'][r]['pixel_pooled']['pixel_error_fraction'] for r in regions]
    bars = ax.bar(['2 px boundary band', 'Interior'], values, color=['#dd8452', '#4c72b0'])
    ax.bar_label(bars, fmt='%.2f%%')
    ax.set(ylabel='Pixel error (%)', ylim=(0, max(values) * 1.2),
           title='Fixed native-pixel regions; descriptive only')
    ax = axes[1, 0]
    mae = [full['image_macro_classes'][c]['area_MAE_pp'] for c in CLASSES]
    bias = [full['image_macro_classes'][c]['area_bias_pp'] for c in CLASSES]
    x = np.arange(4)
    ax.bar(x - .18, mae, .36, label='MAE', color='#4c72b0')
    ax.bar(x + .18, bias, .36, label='Signed bias', color='#dd8452')
    ax.axhline(0, color='#555555', linewidth=.8)
    ax.set(xticks=x, xticklabels=labels, ylabel='Area fraction error (percentage points)',
           title='2D phase area: mean across 90 images')
    ax.legend()
    ax = axes[1, 1]
    keys = ['eut_area_low30', 'eut_area_middle30', 'eut_area_high30']
    positions = np.arange(3)
    for offset, metric, label, color in [(-.18, 'recall_pct', 'Recall', '#4c72b0'),
                                        (.18, 'precision_pct', 'Precision', '#55a868')]:
        values = [result['groups'][k]['image_macro_classes']['Eutectic Si'][metric] for k in keys]
        bars = ax.bar(positions + offset, values, .36, label=label, color=color)
        ax.bar_label(bars, fmt='%.1f', fontsize=9)
    ax.set(xticks=positions, xticklabels=['Low (30)', 'Middle (30)', 'High (30)'],
           ylabel='Image-macro (%)', ylim=(0, 100), title='Eutectic Si by GT area fraction rank')
    ax.legend(loc='lower right')
    fig.suptitle('Frozen MiT 4x + UNet++ 3x, 50:50 | Exploratory full90 OOF diagnostics', fontsize=14)
    fig.savefig(OUT / 'summary.png', dpi=160, metadata={'Software': 'Matplotlib'})
    plt.close(fig)


def main():
    start = time.monotonic()
    os.chdir(ROOT)
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    assert not any((OUT / n).exists() for n in ('results.json', 'per_image.json', 'summary.png', 'all.done'))
    pins = [verify({'path': f'runs/review22_pair/{name}', 'sha256': digest})
            for name, digest in TRUSTED.items()]
    plan, prior, reference, done = [read(PRIOR / name) for name in TRUSTED]
    assert done['outcome'] == 'fixed_two_model_pair_complete'
    assert done['plan_sha256'] == TRUSTED['plan.json']
    assert done['results_sha256'] == TRUSTED['results.json']
    assert plan['weights'] == [0.5, 0.5]
    pins += [verify(r) for r in plan['source_receipts']]
    pins += [verify(r) for r in plan['data_receipts']]
    pins.append(receipt(Path(__file__)))
    fold_of = read(ROOT / plan['split']['path'])
    items = load_split(ROOT / 'data', 'train') + load_split(ROOT / 'data', 'valid')
    assert len(items) == len(fold_of) == 90
    assert {it['name'] for it in items} == set(fold_of)
    source_map = {(r['name'], r['source']): r for r in prior['source_files']}
    rows = {}
    for index, item in enumerate(items, 1):
        name = item['name']; gt = item['lab']
        assert gt.shape == item['gray'].shape and np.isin(gt, np.arange(4)).all()
        probabilities = []
        for source in ('candidate', 'unet'):
            rec = source_map[(name, source)]
            pins.append(verify(rec))
            prob = np.load(ROOT / rec['path'], allow_pickle=False).astype(np.float32)
            assert prob.shape == (4, *gt.shape) and np.isfinite(prob).all()
            assert prob.min() >= 0 and prob.max() <= 1
            assert np.max(np.abs(prob.sum(0) - 1)) < .005
            probabilities.append(prob)
        pred = postprocess((probabilities[0] + probabilities[1]) / 2)
        ious = image_iou(gt, pred)
        expected = np.asarray(reference['rows']['x4_mit_x3_unetpp_pair'][name]['tta_pp'], float)
        np.testing.assert_allclose(ious, expected, atol=1e-10, rtol=0, equal_nan=True)
        edge = np.zeros(gt.shape, np.uint8)
        d = gt[:, 1:] != gt[:, :-1]; edge[:, 1:] |= d; edge[:, :-1] |= d
        d = gt[1:] != gt[:-1]; edge[1:] |= d; edge[:-1] |= d
        band = cv2.dilate(edge, np.ones((5, 5), np.uint8)).astype(bool)
        rows[name] = {'shape': list(gt.shape), 'alloy': item['alloy'], 'step': item['step'],
                      'fold': fold_of[name], 'IoU_pct': (ious * 100).tolist(),
                      'mIoU_pct': float(np.nanmean(ious) * 100),
                      'regions': {'all': describe(matrix(gt, pred)),
                                  'boundary_2px': describe(matrix(gt, pred, band)),
                                  'interior': describe(matrix(gt, pred, ~band))}}
        if index % 15 == 0:
            print(f'Validated {index}/90 images', flush=True)
    names = sorted(rows)
    area_order = sorted(names, key=lambda n: (rows[n]['regions']['all']['classes']['Eutectic Si']['gt_area_fraction'], n))
    groups = {'all90': names,
              'width675': [n for n in names if rows[n]['shape'][1] == 675],
              'other_widths': [n for n in names if rows[n]['shape'][1] != 675],
              'eut_area_low30': area_order[:30], 'eut_area_middle30': area_order[30:60],
              'eut_area_high30': area_order[60:]}
    aggregates = {key: aggregate(rows, names) for key, names in groups.items()}
    baseline = prior['scopes']['full90']['scores']['x4_mit_x3_unetpp_pair']
    assert abs(aggregates['all90']['mIoU_pct_image_macro'] - baseline) < 1e-9
    result = {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'images': 90,
        'recipe': 'MiT 4x crop1536 60 epochs + UNet++ 3x 120 epochs; fixed 50:50 soft voting and existing postprocess',
        'class_order': CLASSES, 'groups': aggregates,
        'boundary_definition': 'Four-neighbor GT class changes marked on both sides, then dilated with a 5x5 square (radius 2 in original image pixels). Conditional region errors, not tolerance metrics.',
        'area_definition': 'Class pixels divided by original-image pixels. Per-image predicted minus GT fraction in percentage points; MAE is mean absolute per-image error. 2D area is not mass fraction or mechanical properties.',
        'group_definition': 'Width675 versus all other native widths. Eutectic GT area fraction sorted ascending; image name breaks ties, then fixed 30-image thirds. Exploratory descriptions, not tuned decision thresholds.',
        'scope': 'Repeatedly examined full90 grouped OOF; diagnostic associations, not independent validation or causal proof. Pixel-pooled confusion and image-macro statistics explicitly separated.',
        'metric': 'mIoU_pct_image_macro is mean of per-image mean class IoU; classes absent from GT and prediction excluded within image. Precision undefined if no predicted pixels; recall undefined if no GT pixels.',
        'verification': {'all90_per_class_IoUs_reproduced': True, 'atol': 1e-10,
                         'reference_mIoU_pct': baseline, 'receipts': pins},
        'gpu_used': False, 'training': False, 'new_model_inference': False,
        'threshold_or_weight_search': False, 'submission_changed': False,
        'duration_seconds_before_plot': time.monotonic() - start,
    }
    write(OUT / 'per_image.json', {'class_order': CLASSES, 'rows': rows})
    write(OUT / 'results.json', result)
    plot(result)
    write(OUT / 'all.done', {'created_utc': datetime.now(timezone.utc).isoformat(),
                           'outcome': 'frozen_final_pair_full90_diagnostics_complete',
                           'duration_seconds': time.monotonic() - start,
                           'artifacts': [receipt(OUT / name) for name in ('results.json', 'per_image.json', 'summary.png')],
                           'script': receipt(Path(__file__)), 'gpu_used': False,
                           'submission_changed': False})
    print(json.dumps({'complete': True, 'duration_seconds': time.monotonic() - start,
                      'mIoU_pct': baseline}), flush=True)


if __name__ == '__main__':
    main()
