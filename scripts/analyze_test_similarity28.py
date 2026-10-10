"""CPU-only image similarity diagnostic; never reads test labels or changes models."""
from pathlib import Path
import hashlib
import json
import math
import re
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.feature import local_binary_pattern

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/review28_test_similarity'


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def features(gray):
    q = np.percentile(gray, [1, 10, 25, 50, 75, 90, 99])
    photo = np.r_[q / 255, gray.mean() / 255, gray.std() / 255]
    norm = np.clip((gray.astype(np.float32) - float(q[0])) / max(float(q[-1] - q[0]), 1e-3), 0, 1).astype(np.float32)
    norm = cv2.resize(norm, (675, round(675 * gray.shape[0] / gray.shape[1])), interpolation=cv2.INTER_AREA)
    hist = np.histogram(norm, bins=16, range=(0, 1))[0].astype(float)
    hist /= hist.sum()
    uint = np.round(norm * 255).astype(np.uint8)
    lbp_blocks = []
    for radius in (1, 3):
        codes = local_binary_pattern(uint, 8, radius, method='uniform')
        h = np.bincount(codes.astype(np.int32).ravel(), minlength=10).astype(float)
        lbp_blocks.extend(h / h.sum())
    spatial = []
    for scale in (3, 9, 31):
        mean = cv2.blur(norm, (scale, scale))
        local_sd = np.sqrt(np.maximum(cv2.blur(norm * norm, (scale, scale)) - mean * mean, 0))
        spatial.extend([local_sd.mean(), *np.percentile(local_sd, [50, 90])])
    for sigma in (0, 1, 3):
        img = norm if sigma == 0 else cv2.GaussianBlur(norm, (0, 0), sigma)
        grad = cv2.magnitude(cv2.Sobel(img, cv2.CV_32F, 1, 0), cv2.Sobel(img, cv2.CV_32F, 0, 1))
        spatial.extend([grad.mean(), *np.percentile(grad, [50, 90, 99])])
    return {'photometry': photo.tolist(), 'normalized_histogram': hist.tolist(),
            'normalized_lbp': lbp_blocks, 'normalized_spatial': list(map(float, spatial))}


def main():
    started = time.monotonic()
    assert not OUT.exists() or not any(OUT.iterdir()), 'Preserve completed diagnostics'
    OUT.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'runs/review27_finetune_full/results.json'
    result = read(source)
    delta = result['scopes']['full90']['comparisons']['interface_cap6_vs_baseline_original']['per_image_delta_pp']
    stats_path = ROOT / 'runs/review25_final_pair_eda/image_statistics.json'
    stats = read(stats_path)
    manifest_path = ROOT / 'delivery/final27_repro/manifest.json'
    manifest = read(manifest_path)
    expected = {name: item['sha256'] for name, item in stats.items()}
    expected.update({Path(x['path']).name.removesuffix('_image.png'): x['sha256'] for x in manifest['test_images']})
    images, rows, paths = {}, {}, {}
    for split in ('train', 'valid', 'test'):
        for p in sorted((ROOT / 'data' / split / 'images').glob('*_image.png')):
            name = p.name.removesuffix('_image.png')
            assert sha(p) == expected[name]
            gray = np.array(Image.open(p).convert('L'))
            alloy, step, sample = map(int, re.fullmatch(r'A(\d+)_step(\d+)_sample(\d+)', name).groups())
            paths[name] = p
            images[name] = gray
            rows[name] = dict(split=split, width=gray.shape[1], height=gray.shape[0], alloy=alloy,
                              step=step, sample=sample, sha256=sha(p), features=features(gray),
                              delta_pp=delta.get(name))
    reference = sorted(delta)
    tests = sorted(set(rows) - set(reference))
    assert len(reference) == 90 and len(tests) == 10 and set(rows) == set(expected)
    # Feature scaling is fit to the90 reference images only, without using outcomes.
    block_distances, scaling = {}, {}
    for key in ('photometry', 'normalized_histogram', 'normalized_lbp', 'normalized_spatial'):
        ref = np.asarray([rows[n]['features'][key] for n in reference], float)
        test = np.asarray([rows[n]['features'][key] for n in tests], float)
        center, scale = ref.mean(axis=0), np.maximum(ref.std(axis=0), 1e-6)
        scaling[key] = dict(mean=center.tolist(), sd=scale.tolist())
        block_distances[key] = np.mean(((test[:, None, :] - ref[None, :, :]) / scale) ** 2, axis=-1)
    texture = sum(block_distances[key] for key in ('normalized_histogram', 'normalized_lbp', 'normalized_spatial')) / 3
    distances = dict(photometry=block_distances['photometry'], normalized_texture=texture,
                     equal_photo_texture=(block_distances['photometry'] + texture) / 2)
    comparisons = {}
    for mode, matrix in distances.items():
        for scope in ('all90', 'same_width'):
            output = []
            for i, test_name in enumerate(tests):
                eligible = [j for j, n in enumerate(reference) if scope == 'all90' or rows[n]['width'] == rows[test_name]['width']]
                order = sorted(eligible, key=lambda j: (matrix[i, j], reference[j]))
                near = [dict(name=reference[j], distance_rms=float(math.sqrt(matrix[i, j])),
                             width=rows[reference[j]]['width'], delta_pp=delta[reference[j]]) for j in order[:5]]
                output.append(dict(test=test_name, width=rows[test_name]['width'], neighbors=near,
                                   reference_count=len(eligible), reference_worsened=sum(delta[reference[j]] < 0 for j in eligible),
                                   nearest5_worsened=sum(x['delta_pp'] < 0 for x in near),
                                   nearest5_mean_delta_pp=float(np.mean([x['delta_pp'] for x in near]))))
            comparisons[mode + '_' + scope] = dict(tests=output, neighbor_slots=50,
                worsened_neighbor_slots=sum(x['nearest5_worsened'] for x in output),
                unique_neighbors=len({n['name'] for x in output for n in x['neighbors']}),
                warning='Neighbor slots repeat reference images and are correlated; counts and mean neighbor deltas are not test accuracy estimates.')
    notes = ['Image-only exploratory diagnostic after model selection; no GT masks loaded.',
             'All similarity definitions and k5 are fixed in this script, without tuning to outcomes.',
             'Each texture block has equal mean-squared standardized-feature weight; combined metric equally weights photometry and texture.',
             'Texture uses per-image p1-p99 normalization then common width675. Physical field of view is not calibrated.',
             'Same-width results control pixel-resolution group, but cannot establish same alloy chemistry or semantic class correctness.',
             'The10test images have no labels. Visual similarity does not determine which model is more accurate.',
             'No training, GPU inference, model-weight selection, mask generation, or existing report/package changes.']
    output = dict(reference_images=90, test_images=10, source_sha256=sha(source), manifest_sha256=sha(manifest_path),
                  image_stats_sha256=sha(stats_path), script_sha256=sha(Path(__file__)), notes=notes,
                  rows=rows, feature_scaling=scaling, comparisons=comparisons)
    for page in range(2):
        examples = comparisons['equal_photo_texture_same_width']['tests'][page*5:(page+1)*5]
        canvas = Image.new('RGB', (1440, 1340), 'white'); draw = ImageDraw.Draw(canvas)
        font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
        font = ImageFont.truetype(font_path, 15)
        for row_id, ex in enumerate(examples):
            tiles = [dict(name=ex['test'], delta_pp=None)] + ex['neighbors'][:3]
            for col, item in enumerate(tiles):
                y = row_id * 268; x = col * 360
                image = Image.fromarray(images[item['name']]).convert('RGB'); image.thumbnail((350, 226))
                canvas.paste(image, (x, y+38))
                draw.text((x+3, y+2), item['name'], fill='black', font=font)
                label = 'TEST' if col == 0 else f"OOF change {item['delta_pp']:+.4f} pp"
                draw.text((x+3, y+20), label, fill='black' if col == 0 else ('#ad2020' if item['delta_pp'] < 0 else '#08653e'), font=font)
        canvas.save(OUT / f'nearest_examples_{page+1}.jpg', quality=93)
    output['duration_seconds'] = time.monotonic() - started
    (OUT / 'results.json').write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: {field: v[field] for field in ('worsened_neighbor_slots', 'unique_neighbors')} for k, v in comparisons.items()}, indent=2))
    print('seconds', output['duration_seconds'])


if __name__ == '__main__':
    main()
