"""Audit and convert only real RG/RGMask pairs from Kaggle dataset version4.

The grain boundary task is distinct from KoMaP phases. Black mask pixels map to
temporary slot1 for binary pretraining. Source groups, including duplicate-linked
groups, are held together across source train/valid. No competition labels used.
"""
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from data import MASK_RGB


def d4(arr):
    return [np.ascontiguousarray(np.rot90(x, k)) for x in [arr, arr[:, ::-1]] for k in range(4)]


def exact_signature(arr):
    return min(hashlib.sha256(str(x.shape).encode() + x.tobytes()).hexdigest() for x in d4(arr))


def phash(arr):
    x = cv2.resize(arr.astype(np.float32), (32, 32), interpolation=cv2.INTER_AREA)
    v = cv2.dct(x)[:8, :8].ravel()[1:]
    bits = v > np.median(v)
    return sum(int(b) << i for i, b in enumerate(bits))


def distance(a, b):
    return min((int(x) ^ int(b)).bit_count() for x in a)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', default='external/kaggle_grains_v4')
    p.add_argument('--out', default='external/kaggle_grains_seg')
    p.add_argument('--audit-out', default='runs/review15/data_audit')
    args = p.parse_args()
    cv2.setNumThreads(1)
    src, out, audit_out = Path(args.src), Path(args.out), Path(args.audit_out)
    assert not out.exists(), 'Output exists: inspect it before converting again.'
    meta = json.loads((src / 'metadata.json').read_text())
    assert meta['currentVersionNumber'] == 4 and meta['licenseName'] == 'CC0: Public Domain'
    raw = src / 'raw/GRAIN DATA SET'
    images, masks = {}, {}
    for sub, pattern, target in [('RG', r'RG(\d+_\d+_\d+)\.jpg', images),
                                  ('RGMask', r'RGMask(\d+_\d+_\d+)\.jpg', masks)]:
        for path in (raw / sub).glob('*.jpg'):
            match = re.fullmatch(pattern, path.name)
            assert match, path
            assert match[1] not in target
            target[match[1]] = path
    assert set(images) == set(masks) and len(images) == 480
    ids = sorted(images, key=lambda k: tuple(map(int, k.split('_'))))
    rows, grays, hashes, canonical = [], [], [], []
    for key in ids:
        gray = np.asarray(Image.open(images[key]).convert('L'))
        mask = np.asarray(Image.open(masks[key]).convert('L'))
        assert gray.shape == mask.shape == (400, 400)
        boundary = mask < 128
        fraction = float(boundary.mean())
        assert 0.005 < fraction < 0.40, (key, fraction)
        grays.append(gray)
        hashes.append([phash(x) for x in d4(gray)])
        canonical.append(exact_signature(gray))
        rows.append({'id': key, 'source_group': int(key.split('_')[0]),
                     'shape': list(gray.shape), 'boundary_fraction': fraction,
                     'image_sha256': hashlib.sha256(images[key].read_bytes()).hexdigest(),
                     'mask_sha256': hashlib.sha256(masks[key].read_bytes()).hexdigest()})
    parents = {r['source_group']: r['source_group'] for r in rows}

    def find(g):
        while parents[g] != g:
            parents[g] = parents[parents[g]]
            g = parents[g]
        return g

    def union(a, b):
        a, b = find(a), find(b)
        parents[max(a, b)] = min(a, b)

    exact_links, near_links = [], []
    for i in range(len(rows)):
        for j in range(i):
            if rows[i]['source_group'] == rows[j]['source_group']:
                continue
            dist = distance(hashes[i], hashes[j][0])
            if canonical[i] == canonical[j] or dist <= 8:
                link = {'a': ids[i], 'b': ids[j], 'phash_distance': dist}
                (exact_links if canonical[i] == canonical[j] else near_links).append(link)
                # Conservative grouping protects the external validation split.
                union(rows[i]['source_group'], rows[j]['source_group'])
    # Compare all competition images, including test, using pixels only for dedup.
    competition = []
    for split in ['train', 'valid', 'test']:
        for path in sorted(Path('data', split, 'images').glob('*_image.png')):
            gray = np.asarray(Image.open(path).convert('L'))
            competition.append((split, path.name, exact_signature(gray), phash(gray)))
    assert len(competition) == 100, len(competition)
    cross = []
    for i, row in enumerate(rows):
        for split, name, sig, h in competition:
            dist = distance(hashes[i], h)
            if canonical[i] == sig or dist <= 8:
                cross.append({'external_id': row['id'], 'competition_split': split,
                              'competition_name': name, 'phash_distance': dist})
    excluded_clusters = {find(int(r['external_id'].split('_')[0])) for r in cross}
    groups = sorted({find(r['source_group']) for r in rows} - excluded_clusters)
    assert len(groups) >= 10, f'Too few independent source clusters: {len(groups)}'
    order = np.random.default_rng(0).permutation(groups).tolist()
    n_valid = max(3, round(len(groups) * 0.2))
    valid = set(order[:n_valid])
    seen = set()
    kept, skipped = [], []
    for i, row in enumerate(rows):
        cluster = find(row['source_group'])
        row['source_cluster'] = cluster
        if cluster in excluded_clusters:
            skipped.append({'id': row['id'], 'reason': 'competition_duplicate_candidate_cluster'})
            continue
        if canonical[i] in seen:
            skipped.append({'id': row['id'], 'reason': 'exact_D4_duplicate_image'})
            continue
        seen.add(canonical[i])
        split = 'valid' if cluster in valid else 'train'
        name = f'A{40000+i}_step1_sample1'
        row.update(split=split, converted_name=name)
        for sub in ['images', 'masks']:
            (out / split / sub).mkdir(parents=True, exist_ok=True)
        Image.fromarray(grays[i]).save(out / split / 'images' / f'{name}_image.png')
        mask = np.asarray(Image.open(masks[row['id']]).convert('L'))
        label = (mask < 128).astype(np.uint8)
        Image.fromarray(MASK_RGB[label]).save(out / split / 'masks' / f'{name}_mask.png')
        kept.append(row)
    train_clusters = {r['source_cluster'] for r in kept if r['split'] == 'train'}
    valid_clusters = {r['source_cluster'] for r in kept if r['split'] == 'valid'}
    assert not train_clusters & valid_clusters
    audit_out.mkdir(parents=True, exist_ok=True)
    # Visual check sheet covers 20 source filename groups.
    selected = [i for i, r in enumerate(rows) if r['source_group'] % 2 == 1 and r['id'].endswith('_1_1')]
    assert len(selected) == 20
    sheet = Image.new('RGB', (1600, 1000), 'white')
    from PIL import ImageDraw
    draw = ImageDraw.Draw(sheet)
    for k, i in enumerate(selected):
        x, y = (k % 4) * 400, (k // 4) * 200
        draw.text((x + 3, y + 3), rows[i]['id'], fill='black')
        im = Image.open(images[rows[i]['id']]).convert('RGB').resize((190, 180))
        ma = Image.open(masks[rows[i]['id']]).convert('RGB').resize((190, 180))
        sheet.paste(im, (x, y + 20)); sheet.paste(ma, (x + 200, y + 20))
    sheet.save(audit_out / 'contact_sheet.jpg', quality=90)
    dup_selected = near_links[:12]
    if dup_selected:
        dup_sheet = Image.new('RGB', (800, 220 * len(dup_selected)), 'white')
        draw = ImageDraw.Draw(dup_sheet)
        for k, link in enumerate(dup_selected):
            draw.text((3, 220*k), f"{link['a']} vs {link['b']}; phash={link['phash_distance']}", fill='black')
            for col, key in enumerate([link['a'], link['b']]):
                dup_sheet.paste(Image.open(images[key]).convert('RGB').resize((390, 200)), (col*400,220*k+20))
        dup_sheet.save(audit_out / 'duplicate_candidates.jpg', quality=90)
    summary = {
        'dataset_ref': 'peterwarren/voronoi-artificial-grains-gen', 'version': 4,
        'license': meta['licenseName'], 'raw_pairs': len(rows), 'shape': [400,400],
        'source_filename_groups': len(parents), 'source_clusters_after_duplicate_links': len(groups),
        'train_valid_counts': dict(Counter(r['split'] for r in kept)),
        'train_clusters': sorted(train_clusters), 'valid_clusters': sorted(valid_clusters),
        'exact_duplicate_links': exact_links, 'near_duplicate_links': near_links,
        'cross_competition_candidates': cross, 'competition_images_checked': len(competition),
        'skipped': skipped, 'boundary_fraction_range': [min(r['boundary_fraction'] for r in kept),max(r['boundary_fraction'] for r in kept)],
        'mask_rule': 'JPEG grayscale <128 is grain boundary; temporary slot1, background slot0.',
        'threshold': 128, 'split_seed': 0, 'near_duplicate_phash_max': 8,
        'ready_for_pretraining': len(kept) >= 200 and not cross,
        'visual_approval_required': True,
        'note': 'Filename groups can share the same field; duplicate-linked groups are unioned. Grain boundaries are not KoMaP phases.',
    }
    (out / 'manifest.json').write_text(json.dumps({'pairs': kept}, indent=2))
    (out / 'audit.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
