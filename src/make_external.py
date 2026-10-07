"""9단계: 외부 데이터를 우리 데이터 형식(load_split)으로 변환한다. 대응하는 상이 없는 픽셀은 IGNORE(255)로 둔다.

MICAST (Theofilatos 외 2026, CC BY 4.0, AlSi6Cu4Fe 합성 LOM): COCO 다각형 → 마스크
  배경 → Al, eutSi(공정 Si) → Eutectic Si, AlFeSi → 무시(255, 우리 상과 대응 없음). 사진은 RGBA → 흑백.
  주의: 배경은 실제 α-Al이 아니라 잡음 무늬이고, 조각을 따로 붙인 합성 사진이다 (직접 확인).
MicroSteel (Fe–TiB2 강, MIT): 정수 마스크 0~4 → 우리 4개 출력 자리에 배치 (의미는 다름, 미세조정 때 출력층을 새로 만든다)
  주의: 광학현미경이 아니라 SEM 사진이다 (사진 정보 줄에 SEM, ×10,000 표기. 직접 확인). 아래 정보 줄은 라벨 크기에 맞춰 잘라낸다.

결과: <out>/train/images/A<번호>_step1_sample1_image.png, masks/..._mask.png (우리 4색 + 무시 색)
예) python src/make_external.py micast --src external/micast/lom --n 1000 --out external/micast_seg
    python src/make_external.py microsteel --src external/microsteel/x --out external/microsteel_seg
"""
import argparse
import glob
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from data import MASK_RGB

IGNORE_RGB = (255, 0, 0)   # 우리 4색에 없는 색 → rgb_to_label에서 255(IGNORE)가 된다


def save(out, k, gray, lab_rgb):
    name = f'A{10000 + k}_step1_sample1'   # load_split의 이름 규칙(A<합금>_step<n>_sample<n>)에 맞춘 가짜 이름
    Image.fromarray(gray).save(Path(out, 'train', 'images', f'{name}_image.png'))
    Image.fromarray(lab_rgb).save(Path(out, 'train', 'masks', f'{name}_mask.png'))


def cmd_micast(args):
    files = sorted(glob.glob(os.path.join(args.src, 'Images', 'TRAIN', 'train*.png')), key=lambda p: int(Path(p).stem[5:]))[:args.n]
    want = {Path(p).name for p in files}
    print('loading json (수 GB, 몇 분 걸림)...', flush=True)
    d = json.load(open(os.path.join(args.src, 'two_phases', 'train.json')))
    cats = {str(c['id']): c['name'] for c in d['categories']}
    img_of = {str(im['id']): im['file_name'] for im in d['images'] if im['file_name'] in want}
    anns = {}
    for a in d['annotations']:
        fid = img_of.get(str(a['image_id']))
        if fid:
            anns.setdefault(fid, []).append(a)
    del d
    stats = np.zeros(3)
    for k, p in enumerate(files):
        im = Image.open(p)
        gray = np.array(im.convert('L'))
        lab = np.zeros((*gray.shape, 3), np.uint8)
        lab[:] = MASK_RGB[0]                                         # 배경 → Al
        for cat, color in [('eutSi', MASK_RGB[3]), ('AlFeSi', IGNORE_RGB)]:   # 겹치면 무시가 이긴다
            for a in anns.get(Path(p).name, []):
                if cats[str(a['category_id'])] != cat:
                    continue
                seg = a['segmentation']
                seg = json.loads(seg) if isinstance(seg, str) else seg
                polys = [np.array(s, np.float32).reshape(-1, 2).round().astype(np.int32) for s in seg if len(s) >= 6]
                cv2.fillPoly(lab, polys, tuple(int(c) for c in color))
        stats += [(lab == MASK_RGB[0]).all(-1).mean(), (lab == MASK_RGB[3]).all(-1).mean(), (lab == IGNORE_RGB).all(-1).mean()]
        save(args.out, k, gray, lab)
    print(f'{len(files)} images -> {args.out} | 픽셀 비율 Al {stats[0]/len(files):.3f}, eutSi {stats[1]/len(files):.3f}, '
          f'무시(AlFeSi) {stats[2]/len(files):.3f}', flush=True)


def cmd_microsteel(args):
    labs = sorted(glob.glob(os.path.join(args.src, 'labels', '*.png')))
    # 값 0 기지, 1 TiB2, 2 TiN(0.17%), 3 FeTiB, 4 Fe2B → 우리 출력 0,1,무시,2,3 자리 (TiN만 너무 적어 무시)
    cls_rgb = [MASK_RGB[0], MASK_RGB[1], IGNORE_RGB, MASK_RGB[2], MASK_RGB[3]]
    k, skipped = 0, []
    for lp in labs:
        lab = np.array(Image.open(lp))
        ip = os.path.join(args.src, Path(lp).stem + '.jpg')
        if not os.path.exists(ip):   # 라벨 82개, 사진 79장: 사진 없는 라벨은 건너뜀
            skipped.append(Path(lp).stem)
            continue
        gray = np.array(Image.open(ip).convert('L'))[:lab.shape[0], :lab.shape[1]]   # 아래 SEM 정보 줄을 잘라 라벨 크기에 맞춤
        rgb = np.zeros((*lab.shape, 3), np.uint8)
        for c, color in enumerate(cls_rgb):
            rgb[lab == c] = color
        save(args.out, k, gray, rgb)
        k += 1
    print(f'{k} images -> {args.out} (사진 없는 라벨 {len(skipped)}개 건너뜀: {skipped})', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['micast', 'microsteel'])
    p.add_argument('--src', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--n', type=int, default=1000)
    args = p.parse_args()
    for sub in ('images', 'masks'):
        Path(args.out, 'train', sub).mkdir(parents=True, exist_ok=True)
    {'micast': cmd_micast, 'microsteel': cmd_microsteel}[args.cmd](args)


if __name__ == '__main__':
    main()
