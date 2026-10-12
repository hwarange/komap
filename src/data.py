"""데이터 로딩, 전처리, 학습용 증강."""
import json
import math
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

CLASSES = ['Al', 'Al3Ni', 'Primary Si', 'Eutectic Si']
MASK_RGB = np.array([(153, 127, 76), (25, 76, 153), (76, 178, 76), (204, 204, 204)], np.uint8)
PAT = re.compile(r'A(\d+)_step(\d)_sample(\d)')

# 정규화된 [0,1] 입력을 모델 입력 분포로 옮길 때 쓰는 값
IN_MEAN, IN_STD = 0.5, 0.25
IGNORE = 255   # 손실에서 빼는 픽셀 (cutout으로 가린 곳). 정답 마스크에는 없는 값


def rgb_to_label(rgb):
    lab = np.full(rgb.shape[:2], 255, np.uint8)
    for i, c in enumerate(MASK_RGB):
        lab[(rgb == c).all(-1)] = i
    return lab


def label_to_rgb(lab):
    return MASK_RGB[lab]


def normalize(gray, lo=1, hi=99):
    """이미지마다 p1→0, p99→1로 늘림. 저대비/고대비 스타일 차이를 줄인다 (EDA 5절)."""
    p_lo, p_hi = (float(v) for v in np.percentile(gray, [lo, hi]))
    return np.clip((gray.astype(np.float32) - p_lo) / max(p_hi - p_lo, 1e-3), 0, 1).astype(np.float32)


def load_split(root, split):
    items = []
    for p in sorted((Path(root) / split / 'images').glob('*_image.png')):
        name = p.name[:-len('_image.png')]
        alloy, step, sample = map(int, PAT.match(name).groups())
        gray = np.array(Image.open(p).convert('L'))
        mp = Path(root) / split / 'masks' / f'{name}_mask.png'
        lab = rgb_to_label(np.array(Image.open(mp).convert('RGB'))) if mp.exists() else None
        items.append(dict(name=name, split=split, alloy=alloy, step=step, sample=sample, gray=gray, lab=lab,
                          low_contrast=bool(np.percentile(gray, 1) > 60)))
    return items


def make_folds(items, n_folds=5, seed=0):
    """합금 번호 단위로 fold를 나눈다. 같은 합금 사진은 항상 같은 fold에 들어간다 (EDA 1절).
    큰 합금 그룹부터, 이미지 수가 가장 적은 fold에 넣는다. 같으면 저대비 사진이 적은 fold를 고른다."""
    groups = {}
    for it in items:
        groups.setdefault(it['alloy'], []).append(it)
    keys = list(groups)
    np.random.default_rng(seed).shuffle(keys)
    keys.sort(key=lambda a: -len(groups[a]))
    count, low, fold_of = [0] * n_folds, [0] * n_folds, {}
    for a in keys:
        n_low = sum(it['low_contrast'] for it in groups[a])
        f = min(range(n_folds), key=lambda f: (count[f], low[f] if n_low else 0, f))
        count[f] += len(groups[a])
        low[f] += n_low
        for it in groups[a]:
            fold_of[it['name']] = f
    return fold_of


def load_cv(root, fold, folds_file):
    """학습+검증 90장을 합쳐 fold번째 묶음은 검증, 나머지는 학습으로 돌려준다."""
    items = load_split(root, 'train') + load_split(root, 'valid')
    fold_of = json.loads(Path(folds_file).read_text())
    return [it for it in items if fold_of[it['name']] != fold], [it for it in items if fold_of[it['name']] == fold]


def scale_for(width, target_width, scale_multiplier=1.0):
    """같은 시야를 해상도만 다르게 찍었으므로 작은 이미지를 정수 배로 키워 배율을 맞춘다 (EDA 4절).
    1096~1440px는 1배(원본 그대로)로 두어 불필요한 재샘플링을 피하고, 675px는 2배로 키운다."""
    if not math.isfinite(scale_multiplier) or scale_multiplier <= 0:
        raise ValueError('scale_multiplier must be finite and positive')
    return (float(max(1, round(target_width / width))) if target_width else 1.0) * scale_multiplier


def to_input(img):
    return ((img - IN_MEAN) / IN_STD).astype(np.float32)


# 9단계 입력 채널: 어두운 가는 구조 강조. 정규화 값은 학습 사진 10장에서 잰 평균·표준편차
# (상별 평균: Eutectic Si는 black top-hat 0.28 / ridge 0.52로 Al3Ni 0.09 / 0.19보다 3배 가량 높다)
FEAT_MEAN, FEAT_STD = (0.066, 0.136), (0.112, 0.185)
_BH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))


def ridge_filter(img, sigmas=(1.0, 2.0)):
    """어두운 선 강조 (Hessian 최대 고윳값, 크기 보정 σ²). 어두운 선을 가로지르는 방향의 2차 미분이 양수로 크다."""
    out = np.zeros_like(img)
    for s in sigmas:
        g = cv2.GaussianBlur(img, (0, 0), s)
        xx, yy = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3)
        xy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3)
        lam = (xx + yy) / 2 + np.sqrt(((xx - yy) / 2) ** 2 + xy ** 2)
        out = np.maximum(out, s * s * np.maximum(lam, 0))
    return out


def to_input_feats(img):
    """[3,H,W] 입력 (9단계): 정규화 밝기 + black top-hat(7×7, 어두운 가는 구조) + ridge(선 모양).
    Eutectic Si는 1~4px 어두운 선이라 주변과 섞여 중간 회색으로 찍히는데(부분 부피), 두 필터는 이를 직접 드러낸다."""
    img = np.ascontiguousarray(img, dtype=np.float32)
    bh = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, _BH_KERNEL)
    r = ridge_filter(img)
    return np.stack([to_input(img), (bh - FEAT_MEAN[0]) / FEAT_STD[0], (r - FEAT_MEAN[1]) / FEAT_STD[1]]).astype(np.float32)


# 밝기 증강 확률. default: 지금까지 쓴 값 / lean: nnU-Net v2 기본값에 가깝게 낮춤 (4단계)
INTENSITY_P = {'default': dict(gamma=0.8, contrast=0.8, illum=0.5, blur=0.2, noise=0.3),
               'lean': dict(gamma=0.3, contrast=0.15, illum=0.15, blur=0.2, noise=0.1)}


def augment_intensity(img, rng, policy='default', drop=()):
    p = {k: (0.0 if k in drop else v) for k, v in INTENSITY_P[policy].items()}  # drop: 끌 증강 이름들 (5단계)
    if rng.random() < p['gamma']:
        img = img ** math.exp(rng.uniform(-0.35, 0.35))
    if rng.random() < p['contrast']:
        c, b = rng.uniform(0.75, 1.25), rng.uniform(-0.08, 0.08)
        img = (img - 0.5) * c + 0.5 + b
    if rng.random() < p['illum']:
        # 조명 불균일: 완만한 기울기 + 가장자리 어두움 (EDA 5절)
        h, w = img.shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        yy, xx = yy / h - 0.5, xx / w - 0.5
        field = 1 + rng.uniform(-0.08, 0.08) * xx + rng.uniform(-0.08, 0.08) * yy - rng.uniform(0, 0.15) * (xx ** 2 + yy ** 2)
        img = img * field
    if rng.random() < p['blur']:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.0))
    if rng.random() < p['noise']:
        img = img + rng.normal(0, rng.uniform(0.005, 0.03), img.shape).astype(np.float32)
    return np.clip(img, 0, 1).astype(np.float32)


def class_medians(img, lab):
    """상별 중간 밝기. 없거나 너무 적은 상은 nan."""
    return np.array([np.median(img[lab == c]) if (lab == c).sum() >= 20 else np.nan for c in range(4)], np.float32)


def phase_anchored_mapping(img, lab, target_medians):
    """상 기준점 단조 매핑 (4단계 증강).
    원본 crop과 목표 사진의 상별 중간 밝기를 기준점으로 하나의 단조 증가 함수를 만들어 crop 전체에 적용한다.
    - 무작위 히스토그램 매칭(RHM)은 히스토그램이 상 분율에 크게 좌우되므로, 분율이 다른 사진에 맞추면 상 밝기가 뒤틀린다.
    - 상별로 따로 매칭하면 정답 경계를 따라 인공 밝기 경계가 생겨 정답이 이미지에 새어 들어간다.
    - 하나의 단조 함수는 밝기 순서(Al > Al3Ni > Si)를 지키고 인공 경계도 만들지 않는다."""
    src = class_medians(img, lab)
    ok = ~np.isnan(src) & ~np.isnan(target_medians)
    if ok.sum() < 2:
        return img
    xs, ys = src[ok], target_medians[ok]
    order = np.argsort(xs)
    xs, ys = xs[order], np.maximum.accumulate(ys[order])  # 단조 증가 강제
    keep = np.concatenate([[True], np.diff(xs) > 1e-3])
    xs, ys = np.concatenate([[0.0], xs[keep], [1.0]]), np.concatenate([[0.0], ys[keep], [1.0]])
    ys = np.maximum.accumulate(ys)
    return np.interp(img, xs, ys).astype(np.float32)


def simulate_lowres(img, rng):
    """이미지만 축소했다가 다시 키워 저해상도 촬영을 흉내 낸다. 정답은 그대로 둔다 (nnU-Net, 4단계)."""
    f = rng.uniform(1.25, 2.0)
    h, w = img.shape
    small = cv2.resize(img, (max(1, round(w / f)), max(1, round(h / f))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def elastic_deform(img, lab, rng, max_disp=4.0, cells=8):
    """탄성 변형 (7단계): 거친 격자(cells×cells)에 무작위 변위를 만들고 부드럽게 키워 조직을 살짝 휘게 한다.
    가는 구조(1~2px)가 끊기지 않도록 변위는 최대 약 max_disp px로 작게 둔다. 정답은 최근접 보간."""
    h, w = img.shape
    d = rng.uniform(-max_disp, max_disp, (2, cells + 1, cells + 1)).astype(np.float32)
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    mx = xx + cv2.resize(d[0], (w, h), interpolation=cv2.INTER_CUBIC)
    my = yy + cv2.resize(d[1], (w, h), interpolation=cv2.INTER_CUBIC)
    return (cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101),
            cv2.remap(lab, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101))


def cutout(img, lab, rng, n_max=3, frac=(0.06, 0.18)):
    """일부 가리기 (7단계): 1~n_max개의 네모를 회색(0.5)으로 가리고, 가린 곳은 정답을 IGNORE로 바꿔 손실에서 뺀다
    (가린 곳의 상은 이미지로 알 수 없으므로 정답을 그대로 두면 라벨 잡음이 된다)."""
    h, w = img.shape
    img, lab = img.copy(), lab.copy()
    for _ in range(int(rng.integers(1, n_max + 1))):
        s = int(rng.uniform(*frac) * min(h, w))
        y0, x0 = int(rng.integers(0, h - s)), int(rng.integers(0, w - s))
        img[y0:y0 + s, x0:x0 + s] = 0.5
        lab[y0:y0 + s, x0:x0 + s] = IGNORE
    return img, lab


class CropDataset(Dataset):
    """이미지마다 무작위 위치·크기·각도로 crop을 뽑는 학습용 데이터셋.

    4단계 증강 옵션 (모두 기본값이면 3단계까지와 동일):
    - rot='rot90': 임의 각도 대신 90° 회전·반전만 (보간으로 가는 구조가 흐려지는 것 방지)
    - policy='lean': nnU-Net식 낮은 확률의 밝기 증강 + 회전·크기 변환도 20% 확률로만
    - hm: 상 기준점 단조 매핑 확률 (목표 사진의 절반은 저대비 사진에서 고름)
    - lowres: 저해상도 흉내 확률
    - no_int: 밝기 증강 끄기
    - copypaste: 다른 crop의 Primary Si·Al3Ni 덩어리를 테두리 2px와 함께 붙여넣을 확률
    - skel: Eutectic Si 뼈대 띠(skeleton tube)를 함께 반환 (Skeleton Recall 손실용)
    7단계 옵션: elastic(탄성 변형 확률), cutout(일부 가리기 확률). mixup·cutmix는 배치 단위라 train.py에서 한다.
    9단계 옵션: hard(오답 위치 중심으로 자를 확률), hard_maps(사진별 오답 지도 폴더, <이름>.png, 0이 아니면 오답)
    """

    def __init__(self, items, crop=512, samples_per_image=8, target_width=1280, scale_jitter=0.15, aug=True,
                 rot='any', policy='default', hm=0.0, lowres=0.0, no_int=False, copypaste=0.0, skel=False, int_drop=(),
                 elastic=0.0, cutout=0.0, hard=0.0, hard_maps='', feats=False, scale_multiplier=1.0, sdf_tau=0.0):
        self.data = [(normalize(it['gray']), it['lab'], scale_for(it['gray'].shape[1], target_width, scale_multiplier)) for it in items]
        self.crop, self.spi, self.jitter, self.aug = crop, samples_per_image, scale_jitter, aug
        self.rot, self.policy, self.hm, self.lowres = rot, policy, hm, lowres
        self.no_int, self.copypaste, self.skel, self.int_drop = no_int, copypaste, skel, tuple(int_drop)
        self.elastic, self.cutout, self.hard, self.feats = elastic, cutout, hard, feats
        self.sdf_tau = sdf_tau   # 36단계: 0보다 크면 상별 부호 거리장(경계까지의 픽셀 거리, ±tau에서 자름)을 함께 반환
        if hard:
            # 오답 위치 (원본 좌표 y, x). 경계 띠 밖에서 앙상블이 틀린 픽셀 (9단계, make_hardmaps)
            self.hard_xy = [np.argwhere(np.array(Image.open(Path(hard_maps) / f"{it['name']}.png")) > 0) for it in items]
        if hm:
            self.medians = [class_medians(img, lab) for img, lab, _ in self.data]
            self.low_idx = [i for i, it in enumerate(items) if it['low_contrast']]

    def __len__(self):
        return len(self.data) * self.spi

    def _warp(self, i, rng, center=None):
        img, lab, base = self.data[i]
        H, W = img.shape
        lean = self.policy == 'lean'
        do_scale = self.aug and (not lean or rng.random() < 0.2)
        do_rot = self.aug and self.rot == 'any' and (not lean or rng.random() < 0.2)
        s = base * (math.exp(rng.uniform(-math.log(1 + self.jitter), math.log(1 + self.jitter))) if do_scale else 1.0)
        theta = rng.uniform(0, 2 * math.pi) if do_rot else 0.0
        # 출력 crop의 중심을 원본의 무작위 위치에 놓고 확대/회전을 한 번에 적용
        half = self.crop / s / 2
        cx, cy = rng.uniform(min(half, W / 2), max(W - half, W / 2)), rng.uniform(min(half, H / 2), max(H - half, H / 2))
        if center is not None:
            # 오답 위치 중심 (9단계): 조각 크기의 ±1/4 안에서 살짝 흔든 뒤 사진 안쪽으로 맞춘다
            cx = float(np.clip(center[0] + rng.uniform(-half / 2, half / 2), min(half, W / 2), max(W - half, W / 2)))
            cy = float(np.clip(center[1] + rng.uniform(-half / 2, half / 2), min(half, H / 2), max(H - half, H / 2)))
        cos, sin = math.cos(theta) * s, math.sin(theta) * s
        M = np.array([[cos, -sin, self.crop / 2 - cos * cx + sin * cy],
                      [sin, cos, self.crop / 2 - sin * cx - cos * cy]], np.float32)
        out = (self.crop, self.crop)
        x = cv2.warpAffine(img, M, out, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        y = cv2.warpAffine(lab, M, out, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
        if self.aug:
            if self.rot == 'rot90' or lean:
                k = int(rng.integers(4))
                x, y = np.rot90(x, k), np.rot90(y, k)
            if rng.random() < 0.5:
                x, y = x[:, ::-1], y[:, ::-1]
        return np.ascontiguousarray(x), np.ascontiguousarray(y)

    def _copy_paste(self, x, y, rng):
        xs, ys = self._warp(int(rng.integers(len(self.data))), rng)
        x, y = x.copy(), y.copy()
        ring = np.ones((5, 5), np.uint8)
        for cls in (2, 1):  # Primary Si, Al3Ni
            n, cc, st, _ = cv2.connectedComponentsWithStats((ys == cls).astype(np.uint8), connectivity=8)
            cand = [k for k in range(1, n) if st[k, 4] >= 50 and st[k, 0] > 2 and st[k, 1] > 2
                    and st[k, 0] + st[k, 2] < self.crop - 2 and st[k, 1] + st[k, 3] < self.crop - 2]
            for k in rng.permutation(cand)[:2]:
                m = cv2.dilate((cc == k).astype(np.uint8), ring).astype(bool)  # 덩어리 + 주변 2px (할로 포함)
                x[m], y[m] = xs[m], ys[m]
        return x, y

    def __getitem__(self, idx):
        rng = np.random.default_rng(np.random.randint(1 << 31))
        i, center = idx % len(self.data), None
        if self.hard and rng.random() < self.hard and len(self.hard_xy[i]):   # hard=0이면 난수를 쓰지 않아 이전과 같다
            yy, xx = self.hard_xy[i][int(rng.integers(len(self.hard_xy[i])))]
            center = (float(xx), float(yy))
        x, y = self._warp(i, rng, center)
        if self.aug:
            if self.elastic and rng.random() < self.elastic:
                x, y = elastic_deform(x, y, rng)
            if self.copypaste and rng.random() < self.copypaste:
                x, y = self._copy_paste(x, y, rng)
            if self.hm and rng.random() < self.hm:
                pool = self.low_idx if self.low_idx and rng.random() < 0.5 else range(len(self.data))
                x = phase_anchored_mapping(x, y, self.medians[int(rng.choice(pool))])
            if self.lowres and rng.random() < self.lowres:
                x = simulate_lowres(x, rng)
            if not self.no_int:
                x = augment_intensity(x, rng, self.policy, self.int_drop)
            if self.cutout and rng.random() < self.cutout:   # 밝기 증강 뒤에 해야 가린 색이 그대로 유지된다
                x, y = cutout(x, y, rng)
        xin = to_input_feats(x) if self.feats else to_input(np.ascontiguousarray(x))[None]   # 필터는 증강 뒤 사진으로
        out = [torch.from_numpy(xin), torch.from_numpy(y.astype(np.int64))]
        if self.skel:
            from skimage.morphology import skeletonize
            tube = cv2.dilate(skeletonize(y == 3).astype(np.uint8), np.ones((3, 3), np.uint8))
            out.append(torch.from_numpy(tube.astype(np.float32)))
        if self.sdf_tau > 0:
            out.append(torch.from_numpy(signed_distance(y, self.sdf_tau)))
        return tuple(out)


def signed_distance(lab, tau, n_classes=len(CLASSES)):
    """상별 부호 거리장 [n_classes, H, W] (36단계 거리장 회귀 목표).
    그 상 안쪽은 +경계까지 거리, 바깥은 −경계까지 거리. ±tau에서 잘라 tau로 나누므로 값은 [-1, 1].
    경계에 붙은 안쪽 픽셀은 +1/tau, 바깥쪽 픽셀은 −1/tau라 0은 두 픽셀 사이에 놓인다."""
    out = np.empty((n_classes, *lab.shape), np.float32)
    for c in range(n_classes):
        m = (lab == c).astype(np.uint8)
        if not m.any():
            out[c] = -1.0
            continue
        if m.all():
            out[c] = 1.0
            continue
        inside = cv2.distanceTransform(m, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        outside = cv2.distanceTransform(1 - m, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        out[c] = np.clip(inside - outside, -tau, tau) / tau
    return out


def worker_init_fn(worker_id):
    np.random.seed((torch.initial_seed() + worker_id) % (1 << 32))
