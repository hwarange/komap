"""Aligned whole-image inputs for the isolated global-context experiment.

The local crop and its RNG stream exactly match data.CropDataset. A compact
whole-image view has the same rotation/flip, with an explicit crop-to-view map.
"""
import copy
from dataclasses import dataclass
import math

import cv2
import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate_fn_map

from data import CropDataset as BaseCropDataset, augment_intensity, normalize, scale_for, to_input
from infer import _dihedral, _undihedral, _window

CONTEXT_SIDE = 384


@dataclass
class ContextInput:
    crop: torch.Tensor
    whole: torch.Tensor
    crop_to_whole: torch.Tensor

    @property
    def shape(self):
        return self.crop.shape

    def to(self, *args, **kwargs):
        return ContextInput(*(x.to(*args, **kwargs) for x in (self.crop, self.whole, self.crop_to_whole)))

    def cuda(self, *args, **kwargs):
        return ContextInput(*(x.cuda(*args, **kwargs) for x in (self.crop, self.whole, self.crop_to_whole)))

    def pin_memory(self):
        return ContextInput(*(x.pin_memory() for x in (self.crop, self.whole, self.crop_to_whole)))


def collate_context(batch, *, collate_fn_map=None):
    return ContextInput(*(torch.stack([getattr(x, k) for x in batch])
                          for k in ('crop', 'whole', 'crop_to_whole')))


default_collate_fn_map[ContextInput] = collate_context


def dihedral_matrix(h, w, rotations, flip, flip_first=False):
    """Pixel-centre (x,y) map, using exactly numpy/torch rot90 conventions."""
    out = np.eye(3, dtype=np.float64)
    if flip and flip_first:
        out = np.array([[-1, 0, w-1], [0, 1, 0], [0, 0, 1]]) @ out
    for _ in range(rotations % 4):
        out = np.array([[0, 1, 0], [-1, 0, w-1], [0, 0, 1]]) @ out
        h, w = w, h
    if flip and not flip_first:
        out = np.array([[-1, 0, w-1], [0, 1, 0], [0, 0, 1]]) @ out
    return out


def transform_image(img, rotations, flip, flip_first=False):
    if flip and flip_first: img = img[:, ::-1]
    img = np.rot90(img, rotations)
    if flip and not flip_first: img = img[:, ::-1]
    return np.ascontiguousarray(img)


def whole_view(img):
    """Aspect-preserving letterbox and original-to-view pixel-centre map."""
    h, w = img.shape
    ratio = CONTEXT_SIDE / max(h, w)
    nh, nw = max(1, round(h*ratio)), max(1, round(w*ratio))
    top, left = (CONTEXT_SIDE-nh)//2, (CONTEXT_SIDE-nw)//2
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    view = cv2.copyMakeBorder(small, top, CONTEXT_SIDE-nh-top, left, CONTEXT_SIDE-nw-left,
                             cv2.BORDER_REFLECT_101)
    resize = np.array([[nw/w, 0, left+(nw/w-1)/2],
                       [0, nh/h, top+(nh/h-1)/2], [0, 0, 1]], np.float64)
    return view, resize


class ContextCropDataset(BaseCropDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.rot == 'rot90' and self.policy == 'default'
        assert not any((self.hm, self.lowres, self.copypaste, self.skel, self.elastic,
                        self.cutout, self.hard, self.feats))

    def _warp(self, i, rng, center=None):
        assert center is None
        state = copy.deepcopy(rng.bit_generator.state)
        x, y = super()._warp(i, rng, center)
        replay = np.random.default_rng(); replay.bit_generator.state = state
        img, _, base = self.data[i]; h, w = img.shape
        s = base * (math.exp(replay.uniform(-math.log(1+self.jitter), math.log(1+self.jitter))) if self.aug else 1.)
        half = self.crop/s/2
        cx = replay.uniform(min(half, w/2), max(w-half, w/2))
        cy = replay.uniform(min(half, h/2), max(h-half, h/2))
        matrix = np.array([[s, 0, self.crop/2-s*cx], [0, s, self.crop/2-s*cy], [0, 0, 1]], np.float32)
        k = int(replay.integers(4)) if self.aug else 0
        flip = bool(replay.random() < .5) if self.aug else False
        assert replay.bit_generator.state == rng.bit_generator.state, 'Local geometry RNG diverged'
        local_d = dihedral_matrix(self.crop, self.crop, k, flip)
        global_d = dihedral_matrix(h, w, k, flip)
        view, resize = whole_view(transform_image(img, k, flip))
        theta = resize @ global_d @ np.linalg.inv(matrix.astype(np.float64)) @ np.linalg.inv(local_d)
        self._context = (view, theta[:2].astype(np.float32), copy.deepcopy(rng.bit_generator.state))
        return x, y

    def __getitem__(self, idx):
        crop, target = super().__getitem__(idx)
        whole, theta, state = self._context
        # Clone the photometric RNG: scalar draws match the local view, while
        # illumination/noise are expressed at each view's own resolution.
        # Extra context draws never change local crops or future sample order.
        if self.aug and not self.no_int:
            rng = np.random.default_rng(); rng.bit_generator.state = state
            whole = augment_intensity(whole, rng, self.policy, self.int_drop)
        return ContextInput(crop, torch.from_numpy(to_input(whole)[None]), torch.from_numpy(theta)), target


def aligned_features(whole, transform, target_hw, stride):
    """Sample matching CNN feature centres, accounting for stride padding.

    A chain of odd-kernel, symmetric-padded strided convolutions places centre
    j at input pixel j*stride. Sampling uses those centres, not ROI box edges.
    """
    h, w = target_hw
    yy, xx = torch.meshgrid(torch.arange(h, device=whole.device, dtype=torch.float32)*stride,
                            torch.arange(w, device=whole.device, dtype=torch.float32)*stride, indexing='ij')
    points = torch.stack([xx, yy, torch.ones_like(xx)]).reshape(1, 3, -1)
    # Keep geometry in float32 even under BF16 training autocast.
    with torch.autocast(device_type=whole.device.type, enabled=False):
        coords = torch.bmm(transform.float(), points.expand(whole.shape[0], -1, -1)) / stride
        coords[:, 0] = 2*(coords[:, 0]+.5)/whole.shape[-1]-1
        coords[:, 1] = 2*(coords[:, 1]+.5)/whole.shape[-2]-1
        grid = coords.transpose(1, 2).reshape(whole.shape[0], h, w, 2)
        sampled = torch.nn.functional.grid_sample(whole.float(), grid, mode='bilinear',
                                                  padding_mode='border', align_corners=False)
    return sampled.to(whole.dtype)


@torch.no_grad()
def predict_context(model, gray, target_width=1280, tile=512, overlap=128, tta=False,
                    device='cuda', extra_scale=1., scale_multiplier=1.):
    original = normalize(gray); h, w = original.shape
    scale = scale_for(w, target_width, scale_multiplier)*extra_scale
    img = cv2.resize(original, (round(w*scale), round(h*scale)), interpolation=cv2.INTER_LINEAR) if scale != 1 else original
    sh, sw = img.shape
    # Exact OpenCV resize pixel-centre convention, including noninteger scales.
    resize = np.array([[sw/w, 0, (sw/w-1)/2], [0, sh/h, (sh/h-1)/2], [0, 0, 1]], np.float64)
    padded = np.pad(img, ((0, max(tile-sh, 0)), (0, max(tile-sw, 0))), mode='reflect')
    hp, wp = padded.shape; step = tile-overlap
    ys, xs = list(range(0, hp-tile+1, step)), list(range(0, wp-tile+1, step))
    if ys[-1] != hp-tile: ys.append(hp-tile)
    if xs[-1] != wp-tile: xs.append(wp-tile)
    coords = [(y, x) for y in ys for x in xs]
    win = torch.from_numpy(_window(tile)).to(device)
    xin = torch.from_numpy(to_input(padded)[None]).to(device)
    norm = torch.zeros(hp, wp, device=device); acc = None
    ks = list(range(8)) if tta else [0]
    views = {}
    for k in ks:
        dg = dihedral_matrix(h, w, k % 4, k >= 4, flip_first=True)
        dl = dihedral_matrix(tile, tile, k % 4, k >= 4, flip_first=True)
        view, rv = whole_view(transform_image(original, k % 4, k >= 4, flip_first=True))
        views[k] = (torch.from_numpy(to_input(view)[None, None]).to(device), rv @ dg @ np.linalg.inv(resize), np.linalg.inv(dl))
    for start in range(0, len(coords), 4):
        batch_coords = coords[start:start+4]
        crops = torch.stack([xin[:, y:y+tile, x:x+tile] for y, x in batch_coords])
        probability = 0
        for k in ks:
            whole, left, right = views[k]
            transforms = [left @ np.array([[1, 0, x], [0, 1, y], [0, 0, 1]]) @ right for y, x in batch_coords]
            theta = torch.from_numpy(np.stack(transforms)[:, :2].astype(np.float32)).to(device)
            inputs = ContextInput(_dihedral(crops, k), whole.expand(len(batch_coords), -1, -1, -1), theta)
            probability = probability + _undihedral(torch.softmax(model(inputs), 1), k)
        probability = probability / len(ks)
        if acc is None: acc = torch.zeros(probability.shape[1], hp, wp, device=device)
        for (y, x), p in zip(batch_coords, probability):
            acc[:, y:y+tile, x:x+tile] += p*win; norm[y:y+tile, x:x+tile] += win
    prob = (acc/norm)[:, :sh, :sw]
    if scale != 1:
        if float(scale).is_integer():
            prob = torch.nn.functional.interpolate(prob[None], size=(h, w), mode='area')[0]
        else:
            prob = torch.nn.functional.interpolate(prob[None], size=(h, w), mode='bilinear', align_corners=False)[0]
    return prob.cpu().numpy()
