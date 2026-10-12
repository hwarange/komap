"""전체 이미지 예측(슬라이딩 윈도우, TTA), 대회 방식 채점, 후처리."""
import cv2
import numpy as np
import torch

from data import normalize, scale_for, to_input, to_input_feats

N_CLASSES = 4
SIP = 2


def image_iou(gt, pr):
    """한 이미지의 상별 IoU. 정답·예측 모두에 없는 상은 nan (평균에서 제외)."""
    out = np.full(N_CLASSES, np.nan)
    for c in range(N_CLASSES):
        g, p = gt == c, pr == c
        u = np.logical_or(g, p).sum()
        if u:
            out[c] = np.logical_and(g, p).sum() / u
    return out


def _window(tile):
    # 타일 가장자리보다 가운데 예측을 더 믿도록 가중
    g = cv2.getGaussianKernel(tile, tile / 4)
    w = (g @ g.T).astype(np.float32)
    return np.maximum(w / w.max(), 1e-3)


def _dihedral(x, k):
    # x: [..., H, W]. k=0..7: 회전 k%4번 + (k>=4면 좌우 반전)
    if k >= 4:
        x = torch.flip(x, dims=[-1])
    return torch.rot90(x, k % 4, dims=[-2, -1])


def _undihedral(x, k):
    x = torch.rot90(x, -(k % 4), dims=[-2, -1])
    if k >= 4:
        x = torch.flip(x, dims=[-1])
    return x


@torch.no_grad()
def _sliding(model, img, tile, overlap, tta, device, batch=4):
    H, W = img.shape
    x = np.pad(img, ((0, max(tile - H, 0)), (0, max(tile - W, 0))), mode='reflect')
    Hp, Wp = x.shape
    stride = tile - overlap
    ys = list(range(0, Hp - tile + 1, stride))
    xs = list(range(0, Wp - tile + 1, stride))
    if ys[-1] != Hp - tile: ys.append(Hp - tile)
    if xs[-1] != Wp - tile: xs.append(Wp - tile)
    coords = [(y, x0) for y in ys for x0 in xs]
    win = torch.from_numpy(_window(tile)).to(device)
    acc = None   # 출력 상 수는 모델에서 정해진다 (4상 모델 / 한 상 전용 모델 2)
    norm = torch.zeros(Hp, Wp, device=device)
    feats = getattr(model, 'input_feats', False)   # 9단계: 필터 채널 입력 모델 (사진 전체에서 한 번 계산)
    xin = torch.from_numpy(to_input_feats(x) if feats else to_input(x)[None]).to(device)
    ks = range(8) if tta else [0]
    for i in range(0, len(coords), batch):
        cs = coords[i:i + batch]
        t = torch.stack([xin[:, y:y + tile, x0:x0 + tile] for y, x0 in cs])
        prob = 0
        for k in ks:
            o = model(_dihedral(t, k))
            if o.shape[1] == 2 * N_CLASSES:
                # 36단계 거리장 모델: 앞 4채널은 상 확률, 뒤 4채널은 부호 거리장(tanh). 둘 다 같은 창으로 평균한다
                o = torch.cat([torch.softmax(o[:, :N_CLASSES], 1), torch.tanh(o[:, N_CLASSES:])], 1)
            else:
                o = torch.softmax(o, 1)
            prob = prob + _undihedral(o, k)
        prob = prob / len(ks)
        if acc is None:
            acc = torch.zeros(prob.shape[1], Hp, Wp, device=device)
        for (y, x0), p in zip(cs, prob):
            acc[:, y:y + tile, x0:x0 + tile] += p * win
            norm[y:y + tile, x0:x0 + tile] += win
    return (acc / norm)[:, :H, :W]


@torch.no_grad()
def predict_probs(model, gray, target_width=1280, tile=512, overlap=128, tta=False, device='cuda', extra_scale=1.0, scale_multiplier=1.0,
                  with_sdf=False):
    """원본 해상도의 상별 확률 [4, H, W]를 돌려준다.
    extra_scale: 학습 배율에 더해 사진을 더 키워 예측 (6단계 크기 키운 TTA, 예: 1.25).
    with_sdf: 거리장 모델(출력 8채널)이면 [8, H, W] (확률 4 + 거리장 4)를 그대로 돌려준다. 아니면 확률 4채널만."""
    img = normalize(gray)
    H, W = img.shape
    s = scale_for(W, target_width, scale_multiplier) * extra_scale
    if s != 1:
        img = cv2.resize(img, (round(W * s), round(H * s)), interpolation=cv2.INTER_LINEAR)
    prob = _sliding(model, img, tile, overlap, tta, device)
    if s != 1:
        # 키운 만큼 되돌린다. 정수 배면 블록 평균(정확히 s×s 평균), 정수가 아니면 bilinear.
        # torch 'area'는 정수가 아닌 배율에서 평균 창이 겹쳐 확률을 흐린다 (정답 왕복 시험, 1.1배: area 97.85 / bilinear 99.95)
        if float(s).is_integer():
            prob = torch.nn.functional.interpolate(prob[None], size=(H, W), mode='area')[0]
        else:
            prob = torch.nn.functional.interpolate(prob[None], size=(H, W), mode='bilinear', align_corners=False)[0]
    if prob.shape[0] == 2 * N_CLASSES and not with_sdf:
        prob = prob[:N_CLASSES]
    return prob.cpu().numpy()


def postprocess(prob, min_comp=1e-4, min_total=5e-3):
    """Primary Si 오탐 제거 (EDA 7절).
    - 이미지 면적의 min_comp 미만인 Primary Si 조각은 지운다.
    - 남은 Primary Si가 전체의 min_total 미만이면 모두 지운다.
    지운 픽셀은 Primary Si를 뺀 나머지 상 중 확률이 가장 높은 상으로 바꾼다."""
    pred = prob.argmax(0).astype(np.uint8)
    sip = (pred == SIP).astype(np.uint8)
    if not sip.any():
        return pred
    area = pred.size
    n, cc, st, _ = cv2.connectedComponentsWithStats(sip, connectivity=8)
    small = np.isin(cc, 1 + np.where(st[1:, 4] < min_comp * area)[0])
    keep = sip.astype(bool) & ~small
    drop = sip.astype(bool) & ~keep if keep.sum() >= min_total * area else sip.astype(bool)
    if drop.any():
        alt = prob.copy()
        alt[SIP] = -1
        pred[drop] = alt.argmax(0)[drop]
    return pred
