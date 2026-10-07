"""분할 모델 학습.

예) python src/train.py --out runs/baseline                          (공식 train/valid 분리)
    python src/train.py --fold 0 --out runs/cv/baseline/fold0 --amp  (90장 합금 단위 5-fold 중 0번)
    python src/train.py --out runs/smoke --epochs 1 --max-iters 5 --val-every 1   (동작 확인용)

교차검증 모드에서는 학습이 끝난 뒤 '마지막 에폭' 모델을 TTA 포함으로 채점해 result.json에 저장한다.
최고 에폭을 고르면 같은 fold로 고르고 채점하게 되어 점수가 부풀기 때문이다.
"""
import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import CLASSES, IGNORE, CropDataset, load_cv, load_split, worker_init_fn
from infer import image_iou, postprocess, predict_probs


class HiResUnet(nn.Module):
    """U-Net 디코더 출력에 원래 해상도·1/2 해상도의 얕은 특징을 더하는 보조 경로 (4단계).
    ConvNeXt·MiT 인코더는 가장 세밀한 연결이 1/4 해상도라서, 1~4px 구조를 위한 세부 정보를 직접 넣어 준다."""

    def __init__(self, base):
        super().__init__()
        self.encoder, self.decoder = base.encoder, base.decoder
        dec_ch = base.segmentation_head[0].in_channels

        def block(i, o, stride=1):
            return nn.Sequential(nn.Conv2d(i, o, 3, stride, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        self.full_res = nn.Sequential(block(1, 32), block(32, 32))
        self.half_res = nn.Sequential(block(32, 64, stride=2), block(64, 64))  # 이름 "half"는 nn.Module.half()와 충돌
        self.fuse = block(dec_ch + 32 + 64, 32)
        self.head = nn.Conv2d(32, len(CLASSES), 1)

    def forward(self, x):
        dec = self.decoder(self.encoder(x))
        f1 = self.full_res(x)
        f2 = F.interpolate(self.half_res(f1), size=x.shape[-2:], mode='bilinear', align_corners=False)
        return self.head(self.fuse(torch.cat([dec, f1, f2], 1)))


MICRONET_S3 = 'https://nasa-public-data.s3.amazonaws.com/microscopy_segmentation_models/'


def micronet_url(enc):
    """MicroNet (Stuckner et al. 2022) 'ImageNet → 현미경' 사전학습 가중치 주소 (6단계)."""
    return MICRONET_S3 + f'{enc}_pretrained_imagenet-microscopynet_v1.0.pth.tar'


def build_model(arch, encoder, weights='imagenet', attention=None, hires=False, n_classes=len(CLASSES), feats=False):
    if feats:
        # 9단계: 밝기 + black top-hat + ridge 3채널 입력. ImageNet RGB 가중치를 그대로 쓴다. infer.py가 input_feats를 보고 필터를 계산
        m = getattr(smp, arch)(encoder_name=encoder, encoder_weights=weights, in_channels=3, classes=n_classes)
        m.input_feats = True
        return m
    kw = {'decoder_attention_type': attention} if attention else {}
    if encoder.startswith('micronet:'):
        # 공식 패키지는 smp를 0.2.1로 내리므로 쓰지 않고, 가중치를 직접 받아 인코더에 넣는다
        enc = encoder.split(':', 1)[1]
        if weights is None:   # 체크포인트에서 불러올 때는 같은 구조만 만들면 된다
            m = getattr(smp, arch)(encoder_name=enc, encoder_weights=None, in_channels=1, classes=n_classes, **kw)
        else:
            m = getattr(smp, arch)(encoder_name=enc, encoder_weights=None, in_channels=3, classes=n_classes, **kw)
            sd = torch.hub.load_state_dict_from_url(micronet_url(enc), map_location='cpu')
            m.encoder.load_state_dict({k.removeprefix('module.'): v for k, v in sd.items()})
            m.encoder.set_in_channels(1, pretrained=True)   # RGB 필터를 합쳐 흑백 1채널로 (smp의 ImageNet 처리와 같음)
    else:
        m = getattr(smp, arch)(encoder_name=encoder, encoder_weights=weights, in_channels=1, classes=n_classes, **kw)
    return HiResUnet(m) if hires else m


def subsample_by_alloy(items, frac, seed=0):
    """학습 사진을 합금 단위로 frac 비율만 남긴다 (6단계 학습 곡선용)."""
    alloys = sorted({it['alloy'] for it in items})
    order = np.random.default_rng(seed).permutation(len(alloys))
    target, keep, n = round(frac * len(items)), set(), 0
    for i in order:
        if n >= target:
            break
        a = alloys[i]
        keep.add(a)
        n += sum(it['alloy'] == a for it in items)
    return [it for it in items if it['alloy'] in keep]


def build_from_args(a, weights=None):
    """체크포인트에 저장된 학습 설정으로 같은 구조의 모델을 만든다 (예측·분석·제출 스크립트용)."""
    two = a.get('binary_class', -1) >= 0 or bool(a.get('pair_classes'))
    return build_model(a['arch'], a['encoder'], weights=weights, attention=a.get('attention'), hires=a.get('hires', False),
                       n_classes=2 if two else len(CLASSES), feats=a.get('feats', False))


def autocast(enabled):
    # BF16 혼합정밀도. L40S 같은 최신 GPU용 (GTX 1060에서는 끈다)
    return torch.autocast('cuda', dtype=torch.bfloat16, enabled=enabled)


class SegLoss(nn.Module):
    """CE(경계·상별 가중 가능) + Dice 또는 Lovász."""

    def __init__(self, second='dice', boundary_weight=0.0, class_weights=None, skel_weight=0.0, use_ignore=False,
                 soft_edge=0.0):
        super().__init__()
        self.bw, self.skel, self.soft = boundary_weight, skel_weight, soft_edge
        self.register_buffer('cw', torch.tensor(class_weights, dtype=torch.float32) if class_weights else None)
        ig = IGNORE if use_ignore else None   # cutout으로 가린 픽셀(IGNORE)을 손실에서 뺀다 (7단계)
        self.second = (smp.losses.LovaszLoss('multiclass', from_logits=True, ignore_index=ig) if second == 'lovasz'
                       else smp.losses.DiceLoss('multiclass', from_logits=True, ignore_index=ig))

    def forward(self, logits, y, tube=None):
        valid = y != IGNORE   # IGNORE가 없으면 모두 True라 이전과 같은 계산
        # 정답 경계에서 1px 이내 픽셀(3×3 이웃에 다른 상이 있음). 오답의 79%가 여기 (기본 모델 오답 분석)
        lab = y[:, None].float()
        edge = (F.max_pool2d(lab, 3, 1, 1) != -F.max_pool2d(-lab, 3, 1, 1))[:, 0]
        edge = edge & ~(F.max_pool2d((~valid)[:, None].float(), 3, 1, 1)[:, 0] > 0)   # 가린 네모의 테두리는 경계가 아님
        if self.soft > 0:
            # 경계 부드러운 정답 (7단계): 경계 띠에서는 정답을 (1-ε)·원래 상 + ε·주변 3×3의 상 비율로 준다.
            # 경계 1px의 라벨이 애매하다는 진단(경계 띠 제외 시 79.7→88.0)에서 출발. 경계 밖은 원래와 같다.
            oh = F.one_hot(y.clamp(max=logits.shape[1] - 1), logits.shape[1]).permute(0, 3, 1, 2).float()
            hist = F.avg_pool2d(oh, 3, 1, 1)
            hist = hist / hist.sum(1, keepdim=True).clamp_min(1e-6)
            tgt = torch.where(edge[:, None], (1 - self.soft) * oh + self.soft * hist, oh)
            ce = F.cross_entropy(logits.float(), tgt, weight=self.cw, reduction='none')
        else:
            ce = F.cross_entropy(logits.float(), y, weight=self.cw, reduction='none', ignore_index=IGNORE)
        if self.bw > 0:
            w = 1 + self.bw * edge.float()
            ce = ce * w / w[valid].mean()
        loss = ce[valid].mean() + self.second(logits.float(), y)
        if self.skel > 0 and tube is not None and tube.sum() > 0:
            # Skeleton Recall: Eutectic Si 중심선 띠에서 Eutectic Si 확률이 낮으면 벌점 (Kirchhoff et al. ECCV 2024)
            p_sie = torch.softmax(logits.float(), 1)[:, 3]
            loss = loss + self.skel * (1 - (p_sie * tube).sum() / (tube.sum() + 1e-6))
        return loss


@torch.no_grad()
def evaluate(model, items, width, tile, overlap, device, tta=False, amp=False, save_dir=None, binary_class=-1, pair=None):
    """save_dir를 주면 사진별 예측 확률 [4,H,W]를 float16 .npy로 저장한다 (앙상블·후처리 실험용).
    binary_class >= 0 (한 상 전용 모델): 확률은 [2,H,W] (0: 나머지, 1: 그 상), 점수는 [나머지 IoU, 그 상 IoU, nan, nan]."""
    model.eval()
    rows = []
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
    for it in items:
        with autocast(amp):
            prob = predict_probs(model, it['gray'], width, tile, overlap, tta, device)
        if save_dir:
            np.save(Path(save_dir) / f"{it['name']}.npy", prob.astype(np.float16))
        if pair:
            # 두 상 전용 모델 (9단계): 정답이 두 상 중 하나인 픽셀에서만 채점 (0: pair[0], 1: pair[1])
            m = np.isin(it['lab'], pair)
            iou = image_iou((it['lab'][m] == pair[1]).astype(np.uint8), prob.argmax(0)[m])
            rows.append(dict(name=it['name'], step=it['step'], low=it['low_contrast'], W=int(it['gray'].shape[1]), iou=iou, iou_pp=iou))
            continue
        if binary_class >= 0:
            iou = image_iou((it['lab'] == binary_class).astype(np.uint8), prob.argmax(0))
            rows.append(dict(name=it['name'], step=it['step'], low=it['low_contrast'], W=int(it['gray'].shape[1]), iou=iou, iou_pp=iou))
            continue
        rows.append(dict(name=it['name'], step=it['step'], low=it['low_contrast'], W=int(it['gray'].shape[1]),
                         iou=image_iou(it['lab'], prob.argmax(0)), iou_pp=image_iou(it['lab'], postprocess(prob))))
    return rows


def summarize(rows, key='iou'):
    """대회 방식: 이미지별 mIoU를 구한 뒤 이미지 수로 평균. 상별·스타일별·Step별 점수도 함께."""
    ious = np.array([r[key] for r in rows])
    per_img = np.nanmean(ious, 1)
    s = {'mIoU': per_img.mean() * 100}
    s.update({c: np.nanmean(ious[:, i]) * 100 for i, c in enumerate(CLASSES)})
    for flag, name in [(True, 'low_contrast'), (False, 'high_contrast')]:
        m = np.array([r['low'] == flag for r in rows])
        if m.any():
            s[name] = per_img[m].mean() * 100
    for st in sorted({r['step'] for r in rows}):
        s[f'step{st}'] = per_img[np.array([r['step'] == st for r in rows])].mean() * 100
    return {k: round(float(v), 2) for k, v in s.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data')
    p.add_argument('--out', default='runs/baseline')
    p.add_argument('--train-splits', nargs='+', default=['train'])
    p.add_argument('--val-split', default='valid')
    p.add_argument('--fold', type=int, default=-1, help='0 이상이면 90장 5-fold 교차검증 모드')
    p.add_argument('--folds-file', default='splits/folds5.json')
    p.add_argument('--arch', default='Unet')
    p.add_argument('--encoder', default='resnet34')
    p.add_argument('--attention', default=None, help='디코더 attention (예: scse)')
    p.add_argument('--loss', default='dice', choices=['dice', 'lovasz'], help='CE와 함께 쓰는 두 번째 손실')
    p.add_argument('--bw', type=float, default=0.0, help='경계 픽셀 CE 가중치 (0이면 끔)')
    p.add_argument('--class-weights', default='', help='상별 CE 가중치 (예: 1,1.5,1,2)')
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--bs', type=int, default=4)
    p.add_argument('--crop', type=int, default=512)
    p.add_argument('--spi', type=int, default=8, help='에폭당 이미지 1장에서 뽑는 crop 수')
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--width', type=int, default=1280, help='배율 기준 가로 길이 (0이면 원본 그대로)')
    p.add_argument('--tile', type=int, default=512)
    p.add_argument('--overlap', type=int, default=128)
    p.add_argument('--val-every', type=int, default=5)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max-iters', type=int, default=0, help='에폭당 최대 반복 수 (0이면 제한 없음)')
    p.add_argument('--amp', action='store_true', help='BF16 혼합정밀도 (L40S 등 최신 GPU)')
    p.add_argument('--save-oof', default='', help='교차검증 마지막 TTA 예측 확률을 저장할 폴더 (예: runs/oof/<실험>)')
    # 4단계 옵션
    p.add_argument('--hires', action='store_true', help='원래·1/2 해상도 보조 경로 추가')
    p.add_argument('--skel', type=float, default=0.0, help='Eutectic Si Skeleton Recall 손실 가중치 (0이면 끔)')
    p.add_argument('--rot', default='any', choices=['any', 'rot90'], help='회전 증강: 임의 각도 / 90도만')
    p.add_argument('--aug-policy', default='default', choices=['default', 'lean'])
    p.add_argument('--hm', type=float, default=0.0, help='상 기준점 단조 매핑 확률')
    p.add_argument('--lowres', type=float, default=0.0, help='저해상도 흉내 확률')
    p.add_argument('--no-int', action='store_true', help='밝기 증강 끄기')
    p.add_argument('--copypaste', type=float, default=0.0, help='Primary Si·Al3Ni 복사-붙여넣기 확률')
    # 5단계 옵션
    p.add_argument('--int-drop', default='', help='끌 밝기 증강 (쉼표 구분: gamma,contrast,illum,blur,noise)')
    p.add_argument('--ema', type=float, default=0.0, help='가중치 이동 평균 decay (예: 0.999, 0이면 끔). 켜면 검증·저장 모두 EMA 모델로')
    # 6단계 옵션
    p.add_argument('--train-frac', type=float, default=1.0, help='학습 사진을 합금 단위로 이 비율만 사용 (학습 곡선)')
    p.add_argument('--enc-lr-mult', type=float, default=1.0, help='인코더 학습률 배수 (예: 0.1)')
    p.add_argument('--full', action='store_true', help='학습+검증 90장 전체로 학습 (검증 없음). --fold 값은 시드 번호로 사용')
    # 7단계 옵션
    p.add_argument('--binary-class', type=int, default=-1, help='0 이상이면 그 상 하나만 구분하는 전용 모델 (예: 3 = Eutectic Si)')
    p.add_argument('--elastic', type=float, default=0.0, help='탄성 변형 확률 (crop마다)')
    p.add_argument('--cutout', type=float, default=0.0, help='일부 가리기 확률 (crop마다, 가린 곳은 손실에서 뺌)')
    p.add_argument('--mixup', type=float, default=0.0, help='mixup 확률 (배치마다, 두 crop을 λ~Beta(0.4,0.4)로 겹침)')
    p.add_argument('--cutmix', type=float, default=0.0, help='CutMix 확률 (배치마다, 다른 crop의 네모 조각을 정답째 붙임)')
    # 9단계 옵션
    p.add_argument('--init', default='', help='먼저 학습한 체크포인트(last.pt)에서 시작 (외부 데이터 사전학습)')
    p.add_argument('--init-skip-head', action='store_true', help='--init에서 출력층은 불러오지 않고 새로 (상 정의가 다른 외부 데이터)')
    p.add_argument('--use-ignore', action='store_true', help='정답의 255(대응 상 없음)를 손실에서 뺀다 (외부 데이터 학습)')
    p.add_argument('--pair-classes', type=int, nargs=2, default=None,
                   help='두 상 전용 모델 (예: 1 3 = Al3Ni vs Eutectic Si). 두 상이 아닌 픽셀은 손실에서 뺀다')
    p.add_argument('--feats', action='store_true', help='입력에 black top-hat·ridge 필터 채널 추가 (3채널)')
    p.add_argument('--hard-crop', type=float, default=0.0, help='오답 위치(경계 띠 밖) 중심으로 자를 확률 (0이면 끔)')
    p.add_argument('--hard-maps', default='runs/hardmaps', help='사진별 오답 지도 폴더 (src/make_hardmaps.py)')
    p.add_argument('--soft-edge', type=float, default=0.0,
                   help='경계 1px 띠의 정답을 주변 3×3 상 비율과 섞는 정도 ε (0이면 끔, cutout과 같이 쓰지 않음)')
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'args.json').write_text(json.dumps(vars(args), indent=2))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = 'cuda'

    if args.full:
        # 90장 전체 학습: 검증 없음. fold 번호는 시드로 쓴다 (같은 설정 × 여러 시드)
        train_items, val_items = load_split(args.data, 'train') + load_split(args.data, 'valid'), []
        torch.manual_seed(args.seed + 1000 * max(args.fold, 0))
        np.random.seed(args.seed + 1000 * max(args.fold, 0))
    elif args.fold >= 0:
        train_items, val_items = load_cv(args.data, args.fold, args.folds_file)
    else:
        train_items = [it for s in args.train_splits for it in load_split(args.data, s)]
        val_items = load_split(args.data, args.val_split) if args.val_split else []
    if args.train_frac < 1.0:
        train_items = subsample_by_alloy(train_items, args.train_frac, args.seed)
    ds = CropDataset(train_items, args.crop, args.spi, args.width, rot=args.rot, policy=args.aug_policy, hm=args.hm,
                     lowres=args.lowres, no_int=args.no_int, copypaste=args.copypaste, skel=args.skel > 0,
                     int_drop=[v for v in args.int_drop.split(',') if v], elastic=args.elastic, cutout=args.cutout,
                     hard=args.hard_crop, hard_maps=args.hard_maps, feats=args.feats)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=args.workers, drop_last=True,
                    worker_init_fn=worker_init_fn, persistent_workers=args.workers > 0, pin_memory=True)

    binary, pair = args.binary_class >= 0, args.pair_classes
    assert not (binary and pair), '--binary-class와 --pair-classes는 같이 쓰지 않는다'
    model = build_model(args.arch, args.encoder, attention=args.attention, hires=args.hires,
                        n_classes=2 if (binary or pair) else len(CLASSES), feats=args.feats).to(device)
    if args.init:
        # 외부 데이터로 먼저 학습한 가중치에서 시작 (9단계). 모양이 같은 층만 불러오고, --init-skip-head면 출력층은 새로
        sd = torch.load(args.init, map_location='cpu', weights_only=False)['model']
        own = model.state_dict()
        keep = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape
                and not (args.init_skip_head and k.startswith('segmentation_head'))}
        model.load_state_dict(keep, strict=False)
        print(f'init from {args.init}: loaded {len(keep)}/{len(own)} tensors'
              f'{" (출력층 새로)" if args.init_skip_head else ""}', flush=True)
    cw = [float(v) for v in args.class_weights.split(',')] if args.class_weights else None
    assert not (args.soft_edge > 0 and (args.cutout > 0 or pair)), '--soft-edge는 무시 픽셀(cutout·pair)과 같이 쓰지 않는다'
    criterion = SegLoss(args.loss, args.bw, cw, args.skel, use_ignore=args.cutout > 0 or bool(pair) or args.use_ignore,
                        soft_edge=args.soft_edge).to(device)
    enc_params = [q for n, q in model.named_parameters() if n.startswith('encoder.')]
    other_params = [q for n, q in model.named_parameters() if not n.startswith('encoder.')]
    opt = torch.optim.AdamW([{'params': enc_params, 'lr': args.lr * args.enc_lr_mult},
                             {'params': other_params, 'lr': args.lr}], weight_decay=args.wd)
    iters = len(dl) if not args.max_iters else min(len(dl), args.max_iters)
    total, warm = args.epochs * iters, iters
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: (i + 1) / warm if i < warm else 0.5 * (1 + math.cos(math.pi * (i - warm) / max(total - warm, 1))))

    ema = copy.deepcopy(model).eval() if args.ema else None
    for q in (ema.parameters() if ema else []):
        q.requires_grad_(False)
    step = 0
    print(f'train {len(train_items)} imgs, {iters} iters/epoch | val {len(val_items)} imgs | {args.arch}-{args.encoder}', flush=True)
    best, last_rows = -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, run, n = time.time(), 0.0, 0
        for batch in dl:
            x, y = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
            if binary:
                # 그 상 = 1, 나머지 = 0 (경계 가중도 그 상의 경계에 걸린다). 가린 픽셀(IGNORE)은 그대로
                y = torch.where(y == IGNORE, y, (y == args.binary_class).long())
            elif pair:
                # 두 상 전용 모델: pair[0] = 0, pair[1] = 1, 나머지 상은 무시 (두 상 사이의 판단만 배운다)
                y = torch.where(y == pair[0], 0, torch.where(y == pair[1], 1, IGNORE))
            tube =batch[2].to(device, non_blocking=True) if len(batch) > 2 else None
            lam, perm = 1.0, None
            if args.mixup and np.random.rand() < args.mixup:
                # mixup: 두 crop을 겹치고 손실도 같은 비율로 섞는다
                lam, perm = float(np.random.beta(0.4, 0.4)), torch.randperm(x.size(0), device=device)
                x = lam * x + (1 - lam) * x[perm]
            elif args.cutmix and np.random.rand() < args.cutmix:
                # CutMix: 다른 crop의 네모 조각(넓이 비율 1-λ, λ~U(0,1))을 정답째 붙인다
                perm, H, W = torch.randperm(x.size(0), device=device), x.size(2), x.size(3)
                r = math.sqrt(1 - np.random.rand())
                h, w = int(H * r), int(W * r)
                y0, x0 = np.random.randint(0, H - h + 1), np.random.randint(0, W - w + 1)
                x, y = x.clone(), y.clone()
                x[:, :, y0:y0 + h, x0:x0 + w] = x[perm, :, y0:y0 + h, x0:x0 + w]
                y[:, y0:y0 + h, x0:x0 + w] = y[perm, y0:y0 + h, x0:x0 + w]
                perm = None   # 정답을 이미 붙였으므로 손실은 한 번만
            with autocast(args.amp):
                out_logits = model(x)
                loss = criterion(out_logits, y, tube)
                if perm is not None:
                    loss = lam * loss + (1 - lam) * criterion(out_logits, y[perm], tube[perm] if tube is not None else None)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if ema is not None:
                # 초반에는 decay를 낮게 시작해 무작위 초기값의 영향을 빨리 지운다
                d = min(args.ema, (1 + step) / (10 + step))
                with torch.no_grad():
                    for pe, pm in zip(ema.parameters(), model.parameters()):
                        pe.mul_(d).add_(pm.detach(), alpha=1 - d)
                    for be, bm in zip(ema.buffers(), model.buffers()):
                        be.copy_(bm)
            run, n = run + loss.item(), n + 1
            if n >= iters:
                break
        log = {'epoch': epoch, 'loss': round(run / n, 4), 'lr': opt.param_groups[0]['lr'], 'train_sec': round(time.time() - t0, 1),
               'gpu_mem_gb': round(torch.cuda.max_memory_allocated() / 1e9, 2)}

        if val_items and (epoch % args.val_every == 0 or epoch == args.epochs):
            t1 = time.time()
            last_rows = evaluate(ema if ema is not None else model, val_items, args.width, args.tile, args.overlap, device, amp=args.amp,
                                 binary_class=args.binary_class, pair=args.pair_classes)
            model.train()
            log['val'], log['val_pp'] = summarize(last_rows), summarize(last_rows, 'iou_pp')
            log['val_sec'] = round(time.time() - t1, 1)
            if log['val']['mIoU'] > best:
                best = log['val']['mIoU']
                torch.save({'model': (ema if ema is not None else model).state_dict(), 'args': vars(args), 'epoch': epoch, 'val': log['val']}, out / 'best.pt')
                log['saved'] = True
        print(json.dumps(log, ensure_ascii=False), flush=True)
        with open(out / 'log.jsonl', 'a', encoding='utf-8') as f:
            f.write(json.dumps(log, ensure_ascii=False) + '\n')

    final = ema if ema is not None else model
    torch.save({'model': final.state_dict(), 'args': vars(args), 'epoch': args.epochs}, out / 'last.pt')

    if args.full:
        # 검증이 없으므로 완료 표시만 남긴다 (launch.py가 result.json으로 완료를 판단)
        (out / 'result.json').write_text(json.dumps({'args': vars(args), 'full': True, 'rows': []}))
    if val_items:
        # 마지막 에폭 모델의 최종 채점: 그대로 / 후처리 / TTA / TTA+후처리
        tta_rows = evaluate(final, val_items, args.width, args.tile, args.overlap, device, tta=True, amp=args.amp,
                            save_dir=args.save_oof or None, binary_class=args.binary_class, pair=args.pair_classes)
        rows = [dict(name=a['name'], step=a['step'], low=a['low'], W=a['W'],
                     plain=a['iou'].tolist(), pp=a['iou_pp'].tolist(), tta=b['iou'].tolist(), tta_pp=b['iou_pp'].tolist())
                for a, b in zip(last_rows, tta_rows)]
        (out / 'result.json').write_text(json.dumps({'args': vars(args), 'rows': rows}))
        print('final', json.dumps({k: summarize([dict(r, iou=np.array(r[k])) for r in rows])['mIoU']
                                   for k in ['plain', 'pp', 'tta', 'tta_pp']}), flush=True)
    print(f'done. best val mIoU {best:.2f}', flush=True)


if __name__ == '__main__':
    main()
