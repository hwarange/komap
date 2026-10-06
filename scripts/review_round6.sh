#!/bin/bash
# 6단계 결과 한 번에 점검 (서버 ~/komap 에서, 6단계가 끝난 뒤 실행)
#   bash scripts/review_round6.sh > runs/review_round6.txt 2>&1
# 실험마다 계획(reports/plan_round6.md)에서 정한 비교 대상과 같은 사진끼리 비교한다.
cd ~/komap
PY=.venv/bin/python
r() { echo "===== $1  vs  $2  ($3)"; $PY src/review.py "$1" --base "$2" 2>&1 | grep -v -i warn; }

echo "##### A1 학습 곡선 판단"; cat runs/learning_curve.json
echo "##### A2 크기 키운 TTA"; cat runs/scale_tta.json
r x2_unetpp_bw_r90 x2_c1024_bw_r90   "B1: 2배 해상도 + UNet++ vs 2배 해상도 U-Net"
r x2_unetpp_bw_r90 unetpp_bw_e120_r90 "B1: vs 최고 단일 모델"
r mn_r34_unetpp    unetpp_bw_r90      "B2: MicroNet ResNet34 vs ImageNet 대조군 (사전학습만 다름)"
r mn_sxt50_unetpp  unetpp_bw_r90      "B2: MicroNet SE-ResNeXt50 vs 대조군"
r unetpp_bw_r90    unetpp_bw          "B2 대조군: 90도 회전 효과 (UNet++ 60에폭)"
r effv2s_unetpp    unetpp_bw_r90      "B3: EfficientNetV2-S (다양성용, 앙상블 기여로 판단)"
r hrnet_unet       unetpp_bw_r90      "B3: HRNet-W18 (다양성용)"
r cnxS_bw_e120     cnxS_bw            "B4: ConvNeXt-Small 120에폭 vs 60에폭"
r m_dinov3_lr      m_dinov3           "B4: DINOv3 인코더 학습률 1/10 vs 3e-4"
r m_dinov3_lr      cnx_bw             "B4: DINOv3(낮은 학습률) vs ImageNet ConvNeXt"
r m_skel_w015      cnx_bw             "B4: Skeleton 손실 0.15 vs 없음"
echo "##### 앙상블 재평가 (기존 16개 + 6단계 모델)"; grep -v -i warn runs/oof/ensemble_log_r6.txt | sed -n '/predefined/,$p'
echo "##### C 단계 (90장 전체 학습) 실행 기록"; cat runs/launch_round6c.log 2>/dev/null || echo "C 단계 실행 안 함"
