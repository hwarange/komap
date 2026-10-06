#!/bin/bash
# 6단계 자동 진행 스크립트 (서버 ~/komap 에서 nohup으로 실행)
#   setsid nohup bash scripts/run_round6.sh > runs/round6_driver.log 2>&1 < /dev/null &
# 진행 상황: runs/round6_driver.log, runs/launch_round6.log, runs/launch_round6c.log
# 끝나면 runs/round6_done.txt 가 생긴다.
cd ~/komap
PY=.venv/bin/python
log() { echo "$(date '+%m-%d %H:%M:%S') $*"; }

log "A2 크기 키운 TTA 평가 시작 (GPU 3, 학습과 함께)"
(CUDA_VISIBLE_DEVICES=3 $PY src/eval_scale_tta.py unetpp_bw_e120_r90 cnx_bw --out runs/scale_tta.json > runs/scale_tta.log 2>&1; echo "A2 exit $?" >> runs/scale_tta.log) &
A2=$!

log "A1 + B + B4 학습 시작 (experiments/round6.txt)"
$PY src/launch.py experiments/round6.txt --gpus 0 1 2 3 --per-gpu 2 > runs/launch_round6.log 2>&1
log "round6 학습 끝: $(grep 'all finished' runs/launch_round6.log)"

log "A1 학습 곡선 판단"
$PY src/learning_curve.py --out runs/learning_curve.json > runs/learning_curve.log 2>&1
cat runs/learning_curve.log
if grep -q '"decision_full90": true' runs/learning_curve.json; then
  log "판단: 90장 전체 학습(C) 진행"
  $PY src/launch.py experiments/round6c_full.txt --root runs/full --folds 3 --gpus 0 1 2 3 --per-gpu 2 > runs/launch_round6c.log 2>&1
  log "C 끝: $(grep 'all finished' runs/launch_round6c.log)"
else
  log "판단: 90장 전체 학습(C) 건너뜀 (기준 미달 또는 결과 불완전)"
fi

wait $A2
log "A2 끝: $(tail -n 1 runs/scale_tta.log)"

log "앙상블 재평가 (기존 16개 + 6단계 새 모델, 예측 확률이 모두 있는 것만 사용)"
$PY src/ensemble.py --max-size 6 --out runs/oof/ensemble_summary_r6.json --exps \
  unetpp_bw_e120_r90 x2_c1024_bw_r90 a_rot90 cnx_bw_e120 unetpp_bw_e120 m_hires bw4_r90 a_lean unetpp_bw cnx_bw \
  cnx_bw_c768 cnx_bw_e120_r90 unet_mit_bw_r90 x2_c1024 unet_mit upp_r50_bw \
  x2_unetpp_bw_r90 mn_sxt50_unetpp cnxS_bw_e120 effv2s_unetpp hrnet_unet mn_r34_unetpp unetpp_bw_r90 m_dinov3_lr m_skel_w015 \
  > runs/oof/ensemble_log_r6.txt 2>&1
$PY src/aggregate.py > runs/cv/summary.txt 2>&1
log "모두 끝"
date '+%m-%d %H:%M:%S' > runs/round6_done.txt
