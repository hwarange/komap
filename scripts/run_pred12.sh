#!/bin/bash
# 12단계 새 모델 4개의 평가 사진 예측 (sub6 후보용, 최종 조합 평가에서 채택될 때 바로 combine 할 수 있게 미리 해 둔다)
# 모델별로 submissions/sub3_pred/parts/models/<실험>/fold<k>/<사진>.npy 에 저장. sub5가 쓰는 기존 예측은 건드리지 않는다.
#   setsid nohup bash scripts/run_pred12.sh > runs/pred12/driver.log 2>&1 < /dev/null &
# 끝나면 runs/pred12/pred12.done
cd ~/komap
PY=.venv/bin/python
log() { echo "$(date '+%m-%d %H:%M:%S') $*"; }
PRED=submissions/sub3_pred
NEW="sie_x2_unetpp_e120 sie_x2_cnx_e120 x2_unetpp_bw_e120_r90_s2 x3_unetpp_bw_r90"
mkdir -p runs/pred12

log "평가 사진 예측: 새 모델 4개 실험 × 5-fold (GPU 0~3)"
PP=()
for i in 0 1 2 3; do
  (CUDA_VISIBLE_DEVICES=$i $PY src/submit.py predict --exps $NEW --split test --out $PRED --shard $i --nshards 4) > runs/pred12/pred_test_gpu$i.log 2>&1 &
  PP+=($!)
done
F=0
for p in "${PP[@]}"; do wait $p || F=$((F + 1)); done
log "평가 사진 예측 끝 (실패한 GPU 작업 $F개): $(for e in $NEW; do echo -n "$e=$(ls $PRED/parts/models/$e/fold*/*.npy 2>/dev/null | wc -l)/50 "; done)"
date '+%m-%d %H:%M:%S' > runs/pred12/pred12.done
