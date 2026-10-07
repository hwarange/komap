#!/bin/bash
# 9단계 자동 평가: 실험마다 끝나면 바로 평가하고 runs/review9/<실험>.done 을 남긴다. 모두 끝나면 앙상블 기여까지.
#   setsid nohup bash scripts/review_round9.sh > runs/review9/driver.log 2>&1 < /dev/null &
cd ~/komap
PY=.venv/bin/python
mkdir -p runs/review9
A="unetpp_bw_e120_r90 x2_c1024_bw_r90 a_rot90 cnx_bw_e120 unetpp_bw_e120 m_hires bw4_r90 a_lean unetpp_bw cnx_bw \
cnx_bw_c768 cnx_bw_e120_r90 unet_mit_bw_r90 x2_c1024 unet_mit upp_r50_bw \
x2_unetpp_bw_e120_r90 x2_unetpp_bw_r90 mn_sxt50_unetpp cnxS_bw_e120 effv2s_unetpp hrnet_unet mn_r34_unetpp unetpp_bw_r90 m_dinov3_lr m_skel_w015"
EXPS="x2_unetpp_bw_e120_r90_s1 upp_hard cnx_hard full_x2_unetpp_bw_e120_r90"

ready() {
  if [ "$1" = full_x2_unetpp_bw_e120_r90 ]; then [ "$(ls runs/full/$1/fold*/result.json 2>/dev/null | wc -l)" -ge 3 ]; return; fi
  [ "$(ls runs/cv/$1/fold*/result.json 2>/dev/null | wc -l)" -ge 5 ] && [ "$(ls runs/oof/$1 2>/dev/null | wc -l)" -ge 90 ]
}
evaluate() {
  case $1 in
    x2_unetpp_bw_e120_r90_s1) echo "### vs 시드 0 (시드 잡음)"; $PY src/review.py $1 --base x2_unetpp_bw_e120_r90
                              echo "### vs 60에폭"; $PY src/review.py $1 --base x2_unetpp_bw_r90 ;;
    upp_hard) $PY src/review.py $1 --base unetpp_bw_r90 ;;
    cnx_hard) $PY src/review.py $1 --base cnx_bw ;;
    full_x2_unetpp_bw_e120_r90) CUDA_VISIBLE_DEVICES=3 $PY src/check_full.py train-score --exps $1 --out runs/review9/full_train_score.json ;;
  esac
}

while true; do
  left=0
  for e in $EXPS; do
    [ -f runs/review9/$e.done ] && continue
    if ready $e; then
      evaluate $e > runs/review9/$e.txt 2>&1
      echo "$(date '+%m-%d %H:%M') $e evaluated" | tee runs/review9/$e.done
    else
      left=$((left + 1))
    fi
  done
  [ $left -eq 0 ] && break
  if [ -f runs/round9.done ] && [ -f runs/round9_full.done ]; then
    for e in $EXPS; do [ -f runs/review9/$e.done ] || echo "$(date '+%m-%d %H:%M') $e INCOMPLETE" | tee runs/review9/$e.done; done
    break
  fi
  sleep 60
done

echo "$(date '+%m-%d %H:%M') 앙상블 기여 평가 (기존 26개 + 시드 1 + 안쪽 오답 모델 2개)"
$PY src/ensemble.py --max-size 6 --out runs/oof/ensemble_summary_r9.json \
  --exps $A x2_unetpp_bw_e120_r90_s1 upp_hard cnx_hard > runs/oof/ensemble_log_r9.txt 2>&1
echo "$(date '+%m-%d %H:%M') all" | tee runs/review9/all.done
