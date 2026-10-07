#!/bin/bash
# 12단계 밤샘: GPU 0·1번은 지금 바로 (시드 2, Eutectic Si 전용 120에폭 2종), GPU 2·3번은 외부 데이터 시험이 끝난 뒤 3배 해상도.
# 실험마다 끝나면 바로 평가하고 runs/review12/<실험>.done 을 남긴다. 모두 끝나면 앙상블 기여 + all.done
#   setsid nohup bash scripts/run_round12.sh > runs/review12/driver.log 2>&1 < /dev/null &
cd ~/komap
PY=.venv/bin/python
log() { echo "$(date '+%m-%d %H:%M:%S') $*"; }
mkdir -p runs/review12
A27=$(cat runs/combo/A27/exps.txt)

(.venv/bin/python src/launch.py experiments/round12.txt --gpus 0 1 --per-gpu 2 > runs/launch_round12.log 2>&1; echo $? > runs/round12.done) &
(until [ -f runs/external/all.done ]; do sleep 60; done
 log "외부 데이터 시험 끝 → 3배 해상도 시작 (GPU 2·3)"
 .venv/bin/python src/launch.py experiments/round12_x3.txt --gpus 2 3 --per-gpu 1 > runs/launch_round12_x3.log 2>&1; echo $? > runs/round12_x3.done) &

ready() { [ "$(ls runs/cv/$1/fold*/result.json 2>/dev/null | wc -l)" -ge 5 ] && [ "$(ls runs/oof/$1 2>/dev/null | wc -l)" -ge 90 ]; }
EXPS="x2_unetpp_bw_e120_r90_s2 sie_e120_pair x3_unetpp_bw_r90"
while true; do
  left=0
  for e in $EXPS; do
    [ -f runs/review12/$e.done ] && continue
    case $e in
      x2_unetpp_bw_e120_r90_s2)
        if ready $e; then
          { echo "### vs 시드 0"; $PY src/review.py $e --base x2_unetpp_bw_e120_r90; echo "### vs 시드 1"; $PY src/review.py $e --base x2_unetpp_bw_e120_r90_s1; } > runs/review12/$e.txt 2>&1
          log "$e evaluated" | tee runs/review12/$e.done
        else left=$((left + 1)); fi ;;
      sie_e120_pair)   # 120에폭 전용 모델 두 개를 섞기 (A27 위). 60에폭 두 개는 +0.091 (runs/combo/A27/3_sie.json)
        if ready sie_x2_unetpp_e120 && ready sie_x2_cnx_e120; then
          $PY src/decision_rules.py sie --ens runs/oof/_ens_A27 --spec sie_x2_unetpp_e120 sie_x2_cnx_e120 --procs 24 \
            --out runs/rules/sie_e120_on_A27.json > runs/review12/$e.txt 2>&1
          $PY src/decision_rules.py sie --ens runs/oof/_ens_A27 --spec sie_x2_unetpp sie_x2_cnx sie_x2_unetpp_e120 sie_x2_cnx_e120 --procs 24 \
            --out runs/rules/sie_e60e120_on_A27.json >> runs/review12/$e.txt 2>&1
          log "$e evaluated" | tee runs/review12/$e.done
        else left=$((left + 1)); fi ;;
      x3_unetpp_bw_r90)
        if ready $e; then
          { echo "### vs 2배 해상도 60에폭"; $PY src/review.py $e --base x2_unetpp_bw_r90; echo "### vs 원래 해상도"; $PY src/review.py $e --base unetpp_bw_r90; } > runs/review12/$e.txt 2>&1
          log "$e evaluated" | tee runs/review12/$e.done
        else left=$((left + 1)); fi ;;
    esac
  done
  [ $left -eq 0 ] && break
  if [ -f runs/round12.done ] && [ -f runs/round12_x3.done ]; then
    for e in $EXPS; do [ -f runs/review12/$e.done ] || log "$e INCOMPLETE" | tee runs/review12/$e.done; done
    break
  fi
  sleep 120
done
log "앙상블 기여 평가 (A27 + 시드 2 + 3배 해상도)"
$PY src/ensemble.py --max-size 6 --out runs/oof/ensemble_summary_r12.json --exps $A27 x2_unetpp_bw_e120_r90_s2 x3_unetpp_bw_r90 \
  > runs/oof/ensemble_log_r12.txt 2>&1
log "all" | tee runs/review12/all.done
