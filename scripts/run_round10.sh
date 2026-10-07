#!/bin/bash
# 10단계: 9단계의 가벼운 작업(안쪽 오답 학습)이 끝나 GPU 칸이 비면 시작하고, 실험마다 끝나면 바로 평가한다.
#   setsid nohup bash scripts/run_round10.sh > runs/review10/driver.log 2>&1 < /dev/null &
# 결과: runs/review10/<실험>.done, .txt / 모두 끝나면 runs/review10/all.done
cd ~/komap
PY=.venv/bin/python
mkdir -p runs/review10
log() { echo "$(date '+%m-%d %H:%M') $*"; }

# 2배 해상도 작업(시드 1, 90장 학습)이 GPU마다 최대 2개라, GPU당 1칸씩만 더 쓴다 (메모리 여유 확인: 작업당 최대 12GB)
until [ "$(ls runs/cv/cnx_hard/fold*/result.json 2>/dev/null | wc -l)" -ge 5 ] || [ -f runs/round9.done ]; do sleep 60; done
log "10단계 학습 시작"
(.venv/bin/python src/launch.py experiments/round10.txt --gpus 0 1 2 3 --per-gpu 1 > runs/launch_round10.log 2>&1; echo $? > runs/round10.done) &

EXPS="pair_al3ni_sie_x2 upp_feats cnx_feats"
ready() { [ "$(ls runs/cv/$1/fold*/result.json 2>/dev/null | wc -l)" -ge 5 ] && [ "$(ls runs/oof/$1 2>/dev/null | wc -l)" -ge 90 ]; }
evaluate() {
  case $1 in
    # 지금 최선 조합 위에서 Al3Ni·Eutectic Si 두 상의 비율만 다시 정하기 / 26개 평균 위에서도 (단독 효과)
    pair_al3ni_sie_x2) $PY src/decision_rules.py sie --pair 1 3 --ens runs/oof/_ens_A26_sie_al --spec $1 --procs 16 \
                         --out runs/rules/pair_on_A26_sie_al.json
                       $PY src/decision_rules.py sie --pair 1 3 --ens runs/oof/_ens_A26 --spec $1 --procs 16 \
                         --out runs/rules/pair_on_A26.json ;;
    upp_feats) $PY src/review.py $1 --base unetpp_bw_r90 ;;
    cnx_feats) $PY src/review.py $1 --base cnx_bw ;;
  esac
}
while true; do
  left=0
  for e in $EXPS; do
    [ -f runs/review10/$e.done ] && continue
    if ready $e; then
      evaluate $e > runs/review10/$e.txt 2>&1
      log "$e evaluated" | tee runs/review10/$e.done
    else
      left=$((left + 1))
    fi
  done
  [ $left -eq 0 ] && break
  if [ -f runs/round10.done ]; then
    for e in $EXPS; do [ -f runs/review10/$e.done ] || log "$e INCOMPLETE" | tee runs/review10/$e.done; done
    break
  fi
  sleep 60
done
log "all" | tee runs/review10/all.done
