#!/bin/bash
# run_combo12.sh 의 Si 분류기 단계 다시 하기.
# 조합 5개 × Si 분류기 3개 = 15개가 동시에 떠서 각자 코어 128개를 쓰려다 부하 700+로 밀렸다 → 중단하고,
# 스레드를 20개로 제한해 단계별로 5개씩 돌린다 (최종 수치인 6번부터).
#   setsid nohup bash scripts/run_combo12_si.sh > runs/combo/driver12_si.log 2>&1 < /dev/null &
# 끝나면 runs/combo/combo12_si.done, 요약은 runs/combo/summary12.txt (덮어씀)
cd ~/komap
PY=.venv/bin/python
log() { echo "$(date '+%m-%d %H:%M:%S') $*"; }
TS="A27chk A27_sie4 A28s2_sie4 A28x3_sie4 A29_sie4"
until [ -f runs/combo/combo12.done ]; do sleep 5; done
export OMP_NUM_THREADS=20
for step in 6_si_sie_al:_sie_al 5_si_sie:_sie 2_si_base:; do
  f=${step%%:*}; suf=${step#*:}
  log "$f 시작"
  for T in $TS; do $PY src/si_classifier.py cv --thr 0.8 --exps _ens_${T}${suf} > runs/combo/$T/$f.txt 2>&1 & done
  wait
done

for T in A27 $TS; do
  echo "##### $T ($(cat runs/combo/$T/alphas.txt))"
  $PY - "$T" <<'EOF'
import json, sys
T = sys.argv[1]
for f in ('3_sie', '4_al3ni'):
    d = json.load(open(f'runs/combo/{T}/{f}.json'))
    h = d['nested (honest)']
    print(f"  {f:8s} base {d['base']['mIoU']:.3f} -> nested {h['mIoU']:.3f} (delta {h['delta']:+.3f} {h['ci95']})")
EOF
  grep -h "Si classifier" runs/combo/$T/6_si_sie_al.txt
done > runs/combo/summary12.txt 2>&1
log "끝"
date '+%m-%d %H:%M:%S' > runs/combo/combo12_si.done
