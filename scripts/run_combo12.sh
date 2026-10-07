#!/bin/bash
# 12단계 결과로 최종 조합 후보를 동시에 평가 (CPU, 조합당 약 3분, 16코어씩). sub5(A27)와 같은 단계별 중첩 검증.
#   setsid nohup bash scripts/run_combo12.sh > runs/combo/driver12.log 2>&1 < /dev/null &
# 끝나면 runs/combo/combo12.done, 요약은 runs/combo/summary12.txt
cd ~/komap
PY=.venv/bin/python
log() { echo "$(date '+%m-%d %H:%M:%S') $*"; }
A27=$(cat runs/combo/A27/exps.txt)
S4="sie_x2_unetpp sie_x2_cnx sie_x2_unetpp_e120 sie_x2_cnx_e120"

log "최종 조합 후보 5개 동시 평가 시작"
bash scripts/eval_final_combo.sh A27chk $A27 &                                   # 스크립트 수정 확인: A27(sub5) 결과와 같아야 함
SIE_SPEC="$S4" bash scripts/eval_final_combo.sh A27_sie4 $A27 &                 # Eutectic Si 전용 모델만 4개로
SIE_SPEC="$S4" bash scripts/eval_final_combo.sh A28s2_sie4 $A27 x2_unetpp_bw_e120_r90_s2 &
SIE_SPEC="$S4" bash scripts/eval_final_combo.sh A28x3_sie4 $A27 x3_unetpp_bw_r90 &
SIE_SPEC="$S4" bash scripts/eval_final_combo.sh A29_sie4 $A27 x2_unetpp_bw_e120_r90_s2 x3_unetpp_bw_r90 &
wait

for T in A27 A27chk A27_sie4 A28s2_sie4 A28x3_sie4 A29_sie4; do
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
date '+%m-%d %H:%M:%S' > runs/combo/combo12.done
