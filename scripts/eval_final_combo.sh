#!/bin/bash
# 최종 조합 평가 (CPU): 기준 앙상블 → Eutectic Si 전용 모델 2개 섞기 → Al3Ni 전용 모델 섞기 → Si 분류기.
# 섞는 비율은 단계마다 중첩 검증으로 정직한 효과를 재고, 다음 단계에는 90장 전체로 고른 비율을 넘긴다.
# 사용: bash scripts/eval_final_combo.sh <이름> <실험들...>   (같은 실험을 두 번 쓰면 가중치 2배)
# 결과: runs/combo/<이름>/ (3_sie.json, 4_al3ni.json, 2·5·6 Si 분류기, alphas.txt, done)
cd ~/komap
PY=.venv/bin/python
T=$1; shift
OUT=runs/combo/$T
mkdir -p $OUT
best() { $PY -c "import json,sys; print(json.load(open(sys.argv[1]))['in_sample_best (optimistic)']['cand'][1])" $1; }

$PY src/decision_rules.py cache --exps "$@" --out runs/oof/_ens_$T --procs 16 > $OUT/1_cache.log 2>&1
echo "$*" > $OUT/exps.txt
$PY src/si_classifier.py cv --thr 0.8 --exps _ens_$T > $OUT/2_si_base.txt 2>&1 &        # 섞기 전 분류기 효과 (참고)
$PY src/decision_rules.py sie --ens runs/oof/_ens_$T --spec sie_x2_unetpp sie_x2_cnx --procs 16 --out $OUT/3_sie.json > /dev/null 2>&1
A1=$(best $OUT/3_sie.json)
$PY src/decision_rules.py blendcache --ens runs/oof/_ens_$T --spec sie_x2_unetpp sie_x2_cnx --alpha $A1 \
  --out runs/oof/_ens_${T}_sie --procs 16 > /dev/null 2>&1
$PY src/decision_rules.py sie --cls 1 --ens runs/oof/_ens_${T}_sie --spec al3ni_x2_unetpp --procs 16 --out $OUT/4_al3ni.json > /dev/null 2>&1
A2=$(best $OUT/4_al3ni.json)
$PY src/decision_rules.py blendcache --cls 1 --ens runs/oof/_ens_${T}_sie --spec al3ni_x2_unetpp --alpha $A2 \
  --out runs/oof/_ens_${T}_sie_al --procs 16 > /dev/null 2>&1
$PY src/si_classifier.py cv --thr 0.8 --exps _ens_${T}_sie > $OUT/5_si_sie.txt 2>&1 &
$PY src/si_classifier.py cv --thr 0.8 --exps _ens_${T}_sie_al > $OUT/6_si_sie_al.txt 2>&1
wait
echo "alpha_sie=$A1 alpha_al3ni=$A2" > $OUT/alphas.txt
date '+%m-%d %H:%M' > $OUT/done
