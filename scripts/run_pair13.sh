#!/bin/bash
# 13단계 최고 두 모델: 전용 모델/Si 분류기 비교 + 제출 코드의 새 CV 예측.
set -euo pipefail
cd ~/komap
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
PY=.venv/bin/python
OUT=runs/pair13
mkdir -p "$OUT"
trap 's=$?; echo "$s" > "$OUT/exit_code.txt"; date -Is > "$OUT/finished.txt"' EXIT
PAIR="x2_mit_bw_r90 x3_unetpp_bw_e120_r90"
S4="sie_x2_unetpp sie_x2_cnx sie_x2_unetpp_e120 sie_x2_cnx_e120"
AL2="al3ni_x2_unetpp al3ni_x2_cnx_e120"
date -Is
PIDS=()
for T in E2_s4a2 E2_s2a1; do
  (
    if [ "$T" = E2_s4a2 ]; then
      export SIE_SPEC="$S4" AL3NI_SPEC="$AL2"
    else
      export SIE_SPEC="sie_x2_unetpp sie_x2_cnx" AL3NI_SPEC="al3ni_x2_unetpp"
    fi
    bash -e scripts/eval_final_combo.sh "$T" $PAIR
    $PY - "$T" <<'PY'
import json, pathlib, sys
root = pathlib.Path('runs/combo', sys.argv[1])
for name in ('3_sie.json', '4_al3ni.json'):
    d = json.loads((root / name).read_text())
    assert 'nested (honest)' in d
for name in ('2_si_base.txt', '5_si_sie.txt', '6_si_sie_al.txt'):
    assert '+ Si classifier mIoU' in (root / name).read_text(), name
print(sys.argv[1], 'evaluation artifacts OK', flush=True)
PY
  ) > "$OUT/$T.log" 2>&1 &
  PIDS+=("$!")
done
for i in 0 1 2 3; do
  (
    export CUDA_VISIBLE_DEVICES=$i
    $PY src/submit.py predict --exps $PAIR --split cv --out submissions/sub6_cv_pair --shard "$i" --nshards 4
    $PY src/submit.py predict --exps $S4 --split cv --out submissions/sub6_cv_sie4 --shard "$i" --nshards 4
    $PY src/submit.py predict --exps $AL2 --split cv --out submissions/sub6_cv_al2 --shard "$i" --nshards 4
  ) > "$OUT/cv_gpu$i.log" 2>&1 &
  PIDS+=("$!")
done
FAILED=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then FAILED=1; fi
done
if [ "$FAILED" -ne 0 ]; then
  echo 'At least one evaluation or prediction job failed; inspect logs.'
  exit 1
fi
date -Is > "$OUT/all.done"
echo 'pair evaluation and independent CV predictions complete'
