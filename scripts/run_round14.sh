#!/bin/bash
# MiT 두 실험 학습 -> 각 실험 완료 시 기존 MiT와 비교 -> 두 모델 조합 기여 평가.
set -euo pipefail
cd ~/komap
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
PY=.venv/bin/python
OUT=runs/review14
mkdir -p "$OUT"
trap 's=$?; echo "$s" > "$OUT/exit_code.txt"; date -Is > "$OUT/finished.txt"' EXIT
NEW=(x2_mit_bw_e120_r90 x2_mit_bw_r90_sj0)
(
  $PY src/launch.py experiments/round14.txt --per-gpu 1 > runs/launch_round14.log 2>&1
  date -Is > "$OUT/train.finished"
) &
TRAIN_PID=$!
ready() {
  [ "$(find "runs/cv/$1" -name result.json 2>/dev/null | wc -l)" -eq 5 ] &&
  [ "$(find "runs/oof/$1" -maxdepth 1 -name '*.npy' 2>/dev/null | wc -l)" -eq 90 ]
}
while true; do
  LEFT=0
  for e in "${NEW[@]}"; do
    [ -f "$OUT/$e.done" ] && continue
    if ready "$e"; then
      $PY src/review.py "$e" --base x2_mit_bw_r90 > "$OUT/$e.txt" 2>&1
      date -Is > "$OUT/$e.done"
      echo "$(date -Is) $e evaluated"
    else
      LEFT=$((LEFT + 1))
    fi
  done
  [ "$LEFT" -eq 0 ] && break
  if [ -f "$OUT/train.finished" ]; then
    echo 'Training launcher finished but some fold/OOF outputs are missing.'
    exit 1
  fi
  if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
    echo 'Training launcher stopped unexpectedly.'
    exit 1
  fi
  sleep 60
done
wait "$TRAIN_PID"
$PY src/ensemble.py --exps x2_mit_bw_r90 x3_unetpp_bw_e120_r90 "${NEW[@]}" --max-size 4 \
  --out runs/oof/ensemble_summary_r14.json > runs/oof/ensemble_log_r14.txt 2>&1
date -Is > "$OUT/all.done"
echo "$(date -Is) round14 complete"
