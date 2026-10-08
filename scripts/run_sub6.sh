#!/bin/bash
# 기본 sub6: 고해상도 MiT + UNet++ 1:1. Si 분류기는 별도 비교 후보.
# 검증/생성 성공 후 사용자 선택한 MiT 두 실험을 시작한다.
set -euo pipefail
cd ~/komap
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
PY=.venv/bin/python
OUT=runs/sub6
mkdir -p "$OUT"
trap 's=$?; echo "$s" > "$OUT/exit_code.txt"; date -Is > "$OUT/finished.txt"' EXIT
until [ -f runs/pair13/all.done ]; do
  if [ -f runs/pair13/exit_code.txt ]; then
    echo 'Pair verification jobs did not complete successfully.'
    exit 1
  fi
  sleep 15
done
$PY scripts/verify_pair13.py > "$OUT/verify_pair.txt" 2>&1
PAIR=(x2_mit_bw_r90 x3_unetpp_bw_e120_r90)
$PY src/si_classifier.py fit --exps _ens_E2_s4a2 --out runs/si_clf/sub6_pair.pkl > "$OUT/si_fit.log" 2>&1
$PY src/submit.py combine --split test --parts submissions/sub3_pred --out submissions/sub6 \
  --recipe "${PAIR[@]}" > "$OUT/combine.log" 2>&1
$PY src/submit.py combine --split test --parts submissions/sub3_pred --out submissions/sub6_si \
  --recipe "${PAIR[@]}" --si-clf runs/si_clf/sub6_pair.pkl --si-thr 0.8 > "$OUT/combine_si.log" 2>&1
$PY - <<'PY'
import json, pathlib
for name in ('sub6', 'sub6_si'):
    d=json.loads(pathlib.Path('submissions', name, 'summary_test.json').read_text())
    assert d['n_images'] == 10 and d['n_models'] == 10 and not d['problems'], d
    assert d['models_per_image'] == [10], d
    print(name, '10 RGB masks, 10 fold models per image, format checks passed')
PY
$PY src/check_full.py diff submissions/sub5 submissions/sub6 > "$OUT/diff_sub5.txt" 2>&1
$PY src/check_full.py diff submissions/sub6 submissions/sub6_si > "$OUT/diff_si.txt" 2>&1
date -Is > "$OUT/all.done"
echo "$(date -Is) sub6 candidates complete"
if [ ! -e runs/review14/pid.txt ]; then
  mkdir -p runs/review14
  $PY - <<'PY'
import pathlib, subprocess
out=pathlib.Path('runs/review14')
with (out/'driver.log').open('w') as log:
    proc=subprocess.Popen(['bash','scripts/run_round14.sh'],stdin=subprocess.DEVNULL,
                          stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(out/'pid.txt').write_text(str(proc.pid))
print('started approved round14 driver PID',proc.pid,flush=True)
PY
fi
