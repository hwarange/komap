#!/bin/bash
# 5차 제출본: 27개 앙상블(A27) + Eutectic Si 전용 모델 2개 섞기(0.4) + Al3Ni 전용 모델 섞기(0.3) + Si 분류기
#   setsid nohup bash scripts/run_sub5.sh > runs/sub5/driver.log 2>&1 < /dev/null &
# 정직한 추정 약 80.23 (runs/combo/A27, 단계별 중첩 검증). sub5_full은 90장 학습(full_x2_unetpp_bw_e120_r90)이 끝나면 이어서 만든다.
# 끝나면 runs/sub5_done.txt
cd ~/komap
PY=.venv/bin/python
log() { echo "$(date '+%m-%d %H:%M:%S') $*"; }
A27=$(cat runs/combo/A27/exps.txt)
PRED=submissions/sub3_pred
NEW="x2_unetpp_bw_e120_r90 x2_unetpp_bw_e120_r90_s1 sie_x2_cnx al3ni_x2_unetpp"
mkdir -p runs/sub5

log "Si 분류기 학습 (CPU, 섞은 OOF 확률 _ens_A27_sie_al)"
($PY src/si_classifier.py fit --exps _ens_A27_sie_al --out runs/si_clf/sub5_model.pkl > runs/sub5/si_fit.log 2>&1) &
SF=$!

log "평가 사진 예측: 새 모델 4개 실험 × 5-fold (GPU 0·1, 2·3번은 90장 학습 중)"
PP=()
for i in 0 1; do
  (CUDA_VISIBLE_DEVICES=$i $PY src/submit.py predict --exps $NEW --split test --out $PRED --shard $i --nshards 2) > runs/sub5/pred_test_gpu$i.log 2>&1 &
  PP+=($!)
done
wait "${PP[@]}"
log "평가 사진 예측 끝: $(for e in $NEW; do echo -n "$e=$(ls -d $PRED/parts/models/$e/fold* 2>/dev/null | wc -l) "; done)"

log "제출 코드 검증용 90장 예측 (각 사진은 학습하지 않은 fold 모델만)"
PP=()
for i in 0 1; do
  (CUDA_VISIBLE_DEVICES=$i $PY src/submit.py predict --exps x2_unetpp_bw_e120_r90 x2_unetpp_bw_e120_r90_s1 --split cv --out submissions/sub5_cv_ens --shard $i --nshards 2
   CUDA_VISIBLE_DEVICES=$i $PY src/submit.py predict --exps sie_x2_cnx --split cv --out submissions/sub5_cv_sie2 --shard $i --nshards 2
   CUDA_VISIBLE_DEVICES=$i $PY src/submit.py predict --exps al3ni_x2_unetpp --split cv --out submissions/sub5_cv_al3ni --shard $i --nshards 2
  ) > runs/sub5/pred_cv_gpu$i.log 2>&1 &
  PP+=($!)
done
wait "${PP[@]}"
wait $SF
log "Si 분류기: $(tail -n 1 runs/sub5/si_fit.log)"

C="$PY src/submit.py combine --split test --parts $PRED --si-thr 0.8 --si-clf runs/si_clf/sub5_model.pkl --recipe $A27 \
   --sie-spec sie_x2_unetpp sie_x2_cnx --sie-alpha 0.4 --al3ni-spec al3ni_x2_unetpp --al3ni-alpha 0.3"
$C --out submissions/sub5 > runs/sub5/combine_sub5.log 2>&1
log "sub5: problems=$(grep -o '"problems": \[[^]]*\]' submissions/sub5/summary_test.json)"

log "제출 코드 검증 (기대: 앙상블만 80.016, Eutectic Si 섞기 후 80.128, Al3Ni 섞기 후 약 80.175)"
$PY src/verify_cv_blend.py --ens submissions/sub3_cv submissions/sub5_cv_ens --sie submissions/sub4_cv_sie submissions/sub5_cv_sie2 \
  --alpha 0.4 --al3ni submissions/sub5_cv_al3ni --alpha-al3ni 0.3 --si-clf runs/si_clf/sub5_model.pkl > runs/sub5/verify_cv.txt 2>&1
{
  for b in sub2 sub4 sub4_full; do echo "##### $b vs sub5"; $PY src/check_full.py diff submissions/$b submissions/sub5; done
  $PY -c "import json; d=json.load(open('submissions/sub5/summary_test.json')); v=[x['pixels_changed_pct'] for x in d['si_classifier']['per_image'].values()]; print('sub5 n_models', d['n_models'], 'Si clf changed mean %.4f max %.4f' % (sum(v)/len(v), max(v)))"
} > runs/sub5/checks.txt 2>&1
log "sub5 끝 (runs/sub5/verify_cv.txt, checks.txt)"
date '+%m-%d %H:%M:%S' > runs/sub5/sub5.done

log "sub5_full: 90장 학습(최고 모델, 시드 3개)이 끝나기를 기다림"
until [ "$(ls runs/full/full_x2_unetpp_bw_e120_r90/fold*/result.json 2>/dev/null | wc -l)" -ge 3 ]; do sleep 60; done
PP=()
for i in 0 1; do
  (CUDA_VISIBLE_DEVICES=$i $PY src/submit.py predict --exps full_x2_unetpp_bw_e120_r90 --root runs/full --folds 3 --split test --out $PRED --shard $i --nshards 2) \
    > runs/sub5/pred_full_gpu$i.log 2>&1 &
  PP+=($!)
done
wait "${PP[@]}"
$C --full-mix 0.5 --out submissions/sub5_full > runs/sub5/combine_sub5_full.log 2>&1
{
  echo "##### sub5 vs sub5_full"; $PY src/check_full.py diff submissions/sub5 submissions/sub5_full
  echo "##### sub2 vs sub5_full"; $PY src/check_full.py diff submissions/sub2 submissions/sub5_full
  echo "##### 시드끼리 불일치 (90장 모델 vs fold 모델)"; $PY src/check_full.py agree --parts $PRED --exps full_x2_unetpp_bw_e120_r90 x2_unetpp_bw_e120_r90
} >> runs/sub5/checks.txt 2>&1
log "sub5_full: problems=$(grep -o '"problems": \[[^]]*\]' submissions/sub5_full/summary_test.json)"
log "모두 끝"
date '+%m-%d %H:%M:%S' > runs/sub5_done.txt
