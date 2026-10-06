#!/bin/bash
# 4차 제출본: 25개 앙상블(A) + Eutectic Si 전용 모델 30% 섞기 + Si 분류기(섞은 확률로 다시 학습)
#   setsid nohup bash scripts/run_sub4.sh > runs/sub4_driver.log 2>&1 < /dev/null &
# 끝나면 runs/sub4_done.txt 가 생긴다. 평가 사진 예측은 sub3에서 저장한 모델별 예측(submissions/sub3_pred)을 다시 쓴다.
cd ~/komap
PY=.venv/bin/python
log() { echo "$(date '+%m-%d %H:%M:%S') $*"; }
A="unetpp_bw_e120_r90 x2_c1024_bw_r90 a_rot90 cnx_bw_e120 unetpp_bw_e120 m_hires bw4_r90 a_lean unetpp_bw cnx_bw \
cnx_bw_c768 cnx_bw_e120_r90 unet_mit_bw_r90 x2_c1024 unet_mit upp_r50_bw \
x2_unetpp_bw_r90 mn_sxt50_unetpp cnxS_bw_e120 effv2s_unetpp hrnet_unet mn_r34_unetpp unetpp_bw_r90 m_dinov3_lr m_skel_w015"
PRED=submissions/sub3_pred
mkdir -p runs/sub4

log "Si 분류기 학습 시작 (CPU, 섞은 OOF 확률 runs/oof/_ens_A_sie03 로)"
($PY src/si_classifier.py fit --exps _ens_A_sie03 --out runs/si_clf/sub4_model.pkl > runs/sub4/si_fit.log 2>&1) &
SF=$!

log "전용 모델 평가 사진 예측 (GPU 4장, fold 5개)"
PP=()
for i in 0 1 2 3; do
  (CUDA_VISIBLE_DEVICES=$i $PY src/submit.py predict --exps sie_x2_unetpp --split test --out $PRED --shard $i --nshards 4) \
    > runs/sub4/predict_gpu$i.log 2>&1 &
  PP+=($!)
done
wait "${PP[@]}"
log "예측 끝: 전용 모델 $(ls -d $PRED/parts/models/sie_x2_unetpp/fold* | wc -l)개 (기대 5)"
grep -h -E "Traceback|Error" runs/sub4/predict_gpu*.log | head -3

wait $SF
log "Si 분류기: $(tail -n 1 runs/sub4/si_fit.log)"

C="$PY src/submit.py combine --split test --parts $PRED --si-thr 0.8 --si-clf runs/si_clf/sub4_model.pkl --recipe $A --sie-spec sie_x2_unetpp --sie-alpha 0.3"
$C --out submissions/sub4                    > runs/sub4/combine_sub4.log 2>&1
$C --out submissions/sub4_full --full-mix 0.5 > runs/sub4/combine_sub4_full.log 2>&1
for s in sub4 sub4_full; do log "$s: problems=$(grep -o '"problems": \[[^]]*\]' submissions/$s/summary_test.json)"; done

{
  for b in sub2 sub3a sub3a_full; do echo "##### $b vs sub4"; $PY src/check_full.py diff submissions/$b submissions/sub4; done
  echo "##### sub4 vs sub4_full (90장 모델을 섞은 효과)"; $PY src/check_full.py diff submissions/sub4 submissions/sub4_full
  echo "##### Si 분류기가 바꾼 픽셀 비율"
  for s in sub3a sub4 sub4_full; do
    $PY -c "import json; d=json.load(open('submissions/$s/summary_test.json')); v=[x['pixels_changed_pct'] for x in d['si_classifier']['per_image'].values()]; print('$s', 'n_models', d['n_models'], 'mean %.4f max %.4f' % (sum(v)/len(v), max(v)), d.get('sie_spec'), d.get('sie_alpha'))"
  done
} > runs/sub4/checks.txt 2>&1
log "모두 끝 (runs/sub4/checks.txt)"
date '+%m-%d %H:%M:%S' > runs/sub4_done.txt
