#!/bin/bash
# 3차 제출본 후보 4개 만들기 (서버 ~/komap 에서 nohup으로 실행)
#   setsid nohup bash scripts/run_sub3.sh > runs/sub3_driver.log 2>&1 < /dev/null &
# 1단계 (GPU): 평가 사진 예측 (fold 모델 125개 + 90장 모델 15개, 모델별 저장) + 90장 모델의 학습 사진 점수
# 2단계 (GPU): 제출 코드 검증 (A 구성 fold 모델로 90장, 그 사진을 학습하지 않은 모델만) → 79.937과 같아야 함
# CPU (1단계와 동시): Si 분류기 학습 (A용, B용). 1단계가 끝나면 조합 4개 + sub2 재현 시험 + 안전 점검
# 끝나면 runs/sub3_done.txt 가 생긴다.
cd ~/komap
PY=.venv/bin/python
log() { echo "$(date '+%m-%d %H:%M:%S') $*"; }

A="unetpp_bw_e120_r90 x2_c1024_bw_r90 a_rot90 cnx_bw_e120 unetpp_bw_e120 m_hires bw4_r90 a_lean unetpp_bw cnx_bw \
cnx_bw_c768 cnx_bw_e120_r90 unet_mit_bw_r90 x2_c1024 unet_mit upp_r50_bw \
x2_unetpp_bw_r90 mn_sxt50_unetpp cnxS_bw_e120 effv2s_unetpp hrnet_unet mn_r34_unetpp unetpp_bw_r90 m_dinov3_lr m_skel_w015"
B_EXPS="x2_unetpp_bw_r90 unet_mit effv2s_unetpp x2_unetpp_bw_r90 cnxS_bw_e120 m_hires"   # 분류기 학습용 (B1 두 번 = 가중치 2)
B_RECIPE="x2_unetpp_bw_r90:2 unet_mit effv2s_unetpp cnxS_bw_e120 m_hires"
SUB2="unetpp_bw_e120_r90 x2_c1024_bw_r90 a_rot90 cnx_bw_e120 unetpp_bw_e120 m_hires bw4_r90 a_lean unetpp_bw cnx_bw \
cnx_bw_c768 cnx_bw_e120_r90 unet_mit_bw_r90 x2_c1024 unet_mit upp_r50_bw"
FULL="full_unetpp_bw_e120_r90 full_x2_c1024_bw_r90 full_cnx_bw_e120 full_unet_mit_bw_r90 full_x2_unetpp_bw_r90"
TS=(full_unetpp_bw_e120_r90 "full_x2_c1024_bw_r90" "full_x2_unetpp_bw_r90" "full_cnx_bw_e120 full_unet_mit_bw_r90")   # GPU별 학습 사진 점수 담당
PRED=submissions/sub3_pred
mkdir -p runs/sub3

log "Si 분류기 학습 시작 (CPU, A용·B용)"
($PY src/si_classifier.py fit --exps $A --out runs/si_clf/sub3a_model.pkl > runs/sub3/si_fit_a.log 2>&1) &
SA=$!
($PY src/si_classifier.py fit --exps $B_EXPS --out runs/si_clf/sub3b_model.pkl > runs/sub3/si_fit_b.log 2>&1) &
SB=$!

log "1단계 (GPU 4장): 평가 사진 예측 + 90장 모델 학습 사진 점수"
P1=()
for i in 0 1 2 3; do
  (export CUDA_VISIBLE_DEVICES=$i
   $PY src/submit.py predict --exps $A --split test --out $PRED --shard $i --nshards 4
   $PY src/submit.py predict --exps $FULL --root runs/full --folds 3 --split test --out $PRED --shard $i --nshards 4
   $PY src/check_full.py train-score --exps ${TS[$i]} --out runs/sub3/check_train_gpu$i.json
  ) > runs/sub3/gpu${i}_phase1.log 2>&1 &
  P1+=($!)
done
wait "${P1[@]}"
log "1단계 끝: 저장된 모델별 예측 $(ls -d $PRED/parts/models/*/fold* | wc -l)개 (기대 140)"
grep -h -E "Traceback|Error" runs/sub3/gpu*_phase1.log | head -5

log "2단계 (GPU 4장, 백그라운드): 제출 코드 검증 (90장, A 구성 fold 모델)"
P2=()
for i in 0 1 2 3; do
  (CUDA_VISIBLE_DEVICES=$i $PY src/submit.py predict --exps $A --split cv --out submissions/sub3_cv --shard $i --nshards 4) \
    > runs/sub3/gpu${i}_phase2.log 2>&1 &
  P2+=($!)
done

wait $SA $SB
log "Si 분류기: $(tail -n 1 runs/sub3/si_fit_a.log) / $(tail -n 1 runs/sub3/si_fit_b.log)"

log "조합 4개 + sub2 재현 시험"
C="$PY src/submit.py combine --split test --parts $PRED --si-thr 0.8"
$C --out submissions/sub3a      --recipe $A        --si-clf runs/si_clf/sub3a_model.pkl                > runs/sub3/combine_sub3a.log 2>&1
$C --out submissions/sub3a_full --recipe $A        --si-clf runs/si_clf/sub3a_model.pkl --full-mix 0.5 > runs/sub3/combine_sub3a_full.log 2>&1
$C --out submissions/sub3b      --recipe $B_RECIPE --si-clf runs/si_clf/sub3b_model.pkl                > runs/sub3/combine_sub3b.log 2>&1
$C --out submissions/sub3b_full --recipe $B_RECIPE --si-clf runs/si_clf/sub3b_model.pkl --full-mix 0.5 > runs/sub3/combine_sub3b_full.log 2>&1
$C --out runs/sub3/sub2_repro   --recipe $SUB2     --si-clf runs/si_clf/sub2_model.pkl                 > runs/sub3/combine_sub2_repro.log 2>&1
for s in sub3a sub3a_full sub3b sub3b_full; do
  log "$s: problems=$(grep -o '"problems": \[[^]]*\]' submissions/$s/summary_test.json)"
done

log "안전 점검"
{
  echo "##### sub2 재현 시험 (새 코드 vs 기존 sub2, 0에 가까워야 함)"; $PY src/check_full.py diff submissions/sub2 runs/sub3/sub2_repro
  for s in sub3a sub3a_full sub3b sub3b_full; do echo "##### sub2 vs $s"; $PY src/check_full.py diff submissions/sub2 submissions/$s; done
  echo "##### sub3a vs sub3a_full (90장 모델을 섞은 효과)"; $PY src/check_full.py diff submissions/sub3a submissions/sub3a_full
  echo "##### 모델끼리 불일치 (90장 모델 시드 3개 vs 같은 구성 fold 모델 5개)"
  $PY src/check_full.py agree --parts $PRED --exps $FULL unetpp_bw_e120_r90 x2_c1024_bw_r90 cnx_bw_e120 unet_mit_bw_r90 x2_unetpp_bw_r90
  echo "##### 학습 사진 점수 (OOF보다 높아야 정상)"; cat runs/sub3/check_train_gpu*.json
} > runs/sub3/checks.txt 2>&1
log "안전 점검 끝 (runs/sub3/checks.txt)"

wait "${P2[@]}"
$PY src/submit.py combine --split cv --out submissions/sub3_cv > runs/sub3/combine_cv.log 2>&1
log "2단계 끝: 제출 코드 검증 $(grep -E '"mIoU"' submissions/sub3_cv/summary_cv.json) (ensemble.py 79.937과 같아야 함)"
log "모두 끝"
date '+%m-%d %H:%M:%S' > runs/sub3_done.txt
