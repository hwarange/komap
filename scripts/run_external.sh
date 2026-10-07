#!/bin/bash
# 외부 데이터 사전학습 시험 (빠른 모델 ConvNeXt-T U-Net, 비교 기준 cnx_bw 79.19).
#   setsid nohup bash scripts/run_external.sh > runs/external/driver.log 2>&1 < /dev/null &
# MicroSteel(SEM, 79장) → 바로 사전학습 / MICAST(합성 LOM, 1000장) → 다운로드·변환 뒤 사전학습.
# 각각 끝나면 우리 90장으로 5-fold 미세조정 → review.py로 cnx_bw와 비교. 결과: runs/external/<실험>.txt, all.done
cd ~/komap
PY=.venv/bin/python
log() { echo "$(date '+%m-%d %H:%M:%S') $*"; }
mkdir -p runs/external runs/pretrain

pretrain() {   # $1 이름, $2 데이터 폴더, $3 에폭, $4 GPU
  log "$1 사전학습 시작 (GPU $4)"
  CUDA_VISIBLE_DEVICES=$4 $PY src/train.py --data $2 --encoder tu-convnext_tiny --bw 2 --amp --workers 6 \
    --train-splits train --val-split "" --use-ignore --epochs $3 --out runs/pretrain/$1 > runs/external/pretrain_$1.log 2>&1
  log "$1 사전학습 끝: $(ls runs/pretrain/$1/last.pt 2>&1)"
}
finetune() {   # $1 실험 이름, $2 추가 옵션, $3 GPU 목록
  echo "$1 | --encoder tu-convnext_tiny --bw 2 $2 --save-oof runs/oof/$1" > experiments/round11_$1.txt
  $PY src/launch.py experiments/round11_$1.txt --gpus $3 --per-gpu 3 > runs/launch_round11_$1.log 2>&1
  $PY src/review.py $1 --base cnx_bw > runs/external/$1.txt 2>&1
  log "$1 미세조정·비교 끝" | tee runs/external/$1.done
}

(pretrain microsteel_cnx external/microsteel_seg 40 2
 finetune cnx_microsteel "--init runs/pretrain/microsteel_cnx/last.pt --init-skip-head" "0 1") &
A=$!

(until [ -f external/micast/fetch_train.done ]; do sleep 30; done
 log "MICAST 다운로드 끝: $(ls external/micast/lom/Images/TRAIN | wc -l)장, 변환 시작"
 $PY src/make_external.py micast --src external/micast/lom --n 1000 --out external/micast_seg > runs/external/convert_micast.log 2>&1
 log "$(tail -n 1 runs/external/convert_micast.log)"
 pretrain micast_cnx external/micast_seg 10 3
 finetune cnx_micast "--init runs/pretrain/micast_cnx/last.pt" "2 3") &
B=$!

wait $A $B
log "모두 끝" | tee runs/external/all.done
