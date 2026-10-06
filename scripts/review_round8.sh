#!/bin/bash
# 8단계 자동 평가: 실험마다 fold 5개가 끝나면 바로 평가하고 runs/review8/<실험>.done 을 남긴다.
#   setsid nohup bash scripts/review_round8.sh > runs/review8/driver.log 2>&1 < /dev/null &
# 모두 끝나면 새 4상 모델의 앙상블 기여까지 재고 runs/review8/all.done 을 남긴다.
cd ~/komap
PY=.venv/bin/python
mkdir -p runs/review8
A="unetpp_bw_e120_r90 x2_c1024_bw_r90 a_rot90 cnx_bw_e120 unetpp_bw_e120 m_hires bw4_r90 a_lean unetpp_bw cnx_bw \
cnx_bw_c768 cnx_bw_e120_r90 unet_mit_bw_r90 x2_c1024 unet_mit upp_r50_bw \
x2_unetpp_bw_r90 mn_sxt50_unetpp cnxS_bw_e120 effv2s_unetpp hrnet_unet mn_r34_unetpp unetpp_bw_r90 m_dinov3_lr m_skel_w015"
EXPS="al3ni_x2_unetpp sie_x2_cnx x2_unetpp_bw_e120_r90 upp_softedge cnx_softedge"

ready() { [ "$(ls runs/cv/$1/fold*/result.json 2>/dev/null | wc -l)" -ge 5 ] && [ "$(ls runs/oof/$1 2>/dev/null | wc -l)" -ge 90 ]; }
evaluate() {
  case $1 in
    x2_unetpp_bw_e120_r90) $PY src/review.py $1 --base x2_unetpp_bw_r90 ;;
    upp_softedge) $PY src/review.py $1 --base unetpp_bw_r90 ;;
    cnx_softedge) $PY src/review.py $1 --base cnx_bw ;;
    # Al3Ni 전용 모델: 지금 최선(앙상블 + Eutectic Si 전용 모델 30%) 위에 Al3Ni 확률을 섞는다
    al3ni_x2_unetpp) $PY src/decision_rules.py sie --cls 1 --ens runs/oof/_ens_A_sie03 --spec al3ni_x2_unetpp --procs 24 \
                       --out runs/rules/al3ni_on_sie03.json ;;
    # 두 번째 Eutectic Si 전용 모델: 단독으로 섞기 / 첫 번째와 평균내서 섞기
    sie_x2_cnx) $PY src/decision_rules.py sie --ens runs/oof/_ens_A_r6 --spec sie_x2_cnx --procs 24 --out runs/rules/sie_A_sie_x2_cnx.json
                $PY src/decision_rules.py sie --ens runs/oof/_ens_A_r6 --spec sie_x2_unetpp sie_x2_cnx --procs 24 \
                  --out runs/rules/sie_A_sie_x2_unetpp+sie_x2_cnx.json ;;
  esac
}

while true; do
  left=0
  for e in $EXPS; do
    [ -f runs/review8/$e.done ] && continue
    if ready $e; then
      evaluate $e > runs/review8/$e.txt 2>&1
      echo "$(date '+%m-%d %H:%M') $e evaluated" | tee runs/review8/$e.done
    else
      left=$((left + 1))
    fi
  done
  [ $left -eq 0 ] && break
  if [ -f runs/round8.done ]; then   # 학습은 끝났는데 결과가 모자란 실험 (실패) 표시
    for e in $EXPS; do [ -f runs/review8/$e.done ] || echo "$(date '+%m-%d %H:%M') $e INCOMPLETE" | tee runs/review8/$e.done; done
    break
  fi
  sleep 60
done

echo "$(date '+%m-%d %H:%M') 앙상블 기여 평가 시작 (기존 25개 + 새 4상 모델 3개)"
$PY src/ensemble.py --max-size 6 --out runs/oof/ensemble_summary_r8.json \
  --exps $A x2_unetpp_bw_e120_r90 upp_softedge cnx_softedge > runs/oof/ensemble_log_r8.txt 2>&1
echo "$(date '+%m-%d %H:%M') all" | tee runs/review8/all.done
