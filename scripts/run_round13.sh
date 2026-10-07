#!/bin/bash
# 13단계 밤샘 (10/8): 학습 → 실험마다 끝나면 바로 평가 → 모두 끝나면 평가 사진 예측 + 최종 조합 후보 + 앙상블 기여 평가.
#   setsid nohup bash scripts/run_round13.sh > runs/review13/driver.log 2>&1 < /dev/null &
# 실험별 결과 runs/review13/<실험>.txt (+ .done), 최종 조합 요약 runs/combo/summary13.txt, 모두 끝나면 runs/review13/all.done
# sub6은 만들지 않는다 (고해상도 비중·최종 구성은 아침에 사용자와 정한다).
cd ~/komap
PY=.venv/bin/python
log() { echo "$(date '+%m-%d %H:%M:%S') $*"; }
CPU() { OMP_NUM_THREADS=32 "$@"; }   # CPU 평가는 스레드를 제한한다 (10/8 Si 분류기 동시 실행으로 부하 717)
mkdir -p runs/review13
A27=$(cat runs/combo/A27/exps.txt)
NEW="x3_unetpp_bw_e120_r90 x2_mit_bw_r90 x2_cnx_bw_e120 al3ni_x2_cnx_e120"
S4="sie_x2_unetpp sie_x2_cnx sie_x2_unetpp_e120 sie_x2_cnx_e120"

(.venv/bin/python src/launch.py experiments/round13.txt --per-gpu 2 --heavy x3_unetpp_bw_e120_r90 x2_mit_bw_r90 \
   > runs/launch_round13.log 2>&1; echo $? > runs/round13.done) &

ready() { [ "$(ls runs/cv/$1/fold*/result.json 2>/dev/null | wc -l)" -ge 5 ] && [ "$(ls runs/oof/$1 2>/dev/null | wc -l)" -ge 90 ]; }
last=""
while true; do
  left=0
  for e in $NEW; do
    [ -f runs/review13/$e.done ] && continue
    if ! ready $e; then left=$((left + 1)); continue; fi
    case $e in
      x3_unetpp_bw_e120_r90)
        { echo "### vs 3배 60에폭"; CPU $PY src/review.py $e --base x3_unetpp_bw_r90
          echo "### vs 2배 120에폭 (최고 단일)"; CPU $PY src/review.py $e --base x2_unetpp_bw_e120_r90; } > runs/review13/$e.txt 2>&1 ;;
      x2_mit_bw_r90)
        { echo "### vs 원래 해상도 MiT (해상도만 다름)"; CPU $PY src/review.py $e --base unet_mit_bw_r90
          echo "### vs 2배 UNet++ 60에폭 (인코더만 다름)"; CPU $PY src/review.py $e --base x2_unetpp_bw_r90; } > runs/review13/$e.txt 2>&1 ;;
      x2_cnx_bw_e120)
        { echo "### vs 원래 해상도 ConvNeXt 120에폭 (해상도만 다름)"; CPU $PY src/review.py $e --base cnx_bw_e120
          echo "### vs 2배 UNet++ 120에폭 (최고 단일)"; CPU $PY src/review.py $e --base x2_unetpp_bw_e120_r90; } > runs/review13/$e.txt 2>&1 ;;
      al3ni_x2_cnx_e120)   # Eutectic Si 전용 4개까지 섞은 앙상블(A27_sie4) 위에서. 기존 al3ni_x2_unetpp 하나는 +0.029
        { echo "### 단독"; CPU $PY src/decision_rules.py sie --cls 1 --ens runs/oof/_ens_A27_sie4_sie --spec $e --procs 24 \
            --out runs/rules/al3ni_cnx_on_A27_sie4.json
          echo "### 기존 al3ni_x2_unetpp와 평균"; CPU $PY src/decision_rules.py sie --cls 1 --ens runs/oof/_ens_A27_sie4_sie \
            --spec al3ni_x2_unetpp $e --procs 24 --out runs/rules/al3ni_pair_on_A27_sie4.json; } > runs/review13/$e.txt 2>&1 ;;
    esac
    log "$e evaluated" | tee runs/review13/$e.done
  done
  [ $left -eq 0 ] && break
  if [ -f runs/round13.done ]; then
    if [ -n "$last" ]; then
      for e in $NEW; do [ -f runs/review13/$e.done ] || log "$e INCOMPLETE" | tee runs/review13/$e.done; done
      break
    fi
    last=1; continue   # 학습이 모두 끝났으면 한 번 더 돌며 남은 평가를 마친다
  fi
  sleep 120
done

# 끝까지 학습된 모델만 쓴다
OK=""; for e in $NEW; do ready $e && OK="$OK $e"; done
GEN=""; for e in x3_unetpp_bw_e120_r90 x2_mit_bw_r90 x2_cnx_bw_e120; do case " $OK " in *" $e "*) GEN="$GEN $e";; esac; done
AL="al3ni_x2_unetpp"; case " $OK " in *" al3ni_x2_cnx_e120 "*) AL="al3ni_x2_unetpp al3ni_x2_cnx_e120";; esac
ALL="$A27 x2_unetpp_bw_e120_r90_s2 x3_unetpp_bw_r90$GEN"
N=$(echo $ALL | wc -w)
log "학습 완료 모델:$OK / 최종 조합 일반 모델 ${N}개 / Al3Ni 전용: $AL"

log "평가 사진 예측 (GPU 0~3)"
PP=()
for i in 0 1 2 3; do
  (CUDA_VISIBLE_DEVICES=$i $PY src/submit.py predict --exps $OK --split test --out submissions/sub3_pred --shard $i --nshards 4) \
    > runs/review13/pred_test_gpu$i.log 2>&1 &
  PP+=($!)
done

log "최종 조합 후보 (CPU, 하나씩 약 3분)"
SIE_SPEC="$S4" AL3NI_SPEC="$AL" CPU bash scripts/eval_final_combo.sh A${N}_s4a2 $ALL
SIE_SPEC="$S4" CPU bash scripts/eval_final_combo.sh A${N}_s4a1 $ALL
for T in A27 A29_sie4 A${N}_s4a1 A${N}_s4a2; do
  echo "##### $T ($(cat runs/combo/$T/alphas.txt))"
  $PY - "$T" <<'EOF'
import json, re, sys
T = sys.argv[1]
d3, d4 = (json.load(open(f'runs/combo/{T}/{f}.json')) for f in ('3_sie', '4_al3ni'))
for f, d in (('3_sie', d3), ('4_al3ni', d4)):
    h = d['nested (honest)']
    print(f"  {f:8s} base {d['base']['mIoU']:.3f} -> nested {h['mIoU']:.3f} (delta {h['delta']:+.3f} {h['ci95']})")
si = open(f'runs/combo/{T}/6_si_sie_al.txt').read()
m = re.search(r'Si classifier.*?delta ([+-][\d.]+)', si)
if m:
    honest = d3['base']['mIoU'] + d3['nested (honest)']['delta'] + d4['nested (honest)']['delta'] + float(m.group(1))
    print(f"  정직한 추정 (평균 + 단계별 중첩 검증 효과의 합): {honest:.3f}  (sub5 80.233 대비 {honest - 80.233:+.3f})")
EOF
  grep -h "Si classifier" runs/combo/$T/6_si_sie_al.txt
done > runs/combo/summary13.txt 2>&1

wait "${PP[@]}"
log "평가 사진 예측 끝: $(for e in $OK; do echo -n "$e=$(ls submissions/sub3_pred/parts/models/$e/fold*/*.npy 2>/dev/null | wc -l)/50 "; done)"

log "앙상블 기여 평가 (ensemble.py, 1시간 이상)"
CPU $PY src/ensemble.py --max-size 6 --out runs/oof/ensemble_summary_r13.json --exps $ALL > runs/oof/ensemble_log_r13.txt 2>&1
log "all" | tee runs/review13/all.done
