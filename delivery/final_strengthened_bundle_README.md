# KoMaP 최종 조합과 마지막 보강 검증 코드

이 묶음은 **MiT 4배 crop1536 + UNet++ 3배의 50:50 soft voting** 제출 재현 코드와, 마지막 오류 분석·배율 점검·짧은 대조 학습의 완료 기록을 함께 보관한다. 두 추가 실험을 검토한 뒤 기존 최종 조합을 유지하기로 확정한 경우에만 생성한다. `final_strengthened_code_bundle_receipt.json`의 `final_decision`이 `retain-current-pair`인지 확인한다. 기존 `final_pair_code_bundle.zip`과 제출 마스크 ZIP은 보존한다.

최종 모델군, 가중치, 추론 설정, 후처리는 기존 조합과 같다. **각 모델군의 5개 fold, 총 10개 체크포인트**로 test 이미지 10장을 처리한다. 합금 단위 OOF 90장 mIoU는 **80.900653378%**이며, 정답이 없는 공식 test 이미지의 점수는 아니다. MiT는 U-Net의 인코더 이름이며 전체 모델을 SegFormer라고 표시하지 않는다.

## 최종 제출 결과 재현

설치와 실행은 [재현 패키지 설명](delivery/final_pair_repro/README.md)을 따른다. [기존 코드 묶음 설명](delivery/final_pair_bundle_README.md)도 보존했다. 필요한 자료는 다음과 같다.

- 공식 test 입력 10장: `data/test/images/`.
- MiT 가중치 5개: `runs/cv/x4_mit_c1536_bw_r90/fold0/last.pt`부터 `fold4/last.pt`.
- UNet++ 가중치 5개: `runs/cv/x3_unetpp_bw_e120_r90/fold0/last.pt`부터 `fold4/last.pt`.
- 정확한 파일 해시는 `delivery/final_pair_repro/manifest.json`, 환경은 같은 폴더의 `environment.yml`.

```bash
python delivery/final_pair_repro/run_submission.py --check \
  --project-root . --out submissions/final_pair_reproduced

python delivery/final_pair_repro/run_submission.py --run --gpu 0 \
  --project-root . --out submissions/final_pair_reproduced
```

기존 기준 서버에서 수행한 test 생성 기록은 `submissions/final_pair_mit4_unet3/run_receipt.json`에 있다. **새 환경 설치 후 전체 실행을 다시 검증한 것은 아니다.** Python 진입점을 제공하며 독립 실행형 EXE를 포함하지 않는다.

## 추가 분석과 대조 실험

| 항목 | 소스 | 완료 기록과 범위 |
|---|---|---|
| 상별 혼동·경계·면적 진단 | `scripts/analyze_final_confusion26.py` | `runs/review26_final_strengthening/diagnostics/`: 기존 OOF 90장, 픽셀 혼동과 사진별 mIoU를 구분 |
| 폭 675px 이미지 배율 점검 | `scripts/refine_scale26.py` | `scale_trials/`: 해당 26장에 MiT 추론 8·6·4배를 적용하고 나머지 64장은 고정; 새 학습 없음 |
| 짧은 추가 학습 대조 | `scripts/finetune_final26.py` | `finetune_plan/`: 기존 fold0의 학습 72장·검증 18장, UNet++ 120에폭 가중치에서 두 조건 각각 6에폭 |

추가 학습은 동일 가중치에서 일반 crop과 **Al3Ni–공정 Si 정답 경계 양쪽을 35% 확률로 중심 삼는 crop**을 비교했다. 지도는 학습 72장의 정답으로만 생성하며, 검증 정답이나 검증 오답을 지도에 넣지 않는다. 학습률 `3e-5`, CE+Dice, 경계 가중치 2, 입력 배율과 crop1536, seed0, 마지막 체크포인트 평가를 유지한다. 새 optimizer와 짧은 스케줄을 사용하므로 기존 optimizer 상태까지 이어 받은 재개 학습은 아니다. 지도 중심을 선택하는 확률이 35%라는 뜻이며, 해당 픽셀의 손실 가중치를 35%로 정한다는 뜻이 아니다.

실험별 `plan.json`·`results.json`·사진별 JSON·`all.done`을 포함했다. 추가 학습은 두 조건의 실제 인자·6에폭 로그·결과도 포함한다. 평균은 사진별 mIoU의 평균이다. 공정 Si 향상만으로 전체 성능 향상을 결론내리지 않고, Al3Ni 등 다른 상과 최종 두 모델 앙상블의 변화도 확인한다. 수치와 최종 비교는 각 `results.json`이 기준이다.

6배 제한은 전체 90장에서 약 **+0.056%p**였고 사전에 정한 참고 점검 항목을 통과했다. 다만 해당 26장 중 12장은 개선되고 14장은 하락했으며, 변화량 중앙값은 약 **−0.137%p**였다. 평균 이득이 일부 사진에 집중되고 신뢰구간이 0을 포함해, 기존 제출 설정을 유지하고 속도 개선 후보로 남겼다. 해당 26장의 MiT 추론 시간은 약 49.3% 줄었지만 이는 전체 앙상블 실행 시간이 절반이 됐다는 뜻은 아니다. 포함된 `cap6_sensitivity_diagnostic.json`은 결과를 본 뒤 수행한 민감도 진단이다. 사전 참고 항목을 실패했다고 바꾸어 해석하지 않는다.

경계 집중 추가 학습은 검증 18장에서 UNet++ 단일 점수를 **79.0342 → 79.2522%**로 높였고, 일반 추가 학습보다 **+0.1710%p**였다. 그러나 MiT와 섞은 최종 조합은 **80.2588 → 80.2879%**, 약 **+0.0292%p**였다. 조합 개선의 합금 그룹 95% 부트스트랩 구간은 **−0.0601~+0.1497%p**로 0을 포함한다. 단일 모델의 개선 신호는 기록하되 최종 조합 교체 근거로는 부족하여 기존 제출 가중치를 유지했다. 위 점수는 18장에 한정되며, 전체 90장 점수와 직접 비교하지 않는다.

검증 자료는 이전부터 반복해서 살펴본 자료다. **배율 점검의 90장과 추가 학습의 18장을 같은 검증 범위로 합치지 않는다.** 합금 그룹 부트스트랩은 불확실성을 설명하는 통계이며, 반복적인 실험 선택에 따른 편향을 제거하지 않는다. 추가 학습의 18장 결과만으로 전체 90장 향상이나 새 사진에서의 우위를 확정할 수 없다. 경계·밝기·면적 분석도 탐색적 관계이며 원인을 증명하지 않는다. 2D 면적은 질량 조성이나 물성의 직접 측정값이 아니다.

## 분석 스크립트 재실행에 필요한 외부 자료

이 코드 ZIP만으로 원래 검증 실험을 즉시 다시 실행할 수는 없다. 원본 프로젝트에서 다음 자료를 별도로 복원해야 한다.

- 공식 train/valid 이미지와 정답 90장, 각 실험의 `args.json`·`result.json`, 기존 OOF `.npy` 확률 파일.
- 진단·배율 분석 계획에 기록된 과거 검증 소스와 완료 기록. 각 파일의 정확한 경로와 SHA-256은 `plan.json` 또는 진단 결과의 `verification.receipts`에 있다.
- 배율 점검에 필요한 해당 fold의 MiT 체크포인트와 기존 UNet++ OOF; 추가 학습에 필요한 UNet++ fold0 `last.pt` 및 기존 MiT OOF.
- Linux 환경, 기록된 패키지 버전과 GPU. 추가 학습 준비는 기존 캐시의 ResNet34 ImageNet 구조 초기화를 확인하므로 관련 캐시도 필요하다. 실제 학습 초기 가중치는 검증한 UNet++ 체크포인트 전체다.

자료 누락이나 해시 차이가 있으면 스크립트는 중단한다. 완료 기록이 이미 있으면 덮어쓰기를 거부하므로, 재실행은 별도의 프로젝트 복사본에서 완료 출력만 분리한 후 명시적으로 진행한다. 배율 스크립트는 준비와 실행을 구분하며, 추가 학습 스크립트의 `--prepare`는 CPU 사전검사와 학습 정답 지도 생성만 한다. GPU 학습은 명시적인 `--run`에서 시작한다. 분석을 보기 위해 학습을 다시 시작할 필요는 없다.

이 묶음에는 원본 이미지·정답 마스크·가중치·중간 확률·학습용 경계 지도·인증정보가 없다. 포함된 것은 코드와 파일 식별자, 해시, 집계·사진별 평가 수치다. 제출 마스크는 별도 `final_pair_masks.zip`을 사용한다. ZIP 생성은 대회 서버로의 공식 제출을 수행하지 않는다.
