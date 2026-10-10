# KoMaP 최종 두 모델 앙상블 코드

최종 선택은 **4배 U-Net/MiT-B2 crop1536 + 3배 UNet++/ResNet34**의 50:50 확률 평균(soft voting)이다. 각 모델군에서 fold 0~4의 확률을 먼저 평균하므로 이미지당 총 10개 체크포인트가 참여한다. 자세한 환경 설치와 실행 방법은 [재현 패키지 README](delivery/final_pair_repro/README.md)를 따른다.

| 모델군 | 필요한 체크포인트 | 학습 에폭 | 추론 설정 |
|---|---|---|---|
| U-Net / MiT-B2 | `runs/cv/x4_mit_c1536_bw_r90/fold0/last.pt` ~ `fold4/last.pt` | 60 | width2560, scale_multiplier2.0, tile1536 |
| UNet++ / ResNet34 | `runs/cv/x3_unetpp_bw_e120_r90/fold0/last.pt` ~ `fold4/last.pt` | 120 | width3840, scale_multiplier1.0, tile1536 |

입력별 p1~p99 정규화, 8방향 TTA, overlap128의 슬라이딩 윈도, 원본 해상도 확률 복원, 기존 Primary Si 후처리를 유지한다. '4배'와 '3배'는 원본 폭에 따른 확대 정책의 이름이며 실제 모든 입력의 배율이 일정한 것은 아니다. MiT는 U-Net의 인코더이며 전체 모델을 SegFormer로 표기하지 않는다.

프로젝트 루트 구조를 유지하여 압축을 풀고 `data/test/images/`에 입력 PNG 10개, 위 경로에 가중치 10개를 준비한다. 정확한 파일 SHA-256은 `delivery/final_pair_repro/manifest.json`에 있다. 추론은 새 출력 폴더에만 생성하며 기존 sub6 예측을 재사용하지 않는다.

```bash
# 자세한 설치 방법은 패키지 README 참고
conda env create -f delivery/final_pair_repro/environment.yml
conda activate komap-final-pair

python delivery/final_pair_repro/run_submission.py --check \
  --project-root . --out submissions/final_pair_reproduced

# 비어 있는 GPU 0 하나를 사용
python delivery/final_pair_repro/run_submission.py --run --gpu 0 \
  --project-root . --out submissions/final_pair_reproduced

# 또는 --gpu 0 대신 --gpus 0 1 2 3으로 여러 빈 GPU에 모델을 나눈다.
```

`environment.yml`은 실제 기준 Linux/Python/CUDA 패키지 버전 기록이다. 루트의 `requirements.txt`는 기존 프로젝트의 간략한 의존성 목록이며 정확한 재현에는 `environment.yml` 또는 manifest의 고정 버전을 사용한다. **새 환경을 설치해서 처음부터 재현하는 검증은 아직 수행하지 않았다.** 이 코드의 실행 파일은 Python 진입점 `run_submission.py`이며 독립 실행형 `.exe`가 아니다.

코드 묶음에는 학습·추론에 필요한 소스 4개, 제출 진입점, 환경 YAML, 입력·가중치 해시, 5-fold 분할 목록, OOF 평가 기록, test 생성 완료 영수증을 포함한다. `scripts/summarize_phase_masks.py`는 생성 마스크의 상별 2D 면적 비율을 요약하는 선택 기능이다. 기존 sub6의 면적 그림은 포함하지 않는다.

```bash
python scripts/summarize_phase_masks.py \
  --mask-dir submissions/final_pair_reproduced/masks \
  --out-dir reports/final_pair_phase_application --expected-count 10 --plot
```

최종 조합의 합금 단위 5-fold OOF 90장 평균 mIoU는 **80.900653378%**다. 기존 sub6의 80.788425601%보다 +0.112227778%p 높았다. 검토를 반복한 90장의 내부 검증 결과이며, 별도로 기록한 36장·54장 결과를 독립적인 외부 시험으로 해석하지 않는다. 정답이 없는 공식 test 10장의 점수는 알 수 없다. 실제 성능 수치는 `runs/review22_pair/results.json`과 `summary.txt`에서 확인할 수 있다.

`delivery/final_pair_repro/manifest.json` 및 사전검사 기록은 실행 전 동결한 문서다. 실제 test 생성 완료 여부·명령·GPU UUID·마스크와 ZIP 해시는 `submissions/final_pair_mit4_unet3/run_receipt.json`, 생성 구성은 같은 폴더의 `summary_test.json`을 기준으로 확인한다. 새 환경 설치 검증과는 별개인 기존 기준 서버의 실행 기록이다.

학습 데이터, test 원본, 가중치, 중간 확률, 마스크 PNG와 마스크 ZIP은 이 **코드 ZIP**에 들어 있지 않다. 결과물 제출용 `final_pair_masks.zip`은 별도다. 상별 면적 비율은 예측 마스크의 2D 픽셀 면적이며 실제 성분비·물성 측정치가 아니다. 이 코드 묶음 생성과 test 마스크 생성은 대회 서버로의 공식 제출을 수행하지 않는다.
