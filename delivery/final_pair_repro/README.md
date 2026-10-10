# 최종 MiT 4배 + UNet++ 3배 제출 재현

사용자가 2026-10-10 선택한 **4배 U-Net/MiT-B2 crop1536 + 3배 UNet++/ResNet34** 두 모델군의 50:50 확률 평균(soft voting)을 고정한다. 각 모델군은 fold 0~4의 `last.pt` 5개를 사용하므로 test 이미지 한 장당 총 10개 체크포인트가 참여한다. 기존 `sub6` 패키지와 제출 파일은 별도로 보존한다.

| 모델군 | 실험 폴더 (`runs/cv/` 아래) | 학습 | 추론 |
|---|---|---|---|
| U-Net / MiT-B2 | `x4_mit_c1536_bw_r90` | 60에폭, crop1536, seed0 | width2560, scale_multiplier2.0, tile1536, overlap128 |
| UNet++ / ResNet34 | `x3_unetpp_bw_e120_r90` | 120에폭, crop1536, seed0 | width3840, scale_multiplier1.0, tile1536, overlap128 |

MiT는 인코더 이름이며 이 모델 전체가 SegFormer인 것은 아니다. '4배'와 '3배'는 실험 정책 이름이다. 실제 이미지별 확대 배율은 `max(1, round(기준 폭 / 원본 폭)) × scale_multiplier`로 정해지므로 모든 사진에서 정확히 4배/3배인 것은 아니다.

입력별 p1~p99 밝기 정규화 → 8방향 회전·반전 TTA 및 슬라이딩 윈도 추론 → 원본 크기 확률 복원 → 각 모델군의 5-fold 확률 평균 → 두 모델군의 50:50 평균 → 기존 Primary Si 후처리 → 원본 크기 4색 RGB PNG 저장 순서다. Primary Si 후처리는 이미지 면적의 0.01% 미만인 연결 성분을 제거하고, 전체 Primary Si 면적이 0.5% 미만이면 해당 상 예측을 제거한다. 제거된 픽셀은 나머지 세 상 중 확률이 가장 높은 상으로 배정한다. 구현은 SHA-256으로 고정한 `src/infer.py`를 따른다.

## 준비 파일과 환경

프로젝트 루트에 `src/{submit,train,data,infer}.py`, 이 패키지, `data/test/images/`의 공식 입력 PNG 10개, 위 두 실험 각각 `fold0/last.pt`~`fold4/last.pt`가 필요하다. `manifest.json`은 소스 4개, 체크포인트 10개, 입력 10개의 정확한 경로·크기·SHA-256과 학습 인자를 기록한다. `reference_server_receipts.json`은 각 fold의 실제 체크포인트 에폭과 학습 인자가 sidecar `args.json`과 일치함을 확인한 기록이다. 가중치·학습 데이터·test 원본은 이 코드 패키지에 포함하지 않는다.

기준 환경은 Linux x86_64, Python 3.12.3, torch 2.14.1+cu126, torchvision 0.29.1+cu126 및 NVIDIA L40S이다. `environment.yml`의 67개 버전은 기준 서버에서 확인했다. 새 환경 설치 자체는 아직 검증하지 않았다.

```bash
conda env create -f delivery/final_pair_repro/environment.yml
conda activate komap-final-pair
```

Linux Bash에서 uv로 설치할 수도 있다.

```bash
uv venv --python 3.12.3 .venv-final-pair
uv pip install --python .venv-final-pair/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cu126 \
  --index-strategy unsafe-best-match \
  -r <(.venv-final-pair/bin/python -c 'import json; m=json.load(open("delivery/final_pair_repro/manifest.json")); print("\n".join(k+"=="+v for k,v in m["packages"].items()))')
source .venv-final-pair/bin/activate
```

PyPI와 공식 PyTorch 인덱스에서 기록된 정확한 버전을 찾도록 지정한 명령이다. 완료된 체크포인트를 읽을 때 모델 구조만 `weights=None`으로 생성하고 전체 상태를 로드하므로 ImageNet 가중치를 다시 다운로드하지 않는다. 실행 환경에 Hugging Face 오프라인 모드를 설정한다.

## 확인과 실행

기본 동작과 `--check`는 읽기 전용이다. 파일 해시, 핵심 패키지 버전, 입력 10개, 새 출력 경로를 확인하며 torch를 불러오거나 CUDA를 초기화하지 않는다. 실패 시 종료 코드는 2이다.

```bash
python delivery/final_pair_repro/run_submission.py --check \
  --project-root . --out submissions/final_pair_mit4_unet3
```

명시적으로 지정한 빈 GPU에서만 추론한다. 실제 실행 직전에 각 장치의 UUID, 다른 계산 프로세스 유무, 여유 메모리 30,000 MiB 이상을 확인한다. 이 기준은 보수적인 실행 조건이며 최소 VRAM 측정값은 아니다. 단일 GPU 또는 여러 GPU 중 하나를 선택한다.

```bash
# GPU 한 개
python delivery/final_pair_repro/run_submission.py --run --gpu 0 \
  --project-root . --out submissions/final_pair_mit4_unet3

# 또는 빈 GPU 네 개로 체크포인트를 나누어 동시에 예측
python delivery/final_pair_repro/run_submission.py --run --gpus 0 1 2 3 \
  --project-root . --out submissions/final_pair_mit4_unet3
```

여러 GPU 실행은 기존 `src/submit.py predict --shard i --nshards N`을 사용한다. 각 프로세스에 확인한 장치 UUID를 전달하며 모델을 중복 배정하지 않는다. 모든 프로세스의 정상 종료와 정확히 100개(10모델×10이미지) 확률 파일의 모양·자료형·유한값·확률합을 확인한 뒤 CPU에서 합친다. 학습이나 추가 fold/seed 실험을 실행하지 않는다.

출력 경로가 이미 있으면 덮어쓰기 없이 중단한다. 이전 예측 파일을 재사용하지 않는다. 중간 실패 시 기록을 보존하며 다른 새 출력 폴더를 지정해 다시 실행한다. 기존 GPU 작업을 종료하지 않는다. 실행 도중 오류가 발생하면 이 진입점에서 만든 자식 프로세스만 정리한다.

성공 시 출력:

- `masks/`: 원본 크기 RGB PNG 10장.
- `final_pair_masks.zip`: 하위 폴더 없이 검증된 PNG 10개만 포함한 제출 결과물.
- `summary_test.json`: 두 모델군 각각 5개, 50:50 조합, 이미지 10개, 마스크 형식 검사 결과.
- `run_receipt.json`: 명령, GPU UUID, 시작·종료 시각, 소스/manifest 해시, shard 종료 코드, 마스크·ZIP 해시와 완료 상태.
- `predict_shard*.log`, `combine.log`, `parts/`: 실행 로그와 모델별 원본 크기 확률. 중간 확률은 제출 마스크 ZIP에 포함하지 않는다.

허용 RGB 색은 Al `(153,127,76)`, Al3Ni `(25,76,153)`, Primary Si `(76,178,76)`, Eutectic Si `(204,204,204)` 네 가지다. 한 사진에 네 상이 모두 나타나야 하는 것은 아니다.

## 검증 범위와 점수 해석

선택된 조합의 기존 합금 단위 5-fold OOF 90장 평균 mIoU는 **80.900653378%**다. 이는 정답이 없는 test 10장의 공식 점수가 아니다. test 결과물에서는 재현 과정과 파일 형식만 확인할 수 있으며 이 도구는 대회에 제출하거나 점수를 전송하지 않는다.

준비 단계에서 기준 서버의 소스 4개·가중치 10개·입력 10개 및 핵심 패키지 버전이 읽기 전용 사전 검사에 통과했다. 체크포인트는 CPU로만 읽어 MiT 60에폭, UNet++ 120에폭과 실제 인자를 확인했고 CUDA는 초기화하지 않았다. 로컬에서는 GPU/서브프로세스를 모의해 네 shard의 배정·UUID 전달, 누락/잘못된 확률 거부, 기존 출력 거부, 실제 검증 함수의 RGB·크기·ZIP 검사에 통과했다. 이 모의 검사는 실제 GPU 추론의 대체가 아니다. 실제 생성 완료 여부는 출력 폴더의 `run_receipt.json`을 기준으로 판단한다. 새 환경 설치 후 전체 GPU 재현 검증은 별도다.
