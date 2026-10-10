# sub6 제출 결과 재생성 도구

2026-10-10 현재 채택한 sub6의 test 이미지 10장에 대한 예측을 재생성하는 실행 안내다. 실행 파일은 **Python 진입점 `run_submission.py`**이며, 단독 실행 바이너리(`.exe`)는 아니다. 프로젝트 소스와 학습 완료 체크포인트를 함께 사용한다. 이 폴더에는 체크포인트·대회 데이터·최종 제출 ZIP을 중복 저장하지 않았다.

## 고정한 모델과 처리 순서

| 모델 | 학습 | 사용 체크포인트 | 추론 설정 |
|---|---|---|---|
| U-Net + MiT-B2 (`x2_mit_bw_r90`) | 60에폭, 합금 단위 5-fold | fold 0~4의 `last.pt` | 기준 폭 2560, tile 1024, overlap 128 |
| UNet++ + ResNet34 (`x3_unetpp_bw_e120_r90`) | 120에폭, 합금 단위 5-fold | fold 0~4의 `last.pt` | 기준 폭 3840, tile 1536, overlap 128 |

입력별 p1~p99 밝기 정규화 → 모델별 8방향 회전·반전 TTA와 슬라이딩 윈도우 추론 → 원본 크기로 확률 복원 → 모델군별 5-fold 확률 평균 → 두 모델군 1:1 확률 평균 → 기존 Primary Si 후처리 → 4색 RGB PNG 저장 순서다. 2배·3배는 실험 이름이며, 실제 이미지별 정수 배율은 `src/data.py`의 `scale_for`가 원본 폭과 기준 폭으로 결정한다. 경계 픽셀의 별도 재추론이나 학습 중인 crop1536 후보는 포함하지 않는다.

Primary Si 후처리는 이미지 면적의 0.01% 미만인 작은 연결 성분을 제거하고, 남은 전체 Primary Si 면적이 0.5% 미만이면 해당 상의 예측을 제거한다. 제거된 픽셀에는 나머지 세 상 중 확률이 가장 높은 상을 부여한다. 세부 구현은 `src/infer.py`에 고정되어 있다.

## 필요한 파일

프로젝트 루트를 아래 구조로 준비한다. 경로와 체크섬의 기준은 `manifest.json`이다.

```text
project/
  src/{submit,train,data,infer}.py
  delivery/sub6_repro/{run_submission.py,environment.yml,manifest.json,README.md}
  data/test/images/<공모전 test 이미지 10장>_image.png
  runs/cv/x2_mit_bw_r90/fold0/last.pt
  runs/cv/x2_mit_bw_r90/fold1/last.pt
  runs/cv/x2_mit_bw_r90/fold2/last.pt
  runs/cv/x2_mit_bw_r90/fold3/last.pt
  runs/cv/x2_mit_bw_r90/fold4/last.pt
  runs/cv/x3_unetpp_bw_e120_r90/fold0/last.pt
  runs/cv/x3_unetpp_bw_e120_r90/fold1/last.pt
  runs/cv/x3_unetpp_bw_e120_r90/fold2/last.pt
  runs/cv/x3_unetpp_bw_e120_r90/fold3/last.pt
  runs/cv/x3_unetpp_bw_e120_r90/fold4/last.pt
```

`src/train.py`는 체크포인트 설정에서 모델 구조를 만드는 함수 때문에 필요하다. 이 실행 도구가 학습을 시작하지는 않는다. test 마스크 정답, train/valid 데이터, 예전 예측 확률은 재생성에 필요하지 않다. 체크포인트 10개의 합계 크기는 약 1.07 GB(십진 기준)다.

학습 시 인코더를 ImageNet 사전학습 가중치로 초기화했다. **완료 체크포인트에서 추론할 때는 `weights=None`으로 구조만 생성한 뒤 저장된 전체 가중치를 불러오므로 ImageNet 파일을 다시 내려받지 않는다.** 실행 환경에도 Hugging Face 오프라인 모드를 설정한다. 별도 외부 학습 데이터 다운로드는 이 추론 절차에 없다.

## 환경 설치

기준 환경은 Linux x86_64, Python 3.12.3, torch 2.14.1+cu126, torchvision 0.29.1+cu126, L40S다. `environment.yml`은 기존 서버의 설치 패키지 67개 버전을 읽어 작성했다. **새 환경 설치 및 이 진입점으로 전체 GPU 추론을 다시 수행하는 검증은 아직 하지 않았다.** 버전 기록만으로 다른 장비에서 동일 결과가 검증되었다고 해석하지 않는다.

Conda를 이용하면 프로젝트 루트에서 다음과 같이 별도 환경을 만든다.

```bash
conda env create -f delivery/sub6_repro/environment.yml
conda activate komap-sub6
```

uv를 사용하는 Linux Bash 환경에서는 같은 manifest의 67개 고정 버전을 설치할 수 있다. 기존 학습 환경과 구분된 `.venv-sub6`를 사용한다.

```bash
uv venv --python 3.12.3 .venv-sub6
uv pip install --python .venv-sub6/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cu126 \
  --index-strategy unsafe-best-match \
  -r <(.venv-sub6/bin/python -c 'import json; m=json.load(open("delivery/sub6_repro/manifest.json")); print("\n".join(k+"=="+v for k,v in m["packages"].items()))')
source .venv-sub6/bin/activate
```

위 명령은 PyPI와 공식 PyTorch의 두 인덱스만 사용한다. `unsafe-best-match`는 일반 패키지가 PyTorch 인덱스에도 있을 때 첫 인덱스의 버전만 검색하는 제약을 풀어, 두 인덱스에서 manifest의 정확한 고정 버전을 찾도록 한다.

CUDA 12.6 계열 PyTorch를 구동할 수 있는 NVIDIA 드라이버가 필요하다. 기존 L40S를 기준으로 준비했으며 최소 VRAM은 별도로 측정하지 않았다. 실행 전 검사에서는 보수적으로 선택 GPU의 여유 메모리 30,000 MiB 이상과 실행 중인 다른 GPU 계산 프로세스가 없음을 요구한다.

## 실행

기본 실행과 `--check`는 같은 읽기 전용 검사다. Python·핵심 패키지 버전, 소스 4개, 체크포인트 10개, test 이미지 10개의 존재·SHA-256·이미지 크기, 새 출력 경로를 검사한다. CUDA를 초기화하거나 모델을 로드하지 않고 출력 파일도 만들지 않는다. 실패 시 종료 코드는 2다.

```bash
python delivery/sub6_repro/run_submission.py --check \
  --project-root . --out submissions/sub6_reproduced
```

GPU가 비어 있을 때 실제 생성을 명시적으로 실행한다. `--gpu`는 `nvidia-smi`의 GPU 번호이며, 검사에서 확인한 장치 UUID를 CUDA에 전달해 같은 GPU를 선택한다. 아래 예시는 GPU 0을 사용한다.

```bash
python delivery/sub6_repro/run_submission.py --run --gpu 0 \
  --project-root . --out submissions/sub6_reproduced
```

실행 순서는 기존 `src/submit.py predict` → 100개(10모델×10이미지) 확률 파일 검사 → `combine` → 마스크·요약 검사 → ZIP 생성이다. 기존 출력 폴더가 있으면 즉시 중단하며, 덮어쓰기·삭제·기존 예측 재사용·자동 재학습은 하지 않는다. 실패한 경우 로그를 확인하고 다른 새 출력 폴더명으로 다시 실행한다. 원래 학습 후속 작업을 포함한 `scripts/run_sub6.sh`는 이 제출용 진입점으로 사용하지 않는다.

성공한 출력 폴더에는 다음이 생성된다.

- `masks/`: 원본 크기 RGB PNG 10장.
- `sub6_masks.zip`: 하위 폴더 없이 검증된 마스크 PNG 10개만 포함. 코드 제출 묶음과는 별도 결과물이다.
- `summary_test.json`: 입력 수 10, 이미지당 모델 수 10, 두 모델군당 5-fold, 1:1 혼합과 형식 검사 결과.
- `run_receipt.json`: 실행 명령, 파일 검증 결과, GPU 정보, 마스크·ZIP SHA-256, 완료 여부.
- `predict.log`, `combine.log`, `parts/`: 실행 로그와 모델별 원본 크기 확률 파일. `parts/`는 재생성을 위한 중간물로 결과물 이미지 ZIP에 들어가지 않는다.

출력 검사에서는 알루미늄 `(153,127,76)`, Al3Ni `(25,76,153)`, Primary Si `(76,178,76)`, Eutectic Si `(204,204,204)` 이외의 색을 허용하지 않는다. 한 이미지에 네 색이 모두 나타나야 하는 것은 아니다.

## 점수 해석과 검증 범위

채택 sub6의 전체 90장 합금 단위 OOF 평균 mIoU는 80.7884256%다. 이 값은 정답이 없는 test 10장의 공식 점수가 아니다. test 결과물에서는 형식과 생성 과정만 검사할 수 있다. 공식 제출이나 점수 전송은 이 도구가 수행하지 않는다.

기존 `src/submit.py` 예측과 OOF 점수의 비교 기록은 `runs/pair13/verify_pair.json` 및 `runs/sub6/verify_pair.txt`에 있다. 이 기록은 기존 예측 코드의 검증이며, 이번 새 진입점의 전체 재실행 검증과 구분한다. 2026-10-10에는 기존 서버에서 이 코드의 읽기 전용 사전 검사를 실행해 소스 4개·체크포인트 10개·test 입력 10개의 해시와 핵심 패키지 버전이 모두 일치함을 확인했다. 서버에 파일을 기록하거나 torch를 불러오거나 GPU를 초기화하지 않았다.

로컬에서는 CLI 기본 동작·GPU 번호 필수 조건·기존 출력 거부, 100개 예측 중 누락 또는 잘못된 확률값 거부, 기존 sub6 마스크 10장의 RGB·4색·크기·혼합 설정 검사를 확인했다. 검증 범위는 `manifest.json`에 기록했다. 최종 배포 전에는 가용 GPU에서 새 환경 설치와 `--run` 완료 기록을 추가하는 것이 남아 있다.
