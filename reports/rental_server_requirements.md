# KoMaP 대여 서버 선정 기준

2026-10-09. 현재 프로젝트 코드, 기존 학습 로그, 실행 중인 서버 조회와 공식 제품 문서를 기준으로 작성했다. 신규 서버의 성능과 대여료는 아직 측정하지 않았다.

## 권장 구성

기본 후보는 **전용 48GB NVIDIA GPU 2개, 16개 정도의 물리 CPU 코어(대략 32 vCPU), RAM 64GB, 여유 공간 500GB 이상의 NVMe SSD**다. 현재처럼 두 seed의 두 fold를 동시에 진행하려면 GPU 4개, CPU/RAM도 대략 두 배로 잡는다. 이는 시작점으로 제안한 구성이며 CPU/RAM의 실측 최소 요구량은 아니다.

| 항목 | 순차 실행 / 비용 절약 | 기본 병렬 실행 | 현재 4개 작업 병렬 실행 |
|---|---|---|---|
| 전용 GPU | 1개 | 2개 | 4개 |
| GPU당 VRAM | 32GB 후보는 검증 필요, 48GB 권장 | 동일 | 동일 |
| CPU 권장 시작점 | 물리 8코어 / 약 16 vCPU | 물리 16코어 / 약 32 vCPU | 물리 32코어 / 약 64 vCPU |
| RAM 권장 시작점 | 32GB | 64GB | 128GB |
| NVMe 여유 공간 | 500GB 이상 | 500GB 이상 | 500GB~1TB |

vCPU는 상품마다 공유 정도와 물리 코어 대응이 다르므로 숫자만 비교하지 않는다. GPU당 DataLoader worker 6개를 쓰는 현재 설정에서 데이터 공급 대기와 epoch 시간을 확인한다. 학습 작업별 OMP 스레드 제한은 4다.

## 현재 작업의 실제 기준

1024×1024 crop, batch 4, BF16, 이미지당 crop 8, fold당 학습 이미지 72개 및 144 iteration/epoch인 설정이다.

| 작업 | 정상 상태 학습 시간 / epoch | 로그의 최대 GPU 할당 메모리 | 주기적 검증 시간 |
|---|---:|---:|---:|
| 2배 MiT-B2 U-Net | 약 54.7초 | 약 26.63GB | 약 5.5초 |
| 4배 MiT-B2 U-Net | 약 54.7초 | 약 26.63GB | 약 18.6~19.1초 |
| 2배 UNet++ / ResNet34 | 약 86.5초 | 약 27.15GB | 약 7.9초 |

초기 epoch는 준비 비용이 포함되므로 제외한 값이다. 검증은 fold의 18개 이미지 기준이며 최종 TTA 평가 시간과 구별한다. MiT의 60 epoch 학습 계산은 약 54.7분이며, 실제 작업 완료에는 초기화, 주기적 검증, 체크포인트 저장, 최종 추론과 결과 처리 시간이 추가된다.

메모리 로그는 `torch.cuda.max_memory_allocated()/1e9`다. 26.63~27.15GB는 약 24.8~25.3GiB이고, PyTorch 예약 메모리 및 CUDA 문맥을 모두 포함한 상한은 아니다. 따라서 **24GB GPU는 기존 설정을 그대로 유지하는 이사 대상에서 제외**한다. 32GB는 후보지만 학습·검증·최종 TTA까지 실행해 OOM 여부를 확인해야 한다. 48GB가 여유 있는 선택이다.

작성 중 조회에서는 L40S 4개 모두 GPU utilization 100%, GPU별 약 22.4GiB 사용, 전력 약 273~283W / 제한 350W였다. 순간 사용량은 최대 메모리 요구량과 다르다. 시스템 RAM 사용량은 약 31GiB였고 전체 RAM은 약 1TiB다. 이 프로젝트를 이유로 1TB RAM을 다시 대여할 근거는 확인되지 않았다.

## GPU 후보

| 후보 | VRAM | 프로젝트에서의 판단 |
|---|---:|---|
| L40S | 48GB | 현재 실측 기준이 있고 기존 환경 이전이 수월하다. 비교 기준 카드다. |
| RTX 6000 Ada | 48GB | 메모리 여유가 있는 비교 후보다. L40S와 다른 전력 제한 및 냉각 조건을 함께 확인한다. |
| RTX 5090 | 32GB | 성능·비용 비교 후보다. 메모리 여유와 Blackwell용 소프트웨어 환경을 검증한다. |
| A100 | 40GB / 80GB | 메모리 대역폭과 용량 측면에서 비교할 가치가 있다. PCIe/SXM 및 전체 GPU/MIG 할당 여부를 확인한다. |
| RTX 4090 | 24GB | 현재 설정 그대로는 제외한다. 배치 축소 등 별도 설정의 검증이 필요하다. |

공식 사양상 L40S의 메모리 대역폭은 864GB/s, RTX 6000 Ada는 960GB/s, RTX 5090은 1,792GB/s, A100 80GB는 PCIe 1,935GB/s 및 SXM 2,039GB/s다. **이는 학습 속도의 배율이 아니다.**

우리 코드는 convolution/attention, 정규화, 업샘플링, FP32 loss 계산, CPU 증강과 tiled inference가 섞여 있다. 연산 처리량, 메모리 이동, 작은 연산 실행 비용, CPU 공급 중 어느 것이 시간을 지배하는지에 따라 GPU 간 순위가 달라질 수 있다. GPU utilization 100%만으로 병목을 판정할 수 없다. 병목의 정확한 비중은 아직 profiler로 측정하지 않았다.

광고의 AI TOPS 또는 FP8·희소 연산 성능은 현재 BF16 학습과 바로 대응하지 않는다. 더 큰 카드의 이론 성능만으로 이 프로젝트에서 더 빠르다고 판정하지 않는다.

## GPU 수의 의미

현재 코드는 GPU 한 개에 fold 또는 seed 작업 한 개를 배정한다. 여러 GPU의 VRAM을 합쳐 사용하는 구현이 아니다. 따라서 24GB 카드 두 개를 48GB 카드 한 개처럼 쓸 수 없다.

GPU 2개 또는 4개의 주된 이점은 독립 실험의 동시 실행이다. 5-fold는 동일 카드 기준으로 1개면 5회, 2개면 3회, 4개면 2회의 실행 구간으로 나뉜다. CPU/SSD 병목과 결과 처리에 따라 전체 시간은 달라진다. 현재 방식에서 NVLink/NVSwitch는 필수 조건이 아니다.

## 대여 전 성능 비교

1. 같은 split, crop 1024, batch 4, BF16, 증강, worker 6 및 iteration 수로 MiT와 UNet++를 각각 확인한다. GPU별로 환경 변경이 필요하면 변경한 버전도 기록한다.
2. 짧은 시험에서 첫 epoch는 준비 시간으로 분리하고, 다음 4개 epoch의 학습 시간을 비교한다. 최소 한 번 주기적 검증과 동일 TTA의 최종 추론도 실행한다.
3. 순간 메모리와 최대 allocated/reserved 메모리, OOM, 실제 전력 제한, 온도 및 성능 저하 여부를 확인한다.
4. 2개/4개를 빌릴 때에는 같은 수의 작업을 함께 실행해 서버 전체 처리량을 측정한다. 단일 GPU 성능만으로 병렬 실행 속도를 예상하지 않는다.
5. 최종 선택은 **전체 서버 시간당 비용 × 동일 실험의 실제 완료 시간**으로 비교한다. 저장소 및 전송 비용이 별도라면 더한다.

예를 들어 같은 작업이 L40S에서 60분, 다른 카드에서 40분이면 속도 향상은 1.5배다. 다른 카드의 시간당 총비용이 1.5배 미만일 때 계산 비용도 줄어든다. 이 수치는 설명용이며 신규 카드에서 측정한 결과가 아니다.

## 저장소·운영·환경

이전 대상은 우리가 학습한 체크포인트·가중치·분류기와 코드·결과·재현 기록이다. 현재 선정한 파일은 약 123GB이며 진행 중인 결과 확정 후 갱신한다. 원본 데이터와 다운로드 가능한 공개 모델, 기존 대용량 OOF 배열은 이전 대상에서 제외한다. 학습 시에는 데이터와 필요한 사전학습 모델을 다시 내려받는다. 새 OOF 생성, 새 체크포인트 및 환경 설치를 위해 NVMe의 실제 여유 공간 500GB를 권장하고, 실험을 오래 누적하면 1TB를 고려한다.

SSH, 서버 간 파일 전송, 외부 데이터 다운로드, 장시간 백그라운드 실행이 가능해야 한다. GPU 대여를 중지하거나 인스턴스를 재생성할 때 파일이 유지되는 저장소의 조건과 별도 요금을 확인한다. 장시간 학습에는 강제 회수 가능성과 작업 재실행 비용도 비교한다. 현재 체크포인트에는 optimizer/scheduler 상태가 저장되지 않아 중간 epoch에서 정확히 이어 학습하는 기능이 완비되지 않았다.

현 서버는 Python 3.12.3, PyTorch 2.14.1+cu126, CUDA runtime 12.6을 사용한다. L40S/Ada 후보는 이 환경을 기준으로 검증한다. RTX 5090은 기존 cu126 환경을 그대로 복사하는 방식으로 진행하지 않고 Blackwell을 지원하는 PyTorch/CUDA 및 드라이버 조합을 준비한다. 현재 PyTorch 공식 안내는 새 Blackwell GPU에 CUDA 13.0+ wheel을 권장한다. 의존성 버전과 체크포인트 출력까지 확인한 뒤 환경을 확정한다.

## 공식 근거

- [NVIDIA GPU performance background](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/): 연산·메모리·실행 지연에 따른 성능 제한.
- [NVIDIA L40S](https://www.nvidia.com/en-us/data-center/l40s/): 48GB, 대역폭 및 전력 사양.
- [NVIDIA RTX 6000 Ada](https://www.nvidia.com/en-us/products/workstations/rtx-6000/) 및 [데이터시트](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/rtx-6000/proviz-print-rtx6000-datasheet-web-2504660.pdf): 48GB, 960GB/s, 300W.
- [NVIDIA RTX 5090](https://www.nvidia.com/en-gb/geforce/graphics-cards/50-series/rtx-5090/) 및 [Blackwell 아키텍처 문서](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf): 32GB 및 1,792GB/s.
- [NVIDIA A100](https://www.nvidia.com/en-us/data-center/a100/) 및 [40GB/80GB 데이터시트](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/a100-80gb-datasheet-update-a4-nvidia-1485612-r12-web.pdf): 용량과 PCIe/SXM별 대역폭.
- [NVIDIA RTX 4090](https://www.nvidia.com/en-au/geforce/graphics-cards/40-series/rtx-4090/): 24GB.
- [PyTorch 2.12 release blog](https://pytorch.org/blog/pytorch-2-12-release-blog/): Blackwell용 CUDA wheel 및 드라이버 안내.

현재 대여료와 신규 서버 성능이 없으므로 가격 대비 최종 순위는 확정하지 않았다. 이 문서는 선정 기준 정리이며 서버 대여·이전 작업을 실행한 기록이 아니다.
