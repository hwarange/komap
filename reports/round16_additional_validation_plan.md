# 4배 MiT 추가 검증 시작

> 범위 변경 2026-10-09 20:46 KST: 사용자가 시간 절약을 위해 seed 0의 fold 2·3·4만 완료하고 보고서 작성 후 종료하도록 결정했다. 아래 최초 6개 계획은 이 변경으로 대체한다. 현재 학습을 이어받는 실행기는 `scripts/run_resolution16_seed0_closeout.py`이며 주 후보는 2배 MiT·3배 UNet++·4배 MiT seed 0을 각각 1/3로 평균한다. seed 1은 기존 pilot fold 0·1만 비교한다. 보고서는 사용자가 제공한 모델링결과서 양식에 맞춰 DOCX로 작성하며 팀명은 김형수없는김형수, 수상경력은 없음이다.

2026-10-09 KST. 사용자가 서버 이전 완료 후 "추가 검증 시작하자"고 요청했다.
RunPod L40S 한 개에서 **seed 0·1 × fold 2·3·4, 총 6개**를 순차 실행한다.
드라이버 PID 20469, 첫 학습 PID 20487, 실행 시작 약 20:10 KST.

## 이유와 고정 범위

이전 fold 0·1에서 유망했던 개선이 남은 fold에서도 이어지는지, 두 시드의 결과가
비슷한지 확인한다. 기존 2배 MiT 및 3배 UNet++ 체크포인트는 비교 기준으로 재사용한다.
완료한 4배 MiT fold 0·1도 재학습하지 않는다.

- 실행 순서: fold 2 seed 0→1, fold 3 seed 0→1, fold 4 seed 0→1.
- 각 실험 60 epoch, MiT-B2 U-Net, BF16, batch 4, crop/tile 1024.
- 원래 설정 그대로 width 2560 × scale_multiplier 2, 크기 증강 0.15, rot90,
  학습률 0.0003, weight decay 0.0001, 경계 가중 2.0.
- 종료 체크포인트는 고정한 마지막 60 epoch다. 결과에 따라 best epoch를 고르지 않는다.
- 초기값은 ImageNet이며 pilot 학습 가중치를 초기값으로 쓰지 않는다.
- OMP 4, BLAS 1, OpenCV 1, DataLoader 6으로 Pod CPU 할당량에 맞춘다.
- 한 GPU에서 한 모델씩 실행하며 기존 출력이 있는 fold는 덮어쓰지 않는다.

## 데이터와 초기값 복원 검증

이전 백업에서는 데이터·공개 모델·OOF를 제외했다. 새 학습에 필요한 작업 자료만 복원했다.

- train/valid 90장과 마스크, 총 180개 파일, 105,004,601바이트.
- 기존 서버에서 RunPod의 인증된 HTTPS API로 직접 복원했다. PC에 원본 데이터를 저장하지 않았다.
- 원본 데이터 식별 기록과 180개 전체 SHA256이 일치한다. test 및 외부 grain 데이터는 제외했다.
- 90개 이미지 이름과 기존 `splits/folds5.json`이 일치하고 각 fold 18장이다.
- 같은 합금이 서로 다른 fold에 걸치지 않으며 마스크의 크기와 클래스 색 변환을 검사했다.
- 공개 초기 모델은 공식 Hugging Face `smp-hub/mit_b2.imagenet`에서 다시 받았다.
- 기존 SMP 설정의 revision `868ab6f13871dcf8c3d9f90ee4519403475b65ef`를 그대로 사용한다.
- `model.safetensors` SHA256은 `ab18b54f81748427bae1183160822903470bcca0e3121944c0d185b4b1816a95`로 기존과 같다.
- 학습 과정은 이 고정 캐시를 offline 모드로 사용해 초기 모델의 자동 갱신을 막는다.

## 종료 후 비교

학습에서 생성한 신규 OOF를 보존하고, 이전 때 제외했던 기존 두 기준 모델의 5-fold OOF와
4배 pilot 두 시드의 fold 0·1 OOF를 같은 마지막 체크포인트와 8방향 TTA로 재생성한다.
전체 90장과 pilot을 제외한 54장(fold 2·3·4)을 별도로 보고한다.

두 개별 시드, seed 1 대비 seed 0, 시드 평균, 기존 sub6 대비 사전 추가 앙상블을 비교한다.
기존 교체 조합도 진단용으로 함께 기록한다. 비중을 새로 탐색하지 않는다.
주 후보 비중은 **2배 MiT 1/3 + 3배 UNet++ 1/3 + 4배 seed 0 1/6 + 4배 seed 1 1/6**다.

평균·fold별·클래스별·저대비 변화와 합금 단위 bootstrap 10,000회의 95% 구간을 기록한다.
주 후보의 검토 기준은 이전 계획 그대로 전체 +0.05%p 이상, 전체·비pilot CI 하한 양수,
비pilot 각 fold 개선, 저대비·Eutectic Si 비악화, 다른 상 하락 0.1%p 이내다.
이는 제출 검토 기준이며 자동 제출이나 현재 sub6 변경은 하지 않는다.

## 자기 점검

- 두 fold에서 후보를 골랐으므로 전체 CV를 완전히 독립적인 검증으로 설명하지 않는다.
  새 세 fold를 별도로 보고하며 공식 평가 점수와 구분한다.
- 앙상블의 모델 수와 비중도 바뀐다. 개선 전부를 해상도 하나의 효과라고 해석하지 않는다.
- 시드 둘은 안정성을 판단할 자료이며 모든 가능한 시드의 분산을 보장하지 않는다.
- 실제 새 서버의 학습 속도는 초기 에폭을 보고 갱신한다. 이전 서버의 54.7초/epoch를
  새 서버의 확정 속도로 사용하지 않는다.
- 원본 코드·가중치·fold 0·1 결과를 보존하고 새 파일·새 fold에만 쓴다.
- 중간에 실패하면 오류와 이미 완료한 작업을 기록한다. optimizer 상태가 없는 기존
  last.pt를 완전한 학습 재개 파일이라고 취급하지 않는다.

## 기록 위치

- 실행: `scripts/run_resolution16_additional.py`, 원본 trainer 래퍼 `scripts/train_runpod.py`.
- 비교: `src/review_resolution16_full.py`.
- 상태·로그·완료: `runs/review16_additional/{state.json,driver.log,seedN_foldK.log,completed_jobs.json,all.done,error.json}`.
- 최종 결과: `runs/review16_additional/{full_results.json,summary.md}`.
- 체크포인트: `runs/cv/x4_mit_bw_r90/fold2..4/`, `runs/cv/x4_mit_bw_r90_s1/fold2..4/`.
- 복원 검사: `backup_metadata/additional_validation_dataset_verified.json`, `imagenet_init_verified.json`, `additional_validation_preflight.json`.
