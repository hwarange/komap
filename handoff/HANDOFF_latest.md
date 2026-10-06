# HANDOFF: KoMaP 상 분할 프로젝트 (다른 노트북에서 이어서 작업하기)

- **작성:** 2026-10-06 06:20 KST
- **제출 마감:** 2026-10-15 (링커리어 공고 기준. 공식 공고로 확인 필요). 예선 결과 10/22, 본선 발표 10/29 KINTEX
- **지금 상태:** **6단계 실험이 서버에서 백그라운드로 돌고 있어요 (10/6 06:13 시작, 12~13시쯤 끝날 예정).** 끝나면 결과를 확인해서 3차 제출본을 만들어야 해요. 그전까지는 **2차 제출본(`submissions/sub2`, CV 79.96)이 현재 최선**이에요.

---

## ★ 가장 먼저: 서버에서 돌고 있는 6단계 결과 확인하기

### 무엇이 돌고 있나 (`scripts/run_round6.sh`, 서버에서 nohup으로 자동 진행)
계획 전체는 `reports/plan_round6.md`에 있어요.

| 단계 | 내용 | 결과 파일 (서버 `~/komap/`) |
|---|---|---|
| A2 | 크기 키운 TTA (원본 + 1.25배) 평가. 학습 없음 | `runs/scale_tta.json` |
| A1 | 학습 곡선 (학습 사진 54장·36장, ConvNeXt·UNet++) | `runs/learning_curve.json` (자동 판단 결과 `decision_full90` 포함) |
| B1 | `x2_unetpp_bw_r90`: 2배 해상도 + UNet++ | `runs/cv/x2_unetpp_bw_r90/` |
| B2 | `mn_r34_unetpp`, `mn_sxt50_unetpp`: MicroNet 인코더 + 대조군 `unetpp_bw_r90` | `runs/cv/...` |
| B3 | `effv2s_unetpp`, `hrnet_unet`: 앙상블 다양성용 | `runs/cv/...` |
| B4 | 두 번째 기회: `cnxS_bw_e120`, `m_dinov3_lr`(인코더 학습률 1/10), `m_skel_w015`(가중치 0.15) | `runs/cv/...` |
| C | **A1에서 54 → 72장 개선이 평균 +0.3 이상일 때만 자동 실행:** 90장 전체 학습. 4개 구성 × 시드 3개 | `runs/full/<실험>/fold<시드>/last.pt` |
| 마지막 | 앙상블 재평가 (기존 16개 + 6단계 모델 9개) | `runs/oof/ensemble_log_r6.txt`, `ensemble_summary_r6.json` |

### 확인 방법
```bash
ssh komap 'cat ~/komap/runs/round6_driver.log'                  # 전체 진행 단계
ssh komap 'grep -c " done " ~/komap/runs/launch_round6.log'     # 학습 완료 수 (총 65)
ssh komap 'grep FAIL ~/komap/runs/launch_round6*.log'           # 실패한 작업 (없어야 정상)
ssh komap 'ls ~/komap/runs/round6_done.txt'                     # 이 파일이 있으면 모두 끝난 것
```

### 끝나면 할 일 (D 단계 → 3차 제출본)
1. **실험별 점검을 한 번에 실행:**
   ```bash
   ssh komap 'cd ~/komap && bash scripts/review_round6.sh > runs/review_round6.txt 2>&1; cat runs/review_round6.txt'
   ```
   계획에서 정한 비교 대상과 같은 사진끼리 비교한 결과가 나와요. **실험마다 결과 + 버그 점검 + 놓친 부분**을 사용자에게 보고하세요(사용자 요청 방식).
2. **판단 기준** (`reports/plan_round6.md`)
   - 단일 모델은 +0.1 이상이고 95% 구간이 0보다 위일 때 채택
   - 앙상블 재료는 다양성이 있고 정직한 추정을 떨어뜨리지 않으면 채택
   - 크기 키운 TTA(A2)는 +0.1 이상이고 95% 구간이 0보다 위일 때 채택
3. **앙상블 구성 결정:** `ensemble_log_r6.txt`에서 "전체 평균"과 "중첩 검증 탐욕적 선택"의 정직한 추정을 비교하세요. 비슷하면 고르는 과정이 없는 쪽을 쓰세요.
4. **Si 덩어리 분류기 다시 학습:**
   ```
   python src/si_classifier.py cv --thr 0.8 --exps <구성>
   python src/si_classifier.py fit --exps <구성> --out runs/si_clf/sub3_model.pkl
   ```
5. **제출본 생성:** `sub2`와 같은 방식이에요(`submissions/sub2/README.md` 참고).
   - `submit.py predict` 실행: `--split cv`로 검증하고, `--split test`로 평가용을 만들어요.
   - 그다음 `combine --si-clf`를 실행해요.
   - **A2를 채택했다면 할 일 (아직 구현 안 됨):** `submit.py`가 `predict_probs(..., extra_scale=1.25)` 예측도 함께 평균내도록 코드를 고쳐야 해요.
   - **C 단계를 했다면 할 일 (아직 구현 안 됨):** 90장 학습 모델(`runs/full`, 실험당 시드 3개)을 섞어야 해요.
     - 계획은 구성마다 "fold 모델 5개 평균 : 90장 모델 3개 평균 = 1 : 1"이에요.
     - 지금 `submit.py`는 모델을 모두 같은 비중으로 평균내요. 비중을 지정하는 기능을 추가하거나, 실험별로 따로 예측한 뒤 합쳐야 해요.
     - **90장 모델은 검증할 수 없으니 안전 점검 세 가지를 하세요:**
       - 학습 사진 점수가 정상인지
       - 평가 사진 예측이 `sub2`와 사진당 2% 이상 다르지 않은지
       - 시드 3개끼리 예측이 일치하는지
6. **교체 기준:** 3차 제출본의 정직한 추정이 `sub2`보다 **+0.1 이상**일 때만 최선으로 바꾸세요.
7. 결과는 `reports/round6_results.md`로 정리하고, `reports/idea_journey.md`에도 반영하세요.

---

## 0. 새 노트북에서 시작하기 (순서대로)

### 0-1. 압축 풀기와 파이썬 환경
1. `Komap_handoff_20261006.zip`을 원하는 곳에 풀어요. 예: `C:\Users\<사용자>\Desktop\Komap`
2. 가상환경을 만들어요. 압축 파일에는 가상환경이 빠져 있어요.
   ```
   cd Komap
   python -m venv .venv
   .venv\Scripts\python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
   .venv\Scripts\python -m pip install -r requirements.txt
   ```
   - **pip 자체를 업그레이드하지 마세요.** Windows에서 가상환경이 깨진 적이 있어요.
   - GPU가 없거나 오래된 노트북이어도 괜찮아요. 학습은 서버에서 하니까요.
   - GTX 10 시리즈를 쓴다면 드라이버를 최신으로 업데이트하세요.

### 0-2. 서버 SSH 연결 (가장 중요)
- 서버: `j-j15a106@70.12.130.104`, 작업 폴더 `~/komap`
- 이전 노트북의 SSH 키는 **보안상 압축에 넣지 않았어요.** 새 노트북에서 새 키를 만들어 등록해야 해요. (서버 계정은 비밀번호 로그인이 안 됐어요.)
1. 새 노트북에서 키를 만들어요 (PowerShell): `ssh-keygen -t ed25519` (엔터 3번, 암호 없이)
2. 공개키 내용을 확인해요: `type $env:USERPROFILE\.ssh\id_ed25519.pub`
3. **회사 JupyterHub에 로그인 → File → New → Terminal**을 열고 실행해요 (`<공개키 한 줄>` 자리에 2번 내용 붙여넣기):
   ```
   echo '<공개키 한 줄>' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys
   ```
4. 새 노트북의 `~/.ssh/config`에 추가해요:
   ```
   Host komap
     HostName 70.12.130.104
     User j-j15a106
     IdentityFile ~/.ssh/id_ed25519
   ```
5. 확인: `ssh komap hostname` → 비밀번호 없이 `jupyter04`가 나오면 성공이에요.

### 0-3. Claude Code에서 이어서 하기
- 이 폴더를 열고 Claude에게 **"handoff/HANDOFF_latest.md를 읽고 이어서 하자"**라고 하면 돼요.
- 이전 대화에서 Claude가 기억해 둔 내용(사용자 선호, 프로젝트 맥락)은 `handoff/claude_memory/`에 있어요.
- 같은 기억을 쓰게 하려면 이 파일들을 Claude 메모리 폴더로 복사하세요. 위치는 `C:\Users\<사용자>\.claude\projects\<프로젝트 경로를 '-'로 바꾼 이름>\memory\`예요. 예: 프로젝트가 `C:\Users\abc\Desktop\Komap`이면 `c--Users-abc-Desktop-Komap`.
- 복사하지 않아도 이 HANDOFF 파일만으로 이어갈 수 있어요.

---

## 1. 과제 요약
- 알루미늄 합금 광학현미경(OM) 사진을 4개 상으로 분할해요: Al(갈색), Al3Ni(파랑), Primary Si(초록), Eutectic Si(마스크상 흰색 `(204,204,204)`).
- 평가는 이미지별 4개 상 mIoU의 평균(평가 10장)이에요. 참고 기준은 75%예요.
- 심사(100점): 과제이해 5, 데이터처리 15, **모델정확도 50**, 인사이트 10, 독창성 20
- 규칙 (사용자 확인):
  - 없는 상은 제외하거나 100%로 처리
  - 외부 사전학습 가중치 사용 가능
  - 검증 데이터 학습 사용 가능
  - **평가 이미지를 학습에 쓰지 않음**
  - 제출 형식은 4색 RGB PNG

## 2. 결과 요약 (90장 합금 단위 5-fold 교차검증, TTA + 후처리)

| 단계 | 내용 | CV mIoU |
|---|---|---|
| 기준 | U-Net + ResNet34 (시드 0·1 평균) | 78.31 |
| 최고 단일 모델 | `unetpp_bw_e120_r90` (UNet++ + 경계 가중 + 120에폭 + 90° 회전) | 79.49 |
| 1차 제출 (`sub1`) | 12개 실험 × 5-fold = 60개 모델 평균 | 79.73 |
| **2차 제출 (`sub2`)** | 16개 실험 × 5-fold = 80개 모델 평균 + **Si 덩어리 분류기** | **79.96** (평가 예상 약 80.6~80.7) |

- **효과 있음:**
  - 이미지별 밝기 정규화, 675px만 2배 확대
  - 8방향 TTA (+0.9), 경계 가중 손실
  - ConvNeXt / UNet++
  - 밝기 증강 (특히 **노이즈 −0.68, 감마 −0.23**: 빼면 손해)
  - 90° 회전만 쓰기 (UNet++·MiT·2배 해상도에서 효과)
  - 구조가 다른 모델끼리의 앙상블
  - Si 덩어리 분류기 (확신 80% 이상일 때만, +0.10)
- **효과 없음:**
  - SegFormer (−1.5), 큰 모델 (ConvNeXt-S/B, MiT-B4), DINOv3
  - 고해상도 경로, Skeleton 손실, 상 기준점 매핑, 저해상도 흉내, 복사-붙여넣기
  - EMA, 큰 배치, 경계 가중 5배
  - 밝기 기반 경계 재판정, 위치 이동 투표, 비인접 후처리
- **남은 벽:**
  - 경계 1px 띠를 빼고 채점하면 79.7 → 88.0. 오차의 상당 부분이 라벨 경계의 애매함으로 추정돼요.
  - Eutectic Si는 약 59%. 남은 큰 문제는 중간 크기 Si 덩어리에서 Primary/Eutectic 혼동이에요(라벨 애매함 추정).

## 3. 환경

| | 노트북 | 서버 |
|---|---|---|
| 위치 | 압축을 푼 폴더 | `ssh komap` → `~/komap` |
| GPU | (보조) | **L40S × 4** (46GB), 128코어, 1TB RAM, Ubuntu 24.04 |
| 파이썬 | `.venv` (requirements.txt) | `~/komap/.venv` (uv, Python 3.12, torch 2.14.1+cu126, smp 0.5.0, scikit-learn, scikit-image) |

- **서버에만 있는 것:**
  - 모든 모델 가중치 (`runs/cv/<실험>/fold*/last.pt`, 수십 GB)
  - OOF 예측 확률 (`runs/oof/<실험>/*.npy`)
  - 제출 예측 중간 파일 (`submissions/*/parts`)
- **코드 동기화 (노트북 → 서버):** `tar czf - --exclude=__pycache__ src splits experiments | ssh komap 'cd ~/komap && tar xzf -'`
- **결과 가져오기 (서버 → 노트북):** 모델 파일은 빼고 json·log만 가져와요. ssh 명령 안에서 다른 출력과 tar 스트림을 섞지 마세요.
- 서버 홈 폴더에 다른 사람 파일이 있으니 **`~/komap`만 쓰세요.** GPU 사용량 모니터링이 돌고 있어요. 사용자가 GPU 4장 사용을 허락했어요.
- 긴 파이썬 스크립트는 ssh 한 줄에 넣지 말고, 파일로 만들어 `scp`로 보내세요. 따옴표 충돌이 자주 났어요.

## 4. 코드 구조 (`src/`)

| 파일 | 역할 |
|---|---|
| `data.py` | 로딩, p1–p99 밝기 정규화, 배율 규칙, 증강(옵션: `rot90`, `lean`, `hm`, `lowres`, `copypaste`, `int_drop`), 합금 단위 5-fold |
| `infer.py` | 슬라이딩 윈도우 예측, 8방향 TTA, 원본 해상도 복원, 대회 방식 IoU, Primary Si 후처리 |
| `train.py` | 학습. `--fold k`, `--amp`(BF16), `--bw`, `--rot rot90`, `--hires`, `--skel`, `--ema`, `--int-drop`, `--save-oof` 등. 끝나면 마지막 에폭 모델을 TTA로 채점해 `result.json` 저장. **6단계 추가:** `--train-frac`(합금 단위 일부만 학습), `--enc-lr-mult`(인코더 학습률 배수), `--full`(90장 전체 학습, fold 번호 = 시드), 인코더 `micronet:<이름>`(MicroNet 가중치) |
| `launch.py` | 실험 파일(`experiments/roundN.txt`)을 5-fold 작업으로 펼쳐 GPU 4장 × 2개씩 실행. 완료된 fold는 건너뜀. **무거운 실험을 파일 앞쪽에** 두세요 |
| `review.py` | 끝난 실험 하나를 기준과 비교하고 이상 징후를 점검 (Δ, 신뢰구간, 상별, 예측 면적 비율, Primary Si 없는 사진) |
| `aggregate.py` | `runs/cv` 전체 집계표 |
| `oof.py` / `ensemble.py` | OOF 확률 생성 / 앙상블 평가 (탐욕적 선택 + 중첩 검증) |
| `si_classifier.py` | **Si 덩어리 분류기.** `cv`로 중첩 평가, `fit`으로 제출용 학습 |
| `submit.py` | 제출 예측 (`predict` → `combine`). `--split cv`로 제출 코드 자체를 OOF 검증, `--si-clf`로 분류기 적용 |
| `predict.py`, `analyze.py` | 예전 단일 모델 예측 / 오답 분석 |
| `eval_scale_tta.py` | 6단계 A2: 크기 키운 TTA 평가 (`infer.predict_probs`의 `extra_scale`) |
| `learning_curve.py` | 6단계 A1: 학습 곡선 → 90장 학습 진행 여부 판단 |
| `scripts/run_round6.sh` | 6단계 자동 진행 (지금 서버에서 실행 중. **실행 중에는 파일을 고치지 마세요**) |
| `scripts/review_round6.sh` | 6단계 끝난 뒤 실험별 점검을 한 번에 |

- 실험 목록: `experiments/round1~6.txt`, `experiments/round6c_full.txt`
- 분할: `splits/folds5.json`

## 5. 문서
- **`reports/idea_journey.md`: 전체 아이디어 과정.** 가설 기록표, 실수와 교정 기록이 있어요. 보고서·발표의 뼈대로 쓸 수 있어요.
- `reports/round1_results.md`, `round2_3_results.md`, `round4_5_results.md`: 단계별 결과
- `reports/research_notes.md`: 증강 논문, U-Net 동향, Al–Si–Ni 소재 지식, OM 관련 모델
- `eda/01_EDA.ipynb` + `eda/figures/`: EDA (보고서용 그림 11장)
- `submissions/sub1`, `submissions/sub2`: 제출본 (각 README에 구성·검증·재생성 방법)

## 6. 6단계 (`reports/plan_round6.md`): **지금 실행 중 (맨 위 ★ 절 참고)**
- 사용자와 합의한 계획이에요.
  - A. 측정: 학습 곡선, 크기 키운 TTA
  - B. 새 앙상블 재료: 2배 해상도 + UNet++, MicroNet 인코더, EfficientNetV2·HRNet
  - C. 90장 전체 학습 (A 결과가 긍정적일 때만)
  - D. 3차 제출본
- 목표는 CV 80.3~80.5예요. 10/7까지 정확도 작업, 10/8부터 보고서에 집중해요.
- 아래는 그 밖의 후보예요.

1. **제출 형식 최종 확인:** 파일명 규칙 `<사진이름>_mask.png`, 제출 방법
2. **90장 전체로 최종 학습:** 지금은 72장으로 학습한 fold 모델들이에요. 검증할 수 없으니 시드 여러 개로 하세요.
3. **MicroNet 인코더 (OM 포함 현미경 사전학습):** 앙상블 다양성용이에요. 불러오는 방법은 `reports/research_notes.md` 8절을 보세요.
4. **보고서용 분석 (학습 불필요)**
   - 분할 → 상 분율 → 탄성률(Hashin–Shtrikman) 추정
   - step별 크기 지표로 "step = 냉각 속도" 가설 검증
5. **보고서·발표 작성:** `idea_journey.md`, EDA 노트북, 단계별 보고서를 바탕으로

## 7. 알려진 함정
- UNet++ + `tu-convnext_*`는 smp 0.5에서 생성 오류가 나요. UNet++에는 ResNet·EfficientNet 계열을 쓰세요.
- 모듈 속성 이름 `half`는 `nn.Module.half()`와 충돌해요.
- 감시 스크립트에서 `grep -c` 결과가 0이면 종료 코드가 1이에요. 끝에 `; true`를 붙이세요.
- 중간 fold 결과로 판단하지 마세요. 기준 모델은 시드 2개 평균을 쓰세요(시드 0 단독은 사진 한 장 사고로 부풀려졌음).
- 손실 항을 추가할 때는 기존 손실 크기에 맞춰 가중치를 보정하고, 예측 면적 비율을 확인하세요.
- Windows 노트북 bash에서는 `cd 'c:\...'` 대신 `/c/Users/...` 경로를 쓰세요.
- GTX 1060 같은 구형 GPU에서는 `--amp`를 끄세요.

## 8. 사용자 선호
- 한국어로 쉽게 설명해 주세요.
- 실험할 때는 **왜 그렇게 골랐는지 + 스스로 놓친 점 점검**을 항상 함께 보고해 주세요. 실험 하나가 끝날 때마다 버그·놓친 부분 점검 결과도 같이요.
- 계획을 먼저 보여주고, 다음 방향은 결과를 보고 사용자가 정해요.
- 기준은 노트북이 아니라 **L40S**로 생각해 주세요.
- 인수인계 파일은 사본 없이 이 파일 하나로 유지해 주세요.
