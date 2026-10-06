# HANDOFF: KoMaP 상 분할 프로젝트 (다른 노트북·다른 날 이어서 작업하기)

- **작성:** 2026-10-06 17:45 KST
- **제출 마감:** 2026-10-15 (링커리어 공고 기준. 공식 공고로 확인 필요). 예선 결과 10/22, 본선 발표 10/29 KINTEX
- **지금 상태**
  - **최선 후보는 4차 제출본 `submissions/sub4`예요.** CV 정직한 추정은 약 80.10이고, 2차 제출본 sub2(79.96)보다 +0.14 높아요.
  - **아직 정하지 않은 것:** 최선 제출본을 `sub4`와 `sub4_full` 중 무엇으로 할지예요. sub4_full은 90장 전체 학습 모델을 비중 10%로 섞은 버전이고, sub4와 픽셀 0.036%만 달라요. Claude의 추천은 sub4_full이에요.
  - **8단계 실험이 서버에서 돌고 있어요.** 10/6 16:37에 시작했고, 20:30~21:00쯤 끝날 예정이에요. 끝나는 대로 **자동 평가 스크립트가 실험마다 평가해 둬요.**
- **코드 관리:** GitHub 비공개 저장소 `https://github.com/hwarange/komap` (`main` 브랜치). 대회 데이터·가상환경·모델 가중치는 빠져 있어요.

---

## ★ 가장 먼저: 8단계 결과 확인하기

### 무엇이 돌고 있나 (`experiments/round8.txt`, 25개 작업)
| 실험 | 내용 | 판단 방법 | 결과 파일 (서버 `~/komap/`) |
|---|---|---|---|
| `x2_unetpp_bw_e120_r90` | 최고 단일 모델(2배 해상도 UNet++, 60에폭 79.71)을 120에폭으로 | `review.py` vs `x2_unetpp_bw_r90` + 앙상블 기여 | `runs/review8/x2_unetpp_bw_e120_r90.txt` |
| `al3ni_x2_unetpp` | **Al3Ni 전용 모델** (`--binary-class 1`, 2배 해상도 UNet++) | 지금 최선(앙상블 + Eutectic Si 전용 모델 30%) 위에 Al3Ni 확률을 섞어 중첩 검증 | `runs/rules/al3ni_on_sie03.json` |
| `sie_x2_cnx` | **Eutectic Si 전용 모델 두 번째** (ConvNeXt, 2배 해상도). 다양성용 | 단독으로 섞기, 그리고 첫 번째 전용 모델(`sie_x2_unetpp`)과 평균내서 섞기 (중첩 검증) | `runs/rules/sie_A_sie_x2_cnx.json`, `runs/rules/sie_A_sie_x2_unetpp+sie_x2_cnx.json` |
| `upp_softedge`, `cnx_softedge` | **경계 부드러운 정답** (`--soft-edge 0.5`. 경계 1px 띠의 정답을 0.5·원래 상 + 0.5·주변 3×3 상 비율로). 논문의 BLS와 같은 방식 | `review.py` vs `unetpp_bw_r90` / `cnx_bw` + 앙상블 기여 | `runs/review8/upp_softedge.txt`, `cnx_softedge.txt` |
| (모두 끝난 뒤) | 새 4상 모델 3개의 앙상블 기여 | `ensemble.py` (25개 + 3개) | `runs/oof/ensemble_log_r8.txt` |

### 확인 방법
```bash
ssh komap 'cat ~/komap/runs/review8/*.done'                 # 평가가 끝난 실험 (INCOMPLETE면 학습 실패)
ssh komap 'ls ~/komap/runs/review8/all.done'                # 있으면 앙상블 기여 평가까지 모두 끝남
ssh komap 'grep -E "FAIL|all finished" ~/komap/runs/launch_round8.log'
ssh komap 'cat ~/komap/runs/review8/<실험>.txt'              # 실험별 결과
```
- Claude에게 **"8단계 결과 확인해줘"**라고 하면 실험별로 결과, 버그 점검, 놓친 점을 정리해 줘요(사용자 요청 방식).

### 판단 기준 (미리 정한 것)
- **4상 모델:** 단일 +0.1 이상이고 95% 구간이 0보다 위면 채택해요. 단일 점수가 낮아도 앙상블 자동 선택에서 꾸준히 뽑히면 앙상블 재료로 써요.
- **전용 모델:** 섞기의 중첩 검증 정직한 추정이 지금보다 오르면 채택해요. 비교 기준은 Eutectic Si 전용 모델 하나로 섞은 80.031(Si 분류기 전)이에요.
- **교체 기준:** 새 제출본(sub5)의 정직한 추정이 sub4(약 80.10)보다 **+0.1 이상**일 때만 최선을 바꿔요.

### 결과가 좋으면 할 일 (sub5)
1. **Si 분류기를 새로 섞은 확률로 다시 학습해야 해요.** 섞은 OOF 확률을 `decision_rules.py blendcache`로 만들고, `si_classifier.py cv`(효과 확인) → `fit` 순서로 해요. sub4의 예시는 `submissions/sub4/README.md`에 있어요.
2. **`submit.py`의 전용 모델 섞기(`--sie-spec`)는 지금 Eutectic Si(3번 상)만 지원해요.** Al3Ni 전용 모델을 쓰려면 상 번호 옵션을 추가해야 해요. `decision_rules.py`에는 `--cls`가 이미 있고, `blend()` 함수를 같이 쓰면 돼요.
3. Eutectic Si 전용 모델을 두 개 쓰면 `--sie-spec sie_x2_unetpp sie_x2_cnx`처럼 넘기면 돼요. 평균은 이미 지원해요.
4. 새 4상 모델이 앙상블에 기여하면 `--recipe`에 추가해요.
5. 검증은 sub4와 같아요: `verify_cv_blend.py`(제출 코드로 90장 재현), 형식 검사, sub4와의 차이.

---

## 0. 새 노트북에서 시작하기

### 0-1. 코드 받기와 파이썬 환경
```
git clone https://github.com/hwarange/komap.git Komap      # 비공개 저장소라 GitHub 로그인 필요
cd Komap
# data/ 폴더(train·valid·test)는 깃에 없어요. 대회 자료에서 따로 복사해 넣으세요.
uv venv Komap --python 3.13          # 가상환경 이름이 Komap (폴더 Komap/Komap)
uv pip install --python Komap/Scripts/python.exe torch==2.14.1 torchvision==0.29.1
uv pip install --python Komap/Scripts/python.exe -r requirements.txt
```
- **노트북에는 CPU용 torch면 충분해요.** 학습과 예측은 모두 서버(L40S)에서 해요. 노트북은 코드 수정, 짧은 시험, EDA, 보고서 그림용이에요.
- Python은 **3.13**으로 해요. requirements는 3.12~3.13 기준이에요.
- **주의:** uv에서 `--extra-index-url`은 `--index-url`보다 **먼저** 찾아봐요. GPU용 torch가 필요하면 `--index-url https://download.pytorch.org/whl/cu126` 하나만 쓰세요. 섞으면 CPU용이 설치돼요.
- Windows 가상환경에서 pip 자체를 업그레이드하지 마세요. 예전에 가상환경이 깨진 적이 있어요.

### 0-2. 서버 SSH 연결
- 서버: `j-j15a106@70.12.130.104`, 작업 폴더 `~/komap` (호스트 이름 `jupyter04`)
- 노트북마다 SSH 키를 새로 만들어 등록해야 해요. 서버 계정은 비밀번호 로그인이 안 돼요.
  1. PowerShell에서 키를 만들어요: `ssh-keygen -t ed25519` (암호 없이)
  2. 공개키 내용을 봐요: `type $env:USERPROFILE\.ssh\id_ed25519.pub`
  3. **회사 JupyterHub → File → New → Terminal**에서 실행해요: `echo '<공개키 한 줄>' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys`
  4. `~/.ssh/config`에 추가해요:
     ```
     Host komap
       HostName 70.12.130.104
       User j-j15a106
       IdentityFile ~/.ssh/id_ed25519
     ```
  5. 확인: `ssh komap hostname` → `jupyter04`가 나오면 성공이에요.
- 10/6 노트북(DESKTOP-MR7IIH9)의 키는 이미 등록돼 있어요.

### 0-3. Claude Code에서 이어서 하기
- 이 폴더를 열고 **"handoff/HANDOFF_latest.md를 읽고 이어서 하자"**라고 하세요.
- 이전 대화에서 Claude가 기억해 둔 내용은 `handoff/claude_memory/`에 있어요.
  - 사용자 선호, 프로젝트 맥락, 노트북 환경, 서버 작업 대기 방식이 들어 있어요.
  - 같은 기억을 쓰려면 이 파일들을 `C:\Users\<사용자>\.claude\projects\<프로젝트 경로를 '-'로 바꾼 이름>\memory\`로 복사하세요.

---

## 1. 과제 요약
- 알루미늄 합금 광학현미경(OM) 사진을 4개 상으로 분할해요: Al(갈색), Al3Ni(파랑), Primary Si(초록), Eutectic Si(마스크상 흰색 `(204,204,204)`).
- 평가는 이미지별 4개 상 mIoU의 평균이에요(평가 10장). 참고 기준은 75%예요.
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
| 기준 | U-Net + ResNet34 (시드 2개 평균) | 78.31 |
| 1차 제출 (`sub1`) | 12개 실험 앙상블 | 79.73 |
| 2차 제출 (`sub2`) | 16개 실험 앙상블 + Si 덩어리 분류기 | 79.96 |
| 최고 단일 모델 | `x2_unetpp_bw_r90` (2배 해상도 + UNet++) | 79.71 |
| 3차 후보 (`sub3a`) | 25개 실험 × 5-fold 앙상블 + Si 분류기 | 80.02 |
| **4차 (`sub4`)** | + **Eutectic Si 전용 모델 30% 섞기** + Si 분류기 다시 학습 | **약 80.10** |
| `sub4_full` | + 90장 전체 학습 모델 (비중 10%, 검증 불가) | 약 80.10 + α |

- **효과 있음**
  - 이미지별 밝기 정규화, 675px만 2배 확대(정수 배율)
  - 8방향 TTA, 경계 가중 손실
  - UNet++, ConvNeXt, **2배 해상도**
  - 노이즈·감마 밝기 증강, 90° 회전만 쓰기
  - 구조가 다른 모델끼리의 앙상블
  - Si 덩어리 분류기(확신 80%)
  - **한 상 전용 모델을 앙상블에 "섞기"** (대조 시험: 같은 구조의 4상 모델로 섞으면 0)
- **효과 없음**
  - 모델·사전학습 바꾸기: SegFormer, 큰 모델, DINOv3(설정 2가지), MicroNet, HRNet, ConvNeXt-S
  - 손실: Skeleton 손실(설정 2가지), EMA
  - 증강: 크기 TTA, 탄성 변형, cutout, mixup, CutMix, 저해상도 흉내, 복사-붙여넣기
  - 판정과 후처리: 상별 판정 기준 조정(이미 최적), "Eutectic Si 먼저 구분하기", Al3Ni↔Eutectic Si 덩어리 분류기(큰 덩어리 상한 +0.14)
- **남은 벽은 라벨 경계의 애매함이에요.**
  - 경계 1px 띠를 빼고 채점하면 79.7 → 88.0이 돼요.
  - 학습한 사진에서도 82점에 그쳐요.
  - 판정 기준을 어느 쪽으로 옮겨도 점수가 떨어져요(오답이 대칭).
  - Eutectic Si 약 59.5, Al3Ni 약 79예요.

## 3. 환경

| | 노트북 | 서버 |
|---|---|---|
| 위치 | 깃 저장소 폴더 | `ssh komap` → `~/komap` |
| GPU | 쓰지 않음 (CPU용 torch) | **L40S × 4** (46GB), 128코어, 1TB RAM, Ubuntu 24.04 |
| 파이썬 | 가상환경 `Komap` (uv, Python 3.13) | `~/komap/.venv` (uv, Python 3.12, torch 2.14.1+cu126, smp 0.5.0) |

- **서버에만 있는 것 (깃에 없음)**
  - 모델 가중치: `runs/cv/<실험>/fold*/last.pt`, `runs/full/...`
  - OOF 예측 확률: `runs/oof/<실험>/*.npy`, 그리고 앙상블 캐시 `runs/oof/_ens_A_r6`, `_ens_A_sie03`
  - Si 분류기: `runs/si_clf/*.pkl`
  - 평가 사진 모델별 예측: `submissions/sub3_pred/parts`
- **코드 동기화 (노트북 → 서버):** 지금은 `scp src/<파일> komap:komap/src/`로 해요. 보낸 뒤 `md5sum`으로 같은지 확인하세요.
  - 서버는 아직 깃 저장소가 아니에요. 서버에서 GitHub로 받게 하려면 배포 키나 토큰을 설정해야 해요.
  - **실행 중인 bash 스크립트 파일은 고치지 마세요.** bash는 실행하면서 스크립트를 한 줄씩 읽어요.
- **결과 가져오기 (서버 → 노트북):** 모델·`npy` 파일은 빼고 json·log·txt와 제출 마스크만 가져와요. `ssh ... 'tar czf - ...' > 파일`처럼 하고, tar 스트림에 다른 출력을 섞지 마세요.
- 서버 홈 폴더에 다른 사람 파일이 있으니 **`~/komap`만 쓰세요.** GPU 사용량 모니터링이 돌고 있어요. 사용자가 GPU 4장 사용을 허락했어요.

## 4. 코드 구조 (`src/`)

| 파일 | 역할 |
|---|---|
| `data.py` | 로딩, p1–p99 밝기 정규화, 배율 규칙(정수 배), 증강. 7단계에서 `elastic_deform`, `cutout`, `IGNORE=255` 추가 |
| `infer.py` | 슬라이딩 윈도우 예측, 8방향 TTA, 원본 해상도 복원, IoU, Primary Si 후처리. **정수가 아닌 배율에서 되돌릴 때는 bilinear**(7단계 버그 수정). 출력 상 수는 모델에서 알아내요 |
| `train.py` | 학습. 이전 옵션(`--bw`, `--rot rot90`, `--amp`, `--save-oof`, `--full`, `--train-frac`, `--enc-lr-mult` …)에 더해 **`--binary-class k`**(한 상 전용 모델), `--elastic`, `--cutout`, `--mixup`, `--cutmix`, **`--soft-edge ε`**(경계 부드러운 정답)이 있어요. 새 옵션을 끄면 손실 계산은 이전과 똑같아요(시험으로 확인) |
| `launch.py` | 실험 파일을 5-fold 작업으로 펼쳐 GPU 4장 × 2개씩 실행해요. 끝난 fold는 건너뛰어요. 무거운 실험을 앞에 두세요 |
| `review.py` | 끝난 실험을 기준과 같은 사진끼리 비교하고 이상 징후를 점검해요 |
| `ensemble.py` | OOF 앙상블 평가 (탐욕적 선택 + 중첩 검증) |
| `si_classifier.py` | Si 덩어리 분류기 (`cv` 중첩 평가 / `fit` 제출용) |
| **`decision_rules.py`** | OOF로 판정 규칙을 실험해요. `cache`(앙상블 확률 저장), `bias`(상별 기준), `sie --cls k`(전용 모델 섞기·먼저 구분하기, 중첩 검증), `blendcache`(섞은 확률 저장) |
| **`oracle_components.py`** | 덩어리 재판정의 상한을 크기별로 계산해요 |
| **`check_full.py`** | 90장 모델 안전 점검(`train-score`, `agree`, `diff`) |
| **`verify_cv_blend.py`** | 섞기까지 포함한 제출 코드 90장 검증 |
| `submit.py` | 제출 예측. **평가 사진은 모델별로 저장**(`parts/models/<실험>/fold<k>`)하고, `combine`에서 `--recipe 실험:가중치`, `--full-mix 0.5`, `--sie-spec <전용 모델> --sie-alpha 0.3`, `--si-clf`로 섞어요 |
| `scripts/run_sub3.sh`, `run_sub4.sh` | 3·4차 제출본 자동 생성 |
| `scripts/review_round8.sh` | 8단계 자동 평가 (지금 서버에서 실행 중) |

- 실험 목록: `experiments/round1~8.txt`, `round6c_full.txt`, `round6c2_full.txt`, `round7.txt`, `round7_aug.txt`
- 분할: `splits/folds5.json`

## 5. 문서
- **`reports/idea_journey.md`: 전체 아이디어 과정.** 가설 기록표, 실수와 교정 기록이 있어요. 보고서·발표의 뼈대로 쓸 수 있어요.
- `reports/round1_results.md`, `round2_3_results.md`, `round4_5_results.md`, **`round6_7_results.md`**: 단계별 결과
- `reports/research_notes.md`: 증강 논문, U-Net 동향, 소재 지식, OM 모델, **9절(10/6 추가): 경계 라벨 스무딩(BLS), 냉각 속도 관계식, 질량 수지 부피 분율, 탄성률 검증 데이터**
- `reports/plan_round6.md`: 6단계 계획
- `eda/01_EDA.ipynb` + `eda/figures/`: EDA (보고서용 그림 11장)
- `submissions/sub1` ~ `sub4_full`: 제출본. sub2와 sub4에는 README와 미리보기가 있어요

## 6. 다음 후보 (8단계 결과를 보고 사용자가 정함)
1. **8단계 결과 반영 → sub5** (위 ★ 절)
2. **경계 라벨 스무딩 완전판:** ③이 효과가 있으면 Dice 손실도 부드러운 정답용으로 바꿔요. 논문에서 CE만 바꾸면 +0.3~0.5, Dice까지 바꾸면 +0.7~1.2였어요. 큰 데이터 기준이라 기대치는 낮춰 잡아야 해요.
3. **상별 앙상블 가중치:** CPU 몇 분이면 돼요. 가중치는 단순하게 두고 중첩 검증으로 확인해요. 픽셀 단위 stacking은 피해요.
4. **앙상블 확률 증류:** 주의가 필요해요. OOF 확률을 교사로 써도 학생의 CV 점수가 부풀어요(다른 fold 모델이 검증 사진을 학습했기 때문). 정직하게 재려면 안쪽 fold로 교사를 다시 만들어야 해요.
5. **보고서용 분석 (학습 불필요, 10/8부터 집중)**
   - 예측 상 분율 vs 정답 분율 → Hashin–Shtrikman 탄성률. Al3Ni 탄성률은 160~190 GPa로 두고, Al-18Si-8Ni 측정값 약 95 GPa로 점검해요.
   - step별 크기 지표(Si 판 두께, Al3Ni 간격)로 "step = 냉각 속도" 가설을 검증해요. λ ∝ V^−0.5 관계를 써요.
   - 라벨 애매함의 근거를 그림으로 정리해요: 경계 1px 제외 88.0, 학습한 사진 82점, 판정 기준 대칭성, Si–Al3Ni 3D 망
   - 독창성 정리: EDA 기반 배율 규칙, 금속학 특징 Si 분류기, 한 상 전용 모델 섞기(대조 시험 포함), 중첩 검증 원칙
6. **대회 멘토링 Q&A로 물어볼 것:** "step 1~4"의 의미, 제출 파일명 규칙(`<사진이름>_mask.png`)
- **일정:** 정확도 작업은 10/7까지, 10/8부터 보고서·발표에 집중, 10/14 최종 점검, 10/15 제출

## 7. 알려진 함정
- UNet++ + `tu-convnext_*`는 smp 0.5에서 생성 오류가 나요. UNet++에는 ResNet·EfficientNet 계열을 쓰세요.
- 모듈 속성 이름 `half`는 `nn.Module.half()`와 충돌해요.
- **서버 작업을 기다릴 때 `while pgrep -f "..."`를 쓰지 마세요.** 기다리는 명령 자신이 그 문자열을 포함해서 영원히 끝나지 않아요. `pkill -f`도 마찬가지로 자기 ssh 세션을 죽여요. **결과 파일이나 done 파일이 생기는 걸 기준으로 기다리세요.**
- 서버의 기본 셸은 `sh`(dash)예요. 배열이나 `[[ ]]`는 `bash -c`나 bash 스크립트 안에서만 쓰세요. `cd ~/komap && A & B`처럼 쓰면 `&` 때문에 B는 다른 폴더에서 실행돼요.
- 크기 TTA처럼 **정수가 아닌 배율**을 쓰면 확률을 되돌리는 방식이 중요해요(torch `area`는 흐려짐). 지금 코드는 고쳐져 있어요.
- numpy의 참/거짓 값(`np.bool_`)은 JSON으로 저장할 수 없어요. `bool()`로 바꾸세요.
- 중간 fold 결과로 판단하지 마세요. 기준 모델은 시드 2개 평균을 쓰세요.
- 손실 항을 추가할 때는 기존 손실 크기에 맞춰 가중치를 보정하고, 예측 면적 비율을 확인하세요.
- Windows 노트북의 git bash에서는 `/c/Users/...` 경로를 쓰세요.
- 고르는 과정(앙상블 구성, 섞는 비율, 기준값)이 있으면 **중첩 검증**으로만 정직한 추정을 내세요. 90장 전체로 고른 값은 부풀려져요.

## 8. 사용자 선호
- **항상 한국어로, 쉽게 설명해 주세요.**
- 실험할 때는 **왜 그렇게 골랐는지 + 스스로 놓친 점 점검**을 항상 함께 보고해 주세요. **실험 하나가 끝날 때마다** 결과, 버그 점검, 놓친 부분을 같이 보고해요.
- 계획을 먼저 보여주고, 다음 방향은 결과를 보고 사용자가 정해요. (사용자가 "다 해 보자", "진행하자"라고 하면 바로 진행)
- "싸게 재 보고 효과 있는 것만" 원칙은 거짓 음성을 만들 수 있어요. 설정 문제로 실패했을 수 있는 아이디어는 다른 설정으로 한 번 더 시험하고, **앙상블 재료는 단일 점수가 아니라 앙상블 기여로** 판단해요.
- 기준은 노트북이 아니라 **L40S**로 생각해 주세요. **학습은 서버에서만** 해요(노트북에 GPU용 torch를 설치하지 않음).
- **작업이 끝나면 알림을 보내 주세요** (PushNotification). 결과는 늦지 않게 바로 보고해요.
- 인수인계 파일은 사본 없이 **이 파일 하나**로 유지해요. 사용자가 요청할 때만 고쳐요.
- 코드는 깃(GitHub 비공개 저장소)으로 관리해요. 단계가 끝나거나 제출본을 만들면 커밋하고 올려요.
