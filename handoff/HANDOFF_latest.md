# HANDOFF: KoMaP 상 분할 프로젝트 (다른 노트북·다른 날 이어서 작업하기)

- **작성:** 2026-10-07 13:20 KST
- **제출 마감:** 2026-10-15 (링커리어 공고 기준. 공식 공고로 확인 필요). 예선 결과 10/22, 본선 발표 10/29 KINTEX
- **일정 원칙:** 정확도 작업은 10/7까지예요. 10/8부터 보고서·발표에 집중하고, 10/14 최종 점검, 10/15 제출이에요.
- **코드 관리:** GitHub 비공개 저장소 `https://github.com/hwarange/komap` (`main`). 대회 데이터·가상환경·모델 가중치는 빠져 있어요.

## 지금 상태 한눈에
- **확정된 최선은 4차 제출본 `submissions/sub4`예요.** CV 정직한 추정은 약 80.10이에요(sub2 79.96).
  - `sub4_full`(90장 모델 비중 10%)과 sub4 중 무엇을 최선으로 할지는 **아직 사용자가 정하지 않았어요.** Claude의 추천은 sub4_full이에요.
- **sub5 후보 구성은 정해졌어요.** CPU 평가가 끝났고, 아래 ★ 절의 순서대로 만들면 돼요. 정직한 추정은 약 80.19(sub4 대비 +0.09)예요.
- **서버에서 아직 돌고 있는 것 (10/7 13:20 기준)**
  - 최고 모델 시드 1(`x2_unetpp_bw_e120_r90_s1`): 마지막 fold가 119/120 에폭이에요. 곧 끝나요.
  - 최고 모델 90장 학습(`full_x2_unetpp_bw_e120_r90`, 시드 3개): 90~97/120 에폭이라 **14:20~14:30쯤** 끝나요.
  - 자동 평가 `scripts/review_round9.sh`: 위 두 작업이 끝나면 평가하고, 앙상블 기여 평가(`runs/oof/ensemble_log_r9.txt`)까지 해요. 끝나면 `runs/review9/all.done`이 생겨요.
- **사용자 결정을 기다리는 것**
  1. 3배 해상도 실험(조각 1536, GPU당 1개, 약 3.5~4시간). Claude 추천은 진행이에요.
  2. 200에폭 학습(2배 해상도 UNet++, 밤사이).
  3. sub4 / sub4_full 중 최선 선택.

---

## ★ 가장 먼저 할 일: 9단계 결과 확인 → sub5 만들기

### 1) 9단계 남은 결과 확인
```bash
ssh komap 'cat ~/komap/runs/review9/*.done; ls ~/komap/runs/review9/all.done'
ssh komap 'cat ~/komap/runs/review9/x2_unetpp_bw_e120_r90_s1.txt'      # 시드 1 vs 시드 0 (시드 잡음), vs 60에폭
ssh komap 'cat ~/komap/runs/review9/full_train_score.json'              # 90장 모델 안전 점검 (학습한 사진 점수 > OOF 이면 정상)
ssh komap 'grep -v -i warn ~/komap/runs/oof/ensemble_log_r9.txt | sed -n "/predefined/,\$p"'
```
- Claude에게 **"9단계 결과 확인해줘"**라고 하면 실험별로 결과, 버그 점검, 놓친 점을 정리해 줘요.

### 2) 최종 조합 다시 평가 (시드 1 포함)
- 시드 1이 단일 기준(+0.1, 95% 구간 > 0)을 넘거나 앙상블에 기여하면 A26에 더해 **A27**로 만들어요.
- 그다음 이걸 실행해요(CPU 약 40분):
```bash
ssh komap 'cd ~/komap && setsid nohup bash scripts/eval_final_combo.sh A27 <A26의 26개 실험> x2_unetpp_bw_e120_r90_s1 > /dev/null 2>&1 < /dev/null &'
# 결과: runs/combo/A27/ (3_sie.json, 4_al3ni.json, 5_si_sie.txt, 6_si_sie_al.txt, alphas.txt, done)
```
- A26의 26개 실험: `runs/combo/A26/exps.txt`

### 3) sub5 만들기
- **구성 (A26 기준, 10/7 평가):**

  | 단계 | 내용 | 효과 (중첩 검증) |
  |---|---|---|
  | 기준 | 26개 평균 | 79.98 |
  | 1 | Eutectic Si 전용 모델 2개(`sie_x2_unetpp`, `sie_x2_cnx`) 평균을 비율 0.4로 섞기 | +0.106 |
  | 2 | Al3Ni 전용 모델(`al3ni_x2_unetpp`)을 비율 0.3으로 섞기 | +0.056 |
  | 3 | Si 분류기 (2단계까지 섞은 OOF 확률로 다시 학습) | +0.050 |
  | | **합계** | **약 80.19** |

- **해야 할 코드 작업:** 지금 `submit.py combine`의 `--sie-spec`은 **Eutectic Si(3번 상) 섞기만** 돼요. Al3Ni(1번 상) 섞기를 추가해야 해요.
  - 예: `--spec2 al3ni_x2_unetpp --spec2-cls 1 --spec2-alpha 0.3`
  - 섞는 함수는 `decision_rules.blend(P, q, alpha, c)`와 같아요. **순서는 Eutectic Si 섞기 → Al3Ni 섞기**예요.
- **Si 분류기:**
  ```
  python src/si_classifier.py fit --exps _ens_A26_sie_al --out runs/si_clf/sub5_model.pkl
  ```
  `_ens_A26_sie_al`은 이미 서버에 있어요. A27로 바꾸면 그에 맞춰 다시 만들어요.
- **평가 사진 예측:** 전용 모델 3개(`sie_x2_cnx`, `al3ni_x2_unetpp`, 그리고 sub4에서 이미 예측한 `sie_x2_unetpp`)와 `x2_unetpp_bw_e120_r90`을 `submissions/sub3_pred`에 추가로 예측해요(`submit.py predict --split test`).
- **검증:** `verify_cv_blend.py`를 두 번 섞기에 맞게 넓혀서 90장 제출 코드 재현을 확인해요. 형식 검사와 sub4와의 차이도 확인해요.
- **교체 기준:** 정직한 추정이 sub4보다 **+0.1 이상**일 때만 최선을 바꿔요. 지금은 +0.09라서 시드 1을 더한 결과로 판단해요.
- **sub5_full:** `full_x2_unetpp_bw_e120_r90`(시드 3개)을 `--full-mix 0.5`로 섞어요. 안전 점검(`check_full.py`) 통과가 조건이에요.

---

## 지난 이틀의 결과 요약 (10/6 저녁 ~ 10/7)
자세한 수치는 `runs/review8~10/`, `runs/rules/`, `runs/combo/`에 있어요. **보고서 문서(`reports/round8_10_results.md`)는 아직 쓰지 않았어요.** 아래 표를 바탕으로 쓰면 돼요.

| 단계 | 실험 | 결과 | 판단 |
|---|---|---|---|
| 8 | 2배 해상도 UNet++ **120에폭** (`x2_unetpp_bw_e120_r90`) | **80.03** (+0.32 [+0.06, +0.57]), Al3Ni +0.56, Eutectic Si +0.52, 675px +0.84 | ✅ 새 최고 단일 모델 |
| 8 | Eutectic Si 전용 모델 두 번째 (ConvNeXt, `sie_x2_cnx`) | 단독으로 섞기 +0.075, 첫 번째와 평균내서 섞기 **+0.115** | ✅ |
| 8 | Al3Ni 전용 모델 (`al3ni_x2_unetpp`) | +0.06 (5개 fold 모두 비율 0.3) | ✅ 약하게 채택 |
| 8 | 경계 부드러운 정답 (BLS, `--soft-edge`) | −0.15 / −0.18 | ❌ 경계 가중과 겹친 게 원인일 수 있어요 |
| 8 | 앙상블 (28개) | 자동 선택(중첩 검증) 80.005 → **80.173** | |
| 최종 조합 | A26 vs G8 (자동 선택 5개) | 둘 다 약 **80.19** | 고르는 과정이 없는 A26 채택 |
| 9 | 안쪽 오답 위주 학습 (`--hard-crop 0.5`) | −0.24 / −0.10 | ❌ |
| 9 | 시드 1, 90장 학습 | 진행 중 | |
| 10 | Al3Ni vs Eutectic Si **두 상 전용 모델** (`--pair-classes 1 3`) | A26 위에서 +0.088, 최선 조합 위에서는 −0.003 | ❌ 기존 전용 모델과 중복 |
| 10 | 가는 어두운 선 강조 입력 (`--feats`: black top-hat + ridge) | −0.23 / −0.24 | ❌ 사전학습 첫 층과 맞지 않은 것으로 추정 |

**분석으로 확인한 것 (보고서 인사이트 후보, 학습 없음)**
- **Eutectic Si 오답의 상대 상**
  - Al3Ni 54%: 70%는 경계, 30%는 조각째 혼동
  - Al 28%: 95%가 경계(두께 문제)
  - Primary Si 18%: 대부분 덩어리째 혼동 → Si 분류기의 영역
- **정답 Eutectic Si를 1px 두껍게 그리면 원래 정답과의 IoU가 62.3이에요.** 우리 모델은 59.6이라 경계 1px 수준의 오차와 같아요. 정답 Eutectic Si 픽셀의 63%가 경계 1px 띠 위에 있어요.
- **예측을 1px 넓히거나 줄이면 크게 손해예요**(−4.6 / −8.1). 지금 칠하는 두께가 이미 최적이에요(면적 비율 1.02).
- **방향 일관성(규칙/불규칙 공정 가설)은 Al3Ni와 Eutectic Si를 구분하지 못해요**(AUC 0.51). 우리 사진은 대부분 3원 공정(두 상 모두 불규칙)으로 보여요.
- **black top-hat과 ridge 필터는 픽셀 단위로는 정보를 더해요**(AUC 0.840 → 0.891). 하지만 CNN 입력으로 넣으면 손해였어요.
- **공통 패턴**
  - 효과 있음: 해상도, 더 오래 학습, 다르게 틀리는 모델을 앙상블에 섞기
  - 효과 없음: 학습 신호를 바꾸거나 더하기(손실 가중, 부드러운 정답, 오답 위주 학습, 증강, 필터 입력)
- **증강:** 오프라인(미리 만들어 재사용)보다 지금 쓰는 온라인(매번 새로)이 맞다는 데 사용자와 합의했어요. 근거는 다양성, 과적합, 120에폭 효과예요.

---

## 0. 새 노트북에서 시작하기

### 0-1. 코드 받기와 파이썬 환경
```
git clone https://github.com/hwarange/komap.git Komap      # 비공개 저장소라 GitHub 로그인 필요
cd Komap
# data/ 폴더(train·valid·test)는 깃에 없어요. 대회 자료에서 따로 복사해 넣으세요.
uv venv Komap --python 3.13
uv pip install --python Komap/Scripts/python.exe torch==2.14.1 torchvision==0.29.1
uv pip install --python Komap/Scripts/python.exe -r requirements.txt
```
- 노트북에는 **CPU용 torch**면 충분해요. 학습과 예측은 모두 서버에서 해요.
- uv에서 `--extra-index-url`은 `--index-url`보다 먼저 찾아봐요. GPU용 torch가 필요하면 cu126 인덱스 하나만 쓰세요.
- Windows 가상환경에서 pip 자체를 업그레이드하지 마세요.

### 0-2. 서버 SSH 연결
- 서버: `j-j15a106@70.12.130.104`, 작업 폴더 `~/komap` (호스트 이름 `jupyter04`)
- 노트북마다 SSH 키를 새로 만들어 등록해야 해요. 비밀번호 로그인은 안 돼요.
  1. `ssh-keygen -t ed25519`
  2. 공개키를 확인해요: `type $env:USERPROFILE\.ssh\id_ed25519.pub`
  3. 회사 JupyterHub → File → New → Terminal에서 실행해요: `echo '<공개키>' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys`
  4. `~/.ssh/config`에 추가해요:
     ```
     Host komap
       HostName 70.12.130.104
       User j-j15a106
       IdentityFile ~/.ssh/id_ed25519
     ```
  5. 확인: `ssh komap hostname` → `jupyter04`
- 10/6 노트북(DESKTOP-MR7IIH9)의 키는 등록돼 있어요.

### 0-3. Claude Code에서 이어서 하기
- **"handoff/HANDOFF_latest.md를 읽고 이어서 하자"**라고 하세요.
- `handoff/claude_memory/`의 메모리 파일을 `C:\Users\<사용자>\.claude\projects\<프로젝트 경로를 '-'로 바꾼 이름>\memory\`로 복사하면 같은 기억을 써요.

---

## 1. 과제 요약
- 알루미늄 합금 광학현미경(OM) 사진을 4개 상으로 분할해요: Al(갈색), Al3Ni(파랑), Primary Si(초록), Eutectic Si(흰색 `(204,204,204)`).
- 평가는 이미지별 4개 상 mIoU의 평균이에요(평가 10장, 그중 6장은 학습에 없는 새 합금). 참고 기준은 75%예요.
- 심사(100점): 과제이해 5, 데이터처리 15, **모델정확도 50**, 인사이트 10, 독창성 20
- 규칙:
  - 없는 상은 제외하거나 100%로 처리
  - 외부 사전학습 가중치 사용 가능
  - 검증 데이터 학습 사용 가능
  - **평가 이미지를 학습에 쓰지 않음**
  - 제출은 4색 RGB PNG

## 2. 성능 흐름 (90장 합금 단위 5-fold, TTA + 후처리)

| 단계 | 내용 | CV mIoU |
|---|---|---|
| 기준 | U-Net + ResNet34 | 78.31 |
| sub1 | 12개 실험 앙상블 | 79.73 |
| sub2 | 16개 + Si 분류기 | 79.96 |
| sub3a | 25개 + Si 분류기 | 80.02 |
| **sub4 (최선)** | 25개 + Eutectic Si 전용 모델 30% + Si 분류기 | **약 80.10** |
| 최고 단일 모델 | `x2_unetpp_bw_e120_r90` (2배 해상도 UNet++ 120에폭) | 80.03 |
| **sub5 후보** | 26개 + Eutectic Si 전용 2개 + Al3Ni 전용 + Si 분류기 | **약 80.19** |

- **상별 IoU (지금 최선 조합):** Al 93.2, Primary Si 89.1, Al3Ni 79.4, **Eutectic Si 59.5**
- **남은 벽:** 경계 라벨 애매함이에요.
  - 경계 1px 제외 시 88.0
  - 학습한 사진에서도 82점
  - 정답을 1px 두껍게 그리면 Eutectic Si 62.3

## 3. 환경
| | 노트북 | 서버 |
|---|---|---|
| GPU | 쓰지 않음 | **L40S × 4** (46GB), 128코어, 1TB RAM |
| 파이썬 | 가상환경 `Komap` (uv, 3.13) | `~/komap/.venv` (uv, 3.12, torch 2.14.1+cu126, smp 0.5.0) |

- **서버에만 있는 것:** 모델 가중치(`runs/cv`, `runs/full`), OOF 확률(`runs/oof/<실험>`), 앙상블 캐시(`runs/oof/_ens_*`), Si 분류기(`runs/si_clf/*.pkl`), 평가 사진 모델별 예측(`submissions/sub3_pred/parts`), 오답 지도(`runs/hardmaps`)
- **코드 동기화:** 지금은 `scp src/<파일> komap:komap/src/`로 해요. 보낸 뒤 `md5sum`으로 같은지 확인해요.
  - 서버는 깃 저장소가 아니에요.
  - **실행 중인 bash 스크립트 파일은 고치지 마세요.**
- **결과 가져오기:** `ssh komap 'cd ~/komap && tar czf - <json·txt·log 경로>' > x.tgz`. `npy`·`pt`는 빼요.
- **GPU 메모리:**
  - 2배 해상도 모델은 작업당 약 15GB(조각 1024)예요. **한 GPU에 2배 해상도 작업 3개는 위험해요(46GB).**
  - 다른 작업이 있을 때 추가하려면 조각 768(약 12GB)로 줄이세요.
- 서버 홈에 다른 사람 파일이 있으니 `~/komap`만 쓰세요. GPU 4장 사용은 허락받았어요.

## 4. 코드 구조 (`src/`)

| 파일 | 역할 |
|---|---|
| `data.py` | 로딩, 정규화, 배율 규칙, 증강. 옵션: `elastic`, `cutout`(IGNORE=255), `hard`(오답 위치 중심 자르기), `feats`(3채널 필터 입력: `to_input_feats`, `ridge_filter`) |
| `infer.py` | 슬라이딩 예측, 8방향 TTA, 원본 해상도 복원(정수가 아닌 배율은 bilinear), IoU, Primary Si 후처리. `model.input_feats`가 있으면 필터 채널을 계산해요 |
| `train.py` | 학습. 주요 옵션은 아래 목록과 같아요. **새 옵션을 끄면 이전과 출력이 같아요**(git 이전 버전과 비교 시험) |
| `launch.py` | 실험 목록을 fold 작업으로 펼쳐 GPU에 배치. `--gpus 3 3 2 --per-gpu 1`처럼 GPU별 칸 수를 조절할 수 있어요 |
| `review.py` | 실험 하나를 기준과 같은 사진끼리 비교 + 이상 징후 점검 |
| `ensemble.py` | OOF 앙상블 평가 (전체 평균, 탐욕적 선택 + 중첩 검증) |
| `si_classifier.py` | Si 덩어리 분류기 (`cv` / `fit`) |
| `decision_rules.py` | OOF 판정 규칙 실험: `cache`, `bias`, `sie`(`--cls`, `--pair a b`), `blendcache`(`--cls`, `--pair`). 모두 중첩 검증 |
| `oracle_components.py` | 덩어리 재판정 상한 (크기별) |
| `make_hardmaps.py` | 안쪽 오답 지도 |
| `check_full.py` | 90장 모델 안전 점검 (`train-score`, `agree`, `diff`) |
| `verify_cv_blend.py` | 섞기 포함 제출 코드 90장 검증 |
| `submit.py` | 제출 예측. 평가 사진은 모델별로 저장하고, `combine`에서 `--recipe`, `--full-mix`, `--sie-spec`/`--sie-alpha`, `--si-clf`를 써요. **Al3Ni 섞기 옵션은 아직 없음** |

**`train.py` 주요 옵션**
- 기존 옵션: `--bw`, `--rot rot90`, `--amp`, `--save-oof`, `--full`, `--train-frac`, `--enc-lr-mult`
- 7단계 이후: `--binary-class k`, `--pair-classes a b`, `--soft-edge`, `--elastic`, `--cutout`, `--mixup`, `--cutmix`, `--hard-crop`, `--feats`

**스크립트**
| 파일 | 역할 |
|---|---|
| `run_sub3.sh`, `run_sub4.sh` | 3·4차 제출본 자동 생성 |
| `eval_final_combo.sh` | 최종 조합 평가 (단계별 중첩 검증) |
| `review_round8.sh`, `review_round9.sh`, `run_round10.sh` | 실험이 끝날 때마다 자동 평가 |

실험 목록은 `experiments/round1~10.txt` 등이에요.

## 5. 문서
- `reports/idea_journey.md`: 전체 아이디어 과정 (7단계까지 반영. 8~10단계는 아직)
- `reports/round1_results.md` ~ `round6_7_results.md`: 단계별 결과 (**8~10단계 문서는 아직 없음**)
- `reports/research_notes.md`: 조사 노트 (9절: BLS, 냉각 속도 관계식, 질량 수지 분율, 탄성률 검증 데이터)
- `submissions/sub4/README.md`: 4차 제출본 구성·검증·재생성 방법

## 6. 보류 목록 추가 (8~10단계)
| 아이디어 | 결과 | 다시 시도할 조건 |
|---|---|---|
| 경계 부드러운 정답 (BLS) | −0.15 / −0.18 | 경계 가중을 끄고, Dice도 부드러운 정답용으로 바꿀 때 |
| 안쪽 오답 위주 학습 | −0.24 / −0.10 | 비율 0.2 이하, 여러 모델이 공통으로 틀리는 안정된 오답만 쓸 때 |
| 두 상 전용 모델 (Al3Ni vs Eutectic Si) | 단독 +0.09, 최선 조합 위에서 0 | 기존 전용 모델을 대체하는 구성을 비교할 때 |
| 필터 채널 입력 | −0.23 / −0.24 | 첫 층이 아니라 별도 가지로 넣을 때 |
| 방향 일관성 채널 | 측정에서 AUC 0.51 | 없음 |
| 손실함수·활성함수 변경 | 시험 안 함 (기대 낮음) | 앙상블 다양성 재료로만 고려 |

## 7. 알려진 함정
- **문자열 치환(파이썬 `str.replace`)으로 코드를 고치면, 맞는 문자열이 없을 때 아무 경고 없이 넘어가요.** 10/7에 `decision_rules.py --pair`가 무시되는 버그가 이것 때문에 생겼어요. 고친 뒤 `grep`으로 반영됐는지 확인하고, **실제 명령 경로까지 시험하세요.**
- 서버 작업을 기다릴 때 `while pgrep -f "..."`를 쓰지 마세요. 자기 자신과 일치해서 끝나지 않아요. `pkill -f`도 자기 ssh 세션을 죽여요. **done 파일이 생기는 걸 기준으로 기다리세요.**
- 서버 기본 셸은 `sh`예요. 배열은 bash 스크립트 안에서만 쓰세요. `cd ~/komap && A & B`처럼 쓰면 `&` 때문에 B가 다른 폴더에서 실행돼요.
- UNet++ + `tu-convnext_*`는 smp 0.5에서 생성 오류가 나요.
- 모듈 이름 `half`는 `nn.Module.half()`와 충돌해요.
- 정수가 아닌 배율에서 확률을 되돌릴 때 torch `area`는 확률을 흐려요(고쳐져 있음).
- numpy의 참/거짓 값은 JSON으로 저장할 수 없어요 → `bool()`로 바꾸세요.
- 고르는 과정이 있는 것(앙상블 구성, 섞는 비율)은 **중첩 검증**으로만 정직한 추정을 내세요.
- 중간 fold 결과로 판단하지 마세요.
- Windows git bash에서는 `/c/Users/...` 경로를 쓰세요.

## 8. 사용자 선호
- **항상 한국어로, 쉽게 설명해 주세요.** 어려운 개념은 그림이나 예시로 설명하면 좋아요(예: 픽셀 쪼개기, 경계 픽셀).
- 실험마다 **왜 골랐는지 + 결과 + 버그 점검 + 놓친 점**을 보고해요. 실험 하나가 끝날 때마다 바로 보고해요.
- 계획을 먼저 보여주고 사용자가 정해요. "다 해 보자", "진행해줘"라고 하면 바로 진행해요.
- 사용자는 아이디어를 많이 내요(전용 모델, 증강 데이터, 무작위 배치, 픽셀 쪼개기 등). **이미 시험한 것과 겹치는지 기록으로 확인하고, 근거와 함께 솔직하게 판단해 주세요.** 효과가 없을 것 같으면 이유를 설명하고, 시험할 가치가 있으면 비용과 함께 제안해요.
- 거짓 음성을 조심해요. 설정 문제로 실패했을 수 있으면 다른 설정으로 다시 시험하고, 앙상블 재료는 앙상블 기여로 판단해요.
- 학습은 서버(L40S)에서만 해요.
- **작업이 끝나면 알림을 보내 주세요**(PushNotification). 결과는 늦지 않게 바로 보고해요.
- 인수인계 파일은 사본 없이 이 파일 하나로, 요청할 때만 고쳐요.
- 깃으로 관리해요. 단계가 끝나거나 제출본을 만들면 커밋하고 올려요.
