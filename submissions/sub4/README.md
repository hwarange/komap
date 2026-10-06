# 4차 제출본 (sub4)

- **생성:** 2026-10-06 14:34 KST
- **파일:** `masks/<사진이름>_mask.png` 10개. 원본과 같은 크기의 RGB PNG이고, 대회 4색만 써요.
- **같은 폴더 계열:** `sub4_full`은 여기에 90장 전체 학습 모델을 섞은 버전이에요. sub4와 픽셀 0.036%만 달라요.

## 구성
1. **4상 앙상블 (A 구성):** 25개 실험 × 5-fold = **125개 모델의 확률 평균**이에요. 모델마다 8방향 TTA를 써요.
   - 실험 목록: `unetpp_bw_e120_r90`, `x2_c1024_bw_r90`, `a_rot90`, `cnx_bw_e120`, `unetpp_bw_e120`, `m_hires`, `bw4_r90`, `a_lean`, `unetpp_bw`, `cnx_bw`, `cnx_bw_c768`, `cnx_bw_e120_r90`, `unet_mit_bw_r90`, `x2_c1024`, `unet_mit`, `upp_r50_bw`, `x2_unetpp_bw_r90`, `mn_sxt50_unetpp`, `cnxS_bw_e120`, `effv2s_unetpp`, `hrnet_unet`, `mn_r34_unetpp`, `unetpp_bw_r90`, `m_dinov3_lr`, `m_skel_w015`
2. **Eutectic Si 전용 모델 섞기 (7단계):** "Eutectic Si인가 아닌가"만 학습한 모델 `sie_x2_unetpp`(2배 해상도 UNet++, 5-fold)를 섞어요.
   - Eutectic Si 확률을 `0.7 × 앙상블 + 0.3 × 전용 모델`로 바꿔요.
   - 나머지 3상은 확률의 합이 1이 되도록 같은 비율로 줄여요.
3. **Primary Si 후처리:** 작은 조각을 지우고, Primary Si가 전체의 0.5% 미만이면 모두 지워요.
4. **Si 덩어리 분류기:** `runs/si_clf/sub4_model.pkl`이에요.
   - 2번에서 섞은 확률로 다시 학습했어요.
   - 확신이 80% 이상일 때만 덩어리 단위로 Primary/Eutectic을 다시 판정해요.
   - 평가 사진에서 바뀐 픽셀은 사진당 0.001~0.14%예요.

## 예상 성능 (90장 OOF, 합금 단위 5-fold)

| 단계 | CV mIoU | 근거 |
|---|---|---|
| 4상 앙상블만 | 79.94 | `ensemble.py`. 제출 코드 검증 79.936과 일치 |
| + 전용 모델 섞기 | 80.03 (정직한 추정) | 섞는 비율을 fold마다 따로 고른 중첩 검증. 비율은 0.2~0.4로 일관됐어요. 90장 전체로 고른 값은 80.068 |
| + Si 분류기 | **약 80.10** | 섞은 확률 위에서 분류기 효과 +0.073 [+0.02, +0.12] (중첩 검증) |

- 2차 제출본(sub2, 79.96)보다 **약 +0.14** 높아요. 미리 정한 교체 기준(+0.1)을 넘어요.
- 평가 사진이 10장뿐이라 실제 점수는 ±2~3%p 달라질 수 있어요.

## 검증
- **제출 코드 검증 (90장):** 각 사진을 "그 사진을 학습하지 않은 모델"(앙상블 25개 + 전용 모델 1개)로만 예측했어요.
  - 앙상블만 **79.936**, 섞은 뒤 **80.068**이에요.
  - 둘 다 평가 코드(`ensemble.py`, `decision_rules.py`)와 일치해요. 결과 파일은 `runs/sub4/verify_cv.txt`예요.
- **섞는 계산:** `submit.py`의 `blend_sie`가 `decision_rules.py`와 같은 결과를 내는 걸 가짜 데이터로 확인했어요.
- **형식 검사:** 크기, RGB, 4색 모두 통과했어요(`summary_test.json`의 `problems: []`).
- **다른 제출본과의 차이 (사진당 다른 픽셀 비율):** sub2와 평균 0.63%(최대 0.77%), sub3a와 0.44%예요.
- **미리보기:** `preview_1.jpg`, `preview_2.jpg` (왼쪽부터 원본 | 예측 | 겹쳐 보기)

## 다시 만드는 방법 (서버 `~/komap`)
평가 사진 예측은 모델별로 `submissions/sub3_pred/parts/models/`에 저장돼 있어요. 다시 예측하려면 아래 1번부터 하세요.
```
# 1) 평가 사진 예측 (GPU별 i=0..3)
python src/submit.py predict --exps <위 25개> --split test --out submissions/sub3_pred --shard i --nshards 4
python src/submit.py predict --exps sie_x2_unetpp --split test --out submissions/sub3_pred --shard i --nshards 4
# 2) Si 분류기 (섞은 OOF 확률로 학습)
python src/decision_rules.py cache --exps <위 25개> --out runs/oof/_ens_A_r6
python src/decision_rules.py blendcache --ens runs/oof/_ens_A_r6 --spec sie_x2_unetpp --alpha 0.3 --out runs/oof/_ens_A_sie03
python src/si_classifier.py fit --exps _ens_A_sie03 --out runs/si_clf/sub4_model.pkl
# 3) 합치기 (sub4_full은 --full-mix 0.5 추가)
python src/submit.py combine --split test --parts submissions/sub3_pred --out submissions/sub4 \
    --recipe <위 25개> --sie-spec sie_x2_unetpp --sie-alpha 0.3 --si-clf runs/si_clf/sub4_model.pkl --si-thr 0.8
```
전체 과정은 `scripts/run_sub3.sh`와 `scripts/run_sub4.sh`에 있어요.

- **확인 필요:** 대회 제출 파일명 규칙이 `<사진이름>_mask.png`인지 공식 안내에서 확인하세요.
