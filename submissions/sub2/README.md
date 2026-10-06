# 2차 제출본 (sub2), 현재 최선

- **생성:** 2026-10-06 04:05 KST
- **파일:** `masks/<사진이름>_mask.png` 10개. 각 파일은 원본과 같은 크기의 RGB PNG이고, 대회 4색만 써요.
- **모델:** 16개 실험 × 5-fold = **80개 모델의 확률 평균**
  - 실험 목록: `unetpp_bw_e120_r90`, `x2_c1024_bw_r90`, `a_rot90`, `cnx_bw_e120`, `unetpp_bw_e120`, `m_hires`, `bw4_r90`, `a_lean`, `unetpp_bw`, `cnx_bw`, `cnx_bw_c768`, `cnx_bw_e120_r90`, `unet_mit_bw_r90`, `x2_c1024`, `unet_mit`, `upp_r50_bw`
  - 모델마다 8방향 TTA를 쓰고, 평균 확률에 Primary Si 후처리(작은 조각 제거 + 전체 0.5% 미만이면 제거)를 적용해요.
- **Si 덩어리 분류기:** `runs/si_clf/sub2_model.pkl`
  - 예측된 Si 덩어리마다 Primary Si인지 Eutectic Si인지 다시 판정해요.
  - 분류기가 80% 이상 확신할 때만 바꿔요.
  - 평가 사진에서 바뀐 픽셀은 사진당 0.002~0.15%예요.
- **예상 성능 (90장 OOF)**

  | 구성 | CV mIoU | 평가 예상 |
  |---|---|---|
  | 앙상블만 | **79.86** | 80.60 |
  | + Si 분류기 | **79.96** | |

  - 제출 코드로 90장을 "그 사진을 학습하지 않은 모델만" 써서 돌린 결과(79.857)가 `ensemble.py` 평가와 일치해요.
  - Si 분류기의 효과(+0.10)는 중첩 교차검증 결과예요.
  - 평가 사진이 10장뿐이라 실제 점수는 ±2~3%p 달라질 수 있어요.
- **1차 제출본(sub1)과의 차이:** 사진당 픽셀의 0.4~0.9%가 달라요.
- **형식 검사:** 크기, RGB, 4색 모두 통과했어요 (`summary_test.json`의 `problems: []`).
- **미리보기:** `preview_1.jpg`, `preview_2.jpg` (왼쪽부터 원본 | 예측 | 겹쳐 보기)
- **다시 만드는 방법 (서버 `~/komap`)**
  ```
  python src/submit.py predict --exps <위 16개> --split test --out submissions/sub2 --shard i --nshards 4   # GPU별 i=0..3
  python src/submit.py combine --split test --out submissions/sub2 --si-clf runs/si_clf/sub2_model.pkl --si-thr 0.8
  ```
  Si 분류기를 다시 학습하려면: `python src/si_classifier.py fit --exps <위 16개> --out runs/si_clf/sub2_model.pkl`
- **확인 필요:** 대회 제출 파일명 규칙이 `<사진이름>_mask.png`인지 공식 안내에서 확인하세요.
