# 1차 제출본 (sub1)

- **생성:** 2026-10-05 23:32 KST
- **파일:** `masks/<사진이름>_mask.png` 10개
  - 각 파일은 원본과 같은 크기의 RGB PNG이고, 대회 4색만 사용해요.
  - 4색: Al `(153,127,76)`, Al3Ni `(25,76,153)`, Primary Si `(76,178,76)`, Eutectic Si `(204,204,204)`
- **모델:** 12개 실험 × 5-fold = **60개 모델의 확률 평균**
  - 실험: `cnx_bw_e120`, `unetpp_bw_e120`, `cnx_bw`, `unetpp_bw`, `cnx_bw_c768`, `x2_c1024`, `unet_mit`, `unet_mitb4_bw`, `cnxS_bw`, `cnxB_bw`, `upp_r50_bw`, `cnx_s1`
  - 모델마다 8방향 TTA를 적용하고, 평균 확률에 Primary Si 후처리(작은 조각 제거 + 전체 0.5% 미만이면 제거)를 해요.
- **예상 성능:** 같은 코드를 학습+검증 90장에 "그 사진을 학습하지 않은 모델만" 써서 돌린 결과 **mIoU 79.73%**
  - `ensemble.py`의 OOF 평가와 일치해서, 제출 코드도 함께 검증됐어요.
  - 평가 세트 구성(저대비 60%)으로 환산하면 **80.51%**예요.
  - 평가 사진이 10장뿐이라 실제 점수는 ±2~3%p 달라질 수 있어요.
- **형식 검사:** 크기, RGB, 4색 모두 통과 (`summary_test.json`의 `problems: []`)
- **미리보기:** `preview_1.jpg`, `preview_2.jpg` (왼쪽부터 원본 | 예측 | 겹쳐 보기)
- **다시 만드는 방법 (서버 `~/komap`):**
  ```
  python src/submit.py predict --exps <위 12개> --split test --out submissions/sub1 --shard i --nshards 4   # GPU별 i=0..3
  python src/submit.py combine --split test --out submissions/sub1
  ```
- **확인 필요:** 대회 제출 파일명 규칙이 `<사진이름>_mask.png`(받은 정답 파일과 같은 규칙)인지 공식 안내로 확인할 것
