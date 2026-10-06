# 3차 제출본 후보 (sub3a · sub3a_full · sub3b · sub3b_full)

- **생성:** 2026-10-06 11:18 KST. 지금 최선은 `sub4`예요(`submissions/sub4/README.md`). 이 폴더들은 비교와 기록용이에요.
- 네 후보 모두 평가 사진 예측을 공유해요(`submissions/sub3_pred`, 서버에만 있음). 섞는 방식만 달라요.

| 후보 | 앙상블 | 90장 모델 | Si 분류기 | 정직한 추정 | sub2와 다른 픽셀 |
|---|---|---|---|---|---|
| sub3a | A: 25개 실험 × 5-fold 평균 | 없음 | `runs/si_clf/sub3a_model.pkl` | 80.02 | 0.27% |
| sub3a_full | A | 5개 실험 1:1 (비중 약 10%) | 〃 | 80.02 + α | 0.28% |
| sub3b | B: 자동 선택 6개 (`x2_unetpp_bw_r90` ×2, `unet_mit`, `effv2s_unetpp`, `cnxS_bw_e120`, `m_hires`) | 없음 | `runs/si_clf/sub3b_model.pkl` | 약 80.06 | 0.83% |
| sub3b_full | B | B1만 1:1 (비중 약 17%) | 〃 | 약 80.06 + α | 0.86% |

- **검증:**
  - 제출 코드로 90장을 예측하니 79.936이에요(`ensemble.py` 79.937).
  - 새 코드로 sub2를 다시 만들면 다른 픽셀이 0.000%예요.
  - 형식 검사와 90장 모델 안전 점검 세 가지 모두 통과했어요(`runs/sub3/checks.txt`).
- **다시 만드는 방법:** `scripts/run_sub3.sh`
- 자세한 결과와 판단은 `reports/round6_7_results.md` 5절을 보세요.
