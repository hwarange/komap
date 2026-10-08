# sub6 후보 (2026-10-08)

평가 사진 10장의 마스크는 `masks/`에 있다. 공식 제출 전 후보이다.

## 구성

- `x2_mit_bw_r90`: 2배 해상도 U-Net + MiT-B2, 60에폭, 5-fold
- `x3_unetpp_bw_e120_r90`: 3배 해상도 UNet++ + ResNet34, 120에폭, 5-fold
- 각 실험의 5개 fold 확률을 평균한 뒤 실험끼리 1:1로 평균한다.
- 8방향 TTA + 기존 Primary Si 후처리. 전용 모델과 Si 분류기는 기본 후보에서 제외.
- 입력과 같은 크기, RGB PNG, 지정 4색 검사 통과. 사진마다 총 10개 fold 모델 사용.

## 성능과 한계

- 두 모델 고정 평균 OOF: 80.788.
- 모델 구성을 fold 밖에서 선택한 검증: 80.766 (선택 과정의 추정치).
- 제출 코드로 90장을 독립 예측: 80.7889, 기존 OOF와 평균 차이 0.0005.
- sub5 대비 평가 마스크 차이: 평균 1.5142%, 최대 1.9770%.
- 실제 평가 사진의 공식 점수는 모른다.

`../sub6_si`는 같은 두 모델에 Si 분류기(확신 기준 0.8)만 추가한 비교판이다.
분류기 효과는 +0.042 [−0.035, +0.115]로 불확실하여 기본 후보에 넣지 않았다.
분류기는 `runs/si_clf/sub6_pair.pkl`이며 가중치와 예측 확률은 서버에 있다.

## 서버에서 재생성

```bash
cd ~/komap
.venv/bin/python src/submit.py combine --split test \
  --parts submissions/sub3_pred --out submissions/sub6 \
  --recipe x2_mit_bw_r90 x3_unetpp_bw_e120_r90
```

Si 비교판은 위 명령에 `--si-clf runs/si_clf/sub6_pair.pkl --si-thr 0.8`을
추가하고 `--out submissions/sub6_si`로 바꾼다.

원본 생성 및 검증 로그: `runs/sub6/`. 14단계 추가 MiT 실험은 별도로 실행 중이며
이 후보에 아직 반영하지 않았다.
