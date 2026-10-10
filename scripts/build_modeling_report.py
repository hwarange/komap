"""Populate the retained form using completed training and fixed inference reviews."""
import argparse
from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import tempfile
from zipfile import ZipFile

from docx import Document
from docx.shared import Inches
from lxml import etree as E

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'reports/modeling_document_work'
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
REL = 'http://schemas.openxmlformats.org/package/2006/relationships'
NS = {'w': W, 'r': R}


def element(tag, **attrs):
    node = E.Element('{' + W + '}' + tag)
    for name, value in attrs.items():
        node.set('{' + W + '}' + name, str(value))
    return node


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_final_strengthening(add, numeric_table, figure):
    """Add completed fixed diagnostics; only finalize after the fixed trials end."""
    folder = ROOT / 'runs/review26_final_strengthening'
    diagnostic_path = folder / 'diagnostics/results.json'
    completed = json.loads((folder / 'diagnostics/all.done').read_text(encoding='utf-8'))
    assert completed['outcome'] == 'frozen_final_pair_full90_diagnostics_complete'
    for record in completed['artifacts'] + [completed['script']]:
        assert sha(ROOT / record['path']) == record['sha256']
    diagnostic = json.loads(diagnostic_path.read_text(encoding='utf-8'))
    assert diagnostic['images'] == 90
    assert diagnostic['verification']['all90_per_class_IoUs_reproduced']
    groups = diagnostic['groups']
    full = groups['all90']
    assert abs(full['mIoU_pct_image_macro'] - 80.900653378248) < 1e-9
    image_macro = full['image_macro_classes']
    eut = image_macro['Eutectic Si']
    eut_pooled = full['regions']['all']['pixel_pooled']['classes']['Eutectic Si']
    fn_blue = 100 * eut_pooled['FN_predicted_as']['Al3Ni'] / eut_pooled['fn']
    fp_blue = 100 * eut_pooled['FP_true_class']['Al3Ni'] / eut_pooled['fp']
    boundary_error = 100 * full['regions']['boundary_2px']['pixel_pooled']['pixel_error_fraction']
    interior_error = 100 * full['regions']['interior']['pixel_pooled']['pixel_error_fraction']
    add('사. 전체 90장의 오류와 면적 분율 검증', 'sub')
    add('최종 조합의 저장된 OOF 확률에 동일한 1:1 혼합과 후처리를 적용하고, 90장 모두 기존 상별 IoU와 일치하는지 검증했다. 아래 정밀도·재현율은 사진별 값을 평균했으며, 면적 오차는 각 사진에서 예측 상 픽셀 비율과 정답 비율을 비교한 값이다.')
    numeric_table(['상', '정밀도 (%)', '재현율 (%)', '면적 MAE (%p)', '면적 편향 (%p)'], [
        [name, f"{value['precision_pct']:.2f}", f"{value['recall_pct']:.2f}",
         f"{value['area_MAE_pp']:.3f}", f"{value['area_bias_pp']:+.3f}"]
        for name, value in image_macro.items()], [2100, 1450, 1450, 1750, 1750])
    add('정밀도는 예측 중 정답 비율, 재현율은 실제 정답 중 찾아낸 비율이다. Primary Si는 88장, 그 밖의 상은 90장을 평균했다. 면적 MAE는 사진별 절대 오차의 평균, 편향은 예측−정답의 평균이며 상이 없는 사진도 포함한 90장 지표다.')
    add(f"공정 규소는 정밀도 {eut['precision_pct']:.2f}%보다 재현율 {eut['recall_pct']:.2f}%가 낮았다. 전체 픽셀을 합친 누락 공정 규소 {eut_pooled['fn']:,}개 중 {fn_blue:.2f}%가 Al3Ni로 분류됐고, 공정 규소 오탐 {eut_pooled['fp']:,}개 중 {fp_blue:.2f}%의 실제 상이 Al3Ni였다. 따라서 흰색을 일괄 늘리는 보정보다 두 상의 혼동을 함께 점검해야 한다.")
    add(f"정답 경계의 양쪽 픽셀을 표시하고 원본 좌표에서 2픽셀 확장한 영역의 오분류율은 {boundary_error:.2f}%, 나머지 내부는 {interior_error:.2f}%였다. 전체 오류의 {100 * full['fraction_of_all_errors_in_boundary']:.2f}%가 이 영역에 있었지만, 영역 자체도 전체 픽셀의 {100 * full['boundary_pixel_fraction']:.2f}%를 차지한다. 이는 경계 주변의 오류 집중을 나타내며 허용 오차 점수나 불규칙성의 인과 증거는 아니다.")
    low = groups['eut_area_low30']['image_macro_classes']['Eutectic Si']
    middle = groups['eut_area_middle30']['image_macro_classes']['Eutectic Si']
    high = groups['eut_area_high30']['image_macro_classes']['Eutectic Si']
    add(f"정답 공정 규소 면적 비율 순서로 30장씩 나누면 작은 집단의 IoU·재현율은 {low['IoU_pct']:.2f}%·{low['recall_pct']:.2f}%였고, 중간 집단은 {middle['IoU_pct']:.2f}%·{middle['recall_pct']:.2f}%, 큰 집단은 {high['IoU_pct']:.2f}%·{high['recall_pct']:.2f}%였다. 면적과 두께·해상도·합금이 함께 달라질 수 있어 면적만을 성능 원인으로 단정하지 않는다.")
    add(f"공정 규소 면적 MAE {eut['area_MAE_pp']:.3f}%p와 편향 {eut['area_bias_pp']:+.3f}%p는 사진별 상 분율을 보조 지표로 사용할 때의 오차를 수치로 보여 준다. 이는 반복 검토한 OOF 정답에서 얻은 값이며 test 10장의 정확도나 실제 수작업 절감 시간을 검증한 결과는 아니다.")
    trials = {}
    trial_evidence = {}
    for name, outcome in [('scale_trials', 'fixed_low_resolution_inference_scale_review_complete'),
                          ('finetune_plan', 'paired_six_epoch_continuation_complete')]:
        path = folder / name / 'results.json'
        trial_done = json.loads((path.parent / 'all.done').read_text(encoding='utf-8'))
        assert trial_done['outcome'] == outcome
        assert sha(path) == trial_done['results_sha256']
        assert sha(path.parent / 'plan.json') == trial_done['plan_sha256']
        assert trial_done['submission_changed'] is False
        trials[name] = json.loads(path.read_text(encoding='utf-8'))
        trial_evidence[name] = {file: sha(path.parent / file) for file in ('plan.json', 'results.json', 'all.done')}
    scale = trials['scale_trials']
    finetune = trials['finetune_plan']
    assert scale['unchanged64_verified'] and not scale['automatic_adoption']
    assert finetune['images'] == 18 and finetune['fold'] == 0
    assert not finetune['independent_validation']
    add('아. 오류 분석에 따른 마지막 보완 실험', 'sub')
    add('원본 폭이 675픽셀인 26장은 기존 MiT 확대 정책에서 실제 8배가 된다. 이 사진에만 추론 배율 6배와 4배를 사전에 고정해 비교했고, 같은 가중치로 8배도 다시 추론했다. UNet++ 확률과 나머지 64장의 결과는 그대로 두었다. 따라서 아래 전체 90장 값은 변경된 26장과 보존한 64장을 합친 점수다.')
    scale_arms = scale['scopes']['full90']['arms']
    affected_arms = scale['scopes']['affected26']['arms']
    numeric_table(['675픽셀 사진의 MiT 배율', '해당 26장 (%)', '전체 90장 (%)'], [
        [label, f"{affected_arms[key]['mIoU']:.4f}", f"{scale_arms[key]['mIoU']:.4f}"]
        for label, key in [('기존 저장 확률 8배', 'saved_baseline'), ('재추론 8배 대조군', 'fresh_scale8'),
                           ('고정 6배', 'cap6'), ('고정 4배', 'cap4')]], [4300, 2100, 2100])
    cap6_full = scale['scopes']['full90']['comparisons_vs_saved']['cap6']
    cap6_affected = scale['scopes']['affected26']['comparisons_vs_saved']['cap6']
    cap6_delta = list(cap6_affected['per_image_delta_pp'].values())
    assert len(cap6_delta) == 26
    lo, hi = cap6_full['group_bootstrap_ci95']
    improved, worsened = sum(d > 0 for d in cap6_delta), sum(d < 0 for d in cap6_delta)
    assert all(scale['diagnostic_review_flags']['cap6'].values())
    add(f"6배 후보는 사전에 정한 여섯 점검 항목을 통과했고 전체 평균도 {cap6_full['delta']:+.4f}%p 높았지만, 합금 그룹 bootstrap 95% 구간은 [{lo:+.4f}, {hi:+.4f}]%p로 0을 포함했다. 변경된 26장 중 {improved}장은 상승하고 {worsened}장은 하락했으며 변화의 중앙값은 {statistics.median(cap6_delta):+.4f}%p여서, 최종 단계에서는 이득의 일관성을 충분히 확인하지 못한 후보로 보존하고 기존 배율을 유지했다.")
    add('학습 가중치는 고정하고 추론 배율만 바꿨다. 8배 재추론으로 수치 차이를 확인했으며, 반복 검토한 OOF 결과이므로 독립 검증 성능으로 해석하지 않는다.')
    add('공정 규소와 Al3Ni의 혼동을 줄일 수 있는지도 고정 fold 0에서 확인했다. UNet++ 3배의 120 epoch 가중치에서 일반 crop 대조군과 두 상의 접경 crop을 35% 확률로 뽑는 후보를 각각 6 epoch 추가 학습했다. 학습률은 0.00003으로 맞추고 새 AdamW와 cosine 일정을 사용했다. 72장 학습 정답만으로 접경 지도를 만들었고, 기존 18장 평가 정답은 crop 선정에 쓰지 않았다.')
    scores = finetune['scores']
    numeric_table(['동일 18장과 MiT 4배 고정', 'mIoU (%)', '기존 대비 (%p)', '공정 Si IoU (%)'], [
        [label, f"{scores[key]['mIoU']:.4f}", f"{scores[key]['mIoU'] - scores['baseline_pair']['mIoU']:+.4f}",
         f"{scores[key]['per_class']['Eutectic Si']:.4f}"]
        for label, key in [('기존 UNet++ 3배', 'baseline_pair'), ('일반 crop 6 epoch 추가', 'control_pair'),
                           ('접경 crop 6 epoch 추가', 'interface_pair')]], [3650, 1450, 1700, 1700])
    sampling = finetune['comparisons']['interface_pair_vs_control_pair']
    add(f"접경 crop 후보와 같은 기간을 추가 학습한 일반 crop 대조군의 차이는 {sampling['delta']:+.4f}%p였다. 추가 학습 자체의 효과와 접경 선택 정책의 효과를 구분하기 위해 기존 가중치 및 일반 crop 대조군을 모두 비교했다.")
    single_gain = finetune['comparisons']['interface_UNet3_vs_baseline_UNet3']
    pair_gain = finetune['comparisons']['interface_pair_vs_baseline_pair']
    pair_lo, pair_hi = pair_gain['group_bootstrap_ci95']
    add(f"접경 crop은 UNet++ 단일 모델에서 {single_gain['improved_images']}/18장이 개선되고 평균 {single_gain['delta']:+.4f}%p였지만, 최종 두 모델을 결합하면 이득은 {pair_gain['delta']:+.4f}%p였다.")
    add(f"앙상블 개선의 합금 그룹 95% 구간은 [{pair_lo:+.4f}, {pair_hi:+.4f}]%p로 0을 포함했다. 단일 모델의 향상이 조합에서도 같은 폭으로 나타나지 않아 제출 가중치를 교체하지 않았다.")
    add('같은 초기 가중치와 시드로 시작해 학습 조각의 선택 정책을 비교했다. 추가 학습의 마지막 6번째 epoch만 평가했으며, 이 18장은 이전에도 검토한 자료다. 한 fold의 결과로 전체 90장 향상이나 새 시편의 안정성을 주장하지 않는다. 실제 관심 영역을 잘라 재추론하는 실험은 수행하지 않았다.')
    add('최종 제출은 전체 90장에서 검증한 MiT 4배와 UNet++ 3배의 1:1 조합과 기존 test 마스크 10장을 유지한다. 보완 실험은 성능이 낮은 상의 혼동과 배율의 영향을 좁혀 보는 근거로 남기고, 반복 fold나 시드 실험으로 자동 확대하지 않는다. 전체 진단과 고정 비교의 설정·확률·해시 및 결과는 review26_final_strengthening에 보관했다.')
    return {'diagnostic_result_sha256': sha(diagnostic_path),
            'diagnostic_completion_sha256': sha(folder / 'diagnostics/all.done'),
            'diagnostic_full90_reproduced': True, 'trials': trial_evidence,
            'additional_training_epochs_per_arm': 6, 'fixed_holdout_images': 18,
            'final_pair_mIoU': full['mIoU_pct_image_macro'],
            'selected_model_changed': False, 'submission_changed': False}


def write_final_pair(add, numeric_table, figure, valid_score, full_crop, final_strengthening=False):
    """Write the selected two-family report with explicit evaluation scopes."""
    reviews = {}
    evidence = {}
    for folder, count in [('review21_deeplab', 36), ('review22_pair', 90),
                          ('review23_unetpp4', 36), ('review24_single_scale', 18)]:
        path = ROOT / 'runs' / folder / 'results.json'
        done = json.loads((path.parent / 'all.done').read_text(encoding='utf-8'))
        assert sha(path) == done['results_sha256']
        value = json.loads(path.read_text(encoding='utf-8'))
        assert value['images'] == count
        reviews[folder] = value
        evidence[folder] = {'path': str(path.relative_to(ROOT)), 'sha256': sha(path), 'images': count}
    pair = reviews['review22_pair']
    all90 = pair['scopes']['full90']
    held = pair['scopes']['remaining54']
    recipe = 'x4_mit_x3_unetpp_pair'
    metrics = all90['arm_metrics'][recipe]
    comparison = all90['pair_vs_sub6']
    rows_path = ROOT / 'runs/review22_pair/per_image_rows.json'
    rows_document = json.loads(rows_path.read_text(encoding='utf-8'))
    rows = rows_document['rows'][recipe]
    assert len(rows) == 90
    recomputed = statistics.mean(statistics.mean(v for v in row['tta_pp'] if v is not None and math.isfinite(v)) for row in rows.values()) * 100
    assert abs(recomputed - metrics['cv_mIoU']) < 1e-8
    evidence['pair_per_image_sha256'] = sha(rows_path)
    evidence['pair_90_image_score_recomputed'] = True

    add('① 분석(EDA) 개요', 'heading')
    add('1-1. 분석 핵심 내용', 'sub')
    add('본 과제는 고탄성 알루미늄 합금의 현미경 이미지에서 Al, Al3Ni, Primary Si, Eutectic Si를 분할하는 것이다. 최종 모델은 4배 U-Net / MiT-B2와 3배 UNet++ / ResNet34의 클래스 확률을 1:1로 평균하는 앙상블로 정했다.')
    add(f"합금 단위로 분리한 전체 90장 OOF 교차검증 mIoU는 {metrics['cv_mIoU']:.4f}%다. 기존 sub6의 {all90['scores']['sub6']:.4f}%보다 {comparison['delta']:+.4f}%p 높다. 평균 성능을 근거로 이 두 모델을 선택했지만 개선 구간에는 0이 포함되므로 새 사진에서의 일관된 우위까지 입증한 것으로 해석하지 않는다.")
    add('1-2. 분석 배경', 'sub')
    add('얇은 조직은 축소 과정과 작은 경계 이동에 민감하고, 촬영 밝기와 원본 해상도도 사진마다 다르다. 조직 두께와 오류가 생기는 위치를 조사한 뒤 밝기 정규화, 입력 배율, 학습 조각 크기 및 모델 조합을 비교했다. 예측 마스크는 상별 면적 분율과 전문가 검토에 활용한다.')
    add('1-3. 분석 방법 및 분석 내용', 'sub')
    numeric_table(['데이터', '사진 수', '정답', '활용'], [
        ['train', '70', '있음', '초기 학습 및 교차검증'],
        ['validation', '20', '있음', '초기 고정분할 평가 및 교차검증'],
        ['test', '10', '없음', '최종 마스크 생성과 형식 확인']], [1800, 1000, 1200, 4500])
    add('파일명의 합금 번호를 그룹으로 사용해 같은 합금의 사진이 학습과 평가에 동시에 들어가지 않도록 했다. 최종 비교는 train과 validation의 정답이 있는 90장을 5개 fold로 나눈 OOF 결과다. 정답 없는 test는 학습이나 성능 계산에 사용하지 않았다.')
    add('원본 폭은 675·1096·1230·1278·1440픽셀이다. 초기 EDA에서 폭 675픽셀 사진의 일부 Al3Ni·Eutectic Si 두께가 약 1.9픽셀이고 기준 모델 오류의 79.1%가 정답 경계 1픽셀 이내에 있음을 확인했다. 이 결과를 확대 입력과 경계 가중 손실의 설계 근거로 삼았다.')
    figure('eda/figures/fig03_thickness.png', '그림 1  원본 EDA 그림 3 재사용  조직 두께에 따른 입력 설계 근거')

    add('② 데이터 전처리', 'heading')
    add('2-1. 전처리 방법', 'sub')
    add('RGBA로 저장된 흑백 사진을 1채널로 읽고 각 이미지의 하위 1%와 상위 99% 밝기를 기준으로 정규화했다. 학습과 추론에 동일하게 적용하며, 무작위 밝기 증강은 학습에만 사용한다. 마스크의 지정된 네 색은 클래스 번호 0~3으로 변환하고 밝기 정규화는 적용하지 않는다.')
    add('4배 MiT는 기준 폭 2560에 가장 가까운 정수 배율을 구한 뒤 2를 곱한다. 3배 UNet++는 기준 폭 3840의 정수 배율을 사용한다. 대부분 사진은 각각 4배·3배이고 폭 675픽셀에서는 8배·6배다. 이 명칭은 입력 정책을 요약한 것이며 모든 사진의 실제 배율이 같은 것은 아니다.')
    add('이미지 확대에는 선형 보간, 정답 마스크에는 최근접 보간을 사용한다. 확대는 원본에 없는 관측 정보를 만들지는 않지만 학습기가 보는 조직의 픽셀 두께와 주변 영역을 바꾼다. 원본 폭이 같은 사진이라도 물리적인 관찰 범위가 같은지는 확인되지 않았다.')
    add('2-2. 전처리 절차', 'sub')
    add('이미지와 마스크의 이름·크기·클래스 확인 → 흑백 변환 및 정규화 → 합금 번호 단위 fold 분리 → 배율 적용 및 무작위 조각 추출 → 증강과 학습 → 타일 예측 및 원본 좌표 복원 → 확률 평균과 후처리 → 지표 계산 순서로 처리했다.')
    add('학습에는 무작위 자르기, 크기 증강, 90도 회전·반전과 밝기 증강을 사용했다. 데이터 180개 파일과 분할 파일의 SHA256을 기록하고 동일한 입력을 재사용했다. 비교마다 학습에 보지 않은 해당 fold의 사진만 평가했다.')

    add('③ 모델링', 'heading')
    add('3-1. 배경/알고리즘 소개', 'sub')
    add('U-Net은 인코더의 특징과 디코더의 위치 정보를 결합해 분할 지도를 예측한다. MiT-B2 인코더를 사용하는 U-Net과 ResNet34 인코더를 사용하는 UNet++를 결합했다. MiT는 SegFormer와 같은 인코더 계열이지만 본 실험의 전체 구조는 SegFormer가 아니다. DeepLab 비교에는 DeepLabV3Plus / ResNet34를 사용했다.')
    add('4배 MiT의 crop1536은 대부분 사진에서 원본 약 384×384 영역을 본다. 초기 4배 crop1024의 약 256×256보다 넓은 맥락을 함께 볼 수 있다. 조각의 면적도 2.25배가 되므로 개선을 주변 맥락 하나의 효과로 단정하지 않는다. 5배에서 같은 crop을 쓰면 원본 영역은 다시 좁아진다.')
    add('본 접근의 차별점은 조직 두께·경계 오류를 바탕으로 입력 표현을 설계하고, 합금 단위 분할을 유지한 상태에서 배율과 crop 및 앙상블 기여를 확인한 데 있다. 기존 구조를 활용했으며 새로운 신경망 구조를 개발했다는 주장은 하지 않는다.')
    add('3-2. 모델링 방법', 'sub')
    numeric_table(['항목', '4배 MiT', '3배 UNet++'], [
        ['구조와 인코더', 'U-Net / MiT-B2', 'UNet++ / ResNet34'],
        ['학습 기간과 시드', '60 epoch / seed 0', '120 epoch / seed 0'],
        ['crop과 추론 tile', '1536 / 1536', '1536 / 1536'],
        ['기준 폭과 추가 배율', '2560 / 2', '3840 / 1'],
        ['검증과 제출', '5개 fold / 5개 가중치', '5개 fold / 5개 가중치']], [2300, 3100, 3100])
    add('각 fold에서 72장으로 학습하고 보지 않은 18장을 평가했다. ImageNet 초기화, AdamW, 교차엔트로피와 Dice 손실, 경계 가중 bw=2를 사용했다. 최종 MiT는 실제 batch 2·누적 2로 유효 batch 4를 유지했다. 모델별 정해진 마지막 epoch 가중치를 사용하고, 검증 점수가 가장 높은 epoch로 다시 고르지 않았다.')
    add('추론은 8방향 TTA와 overlap 128의 타일 예측을 적용한다. 각 모델 계열의 5개 fold 확률을 평균한 뒤 두 계열을 0.5씩 결합하는 soft voting이다. argmax로 클래스를 정한 후 기존 후처리를 적용하며, 개별 모델의 정답 색을 다수결로 합치는 방식이 아니다. 교차검증에서는 사진마다 자신을 학습하지 않은 fold 가중치만 사용한다.')
    add('밝기 증강 제거는 당시 기준보다 −1.03%p였고, MiT 2배의 60→120 epoch 연장은 전체 90장에서 +0.019%p에 그쳤다.')

    add('3-3. 성능', 'sub')
    add('가. 지표와 평가 범위', 'sub')
    add(f'초기 U-Net / ResNet34는 train 70장으로 학습한 validation 20장 평가에서 {valid_score:.4f}%를 기록했다. 이 결과는 초기 고정분할 기록이며 최종 조합의 성능이 아니다. 초기 체크포인트 선택에 validation을 사용했으므로 낙관적인 추정이 포함될 수 있다.')
    add('mIoU는 각 이미지에서 클래스별 교집합/합집합을 구해 평균한 다음 이미지들에 대해 다시 평균한다. 정답과 예측 모두에 없는 클래스는 해당 이미지 평균에서 제외한다. 아래 전체 90장, 일부 36장, 고정 18장 점수는 평가 범위가 다르므로 서로 빼서 개선 폭으로 쓰지 않는다.')
    add('나. 최종 두 모델의 전체 90장 검증', 'sub')
    numeric_table(['구성', '전체 90장 (%)', '나머지 54장 (%)'], [
        ['기존 sub6  MiT 2배 + UNet++ 3배', f"{all90['scores']['sub6']:.4f}", f"{held['scores']['sub6']:.4f}"],
        ['최종  MiT 4배 + UNet++ 3배', f"{all90['scores'][recipe]:.4f}", f"{held['scores'][recipe]:.4f}"]], [4400, 2050, 2050])
    lo, hi = comparison['group_bootstrap_ci95']
    add(f"최종 조합은 전체 {comparison['improved_images']}/90장에서 개선됐고 sub6 대비 {comparison['delta']:+.4f}%p였다. 합금 그룹 bootstrap 10,000회로 구한 개선의 95% 구간은 [{lo:+.4f}, {hi:+.4f}]%p다. 예비 36장을 제외한 나머지 54장에서는 {held['pair_vs_sub6']['delta']:+.4f}%p로 사실상 동률이었다. 이 54장도 이미 반복 검토했으므로 새로운 독립 검증으로 부르지 않는다.")
    numeric_table(['상', '최종 IoU (%)', 'sub6 대비 (%p)', '유효 사진 수'], [
        [name, f"{value:.4f}", f"{comparison['delta_per_class'][name]:+.4f}", str(metrics['class_valid_images'][name])]
        for name, value in metrics['IoU_per_class'].items()], [2800, 2000, 2100, 1600])
    add('Al은 가장 높은 IoU를 보였고 Eutectic Si는 가장 낮았다. 최종 조합은 공정 규소에서 +0.3031%p 높았지만 Al3Ni는 −0.0634%p였다. 상별 평균의 유효 사진 수가 다를 수 있어 이 표의 네 평균을 단순 평균한 수치를 전체 mIoU로 사용하지 않는다.')
    add('fold별 최종 점수는 ' + ', '.join(f"{k}번 {v:.4f}%" for k, v in metrics['mIoU_per_fold'].items()) + '다. sub6 대비 3개 fold는 상승하고 2개는 하락했다.')

    # This retained form flows inside a single table cell. Local spacing keeps
    # the comparison heading and the following evidence on the same page.
    add('')
    add('다. 배율과 crop 및 추가 모델 비교', 'sub')
    full = full_crop['scopes']['full90']
    crop_gain = full['comparisons']['candidate_vs_control_single']
    add(f"crop1024와 crop1536을 같은 seed 0·60 epoch·유효 batch 4·추론 tile1536으로 맞춘 전체 90장 비교에서, 4배 MiT 단일 모델의 crop1536 이득은 {crop_gain['delta']:+.4f}%p였다. 다섯 fold 모두 상승했다. 동일 비중 세 모델 혼합의 최고 관측값은 {full['scores']['crop1536_uniform']:.4f}%였지만 최종 두 모델보다 약 0.0162%p 높을 뿐이며 두 모델 구성을 최종 선택했다.")
    r21 = reviews['review21_deeplab']; r23 = reviews['review23_unetpp4']; r24 = reviews['review24_single_scale']
    numeric_table(['단일 모델 비교', '동일 사진 수', '기존 → 신규 mIoU (%)'], [
        ['UNet++ 3배 → 4배  각 60 epoch', '36', f"{r23['scores']['x3_unet60_single']:.4f} → {r23['scores']['x4_unet60_single']:.4f}"],
        ['MiT 4배 → 5배  각 60 epoch', '18', f"{r24['scores']['mit4_single']:.4f} → {r24['scores']['mit5_single']:.4f}"],
        ['DeepLab 2배 → 4배  각 60 epoch', '18', f"{r24['scores']['deeplab2_single']:.4f} → {r24['scores']['deeplab4_single']:.4f}"]], [4100, 1200, 3200])
    add('UNet++ 4배는 기존 fold 0·1의 36장, MiT 5배와 DeepLab 4배는 기존 fold 0의 train72/validation18 한 번만 평가했다. 반복 fold 검증을 추가하지 않았다. 위 단일 비교의 학습 기간은 같지만 확대가 원본 영역과 처리량도 바꾸므로 성능 저하 원인을 확대 자체로 확정하지 않는다.')
    numeric_table(['동일 18장의 앙상블', 'mIoU (%)'], [
        [label, f"{r24['scores'][key]:.4f}"] for label, key in [
            ('MiT 4배 + UNet++ 3배', 'mit4_unet3_pair'),
            ('MiT 4배 + UNet++ 4배', 'mit4_unet4_pair'),
            ('MiT 5배 + UNet++ 3배', 'mit5_unet3_pair'),
            ('MiT 4배 + UNet++ 3배 + DeepLab 4배', 'mit4_unet3_deeplab4_triple')]], [6200, 2300])
    add(f"쌍은 1:1, 세 모델은 각 1/3의 확률 평균이다. 이 앙상블 표에서 UNet++ 3배는 120 epoch, 4배는 60 epoch여서 배율과 학습 기간의 효과가 함께 포함된다. 앞선 동일 36장에서 DeepLab 2배를 sub6에 추가한 조합도 {r21['scores']['sub6']:.4f}%에서 {r21['scores']['deeplab_equal_three']:.4f}%로 낮아졌다. 시험한 추가 구성은 최종 조합에 넣지 않았다.")
    add('저장된 예측에서 불확실한 픽셀만 세 모델로 혼합하는 규칙도 전체 혼합보다 낮아 채택하지 않았다. 이는 실제 관심 영역을 잘라 다시 추론한 실험과 다르다. 시드 비교는 과거 36장 예비 실험에 한정되고 새 최종 조합의 전체 시드 안정성은 확인하지 않았다.')
    figure('reports/final_pair_oof_example.png', '그림 2  최종 조합의 OOF 오류 사례 A42  원본과 정답 및 예측과 오류 표시 순서  빨강은 오분류 픽셀이며 평균 사례를 대표하지 않음')

    add('라. 채택 판단과 한계', 'sub')
    add('최종 제출 모델은 MiT 4배 crop1536과 UNet++ 3배다. 동일한 전체 90장에서 기존 두 모델보다 관측 평균이 높고, 추가 확대 및 DeepLab 혼합에서 개선 근거를 얻지 못한 점을 반영했다. ' + ('당시 비교한 세 모델 혼합의 미세한 이득보다 두 계열 구성을 선택했다.' if final_strengthening else '전체 최고인 세 모델 혼합의 미세한 이득을 추구하기보다 두 계열 구성을 선택했다.') + ' 새 데이터에 대한 우위를 보장한다는 뜻은 아니다.')
    add('반복 검토한 OOF 결과에 따라 모델을 선택했으므로 선택 과정의 낙관성이 남는다. bootstrap 구간은 합금별 표본 변동을 요약하며 시드·분할·반복 선택의 불확실성을 모두 포함하지 않는다. 독립 일반화 성능은 새로운 합금 또는 시편의 정답 자료로 별도 확인해야 한다.')

    add('마. 상별 면적 분율을 이용한 활용', 'sub')
    application_path = ROOT / 'reports/phase_application_final_pair/phase_area_summary.json'
    application = json.loads(application_path.read_text(encoding='utf-8'))
    assert application['quantity'] == 'predicted_2d_area_fraction' and application['mask_count'] == 10
    assert application['all_masks_exact_four_color_valid']
    assert sha(ROOT / 'scripts/summarize_phase_masks.py') == application['script_sha256']
    for row in application['images']:
        assert sha(ROOT / 'submissions/final_pair_mit4_unet3/masks' / row['file']) == row['sha256']
        assert sum(row['class_pixel_counts'].values()) == row['total_pixels']
        assert abs(sum(row['predicted_area_percent'].values()) - 100) < 1e-8
    add('최종 조합의 test 10장 예측 마스크에서 상별 픽셀 수를 전체 픽셀 수로 나눠 2차원 면적 분율을 산출했다. 색상 검사·상별 집계·JSON 저장과 도표 생성을 구현했다. test 정답이 없으므로 이 값은 예측 마스크의 구성비이며 정확도 평가가 아니다.')
    figure('reports/phase_application_final_pair/phase_area_fractions.png', '그림 3  최종 MiT 4배와 UNet++ 3배 조합의 test 10장 예측 면적 분율')
    example = application['images'][0]; fractions = example['predicted_area_percent']
    add(f"예를 들어 {example['file'].removesuffix('_mask.png')}의 예측 면적 분율은 Al {fractions['Al']:.2f}%, Al3Ni {fractions['Al3Ni']:.2f}%, Primary Si {fractions['Primary Si']:.2f}%, Eutectic Si {fractions['Eutectic Si']:.2f}%다. 모든 사진에서 네 상의 픽셀 수 합이 전체 픽셀 수와 일치했다.")
    add('분석 담당자는 원본과 마스크 및 사진별 분율을 함께 보고 비교할 시편과 수동 검토 대상을 정할 수 있다. ' + ('정답이 있는 90장의 분율 측정 오차는 뒤의 오류 분석에서 확인했으며, 실제 수작업 절감 시간은 아직 평가하지 않았다.' if final_strengthening else '실제 수작업 절감 시간과 분율 측정 정확도는 아직 평가하지 않았다.') + ' 단면의 예측 면적 분율을 합금 조성·3차원 부피 분율 또는 탄성률로 직접 환산하지 않는다.')
    evidence['application_result_sha256'] = sha(application_path)
    evidence['application_figure_sha256'] = sha(ROOT / 'reports/phase_application_final_pair/phase_area_fractions.png')

    add('바. 제출 결과와 재현', 'sub')
    add('두 모델 계열의 5-fold 가중치 총 10개를 사용해 test RGB 마스크 10장을 생성했다. 원본 크기, 지정된 네 색, 파일 목록과 SHA256을 검증하고 ZIP으로 묶었다. 기존 sub6 산출물은 과거 비교 자료로 보존하고 최종 제출용 폴더를 별도로 구분했다.')
    add('학습·추론 소스, 환경 YAML, 실행 스크립트, 분할 ID, 설정·가중치·결과의 해시를 함께 보관한다. Python 3.12, PyTorch 2.14.1+cu126, segmentation_models_pytorch 0.5.0, timm 1.0.30을 사용했고, 사전학습 모델 revision과 SHA256을 고정했다. 새 환경의 전체 설치와 재학습 검증은 별도 수행하지 않았다.')
    add('근거는 review20_crop_full의 동일 조건 crop 비교, review22_pair의 전체 90장 최종 조합, review21_deeplab·review23_unetpp4의 36장 및 review24_single_scale의 18장 결과다. 최종 재현 안내는 delivery/final_pair_repro, test 마스크는 submissions/final_pair_mit4_unet3에 보관한다.')

    if final_strengthening:
        evidence['final_strengthening'] = write_final_strengthening(add, numeric_table, figure)
        evidence['new_training_started'] = True
        evidence['final_pair_mIoU'] = metrics['cv_mIoU']
        return evidence

    add('사. 오류 EDA와 후속 개선 방향', 'sub')
    eda_path = ROOT / 'runs/review25_final_pair_eda/results.json'
    eda = json.loads(eda_path.read_text(encoding='utf-8'))
    assert eda['groups']['all90']['n'] == 90
    for source in eda['sources']:
        assert sha(ROOT / source['path']) == source['sha256']
    area_groups = eda['quantiles']['frac_SiE']['groups']
    fine_groups = eda['quantiles']['SiE_fineness']['groups']
    assert all(g['n'] == 30 for g in [*area_groups, *fine_groups])
    area_iou = [g['pair_class_IoU_pct']['Eutectic Si'] for g in area_groups]
    fine_iou = [g['pair_class_IoU_pct']['Eutectic Si'] for g in fine_groups]
    add(f'최종 조합의 OOF 오류를 정답 조직 형태와 원본 밝기 지표에 연결해 추가 EDA를 수행했다. 공정 규소 면적이 적은 30장의 해당 상 IoU는 {area_iou[0]:.2f}%로 중간·많은 30장의 {area_iou[1]:.2f}%·{area_iou[2]:.2f}%보다 낮았다. 면적당 경계가 촘촘한 30장도 {fine_iou[2]:.2f}%로 성긴 30장의 {fine_iou[0]:.2f}%보다 낮았다. 이는 원인 확정이 아닌 반복 검토한 90장의 탐색적 연관성이다.')
    narrow = eda['groups']['W=675']
    contrast_rho = eda['correlations']['range_p99_p01']['Eutectic_IoU']['spearman_rho']
    add(f"폭 675픽셀의 {narrow['n']}장에서는 최종 조합이 sub6보다 전체 {narrow['delta_pp']:+.4f}%p, Al3Ni {narrow['class_delta_pp']['Al3Ni']:+.4f}%p였다. 반면 p99−p01 밝기 범위와 공정 규소 IoU의 Spearman 상관은 약 {contrast_rho:.3f}로 단순 관계가 없었다. 기존 p01>60 분류는 직접적인 저대비 측정이 아니라 촬영 스타일 지표이므로 밝기만을 주요 오류 원인으로 단정하지 않는다.")
    diagnostics_path = ROOT / 'runs/review25_final_pair_eda/case_diagnostics.json'
    diagnostics = json.loads(diagnostics_path.read_text(encoding='utf-8'))
    case = diagnostics['cases']['A42_step2_sample4']
    confusion = case['models']['final_pair']['confusion']['Eutectic Si']
    missed_primary_pct = confusion['FN_predicted_as']['Primary Si'] / confusion['fn'] * 100
    interior_recall = case['boundary_regions']['2']['interior']['Eutectic Si']['recall_pct']
    add(f'전체 점수 하락이 큰 2장과 공정 규소 IoU가 가장 낮은 2장도 별도로 조사했다. 그림 2의 A42에서는 공정 규소 누락 중 {missed_primary_pct:.1f}%가 Primary Si로 분류됐고, 정답 경계 주변 2픽셀을 제외한 내부의 공정 규소 재현율도 {interior_recall:.2f}%였다. 다른 사례에는 공정 규소의 과검출도 있어 흰색을 일괄 늘리는 보정은 적절하지 않다. 선택한 네 사례의 관찰을 전체 사진의 원인으로 일반화하지 않는다.')
    numeric_table(['확인할 오류', '추가로 볼 지표', '다음 검토'], [
        ['공정 규소의 작은 면적과 미세 조직', '두께별 누락·과검출과 경계 오차', '대표 오류 패치와 혼동행렬'],
        ['저해상도 사진의 Al3Ni 악화', '배율·원본 영역별 상 IoU', '주변 맥락과 확대 정책'],
        ['두 모델의 판단 불일치', '불일치 면적과 오류 포착률', '여백을 포함한 실제 부분 재추론']], [2800, 2900, 2800])
    add('후속 작업은 원본·정답·예측에서 가는 조직 누락, 다른 상으로의 혼동, 경계 위치·두께 오류를 구분하는 순서다. 정답 경계가 애매한 사례는 기록하고 평가 정답을 임의로 바꾸지 않는다. 물리적 배율 정보가 없으므로 정규화된 픽셀 두께를 마이크로미터로 해석하지 않는다.')
    add('실제 부분 재추론은 아직 수행하지 않았다. 시행한다면 예측의 불확실성과 모델 불일치만으로 영역을 정하고 주변 여백을 포함해 재추론한 뒤 원본 좌표에서 확률을 결합한다. 규칙을 고정하고 mIoU·누락·면적 분율 오차·총 처리 시간·재추론 면적을 함께 비교한다. 확대나 epoch를 일괄 증가시키는 추가 학습은 우선하지 않는다.')
    evidence['eda_result_sha256'] = sha(eda_path)
    evidence['case_diagnostics_sha256'] = sha(diagnostics_path)
    evidence['final_pair_oof_figure_sha256'] = sha(ROOT / 'reports/final_pair_oof_example.png')
    evidence['new_training_started'] = False
    evidence['final_pair_mIoU'] = metrics['cv_mIoU']
    return evidence


def write_final_validation(add, numeric_table, figure, valid_score, full_crop):
    """Report the highest full90 configuration after the complete fixed review."""
    folder = ROOT / 'runs/review27_finetune_full'
    result_path = folder / 'results.json'
    result = json.loads(result_path.read_text(encoding='utf-8'))
    done = json.loads((folder / 'all.done').read_text(encoding='utf-8'))
    assert done['outcome'] == 'five_fold_continuation_fixed_2x2_review_complete'
    for name, key in [('results.json', 'results_sha256'), ('per_image_rows.json', 'rows_sha256'),
                      ('plan.json', 'plan_sha256'), ('completed_jobs.json', 'completed_jobs_sha256')]:
        assert sha(folder / name) == done[key]
    assert result['images'] == 90 and result['folds'] == 5
    selected = result['highest_observed_full90_config']
    assert selected == done['highest_observed_full90_config'] == result['ranking'][0]['config']
    scopes = result['scopes']; full = scopes['full90']; metrics = full['scores'][selected]
    assert metrics['images'] == 90
    rows_doc = json.loads((folder / 'per_image_rows.json').read_text(encoding='utf-8'))
    rows = rows_doc[selected]
    assert len(rows) == 90
    recomputed = 100 * statistics.mean(statistics.mean(v for v in row['tta_pp'] if v is not None and math.isfinite(v)) for row in rows.values())
    assert abs(recomputed - metrics['mIoU']) < 1e-8
    configs = ['baseline_original', 'baseline_cap6', 'interface_original', 'interface_cap6']
    assert metrics['mIoU'] == max(full['scores'][key]['mIoU'] for key in configs)
    tuned = selected.startswith('interface'); capped = selected.endswith('cap6')
    labels = {'baseline_original': '기존 가중치와 기존 배율', 'baseline_cap6': '기존 가중치와 675픽셀 6배',
              'interface_original': '접경 추가학습과 기존 배율', 'interface_cap6': '접경 추가학습과 675픽셀 6배'}
    cfg = result['configs'][selected]
    assert cfg['family_weights'] == [0.5, 0.5] and cfg['tta'] == 8 and cfg['unchanged_postprocess']
    assert not result['confidence_intervals_are_adoption_gate']
    base = full['scores']['baseline_original']
    comparison = full['comparisons'][selected + '_vs_baseline_original']
    application_path = ROOT / 'reports/phase_application_final27/phase_area_summary.json'
    application = json.loads(application_path.read_text(encoding='utf-8'))
    assert application['quantity'] == 'predicted_2d_area_fraction' and application['mask_count'] == 10
    assert application['all_masks_exact_four_color_valid']
    assert sha(ROOT / 'scripts/summarize_phase_masks.py') == application['script_sha256']
    for row in application['images']:
        assert sha(ROOT / 'submissions/final27_selected/masks' / row['file']) == row['sha256']
        assert sum(row['class_pixel_counts'].values()) == row['total_pixels']
        assert abs(sum(row['predicted_area_percent'].values()) - 100) < 1e-8
    summary_path = ROOT / 'submissions/final27_selected/summary_test.json'
    test_summary = json.loads(summary_path.read_text(encoding='utf-8'))
    assert test_summary['n_images'] == 10 and test_summary['n_models'] == 10
    assert test_summary['models_per_image'] == [10]
    assert test_summary['models_per_exp'] == {'mit4': 5, 'unet3': 5}
    assert test_summary['mit_policy'] == ('cap6_width675' if capped else 'original')
    assert test_summary['unet_policy'] == ('interface6' if tuned else 'original')
    assert not test_summary['test_ground_truth_used']
    manifest_path = ROOT / 'delivery/final27_repro/manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    receipt_path = ROOT / 'submissions/final27_selected/run_receipt.json'
    test_receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    assert manifest['selected_recipe'] == test_receipt['selected_recipe'] == selected
    assert abs(manifest['selected_mIoU'] - metrics['mIoU']) < 1e-9
    assert test_receipt['status'] == 'complete' and len(test_receipt['masks']) == 10
    assert len(test_receipt['probability_outputs']) == 100
    assert sha(ROOT / 'submissions/final27_selected/final27_masks.zip') == test_receipt['zip_sha256']
    for record in test_receipt['masks']:
        assert sha(ROOT / 'submissions/final27_selected/masks' / record['name']) == record['sha256']

    add('① 분석(EDA) 개요', 'heading')
    add('1-1. 분석 핵심 내용', 'sub')
    add('고탄성 알루미늄 합금의 현미경 이미지에서 Al, Al3Ni, Primary Si, Eutectic Si를 분할했다. 최종 구조는 4배 U-Net / MiT-B2와 3배 UNet++ / ResNet34의 클래스 확률을 1:1로 평균하는 앙상블이다.')
    add(f"동일한 전체 90장 OOF에서 사전에 정한 네 구성을 비교한 결과, ‘{labels[selected]}’의 mIoU {metrics['mIoU']:.4f}%가 가장 높아 이를 최종 선택했다. 이전 기본 조합 {base['mIoU']:.4f}% 대비 {comparison['delta']:+.4f}%p다. 평균 점수를 선택 기준으로 삼고 신뢰구간과 사진별 변화는 결과의 불확실성을 설명하는 지표로 함께 제시한다.")
    add('1-2. 분석 배경', 'sub')
    add('얇은 조직은 축소와 작은 경계 이동에 민감하고, 사진마다 밝기·해상도가 다르다. 조직 두께와 오류 위치를 조사해 정규화, 확대 배율, 학습 조각 크기와 접경 영역의 표집을 설계했다. 자동 분할 결과는 상별 면적 분율을 요약하고 전문가가 검토할 시편을 정하는 데 활용한다.')
    add('1-3. 분석 방법 및 분석 내용', 'sub')
    numeric_table(['데이터', '사진 수', '정답', '활용'], [
        ['train', '70', '있음', '초기 학습 및 교차검증'], ['validation', '20', '있음', '초기 고정분할 및 교차검증'],
        ['test', '10', '없음', '최종 마스크 생성과 형식 확인']], [1800, 1000, 1200, 4500])
    add('정답이 있는 train·validation 90장을 합금 번호로 묶어 5개 fold로 분할했다. 각 fold는 학습 72장과 평가 18장이며, 같은 합금은 두 집합에 동시에 들어가지 않는다. OOF는 사진마다 자신을 학습하지 않은 가중치로 얻은 예측이다. test는 학습과 성능 계산에 사용하지 않았다.')
    add('원본 폭은 675·1096·1230·1278·1440픽셀이다. 초기 EDA에서 675픽셀 사진 일부의 Al3Ni·Eutectic Si 두께가 약 1.9픽셀이었고, 당시 기준 모델 오류의 79.1%가 정답 경계 1픽셀 이내였다. 확대 입력과 경계 가중 손실의 근거로 활용했으며, 이 초기 지표를 최종 모델의 진단값으로 사용하지 않는다.')
    figure('eda/figures/fig03_thickness.png', '그림 1  초기 EDA의 해상도별 조직 두께  원본 그림 3 재사용')

    add('② 데이터 전처리', 'heading')
    add('2-1. 전처리 방법', 'sub')
    add('흑백 사진을 1채널로 읽고 각 이미지의 밝기 하위 1%와 상위 99%를 기준으로 정규화했다. 학습·추론에 동일하게 적용하며, 무작위 밝기 증강은 학습에만 사용한다. 정답 마스크의 네 색은 클래스 번호 0~3으로 바꾸고 밝기 정규화는 적용하지 않는다.')
    add('MiT 학습 배율은 기준 폭 2560에 가장 가까운 정수 배율에 2를 곱하며, UNet++는 기준 폭 3840의 정수 배율을 사용한다. 대부분 사진은 각각 4배·3배이고 폭 675픽셀에서는 8배·6배다. 모델의 배율 명칭은 입력 정책을 요약한 것으로 모든 사진에서 실제 배율이 같지는 않다.')
    add('최종 추론에서는 ' + ('폭 675픽셀 사진의 MiT 배율만 8배에서 6배로 제한했다. 그 밖의 사진과 UNet++ 배율은 기존 정책을 유지했다.' if capped else 'MiT와 UNet++ 모두 학습 때 사용한 기존 배율 정책을 유지했다.') + ' 이미지 확대에는 선형 보간, 정답에는 최근접 보간을 사용한다. 확대는 원본에 없는 관측 정보를 만들지 않으며, 조직의 픽셀 두께와 crop이 보는 원본 범위를 바꾼다.')
    add('2-2. 전처리 절차', 'sub')
    add('파일·크기·클래스 확인 → 흑백 변환과 정규화 → 합금별 fold 분리 → 확대와 조각 추출 → 증강·학습 → 타일 추론과 원본 좌표 복원 → 확률 평균과 후처리 → 지표 계산 순서로 처리했다. 데이터 180개 파일과 분할 파일의 SHA256을 고정해 입력 변경 여부를 검사했다.')
    add('접경 추가학습은 해당 fold의 학습 정답 72장만 사용했다. Al3Ni와 Eutectic Si가 원본 2픽셀 이내에서 맞닿는 양쪽 영역의 지도를 만들고, 35% 확률로 그 영역 중심의 crop을 선택했다. 평가 18장의 정답이나 예측 오류 지도는 학습 조각 선정에 사용하지 않았다.')

    add('③ 모델링', 'heading')
    add('3-1. 배경/알고리즘 소개', 'sub')
    add('U-Net은 인코더 특징과 디코더의 위치 정보를 결합한다. MiT-B2 인코더를 쓰는 U-Net과 ResNet34 인코더를 쓰는 UNet++를 결합했다. MiT는 SegFormer와 같은 인코더 계열이지만 본 실험의 전체 구조는 SegFormer가 아니다. 두 계열의 확률을 결합해 상별 오차를 보완한다.')
    add('4배 MiT의 crop1536은 대부분 사진에서 원본 약 384×384 영역을 본다. 초기 crop1024의 약 256×256보다 주변 조직을 넓게 포함한다. crop 면적과 맥락이 함께 달라지므로 효과를 한 원인에만 돌리지 않는다. 차별점은 기존 구조에 조직 두께·상 간 혼동·원본 범위를 연결하고 동일 분할의 비교 실험으로 검증한 데 있다.')
    add('3-2. 모델링 방법', 'sub')
    numeric_table(['항목', 'MiT 계열', 'UNet++ 계열'], [
        ['구조와 인코더', 'U-Net / MiT-B2', 'UNet++ / ResNet34'],
        ['최종 학습 가중치', '60 epoch / seed 0', '120 epoch + 추가 6 epoch' if tuned else '120 epoch / seed 0'],
        ['crop / 추론 tile', '1536 / 1536', '1536 / 1536'],
        ['추론 배율 정책', '675픽셀만 6배' if capped else '기존 4배 정책', '기존 3배 정책'],
        ['제출 가중치 수', '5개 fold', '5개 fold']], [2300, 3100, 3100])
    add('ImageNet 초기화, AdamW, 교차엔트로피와 Dice 손실, 경계 가중 bw=2를 사용했다. MiT는 batch 2·누적 2로 유효 batch 4를 유지했다. 무작위 자르기·크기 증강·90도 회전·반전·밝기 증강을 사용했으며, 정해진 마지막 epoch 가중치를 평가했다.')
    add('접경 후보는 각 fold의 기존 UNet++ 120 epoch 마지막 가중치에서 6 epoch를 추가 학습했다. 학습률 0.00003, batch 4, seed 0, 새 AdamW와 cosine 일정을 사용했다. 옵티마이저 상태를 이어받은 연속 126 epoch 학습과는 다르다. 먼저 끝난 fold 0을 보존하고 fold 1~4만 동일 설정으로 실행했다.')
    add('제출 추론은 8방향 TTA, overlap 128의 타일 예측을 사용한다. 모델 계열별 5개 fold 확률을 평균한 뒤 두 계열을 0.5씩 결합하고, argmax와 기존 후처리를 적용한다. 전체 계열에 같은 가중치 정책을 적용하며 점수가 좋은 fold나 사진만 골라 섞지 않았다.')

    # Native Word ignores paragraph keep rules in the template's flowing cell.
    # Retain source-style spacing so the performance section starts together.
    for _ in range(4):
        add('')
    add('3-3. 성능', 'sub')
    add('가. 평가 지표와 선택 기준', 'sub')
    add(f'초기 U-Net / ResNet34의 train70 학습·validation20 평가는 {valid_score:.4f}%였다. 초기 가중치 선택에 validation을 사용했으므로 낙관성이 포함될 수 있으며, 아래 최종 90장 OOF 점수와 직접 빼서 개선 폭으로 쓰지 않는다.')
    add('mIoU는 이미지마다 클래스별 교집합/합집합을 평균한 뒤 사진들에 대해 다시 평균한다. 정답과 예측 모두에 없는 클래스는 해당 사진의 평균에서 제외한다. 모든 후보는 같은 float16 저장 확률을 float32로 읽고 같은 비중·후처리로 비교했다.')
    add('최종 선택 기준은 네 고정 구성 중 동일 90장의 평균 mIoU 최대값이다. 신뢰구간에 0이 포함되는지는 자동 제외 조건으로 쓰지 않았다. 순위는 이미 반복 검토한 자료에서 얻은 모델 선택 결과이며, 숨겨진 test 정답에서의 순위를 보장하지 않는다.')
    add('나. 전체 90장과 재사용 18장 및 나머지 72장', 'sub')
    numeric_table(['가중치와 추론 정책', '전체 90장 (%)', '나머지 72장 (%)', '재사용 18장 (%)'], [
        [(('최종 ' if key == selected else '') + labels[key]).replace('접경 추가학습과', '접경 학습과'), f"{full['scores'][key]['mIoU']:.4f}",
         f"{scopes['remaining72']['scores'][key]['mIoU']:.4f}", f"{scopes['reused_fold0_18']['scores'][key]['mIoU']:.4f}"]
        for key in configs], [3400, 1700, 1750, 1650])
    add('재사용 18장은 앞서 접경 후보를 검토한 fold 0이다. 나머지 72장은 fold 1~4에 같은 추가학습 정책을 확장한 결과다. 이 72장도 이전 모델 비교에서 사용했으므로 새로운 독립 데이터로 부르지 않는다. 배율 제한의 직접 대상은 675픽셀 26장이며 나머지 64장은 MiT 추론 배율이 그대로다.')
    lo, hi = comparison['group_bootstrap_ci95']
    add(f"최종 구성은 기본 조합보다 {comparison['delta']:+.4f}%p 높았고, {comparison['improved_images']}장은 개선·{comparison['worsened_images']}장은 하락·{comparison['tied_images']}장은 동률이었다. 합금 그룹 bootstrap 10,000회의 개선 95% 구간은 [{lo:+.4f}, {hi:+.4f}]%p다. 일부 사진의 큰 개선에 평균 이득이 집중됐고, 이 구간도 반복 모델 선택과 시드 변경의 불확실성까지 보정한 값은 아니므로 새 데이터에서의 안정적 개선을 주장하지 않는다.")
    add('최종 fold별 mIoU는 ' + ', '.join(f"{key}번 {value:.4f}%" for key, value in metrics['per_fold'].items()) + '다.')
    add('')
    add('')
    numeric_table(['상', '최종 IoU (%)', '기본 조합 대비 (%p)', '유효 사진 수'], [
        [name, f"{value:.4f}", f"{comparison['delta_per_class'][name]:+.4f}", str(metrics['class_valid_images'][name])]
        for name, value in metrics['class_IoU'].items()], [2500, 2000, 2400, 1600])
    add('유효 사진 수는 최종 상별 평균의 분모이며, 기본 대비 값은 두 구성의 IoU가 모두 정의되는 사진에서 계산했다. 이 표의 네 IoU를 단순 평균한 값을 전체 mIoU로 쓰지 않는다. 상별 이득과 손실을 함께 공개하고 전체 이미지 평균이 가장 높은 구성을 선택했다.')
    add('다. 추가학습 효과의 범위', 'sub')
    single = full['comparisons']['interface_UNet3_vs_baseline_UNet3']
    add(f"UNet++ 단일 모델의 추가학습 이득은 전체 90장에서 {single['delta']:+.4f}%p였다. 단일 모델 이득을 합산하지 않고 MiT와 확률을 평균한 네 앙상블 구성을 직접 평가했다.")
    add('접경 후보는 전체 90장에서 검증했지만, 같은 기간 일반 crop을 더 학습한 대조군은 fold 0의 18장에만 있다. 개선에는 접경 표집·6 epoch 추가·옵티마이저 재설정이 함께 포함되어 접경 표집만의 인과 효과로 해석하지 않는다. 이번 확장은 seed 0이다.')
    crop_gain = full_crop['scopes']['full90']['comparisons']['candidate_vs_control_single']
    add(f"앞선 동일 조건의 4배 MiT crop1024·1536 비교에서는 60 epoch·유효 batch 4·추론 tile1536을 맞췄고, crop1536 단일 모델이 전체 90장에서 {crop_gain['delta']:+.4f}%p 높았다. 반면 MiT 5배·DeepLab 4배는 고정 18장, UNet++ 4배는 36장에서 개선을 얻지 못했다. 이 일부 사진의 점수를 전체 90장과 혼합해 비교하지 않는다.")

    add('라. 개선 출발점이 된 기본 조합의 오류 분석', 'sub')
    diag_path = ROOT / 'runs/review26_final_strengthening/diagnostics/results.json'
    diag_done = json.loads((diag_path.parent / 'all.done').read_text(encoding='utf-8'))
    assert any(record['path'] == str(diag_path.relative_to(ROOT)).replace('\\', '/') and record['sha256'] == sha(diag_path) for record in diag_done['artifacts'])
    diag = json.loads(diag_path.read_text(encoding='utf-8')); old = diag['groups']['all90']
    add('아래 진단은 추가학습과 배율 제한 전의 MiT 4배·UNet++ 3배 기본 조합 80.9007%에 대한 것이다. 최종 구성의 오류율이나 면적 오차로 재사용하지 않는다. 원본·마스크·확률 해시와 90장 각각의 IoU를 재확인한 뒤 개선 가설을 세웠다.')
    numeric_table(['상', '정밀도 (%)', '재현율 (%)', '면적 MAE (%p)'], [
        [name, f"{value['precision_pct']:.2f}", f"{value['recall_pct']:.2f}", f"{value['area_MAE_pp']:.3f}"]
        for name, value in old['image_macro_classes'].items()], [2200, 2100, 2100, 2100])
    add('정밀도는 예측 중 정답 비율, 재현율은 정답 중 검출 비율이다. Primary Si는 88장, 나머지는 90장의 이미지 평균이며 면적 MAE는 상의 부재를 포함한 90장 평균이다. 공정 Si 누락 1,490,534픽셀의 48.73%가 Al3Ni로 예측됐고, 공정 Si 오탐 1,217,106픽셀의 62.76%는 실제 Al3Ni였다. 이 두 상의 접경을 추가학습 대상으로 삼았다.')
    add('기본 조합에서 정답 경계 양쪽을 원본 좌표의 2픽셀로 확장한 영역은 전체 픽셀의 57.35%였다. 이 영역의 오류율은 10.44%, 내부는 1.02%였고 전체 오류의 93.20%가 영역에 포함됐다. 경계 주변 오류 집중이라는 관측이며 조직 불규칙성의 인과 증거는 아니다.')
    figure('reports/final_pair_oof_example.png', '그림 2  기본 조합의 선택된 OOF 오류 사례 A42  원본과 정답 및 기본 예측과 오류 표시  최종 구성의 예측 그림이 아님')
    add('정답 공정 Si 면적이 작은 30장은 기본 조합 IoU 51.28%, 큰 30장은 66.05%였다. 면적·두께·해상도·합금이 함께 달라질 수 있다. 정규화한 픽셀 두께를 물리 단위로 해석하거나 평가 정답의 애매한 경계를 임의로 바꾸지 않았다.')

    add('마. 최종 마스크의 면적 분율과 활용', 'sub')
    add('선택한 구성의 10개 가중치로 test 마스크 10장을 새로 생성했다. 원본 크기와 지정 네 색을 확인하고 상별 픽셀 수를 전체 픽셀 수로 나눠 2차원 면적 분율을 산출했다. test 정답은 없으므로 아래 값은 예측 구성비이며 정확도 지표는 아니다.')
    figure('reports/phase_application_final27/phase_area_fractions.png', '그림 3  최종 선택 구성의 test 10장 예측 면적 분율')
    example = application['images'][0]; fractions = example['predicted_area_percent']
    add(f"{example['file'].removesuffix('_mask.png')}의 예측 분율은 Al {fractions['Al']:.2f}%, Al3Ni {fractions['Al3Ni']:.2f}%, Primary Si {fractions['Primary Si']:.2f}%, Eutectic Si {fractions['Eutectic Si']:.2f}%다. 모든 사진에서 네 상의 픽셀 수 합과 전체 픽셀 수가 일치했다.")
    add('분석 담당자는 원본·마스크·사진별 분율을 함께 보고 비교할 시편과 수동 검토 대상을 정할 수 있다. 실제 작업 시간 절감과 새 시편의 분율 정확도는 별도 평가해야 한다. 단면의 예측 면적 분율을 합금 조성·3차원 부피 분율·탄성률로 직접 환산하지 않는다.')
    add('바. 제출 결과와 재현', 'sub')
    add('최종 마스크 10장과 ZIP, 모델 계열별 가중치 5개씩 총 10개, 입력·설정·확률·결과의 SHA256을 기록했다. 학습·추론 코드, 환경 YAML, 실행 스크립트와 분할 ID를 제출용 묶음에 포함했다. 이전 기본 조합의 결과는 별도 폴더에 보존했다.')
    add('Python 3.12, PyTorch 2.14.1+cu126, segmentation_models_pytorch 0.5.0, timm 1.0.30을 사용했다. 사전학습 모델 revision과 해시를 고정했으며, 새 환경에서 전체 재학습을 다시 실행한 검증은 수행하지 않았다. 주요 근거는 review27_finetune_full이며, 재현 안내는 delivery/final27_repro, 최종 마스크는 submissions/final27_selected에 보관한다.')
    add(f"최종 선택은 {labels[selected]}이며 전체 90장 OOF mIoU는 {metrics['mIoU']:.4f}%다. 사전에 정한 네 후보 중 관측 점수가 가장 높은 구성으로 제출 파일을 만들었다. 실제 관심 영역을 잘라 다시 추론하는 실험은 수행하지 않았다. 이번 접경 추가학습과 최종 구성의 추가 시드 반복도 수행하지 않았다.")
    return {'final_validation': True, 'result_sha256': sha(result_path), 'completion_sha256': sha(folder / 'all.done'),
            'per_image_sha256': sha(folder / 'per_image_rows.json'), 'full90_score_recomputed': True,
            'selected_recipe': selected, 'selected_mIoU': metrics['mIoU'], 'selection_rule': result['selection_rule'],
            'confidence_intervals_are_adoption_gate': False, 'same_family_policy_all5folds': True,
            'interface_weights_selected': tuned, 'cap6_width675_selected': capped,
            'diagnostics_explicitly_original_baseline': True, 'baseline_diagnostics_sha256': sha(diag_path),
            'application_result_sha256': sha(application_path), 'application_figure_sha256': sha(ROOT / 'reports/phase_application_final27/phase_area_fractions.png'),
            'test_summary_sha256': sha(summary_path), 'test_receipt_sha256': sha(receipt_path),
            'manifest_sha256': sha(manifest_path), 'new_training_started': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--draft', action='store_true')
    parser.add_argument('--crop-review', action='store_true', help='Include completed crop and selective-blending pilots')
    crop_status = parser.add_mutually_exclusive_group()
    crop_status.add_argument('--full-crop-pending', action='store_true', help='Describe the authorized remaining-fold validation as pending')
    crop_status.add_argument('--full-crop-complete', action='store_true', help='Include verified completed five-fold crop validation')
    parser.add_argument('--rubric-review', action='store_true', help='Add verified practical outputs and sharpen the competition narrative')
    parser.add_argument('--development-plan', action='store_true', help='Add the proposed eutectic-silicon error analysis and regional reinference plan')
    parser.add_argument('--final-pair', action='store_true', help='Write the selected MiT4 and UNet++3 report without overwriting the prior rubric report')
    parser.add_argument('--final-strengthening', action='store_true', help='Write a separate final report with completed full90 diagnostics and fixed follow-up trials')
    parser.add_argument('--final-validation', action='store_true', help='Write the highest full90 four-configuration report after final27 test masks are verified')
    args = parser.parse_args()
    if args.full_crop_pending and not args.crop_review:
        parser.error('--full-crop-pending requires --crop-review')
    if args.full_crop_complete and (args.draft or not args.crop_review):
        parser.error('--full-crop-complete requires completed --crop-review evidence')
    if args.rubric_review and (args.draft or not args.crop_review):
        parser.error('--rubric-review requires completed --crop-review evidence')
    if args.development_plan and not args.rubric_review:
        parser.error('--development-plan requires --rubric-review')
    if args.final_pair and not (args.full_crop_complete and args.rubric_review and args.development_plan):
        parser.error('--final-pair requires --full-crop-complete --rubric-review --development-plan')
    if args.final_strengthening and not args.final_pair:
        parser.error('--final-strengthening requires --final-pair')
    if args.final_validation and (not args.final_pair or args.final_strengthening):
        parser.error('--final-validation requires --final-pair and excludes --final-strengthening')
    request = json.loads((WORK / 'request.json').read_text(encoding='utf-8'))
    results = ROOT / request['final_result']
    result = None
    weights_path = ROOT / 'runs/review17_weights/results.json'
    tiles_path = ROOT / 'runs/review17_tiles/results.json'
    weights = tiles = None
    crop = selective = full_crop = None
    crop_path = ROOT / 'runs/review18_crop/results.json'
    selective_path = ROOT / 'runs/review19_selective/crop/results.json'
    full_crop_path = ROOT / 'runs/review20_crop_full/results.json'
    full_crop_done_path = full_crop_path.parent / 'all.done'
    full_crop_rows_path = full_crop_path.parent / 'per_image_rows.json'
    if not args.draft:
        done = json.loads((ROOT / 'runs/review16_additional/all.done').read_text())
        assert done['outcome'] == 'three_seed0_jobs_full_cv_and_final_report_complete'
        result = json.loads(results.read_text(encoding='utf-8'))
        assert result['checked_folds'] == list(range(5)) and not result['further_training_scheduled']
        weights = json.loads(weights_path.read_text(encoding='utf-8'))
        tiles = json.loads(tiles_path.read_text(encoding='utf-8'))
        tile_done = json.loads((ROOT / 'runs/review17_tiles/all.done').read_text(encoding='utf-8'))
        assert not weights['training_started'] and not weights['submission_changed']
        assert not tiles['training'] and not tiles['submission_changed'] and not tile_done['training']
        assert tiles['baseline']['images'] == 90
        if args.crop_review:
            crop_done = json.loads((crop_path.parent / 'all.done').read_text())
            selective_done = json.loads((selective_path.parent / 'all.done').read_text())
            assert sha(crop_path) == crop_done['results_sha256']
            assert sha(selective_path) == selective_done['results_sha256']
            crop = json.loads(crop_path.read_text())
            selective = json.loads(selective_path.read_text())
            assert crop['images'] == selective['images'] == 36
            assert crop['folds'] == selective['folds'] == [0, 1]
            assert not crop['submission_changed'] and not selective['submission_changed']
        if args.full_crop_complete:
            full_crop_done = json.loads(full_crop_done_path.read_text(encoding='utf-8'))
            assert full_crop_done['outcome'] == 'matched_crop_five_fold_validation_complete'
            assert sha(full_crop_path) == full_crop_done['results_sha256']
            assert full_crop_done['epochs'] == 60 and full_crop_done['validation_images'] == 90
            assert not full_crop_done['further_training_scheduled'] and not full_crop_done['submission_changed']
            full_crop = json.loads(full_crop_path.read_text(encoding='utf-8'))
            full_crop_rows = json.loads(full_crop_rows_path.read_text(encoding='utf-8'))
            assert full_crop['folds'] == full_crop_rows['folds'] == list(range(5))
            assert full_crop['images'] == 90
            assert not full_crop['further_training_scheduled'] and not full_crop['submission_changed']
            assert full_crop['matched_design']['epochs_both'] == 60
            assert full_crop['matched_design']['seed_both'] == 0
            assert len(full_crop_rows['fold_of']) == 90
            scores = full_crop['scopes']['full90']['scores']
            assert len(scores) == len(full_crop_rows['rows']) == 9
            for recipe, rows_by_name in full_crop_rows['rows'].items():
                assert set(rows_by_name) == set(full_crop_rows['fold_of'])
                means = [statistics.mean(v for v in row['tta_pp'] if v is not None and math.isfinite(v))
                         for row in rows_by_name.values()]
                assert abs(statistics.mean(means) * 100 - scores[recipe]) < 1e-8
    with ZipFile(WORK / 'reference.docx') as z:
        parts = {name: z.read(name) for name in z.namelist()}
        infos = z.infolist()
    xml = E.fromstring(parts['word/document.xml'])
    original_section = E.tostring(xml.find('.//w:sectPr', NS))
    table = xml.find('.//w:tbl', NS)
    rows = table.findall('w:tr', NS)
    body = rows[9].find('w:tc', NS)
    templates = body.findall('w:p', NS)
    regular, heading, subsection = templates[7], templates[3], templates[5]

    def paragraph(text, role='body', size=None):
        sample = heading if role == 'heading' else subsection if role == 'sub' else regular
        node = deepcopy(sample)
        for child in list(node):
            if child.tag != '{' + W + '}pPr':
                node.remove(child)
        pp = node.find('w:pPr', NS)
        if pp is None:
            pp = element('pPr'); node.insert(0, pp)
        for tag in ['ind', 'spacing', 'keepNext', 'keepLines', 'widowControl', 'jc']:
            for child in pp.findall('w:' + tag, NS):
                pp.remove(child)
        pp.append(element('spacing', before=120 if role != 'body' else 0,
                          after=80, line=312, lineRule='auto'))
        pp.append(element('jc', val='left'))
        if text in ['③ 모델링', '3-1. 배경/알고리즘 소개']:
            pp.append(element('keepNext'))
        # Match the source's flowing table cell; Word can otherwise defer an
        # entire long body row when keep-with-next chains include nested tables.
        run = element('r'); rp = element('rPr')
        rp.append(element('rFonts', ascii='돋움', eastAsia='돋움', hAnsi='돋움'))
        rp.append(element('sz', val=size or (24 if role == 'heading' else 22)))
        rp.append(element('color', val='000000'))
        if role in ['heading', 'sub']:
            rp.append(element('b'))
        run.append(rp); text_node = element('t'); text_node.text = text
        text_node.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
        run.append(text_node); node.append(run)
        return node

    def set_cell(row, text):
        cell = rows[row].findall('w:tc', NS)[-1]
        for child in list(cell):
            if child.tag != '{' + W + '}tcPr':
                cell.remove(child)
        cell.append(paragraph(text))

    set_cell(2, '')
    set_cell(3, request['team_name'])
    set_cell(4, request['awards'])
    set_cell(5, 'PyTorch, segmentation_models_pytorch, timm, OpenCV, NumPy, scikit-learn, Pillow')
    summary = rows[7].find('w:tc', NS)
    for child in list(summary):
        if child.tag != '{' + W + '}tcPr':
            summary.remove(child)
    for text in ['◦ 아이디어 제안 배경',
                 '광학현미경 이미지의 Al, Al3Ni, Primary Si, Eutectic Si 자동 분할을 수행했다.',
                 '얇은 조직의 두께와 촬영 대비를 분석해 입력 배율과 학습 조각 크기를 설계했다.',
                 ('합금 단위 교차검증을 거쳐 4배 MiT와 3배 UNet++를 1:1로 결합했다.' if args.final_pair else '합금 단위 교차검증을 거쳐 2배 MiT와 3배 UNet++를 결합한 sub6을 채택했다.')]:
        summary.append(paragraph(text))
    for _ in range(4):
        summary.append(paragraph(''))
    for text in ['◦ 파급효과',
                 '예측 마스크를 상별 면적 분율로 요약해 현미경 조직 분석의 검토 자료로 제공한다.',
                 'test 10장의 분율 산출과 도표 생성을 구현해 결과의 활용 절차를 제시했다.' if args.rubric_review else '촬영 조건이 다른 사진의 정량 비교와 전문가 검토를 지원하는 활용안을 제시한다.',
                 '예측 오차를 함께 확인하며 소재의 조성이나 탄성률을 직접 측정한 값과 구분한다.']:
        summary.append(paragraph(text))
    for _ in range(7):
        summary.append(paragraph(''))
    for child in list(body):
        if child.tag != '{' + W + '}tcPr':
            body.remove(child)
    # The empty form's minimum-height body row must expand to the actual report.
    row_properties = rows[9].find('w:trPr', NS)
    if row_properties is not None:
        for tag in ['trHeight', 'cantSplit']:
            for node in row_properties.findall('w:' + tag, NS):
                row_properties.remove(node)
    vertical = body.find('w:tcPr/w:vAlign', NS)
    if vertical is not None:
        vertical.set('{' + W + '}val', 'top')

    def add(text, role='body'):
        body.append(paragraph(text, role))

    def numeric_table(headers, data, widths):
        t = element('tbl'); pr = element('tblPr')
        pr.append(element('tblW', w=sum(widths), type='dxa'))
        borders = element('tblBorders')
        for tag in ['top', 'left', 'bottom', 'right', 'insideH', 'insideV']:
            borders.append(element(tag, val='single', sz=4, color='D9D9D9'))
        pr.append(borders); margins = element('tblCellMar')
        for tag in ['top', 'bottom', 'left', 'right']:
            margins.append(element(tag, w=80, type='dxa'))
        pr.append(margins); t.append(pr)
        grid = element('tblGrid')
        for width in widths:
            grid.append(element('gridCol', w=width))
        t.append(grid)
        for index, values in enumerate([headers, *data]):
            tr = element('tr')
            if index == 0:
                tp = element('trPr'); tp.append(element('tblHeader')); tr.append(tp)
            for col, (value, width) in enumerate(zip(values, widths)):
                tc = element('tc'); tcp = element('tcPr')
                tcp.append(element('tcW', w=width, type='dxa'))
                tcp.append(element('vAlign', val='center'))
                if index == 0:
                    tcp.append(element('shd', fill='EFEFEF'))
                tc.append(tcp)
                para = paragraph(str(value), size=20)
                pp = para.find('w:pPr', NS)
                pp.find('w:spacing', NS).set('{' + W + '}after', '30')
                if col > 0:
                    pp.find('w:jc', NS).set('{' + W + '}val', 'center')
                if index == 0:
                    para.find('w:r/w:rPr', NS).append(element('b'))
                tc.append(para); tr.append(tc)
            t.append(tr)
        body.append(t); add('')

    rels = E.fromstring(parts['word/_rels/document.xml.rels'])
    image_number = 0

    def figure(path, caption):
        nonlocal image_number
        image_number += 1
        fragment = Document()
        picture_width = 5.15 if ((args.final_strengthening and path == 'reports/phase_application_final_pair/phase_area_fractions.png') or (args.final_validation and path == 'reports/phase_application_final27/phase_area_fractions.png')) else 5.6
        fragment.add_paragraph().add_run().add_picture(str(ROOT / path), width=Inches(picture_width))
        with tempfile.TemporaryDirectory() as temp:
            frag_path = Path(temp) / 'fragment.docx'; fragment.save(frag_path)
            with ZipFile(frag_path) as z:
                root = E.fromstring(z.read('word/document.xml'))
                para = root.find('w:body/w:p', NS)
                image_name = next(n for n in z.namelist() if n.startswith('word/media/'))
                media_name = f'word/media/modeling_report_{image_number}.png'
                parts[media_name] = z.read(image_name)
        relid = f'rIdReportImage{image_number}'
        blip = para.find('.//{http://schemas.openxmlformats.org/drawingml/2006/main}blip')
        blip.set('{' + R + '}embed', relid)
        props = para.find('.//{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}docPr')
        props.set('id', str(100 + image_number)); props.set('descr', caption)
        rel = E.SubElement(rels, '{' + REL + '}Relationship')
        rel.set('Id', relid); rel.set('Type', R + '/image')
        rel.set('Target', f'media/modeling_report_{image_number}.png')
        image_pp = para.find('w:pPr', NS)
        if image_pp is None:
            image_pp = element('pPr'); para.insert(0, image_pp)
        image_pp.append(element('keepNext'))
        body.append(para); add(caption, 'caption')

    final_pair_evidence = None
    if args.final_pair:
        with (ROOT / 'runs/baseline/analysis/per_image.csv').open(encoding='utf-8-sig') as f:
            validation_rows = list(csv.DictReader(f))
        assert len(validation_rows) == 20
        valid_score = statistics.mean(float(row['tta+pp']) for row in validation_rows)
        final_pair_evidence = (write_final_validation(add, numeric_table, figure, valid_score, full_crop) if args.final_validation else write_final_pair(add, numeric_table, figure, valid_score, full_crop, args.final_strengthening))
        oof_path = ROOT / ('runs/review27_finetune_full/per_image_rows.json' if args.final_validation else 'runs/review22_pair/per_image_rows.json')
        eutectic_mean = None
        application_path = ROOT / ('reports/phase_application_final27/phase_area_summary.json' if args.final_validation else 'reports/phase_application_final_pair/phase_area_summary.json')
    else:
        add('① 분석(EDA) 개요', 'heading')
        add('1-1. 분석 핵심 내용', 'sub')
        add('본 과제는 고탄성 알루미늄 합금의 현미경 이미지를 네 상으로 분할하는 것이다. 촬영 대비·해상도 차이와 얇은 Eutectic Si의 경계 손실에 대응해, 밝기 정규화·입력 배율·모델 조합을 비교 실험으로 검증했다.')
        if tiles:
            highest = (full_crop['scopes']['full90']['scores']['crop1536_uniform'] if full_crop
                       else tiles['tile1536_addition_vs_sub6']['full']['cv_mIoU'])
            add(f"현재 채택 모델은 2배 MiT와 3배 UNet++의 동일 비중 앙상블 sub6이며, 합금 단위 90장 교차검증 mIoU는 {tiles['baseline']['cv_mIoU']:.4f}%다. 후속 4배 모델 조합의 최고 관측값 {highest:.4f}%는 탐색 결과로 구분했다. 평균 점수뿐 아니라 합금별 변화와 추가 추론 비용을 확인해 채택 여부를 판단했다.")
        else:
            add('기준 조합은 2배 MiT와 3배 UNet++의 동일 비중 앙상블인 sub6이다. 4배 MiT seed 0의 전체 5-fold 검증과 고정 비중·추론 타일 비교 결과를 반영한다.')
        add('1-2. 분석 배경', 'sub')
        add('같은 소재의 상은 밝기만으로 구분하기 어렵고 얇은 경계는 작은 픽셀 이동에도 영향을 받는다. 따라서 평균 점수와 함께 상별·저대비 사진의 오차와 합금 간 성능 차이를 확인했다.')
        add('1-3. 분석 방법 및 분석 내용', 'sub')
        numeric_table(['데이터', '사진 수', '정답 마스크', '활용'], [
            ['train', '70', '있음', '초기 학습 및 교차검증'],
            ['validation', '20', '있음', '초기 고정분할 평가 및 교차검증'],
            ['test', '10', '없음', '최종 마스크 생성 및 형식 확인']], [1800, 1000, 1400, 4300])
        add('원본의 밝기 분위수·촬영 스타일·폭과 정답 마스크의 상별 조직 두께를 분석했다. 검증 분할에는 파일명의 합금 번호를 사용했다. 정답이 없는 test는 학습이나 성능 계산에 사용하지 않았다.')
        add('원본 폭은 675·1096·1230·1278·1440픽셀로 나뉜다. EDA에서 675픽셀 사진의 일부 Al3Ni·Eutectic Si 두께가 약 1.9픽셀로 확인되어, 사진 전체를 축소하는 접근보다 배율을 맞추고 타일로 처리하는 접근을 택했다. 기준 모델 오답의 79.1%가 정답 경계 1픽셀 이내에 있었다.')
        if args.rubric_review:
            add('촬영 대비 차이에 맞춰 이미지별 밝기 정규화와 학습 시 밝기 증강을 적용했다. 얇은 조직과 경계 오차에는 확대 입력과 경계 가중 손실로 대응했다. 같은 합금의 사진이 학습과 검증에 함께 들어가지 않도록 합금 번호 단위로 분리했다.')
        figure('eda/figures/fig03_thickness.png', '조직 두께 분석  EDA 원본 그림 3  얇은 구조 보존을 위한 해상도 설계의 근거')
        add('② 데이터 전처리', 'heading')
        add('2-1. 전처리 방법', 'sub')
        add('RGBA로 저장된 흑백 사진을 1채널로 읽고, 각 이미지의 하위 1%와 상위 99% 밝기를 기준으로 정규화했다. 이 정규화는 학습과 추론에 동일하게 적용했다. 촬영 조건을 다양하게 만드는 무작위 밝기 증강은 학습에만 사용했다. 지정된 네 색의 마스크는 클래스 번호 0~3으로 변환하며 밝기 정규화는 적용하지 않았다.')
        add('배율은 기준 폭에 가장 가까운 정수 배율로 정한다. 2배 MiT는 기준 폭 2560, 4배 MiT는 같은 기준 폭에 scale_multiplier=2를 적용한다. 따라서 대부분 사진은 각각 2배·4배이고, 폭 675픽셀 사진은 4배·8배다. 확대는 기존 픽셀의 표현을 바꾸며 원본에 없는 관측 정보를 생성하지 않는다.')
        add('2-2. 전처리 절차', 'sub')
        add('① 이미지와 마스크의 이름·크기·클래스 확인 → ② 흑백 변환 및 밝기 정규화 → ③ 합금 번호 단위 fold 분리 → ④ 배율을 반영한 무작위 조각 추출과 증강 → ⑤ 학습 → ⑥ 타일 예측 및 원본 크기 복원 → ⑦ 후처리와 지표 계산 순서로 처리했다.')
        add('학습에는 무작위 자르기, 크기 증강, 90도 회전·반전 및 밝기 증강을 사용했다. 이미지 보간은 선형, 정답 마스크 보간은 최근접 방식으로 클래스 경계를 처리한다. 이번 추가 검증은 기존 데이터 180개 파일의 SHA256 일치를 확인하고 동일한 fold 파일을 재사용했다.')
        # The source template uses one flowing table cell. Its keep-with-next
        # setting is not honored at this boundary by Word; retain heading space.
        if not args.rubric_review:
            add('')
            add('')
        else:
            add('')
            add('')
            add('')
        add('③ 모델링', 'heading')
        add('3-1. 배경/알고리즘 소개', 'sub')
        add('U-Net 계열은 인코더에서 얻은 특징과 디코더의 위치 정보를 결합해 원본 크기의 분할 지도를 예측한다. MiT-B2 인코더를 사용하는 U-Net과 ResNet34 인코더를 사용하는 UNet++를 비교하고, 서로 다른 오류를 보완하도록 두 모델의 클래스 확률을 평균했다.')
        add('4배 MiT의 원리는 모델의 파라미터를 늘리는 것이 아니라 더 확대된 조직을 보여주는 것이다. 기존 배율 비교에서 입력 조각은 2배와 4배 모두 1024×1024였다. 대부분 사진에서 2배는 원본 약 512×512 영역을, 4배는 약 256×256 영역을 본다. 얇은 조직의 표현은 커지지만 주변 맥락은 줄어들 수 있어 검증이 필요하다.')
        if args.rubric_review:
            add('본 접근의 차별점은 현미경 조직의 두께와 촬영 해상도를 근거로 입력 표현을 설계하고, 서로 다른 모델의 결합 효과를 같은 검증 조건에서 확인한 데 있다. 기존 U-Net·UNet++·MiT 구조를 활용했으며 새 신경망 구조를 개발한 것은 아니다. 후속 crop1536은 확대 배율을 유지하면서 주변 맥락을 넓히는 시도다. crop 면적도 2.25배 늘어나므로 같은 에폭이라도 처리 픽셀 양이 달라지며, 효과를 맥락 하나로 단정하지 않는다.')
        add('3-2. 모델링 방법', 'sub')
        add('초기 전처리와 증강을 검토한 뒤 입력 배율과 모델 구조를 개선하고, 학습 기간 및 앙상블 기여를 확인했다. 주요 결과는 다음과 같다.')
        numeric_table(['주요 실험', '결과', '해석'], [
            ['밝기 증강 제거', '−1.03%p', '해당 대조 실험에서 악화'],
            ['2배 MiT / 60 epoch', '80.171%', '최종 조합의 첫 모델'],
            ['3배 UNet++ / 120 epoch', '80.162%', '최종 조합의 둘째 모델'],
            ['두 모델 동일 비중 sub6', '80.7884%', '채택 조합'],
            ['2배 MiT / 120 epoch', '80.190%', '60 epoch 대비 이득 불확실']], [3700, 1800, 3000])
        add('밝기 증강 제거는 당시 기준 대비 변화이며, 나머지는 합금 단위 90장 CV 점수다. MiT의 120 epoch 연장은 약 +0.019%p에 그쳐 개선 근거가 부족했다.')
        if args.rubric_review:
            add('기존 4배 MiT는 ImageNet으로 초기화한 U-Net / MiT-B2를 seed 0, 60 epoch, batch 4, 사진당 8조각으로 학습했다. crop/tile 1024, width 2560, scale_multiplier 2를 사용했다. AdamW(lr 0.0003, weight decay 0.0001), 교차엔트로피와 Dice 손실, 경계 가중 bw=2를 적용했다. L40S 1개와 BF16으로 학습하고 마지막 epoch를 8방향 TTA 및 overlap 128로 평가했다.')
        else:
            numeric_table(['항목', '기존 4배 MiT 전체 검증 설정'], [
                ['모델과 학습', 'U-Net / MiT-B2, ImageNet 초기화; seed 0, 60 epoch, batch 4, 사진당 8조각'],
                ['입력과 배율', 'crop/tile 1024, width 2560, scale_multiplier 2'],
                ['최적화와 손실', 'AdamW, lr 0.0003, weight decay 0.0001; 교차엔트로피 + Dice, 경계 가중 bw=2'],
                ['평가와 환경', '마지막 60 epoch, 8방향 TTA, overlap 128; L40S 1개, BF16']], [2200, 6300])
        add('합금 번호 단위로 90장을 5개 fold에 나눠, 매번 72장으로 학습하고 해당 모델이 보지 않은 18장으로 평가한다. 같은 합금의 사진은 서로 다른 fold에 나누지 않았다. 앞선 시드 비교에서는 비용을 줄이기 위해 seed 1의 남은 세 fold를 생략하고 seed 0만 전체 검증했다.')
        add('4배 학습 후 주 후보는 2배 MiT·3배 UNet++·4배 MiT seed 0을 각각 1/3로 평균해 비교했다. 기존 sub6은 앞의 두 모델을 각각 1/2로 평균한다. 이후 재학습 없이 고정 비중 두 가지와 추론 타일 1536을 추가 비교했다. 이 후속 검토는 이미 관찰한 90장에 대한 탐색이며 독립 검증으로 간주하지 않았다.')
        add('3-3. 성능', 'sub')
        with (ROOT / 'runs/baseline/analysis/per_image.csv').open(encoding='utf-8-sig') as f:
            validation_rows = list(csv.DictReader(f))
        assert len(validation_rows) == 20
        valid_score = sum(float(row['tta+pp']) for row in validation_rows) / 20
        add('가. 초기 기준 모델의 고정분할 평가', 'sub')
        add(f'초기 U-Net + ResNet34 기준 모델은 train 70장으로 학습하고 validation 20장의 분할 마스크를 예측했다. 이미지별 네 상 IoU를 평균한 후 20장에 대해 평균한 TTA 및 후처리 성능은 {valid_score:.4f}%다. 이 값은 기준 모델의 검증 기록이며 최종 sub6의 고정분할 성능으로 표시하지 않는다. 기준 모델의 체크포인트 선택에 validation을 사용해 낙관적인 추정이 포함될 수 있다.')
        numeric_table(['모델 및 평가', '사진 수', 'mIoU (%)'], [
            ['train70 학습 기준 모델 / validation20', '20', f'{valid_score:.4f}'],
            ['기존 sub6 / 합금 단위 OOF CV', '90', f"{tiles['baseline']['cv_mIoU']:.4f}" if tiles else '80.7884 (기존 기록)']], [5300, 1200, 2000])
        add('mIoU는 각 이미지에서 클래스별 교집합/합집합을 계산해 평균한 뒤 이미지들에 대해 다시 평균한다. 정답과 예측 모두에 없는 클래스는 해당 이미지의 평균에서 제외한다. 고정분할 20장과 교차검증 90장은 평가 대상 및 학습 구성이 달라 두 점수의 차이를 개선 폭으로 해석하지 않는다.')
        add('나. 4배 MiT seed 0 전체 교차검증', 'sub')
        if result:
            comparisons = [('seed0_vs_MiT', '4배 seed0 단일'), ('add_seed0_equal_three', 'sub6에 4배 seed0 추가'),
                           ('replace_MiT_seed0', 'sub6 MiT를 4배로 교체')]
            data = []
            for key, label in comparisons:
                full, held = result[key]['full'], result[key]['nonpilot']
                data.append([label, f"{full['cv_mIoU']:.4f}", f"{full['delta']:+.4f}", f"{held['delta']:+.4f}"])
            numeric_table(['비교', '전체 mIoU (%)', '전체 변화 (%p)', '새 fold 변화 (%p)'], data, [3200, 1700, 1700, 1900])
            add('단일 모델의 변화는 2배 MiT 대비이며, 추가·교체 조합의 변화는 기존 sub6 대비다. 새 fold는 이번에 완성한 fold 2·3·4의 54장이다.')
            candidate = result['add_seed0_equal_three']; full, held = candidate['full'], candidate['nonpilot']
            lo, hi = full['group_bootstrap_ci95']; hlo, hhi = held['group_bootstrap_ci95']
            add(f'추가 조합의 개선 폭 95% 구간은 전체 [{lo:+.4f}, {hi:+.4f}]%p, 새 fold [{hlo:+.4f}, {hhi:+.4f}]%p다. fold 안에서 합금 그룹을 재표집한 bootstrap 10,000회로 계산했다. 평균 개선과 함께 구간에 0이 포함되는지도 판단한다.')
            numeric_table(['상 또는 조건', '전체 변화 (%p)', '새 fold 변화 (%p)'],
                [[name, f'{delta:+.4f}', f"{held['delta_per_class'][name]:+.4f}"]
                 for name, delta in full['delta_per_class'].items()] +
                [['저대비', f"{full['delta_low_contrast']:+.4f}", f"{held['delta_low_contrast']:+.4f}"]], [3900, 2300, 2300])
            fold_changes = ', '.join(f"fold {f} {d:+.4f}" for f, d in full['delta_per_fold'].items())
            add('초기 두 fold의 개선이 전체 평균 상승을 이끌었지만, 추가 fold 중 2와 4에서는 소폭 하락했다.' if args.rubric_review else f'추가 조합의 fold별 변화는 {fold_changes}%p였다. 초기 두 fold의 개선이 전체 평균 상승을 이끌었고, 새 fold 중 2와 4에서는 소폭 하락했다.')
        else:
            add('전체 seed 0 검증이 진행 중이다. 전체 및 새 fold의 점수, 상별·저대비 변화와 95% 구간은 학습 및 OOF 평가가 모두 완료된 결과로 반영한다.')
        add('다. 재학습 없는 고정 비중 및 추론 타일 비교', 'sub')
        if weights and tiles:
            a = weights['comparisons']['candidate_A_x4_0_1']
            b = weights['comparisons']['candidate_B_x4_0_2']
            tile = tiles['tile1536_addition_vs_sub6']
            latest_rows = []
            for label, candidate, held_key in [
                    ('고정 비중 A  0.4 / 0.1 / 0.5', a, 'nonpilot'),
                    ('고정 비중 B  0.3 / 0.2 / 0.5', b, 'nonpilot'),
                    ('4배 타일 1536  각 1/3', tile, 'previous_nonpilot')]:
                f, h = candidate['full'], candidate[held_key]
                latest_rows.append([label, f"{f['cv_mIoU']:.4f}", f"{f['delta']:+.4f}", f"{h['delta']:+.4f}"])
            numeric_table(['구성', '전체 mIoU (%)', 'sub6 대비 (%p)', '기존 54장 변화 (%p)'], latest_rows, [3200, 1700, 1700, 1900])
            add('비중은 2배 MiT / 4배 MiT / 3배 UNet++ 순서다. A와 B는 UNet++ 비중을 0.5로 유지한 고정 비교이며 격자 탐색은 하지 않았다. 모든 후보의 전체 개선 95% 구간은 0을 포함했다. 기존 54장은 앞선 실험의 fold 2·3·4로, 이번에는 미관측 검증 데이터가 아니다.')
            tf, th = tile['full'], tile['previous_nonpilot']
            tlo, thi = tf['group_bootstrap_ci95']
            tile_change = tiles['tile1536_addition_vs_tile1024_addition']['full']['delta']
            historical_prefix = '당시 ' if full_crop else ''
            add(f"추론 타일만 1024에서 1536으로 넓힌 추가 조합이 {historical_prefix}최고 {tf['cv_mIoU']:.4f}%를 기록했다. sub6 대비 95% 구간은 [{tlo:+.5f}, {thi:+.5f}]%p, 기존 54장 변화는 {th['delta']:+.5f}%p였다. 같은 추가 조합에서 타일 변경만의 이득은 {tile_change:+.6f}%p로 작았다. 타일 크기는 주변 맥락뿐 아니라 경계 위치와 확률 혼합도 바꾸므로 개선 원인을 하나로 확정하지 않았다.")
            add('같은 표 안의 비교는 동일한 저장 OOF와 평가 코드로 재계산했다.' if args.rubric_review else f"이 표는 원본 서버의 동일 기준 OOF로 재계산했다. sub6은 {tiles['baseline']['cv_mIoU']:.4f}%이며 앞선 RunPod 재계산 {result['baseline']['cv_mIoU']:.4f}%와의 차이는 {tiles['baseline_drift_pp']:+.6f}%p다. 같은 표 안에서 동일 기준과 비교했으며 계산 환경의 미세한 차이를 모델 개선으로 해석하지 않았다.")
        else:
            add('고정 비중 및 추론 타일 비교의 완료 결과를 최종본에 반영한다.')
        if not args.rubric_review:
            add('')
        add('라. 학습 조각과 선택적 혼합의 전체 검증' if full_crop else ('라. 시드와 학습 조각 및 선택적 혼합의 예비 검증' if crop else '라. 예비 결과와 시드 비교 범위'), 'sub')
        if args.rubric_review:
            add('시드 비교는 fold 0·1의 36장에 한정했다. 평균 차이는 작았지만 fold별 상승과 하락이 상쇄됐고 seed 1의 나머지 fold는 수행하지 않았다. 따라서 전체 데이터의 시드 안정성은 아직 확인하지 못했다.' + (' 이후 crop 비교는 seed 0으로 전체 90장 검증을 완료했다.' if full_crop else ' 아래 36장 점수는 전체 90장 점수와 직접 비교하지 않는다.'))
        else:
            add('기존 fold 0·1의 36장에서 4배 seed 0은 81.3048%, seed 1은 81.3063%로 평균 차이가 약 +0.0015%p였다. 그러나 fold별 차이 −0.3219%p와 +0.3248%p가 상쇄된 값이다. seed 1의 나머지 fold는 수행하지 않았으므로 전체 데이터에 대한 시드 안정성을 입증했다고 표현하지 않는다.')
            add('기존 두 시드 평균을 추가한 예비 조합의 +0.3069%p 개선은 이번 seed 0 단독 추가 조합과 구성이 다르다. 예비 점수나 36장의 점수를 이번 전체 90장 결과와 직접 빼서 비교하지 않는다.')
        if full_crop:
            f90 = full_crop['scopes']['full90']
            r54 = full_crop['scopes']['remaining54']
            fc = f90['comparisons']
            single = fc['candidate_vs_control_single']
            uniform = fc['candidate_uniform_vs_control_uniform']
            vs_sub6 = fc['candidate_uniform_vs_sub6']
            numeric_table(['구성', '전체 90장 (%)', '추가 54장 (%)'], [
                [label, f"{f90['scores'][key]:.4f}", f"{r54['scores'][key]:.4f}"]
                for label, key in [('기존 sub6', 'sub6'), ('crop1024 대조군 전체 혼합', 'matched_crop1024_uniform'),
                                   ('crop1536 후보 전체 혼합', 'crop1536_uniform'), ('crop1536 선택적 혼합', 'crop1536_selected')]],
                [4100, 2200, 2200])
            slo, shi = single['group_bootstrap_ci95']
            add(f"두 crop 조건 모두 seed 0, 60 epoch, 실제 batch 2·누적 2, 추론 tile1536을 적용하고 마지막 epoch를 평가했다. 예비 36장을 보존하고 fold 2·3·4의 54장을 추가했다. 단일 4배 MiT에서 crop1536은 crop1024보다 {single['delta']:+.4f}%p 높고 합금 그룹 bootstrap 95% 구간은 [{slo:+.4f}, {shi:+.4f}]%p였다. 5개 fold 모두 상승했고 공정 규소는 {single['delta_per_class']['Eutectic Si']:+.4f}%p였다.")
            ulo, uhi = uniform['group_bootstrap_ci95']
            blo, bhi = vs_sub6['group_bootstrap_ci95']
            add(f"세 모델을 각 1/3로 섞으면 후보의 이득은 대조군 대비 {uniform['delta']:+.4f}%p, sub6 대비 {vs_sub6['delta']:+.4f}%p로 줄었다. 각각의 95% 구간 [{ulo:+.4f}, {uhi:+.4f}]%p와 [{blo:+.4f}, {bhi:+.4f}]%p는 0을 포함한다. 추가 54장에서 sub6 대비 {r54['comparisons']['candidate_uniform_vs_sub6']['delta']:+.4f}%p여서 예비 36장의 개선은 그대로 재현되지 않았다.")
            add(f"선택적 혼합은 sub6의 상위 두 클래스 확률 차이 0.15 미만에서만 세 모델 평균을 적용했다. 전체 혼합 대비 {fc['candidate_selected_vs_uniform']['delta']:+.4f}%p로 채택하지 않았다. 저장된 확률의 혼합이며 실제 부분 재추론은 아니다. 반복 검토한 90장의 내부 비교로, 독립 검증이나 시드 안정성의 증거로 해석하지 않는다.")
        elif crop and selective:
            comparisons = crop['comparisons']
            matched = comparisons['crop1536_addition_vs_matched_crop1024_addition']
            historical = comparisons['crop1536_addition_vs_original_tile1536_addition']
            selective_delta = selective['comparisons']['crop1536_selected_vs_uniform']
            numeric_table(['동일한 fold 0·1의 36장 비교', 'mIoU (%)'], [
                ['기존 sub6', f"{crop['sub6_reference']['cv_mIoU']:.4f}"],
                ['새 crop1024 대조군 전체 혼합', f"{matched['base_cv_mIoU']:.4f}"],
                ['새 crop1536 후보 전체 혼합', f"{matched['cv_mIoU']:.4f}"],
                ['crop1536 선택적 혼합', f"{selective_delta['cv_mIoU']:.4f}"]], [6100, 2400])
            lo, hi = matched['group_bootstrap_ci95']
            hlo, hhi = historical['group_bootstrap_ci95']
            add(f"학습 조각은 crop1024와 crop1536으로 비교했다. 두 조건 모두 seed 0, 60 epoch, 실제 batch 2·누적 2, 추론 tile1536을 적용하고 마지막 epoch를 평가했다. crop1536의 전체 혼합은 대조군보다 {matched['delta']:+.4f}%p 높았으며 합금 그룹 bootstrap 95% 구간은 [{lo:+.4f}, {hi:+.4f}]%p였다. 다만 기존 4배 모델의 같은 36장 혼합 {historical['base_cv_mIoU']:.4f}% 대비 이득은 {historical['delta']:+.4f}%p이고 구간 [{hlo:+.4f}, {hhi:+.4f}]%p에 0이 포함됐다.")
            add(f"선택적 혼합은 sub6의 상위 두 클래스 확률 차이가 0.15 미만인 픽셀에만 세 모델의 동일 비중 평균을 적용했다. 전체 혼합보다 {selective_delta['delta']:.4f}%p 낮아 이번 규칙은 채택하지 않았다. 저장된 예측을 섞은 평가이며 관심 영역을 잘라 확대 재추론하지는 않았다. 반복 검토한 36장의 예비 비교이므로 전체 90장 성능이나 독립 검증으로 해석하지 않는다.")
        if (WORK / 'sub6_oof_examples.png').exists():
            figure('reports/modeling_document_work/sub6_oof_examples.png', '그림 2  채택 모델 sub6의 OOF 예측  90장 중 mIoU 중앙값 부근과 최저 점수 사례를 고정 기준으로 선정')
        else:
            figure('runs/baseline/analysis/worst_A71_step2_sample1.png', '그림 2  초기 기준 모델의 validation 오류 사례  최신 후보의 결과와 구분')
        add('마. 채택 판단과 한계', 'sub')
        if result:
            conclusion = f"완료된 전체 90장 검증의 최고 관측 점수는 {highest:.4f}%였지만, 현재 채택 모델은 sub6({tiles['baseline']['cv_mIoU']:.4f}%)이다. 평균 상승만으로 교체하지 않고 합금별 변동과 개선 구간 및 추가 추론 비용을 함께 판단했다."
            if full_crop:
                conclusion += f" 새 최고값은 종전 최고보다 {fc['candidate_uniform_vs_old_uniform']['delta']:+.4f}%p 높지만, 추가 54장과 개선 구간이 일관된 우위를 뒷받침하지 않아 sub6을 유지했다. crop 변경은 주변 맥락과 학습 픽셀 수도 바꾸므로 단일 원인의 효과로 단정하지 않는다."
            elif args.full_crop_pending:
                conclusion += ' crop1536과 동일 조건 crop1024의 남은 fold 2·3·4를 각각 60 epoch로 학습해 전체 5-fold 검증을 진행하며, 완료 후 채택 여부를 다시 판단한다. 전체 결과는 아직 확정하지 않았다.'
            elif crop:
                conclusion += ' 학습 조각 및 선택적 혼합의 예비 비교만으로 채택 모델을 변경하지 않았다.'
            else:
                conclusion += ' 추가 학습 없이 이번 검토를 종료했다.'
            add(conclusion)
            checks = result['add_seed0_equal_three']['predeclared_checks']
            if not args.rubric_review:
                add(f'앞선 4배 학습 후 동일 비중 조합은 사전 판정 기준 {sum(checks.values())}/{len(checks)}개를 충족했다. 전체 +0.05%p 이상, 전체·새 fold의 구간 하한 양수, 새 fold 각각 개선, 저대비 및 Eutectic Si 악화 없음, 다른 상의 하락 0.1%p 이내를 함께 확인했다.')
        else:
            add('최종 채택 판단은 진행 중인 전체 검증 결과로 확정한다. 보고서용 임시 배치에서는 결과와 결론을 미리 확정하지 않는다.')
        add('추가 앙상블은 모델 수와 구조별 비중도 바꾸므로 개선 전체를 확대율만의 효과로 설명할 수 없다. 확대 모델은 처리할 타일과 예측 비용이 늘어난다. 결과는 상별 면적 분율과 형태 정량화에 활용할 수 있으며, 탄성률이나 조성의 직접 측정값과 구분한다.')
        if args.rubric_review:
            add('바. 상별 면적 분율을 이용한 활용', 'sub')
            add('예측 마스크에서 각 상의 픽셀 수를 전체 픽셀 수로 나눈 뒤 100을 곱해 2차원 면적 분율을 산출했다. test 10장의 기존 sub6 마스크를 입력으로 색상 검사, 상별 픽셀 집계, JSON 저장 및 도표 생성을 구현했다. 아래 값은 예측 마스크의 구성비이며 test 정답을 이용한 정확도 검증 결과가 아니다.')
            application_path = ROOT / 'reports/phase_application/phase_area_summary.json'
            application = json.loads(application_path.read_text(encoding='utf-8'))
            assert application['quantity'] == 'predicted_2d_area_fraction' and application['mask_count'] == 10
            assert application['all_masks_exact_four_color_valid']
            assert sha(ROOT / 'scripts/summarize_phase_masks.py') == application['script_sha256']
            assert len(application['images']) == 10
            for row in application['images']:
                assert sha(ROOT / 'submissions/sub6/masks' / row['file']) == row['sha256']
                assert sum(row['class_pixel_counts'].values()) == row['total_pixels']
                assert abs(sum(row['predicted_area_percent'].values()) - 100) < 1e-8
            figure('reports/phase_application/phase_area_fractions.png', '그림 3  sub6의 test 10장 예측 마스크에서 산출한 2차원 상별 면적 분율')
            example = application['images'][0]
            fractions = example['predicted_area_percent']
            add(f"예를 들어 {example['file'].removesuffix('_mask.png')}의 예측 면적 분율은 Al {fractions['Al']:.2f}%, Al3Ni {fractions['Al3Ni']:.2f}%, Primary Si {fractions['Primary Si']:.2f}%, Eutectic Si {fractions['Eutectic Si']:.2f}%다. 모든 사진에서 네 상의 픽셀 수 합이 전체 픽셀 수와 일치하는지 확인했다.")
            add('분석 담당자는 사진별 분율과 원본·마스크를 함께 검토해 비교할 시편을 정할 수 있다. 향후 경계 오차가 큰 사진의 수동 검토와 촬영 조건별 분율 오차 검증을 추가할 수 있다. 실제 수작업 절감 시간과 분율 측정 정확도는 아직 평가하지 않았다. 단면의 예측 면적 분율을 합금 조성, 3차원 부피 분율 또는 탄성률로 환산하지 않는다.')
        add('사. 제출 결과와 재현' if args.rubric_review else '바. 제출 결과와 재현', 'sub')
        add('sub6은 2배 MiT와 3배 UNet++의 각 5-fold 확률 평균을 1:1로 결합한다. 생성한 test 10장의 RGB 마스크는 원본 크기와 지정된 네 색을 검증했다. sub6의 90장 교차검증과 초기 기준 모델의 validation 20장 점수는 구분하며, sub6 자체의 고정분할 20장 성능은 별도 측정하지 않았다.')
        add('학습 설정·코드·체크포인트·fold·결과를 보관한다. Python 3.12, PyTorch 2.14.1+cu126, segmentation_models_pytorch 0.5.0, timm 1.0.30을 사용했다. MiT 초기화는 고정한 smp-hub/mit_b2.imagenet revision을 재사용하고 SHA256을 확인했다.')
        if crop:
            if args.rubric_review:
                add('환경 YAML, 실행 스크립트와 면적 분율 분석 코드를 포함했다. 원본 환경에서 입력·가중치·핵심 버전 검사를 통과했으며, 새 환경 설치와 전체 GPU 재생성 검증은 아직 수행하지 않았다.')
            else:
                add('자체 학습 체크포인트와 결과·코드·재현 기록을 보관한다. 주요 근거는 baseline/analysis, review13~17, seed0_full_results.json, seed0_artifacts.json 및 학습 조각 비교 review18_crop, 선택적 혼합 review19_selective 결과다. 전체 검증은 완료된 fold 0·1을 보존하고 남은 fold 2·3·4만 동일 설정으로 추가하며, 실행 기록은 review20_crop_full에 보관한다.' if args.full_crop_pending else '자체 학습 체크포인트와 결과·코드·재현 기록을 보관한다. 주요 근거는 baseline/analysis, review13~17, seed0_full_results.json, seed0_artifacts.json, review18_crop 및 review19_selective 결과다.')
        else:
            add('추가 학습은 seed 0의 세 fold로 종료했고, 이어서 재학습 없는 비중·타일 비교만 완료했다. 자체 학습 체크포인트와 결과·코드·재현 기록을 보관하며 새로운 실험을 예약하지 않았다. 주요 근거는 baseline/analysis, review13~17 결과, seed0_full_results.json 및 seed0_artifacts.json이다.')
        if args.development_plan:
            oof_path = ROOT / 'runs/review17_tiles/per_image_rows.json'
            oof = json.loads(oof_path.read_text(encoding='utf-8'))['sub6']
            assert len(oof) == 90
            eutectic_iou = [row['tta_pp'][3] for row in oof.values()]
            assert all(value is not None and math.isfinite(value) for value in eutectic_iou)
            eutectic_mean = statistics.mean(eutectic_iou) * 100
            # Source table flow needs local spacing to keep this final heading
            # from being isolated at the foot of the preceding page.
            add('')
            add('')
            add('아. 공정 규소 분할의 후속 개선 계획', 'sub')
            development_intro = ('전체 검증에서 유지하기로 한 sub6을 기준으로 공정 규소(Eutectic Si)의 오류 원인을 분석한다.' if full_crop else '후속 개발은 전체 검증으로 기준 모델을 확정한 뒤 공정 규소(Eutectic Si)의 오류 원인을 분석하는 순서로 진행한다.')
            add(f'{development_intro} 현재 sub6의 90장 OOF에서 공정 규소의 이미지별 IoU 평균은 {eutectic_mean:.2f}%다. 불규칙성과 얇은 두께의 영향을 구분해 확인하며, 아래 내용은 아직 수행하지 않은 개선 계획이다.')
            numeric_table(['오류 유형', '확인할 지표', '검토할 방향'], [
                ['가는 조직 누락', '두께별 재현율·누락 면적', '미세 구조와 원본 해상도'],
                ['다른 상으로 혼동', '혼동행렬·영역별 오분류', '주변 조직과 전체 형태'],
                ['경계 위치·두께 오차', '경계 IoU·두께 편향', '경계 주변 정밀 보정']], [2400, 3100, 3000])
            add('원본·정답·OOF 예측을 함께 검토해 오류를 위 유형으로 나눈다. 경계가 애매한 사례는 별도로 기록하며 평가 정답을 임의로 수정하지 않는다. 우선순위는 오류의 빈도와 면적, 기존 모델로 보정할 가능성에 따라 정한다.')
            add('다음 후보는 실제 부분 재추론이다. 전체 예측에서 불확실하거나 모델 간 판단이 다른 관심 영역을 선정하고 주변 여백을 포함해 기존 학습 모델로 다시 예측한다. 입력 배율·crop·여백·결합 규칙을 고정하고, 원본 좌표로 복원한 확률을 겹침 영역에서 결합해 이음새를 점검한다. 영역 선택에는 정답을 쓰지 않으며, 사후에 선택 면적과 전체 오류 중 포착한 비율을 계산해 확신하면서 틀린 영역의 누락을 확인한다.')
            add('앞선 선택적 혼합은 저장된 확률을 섞은 실험으로 실제 부분 재추론과 구분한다. 기존의 오답 조각 집중 학습·경계 정답 완화·Skeleton 손실은 시험한 설정에서 개선 근거가 부족했으므로, 동일 조건의 반복은 우선하지 않는다.')
            add('작은 탐색에서 가능성을 확인한 구성만 규칙을 고정해 전체 평가로 확장한다. 같은 사진에서 전체 mIoU, 상별 누락·과검출, 면적 분율의 평균 절대오차와 평균 편향(%p), 사진당 총 처리 시간과 재추론 면적을 비교한다. 다른 상의 악화와 합금별 변동도 함께 확인한다. 이미 반복 검토한 90장의 결과는 내부 비교로 해석하며, 독립 일반화 성능은 새로운 합금·시편 자료로 별도 검증한다.')
    assert E.tostring(xml.find('.//w:sectPr', NS)) == original_section
    parts['word/document.xml'] = E.tostring(xml, xml_declaration=True, encoding='UTF-8', standalone=True)
    parts['word/_rels/document.xml.rels'] = E.tostring(rels, xml_declaration=True, encoding='UTF-8', standalone=True)
    types = E.fromstring(parts['[Content_Types].xml'])
    ct = 'http://schemas.openxmlformats.org/package/2006/content-types'
    if not any(n.get('Extension') == 'png' for n in types):
        node = E.SubElement(types, '{' + ct + '}Default'); node.set('Extension', 'png'); node.set('ContentType', 'image/png')
        parts['[Content_Types].xml'] = E.tostring(types, xml_declaration=True, encoding='UTF-8', standalone=True)
    output = (WORK / 'layout_draft.docx' if args.draft else
              ROOT / ('reports/모델링결과서_김형수없는김형수_최종검증.docx' if args.final_validation else ('reports/모델링결과서_김형수없는김형수_최종보강.docx' if args.final_strengthening else ('reports/모델링결과서_김형수없는김형수_최종조합.docx' if args.final_pair else ('reports/모델링결과서_김형수없는김형수_심사기준보완.docx' if args.rubric_review else 'reports/모델링결과서_김형수없는김형수.docx')))))
    with ZipFile(output, 'w') as z:
        for info in infos:
            z.writestr(info, parts.pop(info.filename))
        for name, content in parts.items():
            z.writestr(name, content)
    with ZipFile(output) as z, ZipFile(WORK / 'reference.docx') as reference:
        unchanged = [name for name in reference.namelist() if name not in
                     ['word/document.xml', 'word/_rels/document.xml.rels', '[Content_Types].xml']]
        assert all(z.read(name) == reference.read(name) for name in unchanged)
    evidence = {'reference_sha256': sha(WORK / 'reference.docx'), 'source_sha256': sha(WORK / 'reference_original.doc'),
                'output': str(output), 'output_sha256': sha(output), 'draft': args.draft,
                'preserve_only_parts_match': True, 'section_unchanged': True, 'validation20_mIoU': valid_score,
                'result_sha256': sha(results) if result else None,
                'weights_result_sha256': sha(weights_path) if weights else None,
                'tiles_result_sha256': sha(tiles_path) if tiles else None,
                'crop_result_sha256': sha(crop_path) if crop else None,
                'selective_result_sha256': sha(selective_path) if selective else None,
                'full_crop_validation_pending': args.full_crop_pending,
                'full_crop_validation_complete': args.full_crop_complete,
                'full_crop_result_sha256': sha(full_crop_path) if full_crop else None,
                'full_crop_completion_sha256': sha(full_crop_done_path) if full_crop else None,
                'full_crop_per_image_sha256': sha(full_crop_rows_path) if full_crop else None,
                'full_crop_90_image_scores_recomputed': bool(full_crop),
                'rubric_revision': args.rubric_review,
                'development_plan_included': args.development_plan,
                'development_plan_executed': False,
                'final_pair_eda_executed': args.final_pair,
                'final_strengthening_included': args.final_strengthening,
                'final_validation_included': args.final_validation,
                'development_oof_source_sha256': sha(oof_path) if args.development_plan else None,
                'sub6_eutectic_mean_IoU': eutectic_mean if args.development_plan and not args.final_pair else None,
                'final_pair': final_pair_evidence,
                'application_result_sha256': sha(application_path) if args.rubric_review else None,
                'application_figure_sha256': sha(ROOT / ('reports/phase_application_final27/phase_area_fractions.png' if args.final_validation else ('reports/phase_application_final_pair/phase_area_fractions.png' if args.final_pair else 'reports/phase_application/phase_area_fractions.png'))) if args.rubric_review else None,
                'sub6_oof_figure_used': (WORK / 'sub6_oof_examples.png').exists() and not args.final_pair}
    (WORK / ('draft_evidence.json' if args.draft else 'final_build_evidence.json')).write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(evidence, ensure_ascii=False))


if __name__ == '__main__':
    main()
