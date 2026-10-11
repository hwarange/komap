"""Fill report prose with audited train70, validation20 and test evidence."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORK = Path(__file__).resolve().parent
EVID = ROOT / 'reports/final35_evidence'
content = json.loads((WORK / 'content_template.json').read_text(encoding='utf-8'))
eda = json.loads((EVID / 'train70_eda.json').read_text(encoding='utf-8'))
cases = json.loads((EVID / 'validation_cases.json').read_text(encoding='utf-8'))
area = json.loads((ROOT / 'submissions/final35_train70/prediction_area_fractions.json').read_text(encoding='utf-8'))
done = json.loads((ROOT / 'submissions/final35_train70/all.done').read_text(encoding='utf-8'))
assert len(done['masks']) == 10
assert eda['images'] == 70 and eda['alloys'] == 59
example = area['images'][0]
v = example['area_fraction_percent']
substitutions = {
    '{{eda_core}}': '학습 자료는 70장·59개 합금이며 원본 크기는 1230×875 20장, 1440×1024 9장, 1278×909 24장, 675×480 17장이다. 전체 68,184,888픽셀을 합친 상별 비율은 Al 57.28%, Al3Ni 26.31%, 초정 Si 9.43%, 공정 Si 6.98%다. 초정 Si가 없는 학습 사진도1장 있어 빈 클래스를 평가 과정에서 구분했다.',
    '{{eda_figure}}': 'reports/final35_evidence/report_eda.png',
    '{{eda_interpretation}}': '사진별 평균 밝기는 8비트 기준138.90~200.02, p99−p1 밝기 범위는137~248로 차이가 있었다. 공정 Si 픽셀의63.16%는3×3 이웃에 다른 상이 포함되는 경계 근처에 있었다. 이는 공정 Si에서 경계에 인접한 영역의 비중이 크다는 관찰이며, 낮은 성능의 원인이 불규칙한 모양 하나에 있다는 뜻은 아니다.',
    '{{case_description}}': f"공식20장의 점수를 오름차순으로 정렬해 최저1위와 중앙 부근10위를 고정 규칙으로 골랐다. {cases[0]['name']}은{cases[0]['mIoU']:.2f}%, {cases[1]['name']}은{cases[1]['mIoU']:.2f}%다. 최저 사례에서는 Al3Ni IoU37.23%, 공정Si43.21%로 두 상의 구분이 특히 어려웠다. 이 두 그림이 모든 사진을 대표하거나 오류 원인을 입증하는 것은 아니다.",
    '{{case_figure}}': 'reports/final35_evidence/report_validation_cases.png',
    '{{test_application}}': f"예를 들어 {example['name']}의 예측 면적 분율은 Al {v['Al']:.2f}%, Al3Ni {v['Al3Ni']:.2f}%, 초정 Si {v['Primary Si']:.2f}%, 공정 Si {v['Eutectic Si']:.2f}%다. test10 모두에서 네 상의 픽셀 수 합이 전체 픽셀 수와 일치하는지 확인했다. 이처럼 분할 결과를 분율표로 변환하는 과정까지 구현했다.",
    '{{reproduction}}': '실행 환경은 Linux, Python3.12.3, PyTorch2.14.1+cu126, segmentation_models_pytorch0.5.0, timm1.0.30이다. 실행 진입점 run_submission.py, 환경YAML과 버전·체크포인트·입력 해시를 기록한 manifest를 제공한다. L40S 두 GPU에서 최종 test10 추론과 마스크 검증을 완료했다. 환경YAML은 실제 환경의 기록이며, 별도의 깨끗한 환경에서 재설치하는 검증은 수행하지 않았다.'
}
corrections = {
    '최종 비교 모델은 모두 ImageNet 사전학습에서 새로 초기화한 뒤': '최종 비교 모델의 인코더는 ImageNet 사전학습으로, 디코더는 새로 초기화한 뒤',
    'U-Net은 인코더의 특징과 디코더의 위치 정보를 연결해': 'U-Net은 인코더의 여러 해상도 특징을 디코더에 연결해',
    '이후 평균 0.5, 표준편차 0.25로 변환해 모델에 입력했다.': '이후 고정 상수를 사용한 (x−0.5)/0.25 변환을 적용해 모델에 입력했다.',
    '크기 변화 ±15%': '기본 배율의 약0.87~1.15배 크기 변화',
    'cosine 학습률 일정': '1epoch warm-up 후 cosine 학습률 일정',
    '경계 픽셀의 교차엔트로피에는 1+2=3배 가중치를 적용했다.': '경계 픽셀의 교차엔트로피에는 비경계 대비3배 상대 가중치를 적용한 뒤 평균 가중치로 정규화했다.',
    '추론 배율을 일부 변경한 후보 역시': 'MiT4·Hybrid의 일부 원본 폭에 추론 배율을 변경한 후보 역시',
    '이번5-fold의10개 모델을': '선택한 두 계열의5-fold 가중치10개를',
    'ImageNet / train70': '인코더 ImageNet / train70'
}

def apply(value):
    if isinstance(value, str):
        for a, b in {**substitutions, **corrections}.items():
            value = value.replace(a, b)
        if '/' not in value or len(value) > 100:
            import re
            value = re.sub(r'(?<=[가-힣])(?=[0-9])', ' ', value)
            for a, b in {
                'train70': 'train 70', 'validation20': 'validation 20', 'test10': 'test 10',
                '공정Si': '공정 Si', '초정Si': '초정 Si', 'IoU37': 'IoU 37',
                'Si43': 'Si 43', 'MiT2와UNet': 'MiT2와 UNet',
                '1epoch': '1 epoch', '60epoch': '60 epoch', '120epoch': '120 epoch',
                'Python3.12': 'Python 3.12', 'PyTorch2.14': 'PyTorch 2.14',
                'segmentation_models_pytorch0.5': 'segmentation_models_pytorch 0.5',
                'timm1.0': 'timm 1.0', '환경YAML': '환경 YAML',
                '체크포인트SHA256': '체크포인트 SHA256', '10장을ZIP': '10장을 ZIP',
                'mIoU80': 'mIoU 80', 'OOF80': 'OOF 80', '차이95': '차이 95',
                '95%구간': '95% 구간', 'Si +': 'Si +',
                '이 두 그림이': '이 두 사례가',
            }.items():
                value = value.replace(a, b)
        assert '{{' not in value and '}}' not in value
        return value
    if isinstance(value, list):
        return [apply(v) for v in value]
    if isinstance(value, dict):
        return {k: apply(v) for k, v in value.items()}
    return value

(WORK / 'content.json').write_text(json.dumps(apply(content), ensure_ascii=False, indent=2), encoding='utf-8')
print('Verified evidence inserted; report content is ready.')
