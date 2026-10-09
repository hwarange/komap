"""Populate the retained form using completed training and fixed inference reviews."""
import argparse
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--draft', action='store_true')
    args = parser.parse_args()
    request = json.loads((WORK / 'request.json').read_text(encoding='utf-8'))
    results = ROOT / request['final_result']
    result = None
    weights_path = ROOT / 'runs/review17_weights/results.json'
    tiles_path = ROOT / 'runs/review17_tiles/results.json'
    weights = tiles = None
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
        for tag in ['ind', 'spacing', 'keepNext', 'keepLines', 'jc']:
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
                 '얇은 공정 Si와 촬영 대비 차이에 대응하는 전처리 및 해상도 설계를 검증했다.',
                 '합금 단위 교차검증과 추가 비교를 거쳐 두 모델 조합인 sub6을 채택했다.']:
        summary.append(paragraph(text))
    for _ in range(4):
        summary.append(paragraph(''))
    for text in ['◦ 파급효과',
                 '상별 면적 분율과 조직 형상을 일관되게 정량화해 합금 조직 분석을 지원한다.',
                 '수작업 분석 부담을 줄이고 촬영 조건이 다른 사진의 비교 가능성을 높인다.',
                 '분할 오차와 검증 범위를 함께 제시해 후속 소재 분석의 판단 근거를 제공한다.']:
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
        fragment.add_paragraph().add_run().add_picture(str(ROOT / path), width=Inches(5.6))
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

    add('① 분석(EDA) 개요', 'heading')
    add('1-1. 분석 핵심 내용', 'sub')
    add('본 과제는 고탄성 알루미늄 합금의 광학현미경 이미지를 네 상으로 구분하는 픽셀 단위 분할이다. 촬영 대비와 해상도의 차이, 얇은 Eutectic Si의 경계 손실을 핵심 문제로 보았다. 이미지 정규화와 배율 조정, 서로 다른 구조의 모델 조합을 실제 비교 실험으로 검증했다.')
    if tiles:
        add(f"4배 MiT seed 0의 전체 5-fold 학습과 고정 비중·추론 타일 비교를 완료했다. 전체 90장 최고 관측 점수는 {tiles['tile1536_addition_vs_sub6']['full']['cv_mIoU']:.4f}%였지만, 개선의 일관성이 부족해 최종 조합은 2배 MiT와 3배 UNet++의 동일 비중 앙상블인 sub6({tiles['baseline']['cv_mIoU']:.4f}%)을 유지했다.")
    else:
        add('기준 조합은 2배 MiT와 3배 UNet++의 동일 비중 앙상블인 sub6이다. 4배 MiT seed 0의 전체 5-fold 검증과 고정 비중·추론 타일 비교 결과를 반영한다.')
    add('1-2. 분석 배경', 'sub')
    add('같은 소재의 여러 상은 밝기만으로 완전히 구분되지 않는다. 특히 얇고 불규칙한 Eutectic Si와 주변 상의 경계가 작은 픽셀 이동에도 영향을 받는다. 따라서 평균 성능뿐 아니라 상별 오차, 저대비 사진의 변화 및 합금 간 일반화 성능을 함께 확인했다.')
    add('1-3. 분석 방법 및 분석 내용', 'sub')
    numeric_table(['데이터', '사진 수', '정답 마스크', '활용'], [
        ['train', '70', '있음', '초기 학습 및 교차검증'],
        ['validation', '20', '있음', '초기 고정분할 평가 및 교차검증'],
        ['test', '10', '없음', '최종 마스크 생성 및 형식 확인']], [1800, 1000, 1400, 4300])
    add('밝기·대비 분석에는 원본 이미지의 분위수와 촬영 스타일을, 해상도 분석에는 이미지 폭과 마스크의 조직 두께를 사용했다. 클래스 분석에는 네 상의 정답 마스크를, 검증 분할에는 파일명에 포함된 합금 번호를 사용했다. test의 정답은 없으며 학습이나 성능 계산에 사용하지 않았다.')
    add('원본 폭은 675·1096·1230·1278·1440픽셀로 나뉜다. EDA에서 675픽셀 사진의 일부 Al3Ni·Eutectic Si 두께가 약 1.9픽셀로 확인되어, 사진 전체를 축소하는 접근보다 배율을 맞추고 타일로 처리하는 접근을 택했다. 기준 모델 오답의 79.1%가 정답 경계 1픽셀 이내에 있었다.')
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
    add('')
    add('')
    add('③ 모델링', 'heading')
    add('3-1. 배경/알고리즘 소개', 'sub')
    add('U-Net 계열은 인코더에서 얻은 특징과 디코더의 위치 정보를 결합해 원본 크기의 분할 지도를 예측한다. MiT-B2 인코더를 사용하는 U-Net과 ResNet34 인코더를 사용하는 UNet++를 비교하고, 서로 다른 오류를 보완하도록 두 모델의 클래스 확률을 평균했다.')
    add('4배 MiT의 원리는 모델의 파라미터를 늘리는 것이 아니라 더 확대된 조직을 보여주는 것이다. 입력 조각은 2배와 4배 모두 1024×1024다. 대부분 사진에서 2배는 원본 약 512×512 영역을, 4배는 약 256×256 영역을 본다. 얇은 조직의 표현은 커지지만 주변 맥락은 줄어들 수 있어 검증이 필요하다.')
    add('3-2. 모델링 방법', 'sub')
    add('초기 전처리와 증강을 검토한 뒤 입력 배율과 모델 구조를 개선하고, 학습 기간 및 앙상블 기여를 확인했다. 주요 결과는 다음과 같다.')
    numeric_table(['주요 실험', '결과', '해석'], [
        ['밝기 증강 제거', '−1.03%p', '해당 대조 실험에서 악화'],
        ['2배 MiT / 60 epoch', '80.171%', '최종 조합의 첫 모델'],
        ['3배 UNet++ / 120 epoch', '80.162%', '최종 조합의 둘째 모델'],
        ['두 모델 동일 비중 sub6', '80.7884%', '채택 조합'],
        ['2배 MiT / 120 epoch', '80.190%', '60 epoch 대비 이득 불확실']], [3700, 1800, 3000])
    add('밝기 증강 제거는 당시 기준 대비 변화이며, 나머지는 합금 단위 90장 CV 점수다. MiT의 120 epoch 연장은 약 +0.019%p에 그쳐 개선 근거가 부족했다.')
    numeric_table(['항목', '4배 MiT 추가 검증 설정'], [
        ['모델과 학습', 'U-Net / MiT-B2, ImageNet 초기화; seed 0, 60 epoch, batch 4, 사진당 8조각'],
        ['입력과 배율', 'crop/tile 1024, width 2560, scale_multiplier 2'],
        ['최적화와 손실', 'AdamW, lr 0.0003, weight decay 0.0001; 교차엔트로피 + Dice, 경계 가중 bw=2'],
        ['평가와 환경', '마지막 60 epoch, 8방향 TTA, overlap 128; L40S 1개, BF16']], [2200, 6300])
    add('합금 번호 단위로 90장을 5개 fold에 나눠, 매번 72장으로 학습하고 해당 모델이 보지 않은 18장으로 평가한다. 같은 합금의 사진은 서로 다른 fold에 나누지 않았다. 이번에는 비용을 줄이기 위해 seed 1의 남은 세 fold를 생략하고 seed 0만 전체 검증했다.')
    add('4배 학습 후 주 후보는 2배 MiT·3배 UNet++·4배 MiT seed 0을 각각 1/3로 평균해 비교했다. 기존 sub6은 앞의 두 모델을 각각 1/2로 평균한다. 이후 재학습 없이 고정 비중 두 가지와 추론 타일 1536을 추가 비교했다. 이 후속 검토는 이미 관찰한 90장에 대한 탐색이며 독립 검증으로 간주하지 않았다.')
    add('3-3. 성능', 'sub')
    with (ROOT / 'runs/baseline/analysis/per_image.csv').open(encoding='utf-8-sig') as f:
        validation_rows = list(csv.DictReader(f))
    assert len(validation_rows) == 20
    valid_score = sum(float(row['tta+pp']) for row in validation_rows) / 20
    add('가. 양식에서 요구하는 train 70장 / validation 20장 고정분할', 'sub')
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
        add(f'추가 조합의 fold별 변화는 {fold_changes}%p였다. 초기 두 fold의 개선이 전체 평균 상승을 이끌었고, 새 fold 중 2와 4에서는 소폭 하락했다.')
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
        add(f"추론 타일만 1024에서 1536으로 넓힌 추가 조합이 최고 {tf['cv_mIoU']:.4f}%를 기록했다. sub6 대비 95% 구간은 [{tlo:+.5f}, {thi:+.5f}]%p, 기존 54장 변화는 {th['delta']:+.5f}%p였다. 같은 추가 조합에서 타일 변경만의 이득은 {tile_change:+.6f}%p로 작았다. 타일 크기는 주변 맥락뿐 아니라 경계 위치와 확률 혼합도 바꾸므로 개선 원인을 하나로 확정하지 않았다.")
        add(f"이 표는 원본 서버의 동일 기준 OOF로 재계산했다. sub6은 {tiles['baseline']['cv_mIoU']:.4f}%이며 앞선 RunPod 재계산 {result['baseline']['cv_mIoU']:.4f}%와의 차이는 {tiles['baseline_drift_pp']:+.6f}%p다. 같은 표 안에서 동일 기준과 비교했으며 계산 환경의 미세한 차이를 모델 개선으로 해석하지 않았다.")
    else:
        add('고정 비중 및 추론 타일 비교의 완료 결과를 최종본에 반영한다.')
    add('')
    add('라. 예비 결과와 시드 비교 범위', 'sub')
    add('기존 fold 0·1의 36장에서 4배 seed 0은 81.3048%, seed 1은 81.3063%로 평균 차이가 약 +0.0015%p였다. 그러나 fold별 차이 −0.3219%p와 +0.3248%p가 상쇄된 값이다. seed 1의 나머지 fold는 수행하지 않았으므로 전체 데이터에 대한 시드 안정성을 입증했다고 표현하지 않는다.')
    add('기존 두 시드 평균을 추가한 예비 조합의 +0.3069%p 개선은 이번 seed 0 단독 추가 조합과 구성이 다르다. 예비 점수나 36장의 점수를 이번 전체 90장 결과와 직접 빼서 비교하지 않는다.')
    if (WORK / 'sub6_oof_examples.png').exists():
        figure('reports/modeling_document_work/sub6_oof_examples.png', '그림 2  채택 모델 sub6의 OOF 예측  90장 중 mIoU 중앙값 부근과 최저 점수 사례를 고정 기준으로 선정')
    else:
        figure('runs/baseline/analysis/worst_A71_step2_sample1.png', '그림 2  초기 기준 모델의 validation 오류 사례  최신 후보의 결과와 구분')
    add('마. 결론 및 활용 범위', 'sub')
    if result:
        add(f"최고 관측 점수는 {tiles['tile1536_addition_vs_sub6']['full']['cv_mIoU']:.4f}%였지만, 최종 채택 모델은 sub6({tiles['baseline']['cv_mIoU']:.4f}%)이다. 평균 상승만으로 교체하지 않고, 합금별 변동과 개선 구간 및 추가 추론 비용을 함께 판단했다. 추가 학습 없이 이번 검토를 종료했다.")
        checks = result['add_seed0_equal_three']['predeclared_checks']
        add(f'앞선 4배 학습 후 동일 비중 조합은 사전 판정 기준 {sum(checks.values())}/{len(checks)}개를 충족했다. 전체 +0.05%p 이상, 전체·새 fold의 구간 하한 양수, 새 fold 각각 개선, 저대비 및 Eutectic Si 악화 없음, 다른 상의 하락 0.1%p 이내를 함께 확인했다.')
    else:
        add('최종 채택 판단은 진행 중인 전체 검증 결과로 확정한다. 보고서용 임시 배치에서는 결과와 결론을 미리 확정하지 않는다.')
    add('추가 앙상블은 모델 수와 구조별 비중도 바꾸므로 개선 전체를 확대율만의 효과로 설명할 수 없다. 확대 모델은 처리할 타일과 예측 비용이 늘어난다. 결과는 상별 면적 분율과 형태 정량화에 활용할 수 있으며, 탄성률이나 조성의 직접 측정값과 구분한다.')
    add('바. 제출 결과와 재현', 'sub')
    add('기존 sub6은 2배 MiT와 3배 UNet++ 각각의 5-fold 예측을 평균한 뒤 두 계열을 같은 비중으로 결합한다. test 10장의 원본 크기 RGB 마스크를 submissions/sub6에 생성하고 지정된 네 색과 크기를 확인했다. 이번 검증으로 제출 파일을 자동 변경하거나 공식 제출하지 않았다. 이 마스크는 초기 train70 전용 기준 모델의 예측과 구분한다.')
    add('최종 sub6의 train70 학습 / validation20 고정분할 성능은 별도로 측정하지 않았다. 따라서 앞서 제시한 초기 기준 모델의 20장 점수와 sub6의 test 마스크를 동일 모델의 검증·제출 결과로 연결할 수 없다.')
    add('재현에 필요한 학습 설정, 코드, 체크포인트, fold 및 결과를 보관한다. Python 3.12, PyTorch 2.14.1+cu126, segmentation_models_pytorch 0.5.0, timm 1.0.30 환경을 사용했다. MiT 초기화는 smp-hub/mit_b2.imagenet의 기존 고정 revision을 재사용하고 원본 SHA256을 확인했다.')
    add('추가 학습은 seed 0의 세 fold로 종료했고, 이어서 재학습 없는 비중·타일 비교만 완료했다. 자체 학습 체크포인트와 결과·코드·재현 기록을 보관하며 새로운 실험을 예약하지 않았다. 주요 근거는 baseline/analysis, review13~17 결과, seed0_full_results.json 및 seed0_artifacts.json이다.')
    assert E.tostring(xml.find('.//w:sectPr', NS)) == original_section
    parts['word/document.xml'] = E.tostring(xml, xml_declaration=True, encoding='UTF-8', standalone=True)
    parts['word/_rels/document.xml.rels'] = E.tostring(rels, xml_declaration=True, encoding='UTF-8', standalone=True)
    types = E.fromstring(parts['[Content_Types].xml'])
    ct = 'http://schemas.openxmlformats.org/package/2006/content-types'
    if not any(n.get('Extension') == 'png' for n in types):
        node = E.SubElement(types, '{' + ct + '}Default'); node.set('Extension', 'png'); node.set('ContentType', 'image/png')
        parts['[Content_Types].xml'] = E.tostring(types, xml_declaration=True, encoding='UTF-8', standalone=True)
    output = WORK / 'layout_draft.docx' if args.draft else ROOT / 'reports/모델링결과서_김형수없는김형수.docx'
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
                'sub6_oof_figure_used': (WORK / 'sub6_oof_examples.png').exists()}
    (WORK / ('draft_evidence.json' if args.draft else 'final_build_evidence.json')).write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(evidence, ensure_ascii=False))


if __name__ == '__main__':
    main()
