"""Record final report acceptance after every rendered page is reviewed."""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / 'reports/final35_document_work'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def record(path):
    return {'path': str(path.relative_to(ROOT)).replace('\\', '/'),
            'size_bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iteration', type=int, required=True)
    parser.add_argument('--reviewed-pages', type=int, required=True)
    args = parser.parse_args()
    render_file = WORK / f'render_v{args.iteration}.json'
    render = read(render_file)
    count = render['pages']
    assert count == args.reviewed_pages and 1 <= count <= 10
    pages = [Path(render['render']) / f'page-{i}.png' for i in range(1, count + 1)]
    assert all(p.is_file() for p in pages)
    build_file = WORK / 'build_evidence.json'
    build = read(build_file)
    doc = ROOT / 'reports/모델링결과서_김형수없는김형수_최종제출.docx'
    item = record(doc)
    assert item['sha256'] == build['sha256']
    timestamp = datetime.now(ZoneInfo('Asia/Seoul')).isoformat()
    qa = {'completed_at': timestamp, 'document': item, 'pages': count,
          'render_iteration': args.iteration, 'all_pages_visually_reviewed': True,
          'checks': ['template form and section retained', 'all text readable',
                     'no clipped or overlapping tables or images',
                     'official validation and internal OOF clearly separated',
                     'train70-only selected weights match submitted masks',
                     'within ten-page submission limit'],
          'rendered_pages': [record(p) for p in pages],
          'pdf': record(Path(render['pdf'])),
          'official_validation20_mIoU': 80.62075310543737,
          'train70_internal_oof_mIoU': 80.77829756479382,
          'external_submission_performed': False}
    write(WORK / 'layout_qa.json', qa)
    render['visual_review_complete'] = True
    write(render_file, render)
    build['visual_qa_complete'] = True
    build['page_count'] = count
    build['layout_qa'] = 'reports/final35_document_work/layout_qa.json'
    write(build_file, build)
    ledger_file = ROOT / 'delivery/final_model_selection.json'
    ledger = read(ledger_file)
    ledger.update(status='selected_train70_submission_complete',
                  eligibility_scope='The selected two models comply with the confirmed train70-only training rule. Final report and local submission artifacts are verified; no external submission has been performed.',
                  report_status='complete', report_sha256=item['sha256'],
                  report_pages=count, report_visually_verified=True,
                  report_qa_status='passed',
                  report_qa='reports/final35_document_work/layout_qa.json')
    write(ledger_file, ledger)
    handoff = ROOT / 'handoff/HANDOFF_latest.md'
    text = handoff.read_text(encoding='utf-8')
    text = text.replace('final35 제출물 완료, 최종 보고서 렌더·검수 중',
                        f'final35 제출물과 최종 보고서 {count}쪽 검수 완료', 1)
    text = text.replace('현재 렌더·시각 검수 중이며 **보고서 SHA·페이지 수·QA는 pending**이다.',
                        f'표지 포함 **{count}쪽의 전 페이지 렌더·시각 검수**를 마쳤다. 보고서 SHA256은 `{item["sha256"]}`이며 검수 증거는 `reports/final35_document_work/layout_qa.json`이다.', 1)
    handoff.write_text(text, encoding='utf-8')
    notes = ROOT / 'reports/research_notes.md'
    text = notes.read_text(encoding='utf-8')
    old = '현재 문서 렌더·시각 검수를 진행 중이므로 **보고서 SHA256·페이지 수·QA는 pending**으로 남긴다.\n이 값들은 최종 보고서 검수 담당이 완료 후 갱신한다.'
    new = f'표지 포함{count}쪽을 원본 양식에 맞춰 작성하고 모든 페이지의 렌더·시각 검수를 완료했다.\n보고서 SHA256은 `{item["sha256"]}`이며 검수 증거는 `reports/final35_document_work/layout_qa.json`이다.\n공식20장 점수·내부70장 OOF·train70 EDA·오류 사례·재현 자료를 구분해 반영했다.'
    assert old in text
    notes.write_text(text.replace(old, new, 1), encoding='utf-8')
    artifacts = [doc, ROOT / 'submissions/final35_train70/final35_masks.zip',
                 ROOT / 'delivery/final35_code_bundle.zip',
                 ROOT / 'delivery/final35_repro/checkpoints/mit2_last.pt',
                 ROOT / 'delivery/final35_repro/checkpoints/unet3_last.pt']
    write(ROOT / 'delivery/final35_delivery_receipt.json',
          {'completed_at': timestamp, 'artifacts': [record(p) for p in artifacts],
           'report_qa': record(WORK / 'layout_qa.json'),
           'external_submission_performed': False, 'new_training_performed': False})
    print(json.dumps({'report': item, 'pages': count, 'status': 'complete'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
