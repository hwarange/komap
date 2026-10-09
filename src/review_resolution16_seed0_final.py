"""Fixed seed0 full-CV closeout with a separate pilot-only seed comparison."""
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from data import load_split
from infer import image_iou, postprocess
from review_grains15 import BASE, metrics, read_rows
from review_resolution16 import EXP, UNET

OUT = Path('runs/review16_additional')


def checks(full, held):
    return {
        'full_delta_at_least_0_05pp': full['delta'] >= 0.05,
        'full_group_ci_positive': full['group_bootstrap_ci95'][0] > 0,
        'nonpilot_group_ci_positive': held['group_bootstrap_ci95'][0] > 0,
        'all_nonpilot_folds_improve': all(x > 0 for x in held['delta_per_fold'].values()),
        'low_contrast_not_worse': full['delta_low_contrast'] is not None and full['delta_low_contrast'] >= 0,
        'eutectic_si_not_worse': full['delta_per_class']['Eutectic Si'] >= 0,
        'other_class_drop_at_most_0_1pp': all(full['delta_per_class'][c] >= -0.1
                                           for c in ['Al', 'Al3Ni', 'Primary Si'])}


def main():
    fold_of = json.loads(Path('splits/folds5.json').read_text())
    expected = set(fold_of)
    single, base = (read_rows(e, list(range(5))) for e in [EXP, BASE])
    assert set(single) == set(base) == expected
    for fold in range(5):
        a = json.loads(Path(f'runs/cv/{EXP}/fold{fold}/args.json').read_text())
        assert a['seed'] == 0 and a['fold'] == fold and a['epochs'] == 60
        assert a['width'] == 2560 and a['scale_multiplier'] == 2.0 and not a['init']
    sub6, addition, replacement = {}, {}, {}
    items = load_split('data', 'train') + load_split('data', 'valid')
    assert len(items) == len(expected) == 90 and {it['name'] for it in items} == expected
    for it in items:
        name = it['name']
        probabilities = []
        for e in [BASE, UNET, EXP]:
            p = np.load(Path('runs/oof') / e / f'{name}.npy').astype(np.float32)
            assert p.shape == (4, *it['gray'].shape) and np.isfinite(p).all()
            assert p.min() >= 0 and p.max() <= 1 and np.max(np.abs(p.sum(0) - 1)) < 0.005
            probabilities.append(p)
        pb, pu, p0 = probabilities
        for rows, p in [(sub6, (pb + pu) / 2), (addition, (pb + pu + p0) / 3),
                        (replacement, (p0 + pu) / 2)]:
            rows[name] = {'tta_pp': image_iou(it['lab'], postprocess(p)).tolist(), 'low': it['low_contrast']}

    def compare(a, b):
        return {'full': metrics(a, b, fold_of),
                'nonpilot': metrics({n: r for n, r in a.items() if fold_of[n] in [2, 3, 4]}, b, fold_of)}

    result = {'created_utc': datetime.now(timezone.utc).isoformat(),
              'seed0_vs_MiT': compare(single, base),
              'add_seed0_equal_three': compare(addition, sub6),
              'replace_MiT_seed0': compare(replacement, sub6),
              'baseline': metrics(sub6, sub6, fold_of),
              'pilot_seed1_vs_seed0': metrics(read_rows(EXP + '_s1', [0, 1]),
                                             read_rows(EXP, [0, 1]), fold_of),
              'recipe': 'x2 MiT 1/3 + x3 UNet++ 1/3 + x4 MiT seed0 1/3',
              'checked_folds': list(range(5)), 'new_training_folds': [2, 3, 4],
              'scope': 'User reduced scope before fold2 completed. Only seed0 has full CV. Pilot-screened candidate; nonpilot folds reported separately. No weight search.',
              'submission_changed': False, 'further_training_scheduled': False}
    candidate = result['add_seed0_equal_three']
    candidate['predeclared_checks'] = checks(candidate['full'], candidate['nonpilot'])
    qualified = candidate['qualified_for_submission_review'] = all(candidate['predeclared_checks'].values())
    result['conclusion'] = ('4배 seed0 추가 앙상블은 사전 기준을 충족해 채택 검토 후보로 보관한다. '
                           '이번 학습은 종료하며 기존 제출은 유지한다.' if qualified else
                           '4배 seed0 추가 앙상블의 채택 근거가 사전 기준에 미달한다. '
                           '기존 sub6을 유지하고 추가 학습 없이 종료한다.')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'seed0_full_results.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    lines = ['# 4배 MiT seed 0 전체 검증 및 최종 보고서', '', result['conclusion'], '',
             '## 검증 범위와 방법', '',
             '- 기존 seed 0 fold 0·1을 재사용하고 fold 2·3·4만 각각 60 epoch 추가 학습했다.',
             '- 총 90장, 5-fold 합금 분리 검증이다. 새 검증 fold 2·3·4의 54장을 별도로 보고한다.',
             '- MiT는 마지막 60 epoch, 3배 UNet++는 마지막 120 epoch 체크포인트를 사용하며, 8방향 TTA 및 동일 후처리로 평가한다.',
             '- 기준 sub6은 2배 MiT와 3배 UNet++의 확률 평균이다.',
             '- 주 후보는 2배 MiT·3배 UNet++·4배 MiT seed 0의 동일 비중 확률 평균이다. 비중 탐색은 하지 않았다.',
             '- 시간 절약을 위해 seed 1의 남은 fold 3개는 학습하지 않았다.', '',
             '## 전체 및 추가 fold 비교', '',
             f"기존 sub6 전체 mIoU: {result['baseline']['cv_mIoU']:.4f}%.", '',
             '| 비교 | 전체 mIoU (%) | 전체 변화 (%p) | 전체 95% CI (%p) | 새 fold 변화 (%p) | 새 fold 95% CI (%p) |',
             '|---|---:|---:|---|---:|---|']
    labels = {'seed0_vs_MiT': '4배 seed0 vs 2배 MiT',
              'add_seed0_equal_three': 'sub6에 4배 seed0 추가',
              'replace_MiT_seed0': 'sub6의 MiT를 4배 seed0으로 교체'}
    for key, label in labels.items():
        full, held = result[key]['full'], result[key]['nonpilot']
        lo, hi = full['group_bootstrap_ci95']
        hlo, hhi = held['group_bootstrap_ci95']
        lines.append(f"| {label} | {full['cv_mIoU']:.4f} | {full['delta']:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {held['delta']:+.4f} | [{hlo:+.4f}, {hhi:+.4f}] |")
    full, held = candidate['full'], candidate['nonpilot']
    lines += ['', '## 주 후보의 fold·상별 변화', '', '| fold | 변화 (%p) |', '|---|---:|']
    lines += [f'| {fold} | {delta:+.4f} |' for fold, delta in full['delta_per_fold'].items()]
    lines += ['', '| 상 | 전체 변화 (%p) | 새 fold 변화 (%p) |', '|---|---:|---:|']
    lines += [f"| {name} | {delta:+.4f} | {held['delta_per_class'][name]:+.4f} |"
              for name, delta in full['delta_per_class'].items()]
    lines += ['', f"저대비 변화: 전체 {full['delta_low_contrast']:+.4f}%p, 새 fold {held['delta_low_contrast']:+.4f}%p.",
              f"개선 사진: 전체 {full['improved_images']}/{full['images']}장, 새 fold {held['improved_images']}/{held['images']}장.",
              '', '## 판정 기준', '', '| 기준 | 충족 여부 |', '|---|---|']
    names = {'full_delta_at_least_0_05pp': '전체 +0.05%p 이상',
             'full_group_ci_positive': '전체 95% 구간 하한 양수',
             'nonpilot_group_ci_positive': '새 fold 95% 구간 하한 양수',
             'all_nonpilot_folds_improve': '새 fold 각각 개선',
             'low_contrast_not_worse': '전체 저대비 악화 없음',
             'eutectic_si_not_worse': '전체 Eutectic Si 악화 없음',
             'other_class_drop_at_most_0_1pp': '다른 상별 하락 0.1%p 이내'}
    lines += [f"| {names[key]} | {'충족' if value else '미충족'} |"
              for key, value in candidate['predeclared_checks'].items()]
    pilot = result['pilot_seed1_vs_seed0']
    plo, phi = pilot['group_bootstrap_ci95']
    lines += ['', '## 시드 비교의 범위와 해석', '',
              f"기존 fold 0·1의 36장에서 seed 1−seed 0 차이는 {pilot['delta']:+.4f}%p이며 95% 구간은 [{plo:+.4f}, {phi:+.4f}]%p다.",
              '시드 비교는 기존 두 fold에 한정된다. 전체 5-fold에서 시드 안정성이 검증됐다고 결론 내릴 수 없다.',
              '주 후보는 모델 수와 비중이 함께 변하므로 앙상블 개선 전체를 4배 확대 자체의 효과로 해석하지 않는다.',
              'fold 0·1에서 후보를 탐색했으므로 전체 CV는 탐색의 영향을 받는다. 새 fold의 결과를 함께 판단한다.',
              '95% 구간은 fold 안에서 합금 그룹을 재표집한 bootstrap 10,000회 기준이며 공식 평가 점수가 아니다.',
              '', '## 실행 비용과 보관', '']
    for fold in [2, 3, 4]:
        rows = [json.loads(line) for line in Path(f'runs/cv/{EXP}/fold{fold}/log.jsonl').read_text().splitlines()]
        lines.append(f"- fold {fold}: 학습 합계 {sum(row['train_sec'] for row in rows)/60:.1f}분, 최대 기록 GPU 메모리 {max(row['gpu_mem_gb'] for row in rows):.2f}GB. 검증·OOF 시간은 이 학습 합계에 포함되지 않는다.")
    lines += ['- 체크포인트와 설정·로그·결과는 runs/cv/x4_mit_bw_r90/fold0..4에 보관한다.',
              '- 이전 원본, 기존 모델 및 제출 파일은 유지한다. 추가 학습은 예약하지 않았다.',
              '- 재현 설정과 범위 변경은 runs/review16_additional/plan_seed0_closeout.json에 기록한다.',
              '- 기계 판독 결과: runs/review16_additional/seed0_full_results.json.',
              '- 완료 후 체크포인트 SHA256: runs/review16_additional/seed0_artifacts.json.', '',
              f"보고서 생성 UTC: {result['created_utc']}", '']
    report = '\n'.join(lines)
    Path('reports/round16_seed0_final.md').write_text(report, encoding='utf-8')
    (OUT / 'summary.md').write_text(report, encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
