"""6단계 A1: 학습 곡선 판단. 72장(전체 fold 학습) vs 54장(75%) vs 36장(50%)의 CV 점수로
'90장 전체 학습'의 기대 효과를 추정하고, 계획의 판단 기준(54→72장 +0.3 이상)에 따라 C 단계 진행 여부를 정한다.

예) python src/learning_curve.py --out runs/learning_curve.json
"""
import argparse
import glob
import json

import numpy as np

PAIRS = {'cnx_bw': ['cnx_bw_f50', 'cnx_bw_f75', 'cnx_bw'], 'unetpp_bw': ['unetpp_bw_f50', 'unetpp_bw_f75', 'unetpp_bw']}
N_IMG = [36, 54, 72]


def score(exp):
    rows = []
    for f in sorted(glob.glob(f'runs/cv/{exp}/fold*/result.json')):
        rows += json.load(open(f))['rows']
    if len(rows) != 90:
        return None
    return float(np.mean([np.nanmean(r['tta_pp']) for r in rows]) * 100)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--threshold', type=float, default=0.3, help='54→72장 개선이 이 값 이상이면 C 단계 진행')
    p.add_argument('--out', default='runs/learning_curve.json')
    args = p.parse_args()
    out, gains = {}, []
    for arch, exps in PAIRS.items():
        sc = [score(e) for e in exps]
        if None in sc:
            out[arch] = {'scores': sc, 'note': 'incomplete'}
            continue
        # 점수 ≈ a + b·log(N) 으로 맞춰 90장을 외삽 (거친 근사)
        b, a = np.polyfit(np.log(N_IMG), sc, 1)
        pred90 = a + b * np.log(90)
        gain_54_72 = sc[2] - sc[1]
        gains.append(gain_54_72)
        out[arch] = {'scores_36_54_72': [round(x, 3) for x in sc], 'gain_54_to_72': round(gain_54_72, 3),
                     'extrapolated_90': round(float(pred90), 3), 'expected_gain_72_to_90': round(float(pred90 - sc[2]), 3)}
    out['mean_gain_54_to_72'] = round(float(np.mean(gains)), 3) if gains else None
    out['decision_full90'] = bool(gains) and float(np.mean(gains)) >= args.threshold
    out['rule'] = f'54→72장 평균 개선 >= {args.threshold} 이면 90장 전체 학습(C 단계) 진행'
    print(json.dumps(out, indent=1, ensure_ascii=False))
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
