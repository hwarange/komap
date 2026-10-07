"""실험 목록을 5-fold 작업으로 펼쳐 GPU 여러 장에 나눠 실행한다.

실험 파일 형식 (한 줄에 실험 하나, '#'은 주석):
    이름 | train.py에 추가로 넘길 옵션
    baseline |
    lovasz   | --loss lovasz

예) nohup .venv/bin/python src/launch.py experiments/round1.txt --gpus 0 1 2 3 --per-gpu 2 > runs/launch_round1.log 2>&1 &
이미 result.json이 있는 fold는 건너뛰므로, 중간에 멈춰도 같은 명령으로 이어서 돌릴 수 있다.
--heavy <이름들>: 메모리가 커서 GPU 한 장을 통째로 써야 하는 실험 (예: 3배 해상도 27GB).
  그 GPU의 다른 칸이 모두 비었을 때만 시작하고, 도는 동안 그 GPU에 다른 작업을 넣지 않는다.
  앞 작업이 들어갈 자리가 없으면 뒤의 가벼운 작업이 먼저 빈 칸에 들어간다. 옵션을 안 주면 이전과 같다.
"""
import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


def log(msg):
    print(time.strftime('%m-%d %H:%M:%S'), msg, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('exp_file')
    p.add_argument('--gpus', nargs='+', default=['0', '1', '2', '3'])
    p.add_argument('--per-gpu', type=int, default=2, help='GPU 한 장에 동시에 돌릴 작업 수')
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--root', default='runs/cv')
    p.add_argument('--common', default='--amp --workers 6', help='모든 작업에 붙는 옵션')
    p.add_argument('--heavy', nargs='*', default=[], help='GPU 한 장을 통째로 쓰는 실험 이름')
    args = p.parse_args()

    jobs = []
    for line in Path(args.exp_file).read_text(encoding='utf-8').splitlines():
        line = line.split('#')[0].strip()
        if not line:
            continue
        name, _, extra = (s.strip() for s in line.partition('|'))
        for k in range(args.folds):
            out = Path(args.root) / name / f'fold{k}'
            if (out / 'result.json').exists():
                continue
            cmd = [sys.executable, 'src/train.py', '--fold', str(k), '--out', str(out),
                   *shlex.split(args.common), *shlex.split(extra)]
            jobs.append((f'{name}/fold{k}', out, cmd, name in args.heavy))
    log(f'{len(jobs)} jobs queued on GPUs {args.gpus} x {args.per_gpu}'
        + (f' (heavy: {args.heavy})' if args.heavy else ''))

    slots = [g for g in args.gpus for _ in range(args.per_gpu)]
    running = {}  # slot index -> (tag, proc, start, heavy)
    failed = []
    while jobs or running:
        for i, g in enumerate(slots):
            if i in running or not jobs:
                continue
            busy = [j for j in running if slots[j] == g]   # 같은 GPU에서 도는 작업
            if any(running[j][3] for j in busy):
                continue                                    # 무거운 작업이 GPU를 통째로 쓰는 중
            pick = next((n for n, jb in enumerate(jobs) if not jb[3] or not busy), None)
            if pick is not None:
                tag, out, cmd, heavy = jobs.pop(pick)
                out.mkdir(parents=True, exist_ok=True)
                f = open(out / 'stdout.txt', 'w')
                proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT,
                                        env={**os.environ, 'CUDA_VISIBLE_DEVICES': g})
                running[i] = (tag, proc, time.time(), heavy)
                log(f'start {tag} on GPU {g}')
        time.sleep(5)
        for i in list(running):
            tag, proc, t0, _ = running[i]
            if proc.poll() is not None:
                ok = proc.returncode == 0
                log(f"{'done' if ok else 'FAIL'} {tag} ({(time.time() - t0) / 60:.1f} min)")
                if not ok:
                    failed.append(tag)
                del running[i]
    log(f'all finished. failed: {failed if failed else "none"}')


if __name__ == '__main__':
    main()
