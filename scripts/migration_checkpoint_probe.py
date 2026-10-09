"""Compare trained checkpoint inference across servers without using original data."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

CHECKPOINTS = [
    'runs/cv/x2_mit_bw_r90/fold0/last.pt',
    'runs/cv/x2_unetpp_bw_e120_r90/fold0/last.pt',
    'runs/cv/x4_mit_bw_r90/fold0/last.pt',
    'runs/cv/x4_mit_bw_r90_s1/fold0/last.pt',
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reference', type=Path)
    args = p.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root / 'src'))
    import numpy as np
    import torch
    from train import build_from_args

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    reference = json.loads(args.reference.read_text()) if args.reference else None
    x = torch.linspace(-1, 1, 128 * 128, device='cuda').reshape(1, 1, 128, 128)
    result = {'input': 'linspace(-1,1), shape (1,1,128,128), FP32',
        'python': sys.version.split()[0], 'torch': torch.__version__,
        'gpu': torch.cuda.get_device_name(0), 'checkpoints': {}}
    for name in CHECKPOINTS:
        ck = torch.load(root / name, map_location='cpu', weights_only=False)
        model = build_from_args(ck['args'], weights=None)
        model.load_state_dict(ck['model'], strict=True)
        del ck
        model = model.cuda().eval()
        with torch.inference_mode():
            output = model(x).float().cpu().numpy()
        if not np.isfinite(output).all() or output.shape != (1, 4, 128, 128):
            raise RuntimeError('Invalid checkpoint inference: ' + name)
        values = output.reshape(-1)[np.linspace(0, output.size - 1, 512, dtype=int)]
        row = {'strict_state_dict_load': True, 'finite': True, 'shape': list(output.shape),
            'samples': values.tolist(), 'mean': float(output.mean())}
        if reference:
            expected = reference['checkpoints'][name]
            diff = np.abs(values - np.asarray(expected['samples']))
            row['max_absolute_sample_difference'] = float(diff.max())
            row['source_inference_matches'] = bool(np.allclose(values, expected['samples'], atol=1e-4, rtol=1e-4))
            if not row['source_inference_matches'] or abs(row['mean'] - expected['mean']) > 1e-4:
                raise RuntimeError('Inference changed after migration: ' + name)
        result['checkpoints'][name] = row
        print('CHECKPOINT_FORWARD_PASSED', name, flush=True)
        del model
        torch.cuda.empty_cache()
    result['status'] = 'source_inference_match_passed' if reference else 'source_reference_ready'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(result['status'], flush=True)


if __name__ == '__main__':
    main()
