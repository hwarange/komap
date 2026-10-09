#!/usr/bin/env bash
# Run on the destination after the selected assets have been transferred.
set -euo pipefail
TASK_PROJECT_ROOT="${1:?Usage: bash restore_uv.sh /workspace/komap}"
cd "$TASK_PROJECT_ROOT"
test -f backup_metadata/requirements-migration.lock
command -v uv >/dev/null
export UV_CACHE_DIR="$TASK_PROJECT_ROOT/.uv-cache"
if [[ ! -x .venv/bin/python ]]; then
    uv venv --python 3.12.3 .venv
fi
# Both indexes are official; exact versions and hashes constrain every package.
uv pip sync --python .venv/bin/python --require-hashes \
    --index https://download.pytorch.org/whl/cu126 \
    --index-strategy unsafe-best-match \
    backup_metadata/requirements-migration.lock
uv pip check --python .venv/bin/python
.venv/bin/python - <<'PY'
import importlib.metadata as md
import json
from pathlib import Path
import platform
import torch
import segmentation_models_pytorch as smp
import timm
expected = {}
for line in Path('backup_metadata/requirements-migration-freeze.txt').read_text().splitlines():
    if line and not line.startswith('#'):
        name, version = line.split('==', 1)
        expected[name] = version
mismatch = {name: {'expected': version, 'actual': md.version(name)}
            for name, version in expected.items() if md.version(name) != version}
if mismatch:
    raise RuntimeError(mismatch)
assert platform.python_version() == '3.12.3', platform.python_version()
assert torch.cuda.is_available(), 'CUDA is unavailable'
net = smp.Unet(encoder_name='mit_b2', encoder_weights=None, in_channels=1, classes=4).cuda().eval()
with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
    result = net(torch.zeros(1, 1, 128, 128, device='cuda'))
assert result.shape == (1, 4, 128, 128) and torch.isfinite(result).all()
report = {'python': platform.python_version(), 'torch': torch.__version__,
          'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
          'packages': len(expected), 'all_package_versions_match': True,
          'mit_b2_cuda_bf16_forward': True, 'status': 'environment_smoke_passed'}
Path('backup_metadata/uv_restore_verified.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report))
PY
