---
name: local-env-cpu-only
description: "Laptops are for editing/report only (CPU); all GPU work runs on the company server via `ssh komap`"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 115359f4-9d03-43f3-ae46-b50d8a5ba8bb
  modified: 2026-10-07T14:19:17.160Z
---

All training and inference run on the company server (`ssh komap`, 4× L40S). Laptops are only for editing code, import checks, EDA, and report figures.

- Old laptop (DESKTOP-MR7IIH9): venv `Komap/` (uv, Python 3.13), CPU torch.
- Current laptop (hostname `hwarang`, from 2026-10-07 evening): `ssh komap` works with ~/.ssh/id_ed25519. No project venv and no uv yet; system Python 3.13 only.

**Why:** On 2026-10-06 the user stopped a GPU (cu126) torch install: "회사 GPU를 쓸건데 왜 이렇게 하고 있는거야". Also, with uv, `--extra-index-url pypi` takes priority over `--index-url`, and that pulled in the CPU torch by mistake.

**How to apply:** Don't set up local GPU training or suggest it. If local Python is needed (report figures), make a CPU-only venv. Check the server state before following handoff steps mechanically. Related: [[feedback-wait-notify]], [[komap-status]].
