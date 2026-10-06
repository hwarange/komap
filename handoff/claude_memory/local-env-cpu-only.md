---
name: local-env-cpu-only
description: "Laptop venv is \"Komap\" (uv, Python 3.13) with CPU torch only; all GPU work runs on the company server"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 89f50a3c-b870-47bf-a4b0-c4aa7fae0988
  modified: 2026-10-06T00:59:34.781Z
---

On this laptop (DESKTOP-MR7IIH9), the project venv is `Komap/` (not `.venv`). It was made with uv on Python 3.13.15 and holds torch 2.14.1+cpu plus requirements.txt. Install packages with `uv pip install --python Komap/Scripts/python.exe ...`.

**Why:** On 2026-10-06 the user stopped a GPU (cu126) torch install: "회사 GPU를 쓸건데 왜 이렇게 하고 있는거야". Training and inference run on the server (`ssh komap`, 4× L40S). The laptop is only for editing code, import checks, EDA, and report figures. Also, with uv, `--extra-index-url pypi` takes priority over `--index-url`, and that pulled in the CPU torch by mistake.

**How to apply:** Don't set up local GPU training or suggest it. Check the server context before following handoff steps mechanically. The SSH alias `komap` works with ~/.ssh/id_ed25519 (registered 2026-10-06).
