---
name: feedback-parallel-and-record
description: Use idle server CPU/GPU in parallel instead of waiting (limit threads on the shared server); record idea discussions in reports/idea_journey.md
metadata:
  node_type: memory
  type: feedback
  originSessionId: 115359f4-9d03-43f3-ae46-b50d8a5ba8bb
  modified: 2026-10-07T16:27:22.716Z
---

1. When a job is slow and the server has idle CPU or GPU, don't just wait. Start the next useful step in parallel. Examples: run the decision-making evaluation directly, or run test predictions ahead of time.
2. When the user and I discuss ideas or answer their questions (e.g. "큰 모델은?", "ViT는?", upgrade candidates), record the discussion. Put the question, the answer with evidence, and the decision in reports/idea_journey.md.

**Why:** On 2026-10-08 the user asked "더 빠르게는 못해? cpu든 gpu든 사용량들이 넉넉한데" while a single-threaded ensemble.py held things up. Later they asked to "이런 아이디어를 나눈것도 기록을 해두고".

**How to apply:**
- Before launching parallel CPU jobs, check how many cores each one uses. The server is shared, and 15 HistGradientBoosting runs × 128 threads pushed the load to 717. Set `OMP_NUM_THREADS`.
- Record discussions in the same session, and commit and push them.

Related: [[feedback-wait-notify]], [[feedback-explain-and-self-review]].
