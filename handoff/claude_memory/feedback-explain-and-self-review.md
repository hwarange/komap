---
name: feedback-explain-and-self-review
description: "When running experiments or making choices, always explain the reasoning in detail and self-review for mistakes or gaps"
metadata:
  node_type: memory
  type: feedback
  originSessionId: df8d7375-2335-419b-861a-6a4d16ad2a94
  modified: 2026-10-05T21:00:43.062Z
---

While running work (experiments, design choices), always explain in detail WHY each choice was made. In the same explanation, critically check my own choices for missed points or mistakes. This covers earlier decisions too.

**Why:** The user wants to understand the reasoning and to catch flawed assumptions early (said 2026-10-05, "항상").

Two follow-ups from 2026-10-06:
- When an experiment round runs, report EACH experiment as it finishes. Include the result plus a bug check and a missed-points check.
- Keep the handoff as ONE file (handoff/HANDOFF_latest.md) with no timestamped copies. Don't auto-update it unless asked.

Third follow-up (2026-10-06): the user pointed out that a "cheap screen, keep only what works" rule causes false negatives.
- If an idea failed and the failure is plausibly a setting problem (weight, LR, training length), give it a second test with another setting before discarding it. Example: the Si classifier went from +0.02 to +0.10 once the application scope changed.
- Judge ensemble members by ensemble contribution, not single-model score.
- Keep a parking-lot list of rejected ideas, each with its retry condition.

**How to apply:**
- Every experiment launch or decision gets a "why" section and a "자기 점검" section.
- List confounds, unverified assumptions, and possible errors honestly, and fix them when cheap.

Related: [[user-prefers-plain-korean]], [[komap-competition]].
