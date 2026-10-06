---
name: komap-status-r7
description: KoMaP project status after round 7 (2026-10-06 16:00) — sub4 (~80.10 CV) is the best candidate; where the results live
metadata:
  node_type: memory
  type: project
  originSessionId: 89f50a3c-b870-47bf-a4b0-c4aa7fae0988
  modified: 2026-10-06T06:44:31.306Z
---

Status as of 2026-10-06 16:00 KST:

- **Best candidate is sub4** (honest estimate about 80.10; sub2 was 79.96). It combines:
  - the 25-experiment ensemble (A),
  - a 30% blend of the Eutectic Si specialist `sie_x2_unetpp` (`train.py --binary-class 3`),
  - the Si classifier refit on the blended probabilities.
- sub4_full adds the full-90 models (about 10% weight, 0.036% of pixels differ). The user has not yet chosen between sub4 and sub4_full.
- Rounds 6–7 are written up in `reports/round6_7_results.md`, and `reports/idea_journey.md` is updated.
- Rejected in round 7:
  - per-class decision bias (already optimal)
  - cascade "SiE first" (−0.3 to −4.7)
  - Al3Ni/SiE component classifier (upper bound only +0.14 on large components)
  - elastic, cutout, mixup and CutMix augmentation (−0.06 to −0.20)
- New tools: `decision_rules.py`, `oracle_components.py`, `check_full.py`, `verify_cv_blend.py`, and the `submit.py` options `--recipe`, `--full-mix`, `--sie-spec`.
- Schedule: accuracy work until 10/7, then the report from 10/8. The deadline is 10/15.

**Why:** Lets a new session pick up without re-deriving the state.

**How to apply:** Check `reports/round6_7_results.md` before proposing experiments, so rejected ideas aren't repeated. Related: [[local-env-cpu-only]], [[feedback-wait-notify]].
