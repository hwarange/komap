---
name: komap-status
description: "KoMaP project status (2026-10-08 01:40) — sub5 80.233 best; round 13 running overnight (x3 e120, x2 MiT, x2 ConvNeXt, Al3Ni specialist 2); sub6 decision in the morning"
metadata:
  node_type: memory
  type: project
  originSessionId: 115359f4-9d03-43f3-ae46-b50d8a5ba8bb
  modified: 2026-10-07T16:27:11.959Z
---

Status as of 2026-10-08 01:40 KST. Full detail is in handoff/HANDOFF_latest.md (rewritten at 01:40), section ★.

**Submissions**
- sub5 is the best, at 80.233 honest. It is backed up in git.
- The user will choose among sub5, sub5_full, sub6 and sub6_full together, once sub6 exists.
- Replacement rule: switch only at +0.1 or more honest.
- How the honest estimate is computed: the 27-model mean plus the nested deltas of the SiE blend, the Al3Ni blend and the Si classifier.

**Round 12 (done)**
- Best combo candidates reached 80.325 (+0.09), so sub5 was kept.
- High-res models dominate in the ensemble. The mean of 4 high-res models (80.23) beat the 29-model mean (80.10), and x3 was picked in all 5 folds.

**Round 13 (started 10/8 01:22 on the server; user approved ideas 2–5)**
- Models:
  - `x3_unetpp_bw_e120_r90`
  - `x2_mit_bw_r90` (26.6GB, gets a whole GPU)
  - `x2_cnx_bw_e120`
  - `al3ni_x2_cnx_e120`
- `scripts/run_round13.sh` evaluates each model automatically, then runs test predictions, the final combos (`runs/combo/summary13.txt`, about 09:00) and `ensemble.py` (r13). It writes `runs/review13/all.done` at about 10:00–10:30.
- `launch.py --heavy` was added for whole-GPU jobs.

**Morning steps**
1. Check the results.
2. Pick the high-res weight k by nested CV (code not written yet).
3. Build sub6 from the `run_sub5.sh` template. `verify_cv_blend.py` needs a weight option if k > 1.
4. The user picks the final submission.
5. Then focus on the report. Accuracy work ends 10/9 morning; deadline 10/15.

**Why:** Lets a new session pick up without re-deriving the state.
**How to apply:** Read the handoff ★ first, and check the server before trusting the timing estimates. Related: [[komap-competition]], [[local-env-cpu-only]], [[feedback-wait-notify]].
