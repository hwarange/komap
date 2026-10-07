---
name: komap-status-r7
description: "KoMaP project status (updated 2026-10-07 13:20) — sub4 best, sub5 recipe (~80.19), what failed in rounds 8–10, where results live"
metadata:
  node_type: memory
  type: project
  originSessionId: 89f50a3c-b870-47bf-a4b0-c4aa7fae0988
  modified: 2026-10-07T04:19:50.197Z
---

Status as of 2026-10-07 13:20 KST (full detail in handoff/HANDOFF_latest.md).

**Submissions**
- Confirmed best: sub4 (~80.10 CV). The user has not yet chosen between sub4 and sub4_full.
- sub5 recipe, evaluated with nested CV:
  - A26 = 25 earlier experiments + `x2_unetpp_bw_e120_r90`
  - blend the Eutectic Si specialists (`sie_x2_unetpp` + `sie_x2_cnx`) at α 0.4
  - blend the Al3Ni specialist (`al3ni_x2_unetpp`, class 1) at α 0.3
  - refit the Si classifier → about 80.19
- `submit.py` still lacks the class-1 (Al3Ni) blend option.

**Round 8 (adopted)**
- `x2_unetpp_bw_e120_r90` is the new best single model at 80.03 (+0.32 over 60 epochs).
- A26 and the greedy G8 ensemble both reach about 80.19, so A26 was chosen because it involves no selection.

**Rounds 9–10 (rejected or redundant)**
- Failed: soft-edge BLS, hard-crop, `--feats` filter channels.
- The Al3Ni-vs-SiE pair specialist adds nothing on top of the existing specialists.
- Still running: seed 1 of the best model and its full-90 version. The review watcher writes `runs/review9/all.done` when finished.

**Pattern:** resolution, longer training and diverse specialists help. Changing or adding training signal (loss weighting, augmentation, input filters, hard mining) hurts, especially Eutectic Si.

**Awaiting user decisions**
- 3× resolution run (recommended)
- 200-epoch run
- sub4 vs sub4_full

**Schedule:** the report starts 10/8 and `reports/round8_10_results.md` is not written yet. Deadline 10/15.

**Why:** Lets a new session pick up without re-deriving the state.

**How to apply:** Read the handoff ★ section first. Check the rejected lists before proposing ideas. Related: [[local-env-cpu-only]], [[feedback-wait-notify]].
