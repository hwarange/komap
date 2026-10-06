---
name: feedback-wait-notify
description: "Wait on server jobs by checking result/done files; never use `pgrep -f` loops, which match themselves and hang"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 89f50a3c-b870-47bf-a4b0-c4aa7fae0988
  modified: 2026-10-06T05:28:54.061Z
---

When waiting for server jobs, wait until a result or done file appears, using a local background Bash until-loop. Never use `while pgrep -f "<pattern>"`, because it matches the waiting shell's own command line and never ends.

**Why:** On 2026-10-06 a `pgrep -f` wait loop hung, so the Si-classifier and augmentation results sat unreported for over an hour. The user noticed first and asked "뭐야 끝났는데 알람설정같은거 안되어있어?". A foreground command that hits the 10-minute timeout also moves to the background and reports late.

**How to apply:**
- For long jobs, start them detached (setsid nohup). Write an `exit $?` line or a done file when each finishes.
- Watch with `run_in_background` and an until-loop that checks for that file, so one notification arrives at completion.
- Report results as soon as they land.

Related: [[feedback-explain-and-self-review]]
