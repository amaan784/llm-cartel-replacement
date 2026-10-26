"""Stop the study early without losing finished work.

    python src/stop.py graceful   # start nothing new; running branches and base runs finish their current job
    python src/stop.py now        # stop every pipeline, launcher and harness process right away
    python src/stop.py resume     # undo 'graceful' (the running pipeline starts launching again)
    python src/stop.py now --dry_run   # only list what 'now' would stop

What is kept either way: every completed round (each round is appended to its run file as it is played),
every finished branch, and the pipeline state. After 'now', branches that were half-way through are rerun
from scratch when the pipeline is restarted (`python src/pipeline.py start`); base runs resume where they
stopped. To build the report from whatever is finished: `bash run_all.sh --skip_runs`.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LOGS, NO_WINDOW, NOTES, REPO, kill_tree, now, read_json, write_json  # noqa: E402

PATTERN = r"pipeline\.py|launch\.py|branch_game\.py|run_game\.py"


def our_processes():
    """(pid, command line) of python processes of this project (matched on script name and repo path)."""
    ps = ("Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | "
          "Select-Object ProcessId, CommandLine | ConvertTo-Json -Compress")
    out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps], capture_output=True,
                         text=True, creationflags=NO_WINDOW).stdout.strip()
    rows = json.loads(out) if out else []
    rows = rows if isinstance(rows, list) else [rows]
    import re

    repo = str(REPO).lower()
    keep = []
    for r in rows:
        cmd = r.get("CommandLine") or ""
        if re.search(PATTERN, cmd) and (repo in cmd.lower() or "llm-cartel" in cmd.lower()) and \
                r["ProcessId"] != os.getpid():
            keep.append((r["ProcessId"], cmd))
    return keep


def control(update: dict) -> int:
    path = LOGS / "pipeline_control.json"
    c = read_json(path, {}) or {}
    st = read_json(LOGS / "pipeline_state.json", {}) or {}
    ver = max(int(c.get("version", 0)), int(st.get("control_version", 0))) + 1
    c = {k: v for k, v in c.items() if k in ("budget_limit", "budget_pause", "max_branches", "deadline")}
    c.update(update, version=ver)
    write_json(path, c)
    return ver


def note(msg: str) -> None:
    with open(NOTES, "a", encoding="utf-8") as f:
        f.write(f"- {now()[:16]} [stop.py] {msg}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["graceful", "now", "resume"])
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    if a.mode in ("graceful", "resume"):
        if a.dry_run:
            print(f"would write logs/pipeline_control.json with {'pause' if a.mode == 'graceful' else 'unpause'}")
            return
        ver = control({"pause": True} if a.mode == "graceful" else {"unpause": True})
        if a.mode == "graceful":
            (LOGS / "PAUSE").write_text(now(), encoding="utf-8")  # launchers check this before relaunching
            note("graceful stop requested: no new launches; running jobs finish")
            print(f"graceful stop requested (control v{ver}); running jobs will finish. Check: python src/status.py")
        else:
            note("resume requested")
            print(f"resume requested (control v{ver}); the running pipeline will start launching again")
        return
    procs = our_processes()
    if a.dry_run:
        print(f"would stop {len(procs)} processes:")
        for pid, cmd in procs:
            print(f"  {pid}  {cmd[:140]}")
        return
    order = lambda pc: (0 if "pipeline.py" in pc[1] else 1 if "launch.py" in pc[1] else 2)  # stop the boss first
    for pid, _ in sorted(procs, key=order):
        kill_tree(pid)
    left = our_processes()
    note(f"stopped now: {len(procs)} processes ({len(left)} left); finished rounds and branches are kept")
    print(f"stopped {len(procs) - len(left)} of {len(procs)} processes. Restart later with: python src/pipeline.py start")


if __name__ == "__main__":
    main()
