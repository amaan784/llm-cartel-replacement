"""One command for the whole study: experiments, then results, analysis and the Word export.

    uv run python src/run_all.py              # start (or reattach to) the pipeline, wait, then collect/analyze/export
    uv run python src/run_all.py --no_wait    # only make sure the pipeline is running, then exit
    uv run python src/run_all.py --skip_runs  # only collect, analyze and export what exists now
    bash run_all.sh                           # same as the first line

Steps:
 1. check that OPENROUTER_API_KEY is set in vendor/collusion_hackathon/compute.env (never printed);
 2. start src/pipeline.py outside this terminal if it is not already running (it survives closing it);
 3. wait until the pipeline reports that every base run and branch has finished (status every 10 min);
 4. src/collect.py   -> copy run/branch/snapshot files and the design state into results/
 5. src/analyze.py   -> out/ tables, figures, checks.md, numbers.json
 6. src/notice.py    -> (optional, --notice) does the incumbent's reasoning notice the swap?
 7. src/export_docx.py -> report/report.docx
Safe to rerun at any point: the pipeline resumes from its state file and the later steps are idempotent.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HARNESS_PY, LOGS, REPO, key_is_set, now, pid_alive, read_json  # noqa: E402

PY = sys.executable  # the analysis environment (uv run) has numpy + matplotlib + pandoc


def step(title: str, cmd: list) -> None:
    print(f"\n{now()} == {title}: {' '.join(str(c) for c in cmd[1:])}", flush=True)
    rc = subprocess.call([str(c) for c in cmd], cwd=str(REPO))
    if rc != 0:
        sys.exit(f"{title} failed (exit code {rc})")


def pipeline_running() -> bool:
    lock = read_json(LOGS / "pipeline.lock", {}) or {}
    return pid_alive(lock.get("pid"), lock.get("start"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no_wait", action="store_true", help="start the pipeline if needed and exit")
    ap.add_argument("--skip_runs", action="store_true", help="do not start or wait for the pipeline")
    ap.add_argument("--notice", action="store_true", help="also run the reasoning classifier (paid, small)")
    a = ap.parse_args()

    if not a.skip_runs:
        if not key_is_set():
            sys.exit("OPENROUTER_API_KEY is empty: add it to vendor/collusion_hackathon/compute.env, then rerun.")
        if pipeline_running():
            print(f"{now()} pipeline already running; attaching")
        else:
            step("start pipeline", [PY, "src/pipeline.py", "start"])
            time.sleep(10)
        if a.no_wait:
            print("pipeline running; check progress with: python src/status.py")
            return
        last, restarts = 0.0, 0
        while True:  # watchdog: while this window is open, a pipeline that disappears is restarted
            st = read_json(LOGS / "pipeline_state.json", {}) or {}
            if st.get("finished"):
                print(f"{now()} pipeline finished at {st['finished']}")
                break
            if not pipeline_running():
                time.sleep(60)  # allow for a restart in progress
                st = read_json(LOGS / "pipeline_state.json", {}) or {}
                if not pipeline_running() and not st.get("finished"):
                    if st.get("smoke") == "failed":
                        sys.exit("the smoke test failed 3 times (bad key, no credit or no internet?): see "
                                 "logs/smoke.log, fix, then rerun this script")
                    if restarts >= 10:
                        sys.exit("the pipeline stopped 10 times: see logs/pipeline.log and NOTES.md")
                    restarts += 1
                    with open(REPO / "NOTES.md", "a", encoding="utf-8") as f:
                        f.write(f"- {now()[:16]} [run_all] pipeline process had disappeared; restarted it "
                                f"(restart {restarts}); state and finished work are kept\n")
                    step(f"restart pipeline ({restarts})", [PY, "src/pipeline.py", "start"])
                    time.sleep(15)
            if time.time() - last > 600:
                last = time.time()
                subprocess.call([PY, "src/status.py"], cwd=str(REPO))
            time.sleep(60)

    step("collect results", [PY, "src/collect.py"])
    step("analysis", [PY, "src/analyze.py"])
    if a.notice:
        step("reasoning classifier", [str(HARNESS_PY), "src/notice.py", "classify"])
    step("Word export", [PY, "src/export_docx.py"])
    print(f"\n{now()} done: see out/ (figures, table_main.md, numbers.json, checks.md) and report/report.docx")


if __name__ == "__main__":
    main()
