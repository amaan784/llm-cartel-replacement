"""Resume loop around the organizer's run_game.py: one process per seed, relaunch until done.

    # foreground loop (what the detached process runs)
    python src/launch.py run --name O_qwen3-14b_P0_s0 --models qwen/qwen3-14b --prefix P0 --rounds 150 --seed 0

    # same, detached from this shell, logging to logs/<name>.log
    python src/launch.py spawn --name O_qwen3-14b_P0_s0 --models qwen/qwen3-14b --prefix P0 --rounds 150 --seed 0

run_game.py resumes a run of the same --name (keeping only complete, successful rounds) and extends a
finished run when --rounds grows, so relaunching after any failure is safe. A state file
logs/<name>.state.json records pid, target, attempts and the last exit code for src/status.py.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CARTEL,
    DEFAULT_RESULTS,
    HARNESS_PY,
    LOGS,
    NO_WINDOW,
    child_env,
    now,
    pid_alive,
    proc_start,
    read_json,
    read_records,
    redirect_output,
    spawn_detached,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["run", "spawn"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--models", required=True, help="OpenRouter id(s), passed to run_game --models")
    ap.add_argument("--prefix", default="P0")
    ap.add_argument("--rounds", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--max_retries", type=int, default=8)
    ap.add_argument("--max_attempts", type=int, default=40, help="give up after this many relaunches")
    ap.add_argument("--max_stuck", type=int, default=6,
                    help="give up after this many consecutive relaunches without a new round (state 'stuck')")
    ap.add_argument("--backoff", type=float, default=30.0, help="seconds between relaunches")
    ap.add_argument("--results_dir", default=str(DEFAULT_RESULTS))
    ap.add_argument("--logs_dir", default=str(LOGS))
    ap.add_argument("--base_url", default=None, help="testing only: route custom: models to this endpoint")
    ap.add_argument("--log", default=None, help="redirect this process's output (used when detached)")
    return ap


def run_path(args) -> Path:
    return Path(args.results_dir) / "jsonl" / f"{args.name}.jsonl"


def state_path(args) -> Path:
    return Path(args.logs_dir) / f"{args.name}.state.json"


def harness_cmd(args) -> list:
    cmd = [
        str(HARNESS_PY), "run_game.py",
        "--models", args.models,
        "--prefix", args.prefix,
        "--rounds", str(args.rounds),
        "--seeds", str(args.seed),
        "--name", args.name,
        "--out_dir", str(Path(args.results_dir) / "jsonl"),
        "--max_retries", str(args.max_retries),
        "-y",
    ]
    if args.think:
        cmd.append("--think")
    return cmd


def repair(path: Path) -> int:
    """Drop a partial/unparseable tail (e.g. after a power cut mid-append) so run_game can resume.

    run_game's own reader json-parses every line and would crash on it. The original file is kept as .bak.
    Returns the number of lines removed."""
    import json
    import shutil

    if not path.exists():
        return 0
    good = read_records(path)
    lines = [ln for ln in open(path, encoding="utf-8") if ln.strip()]
    if len(lines) == len(good):
        return 0
    shutil.copyfile(path, str(path) + ".bak")
    with open(path, "w", encoding="utf-8") as f:
        for r in good:
            f.write(json.dumps(r) + "\n")
    return len(lines) - len(good)


def run(args) -> int:
    import os

    redirect_output(args.log)
    sp = state_path(args)
    state = read_json(sp, {}) or {}
    if state.get("pid") and state.get("pid") != os.getpid() and state.get("state") == "running" and pid_alive(
            state["pid"], state.get("start")):
        print(f"{now()} {args.name}: another launcher (pid {state['pid']}) is running; exiting")
        return 1
    extra = {"LLM_BASE_URL": args.base_url} if args.base_url else None
    env = child_env(extra)
    state = {
        "name": args.name, "pid": os.getpid(), "start": proc_start(os.getpid()), "target": args.rounds,
        "state": "running", "attempts": 0, "no_progress": 0, "started": now(), "cmd": harness_cmd(args),
        "results": str(run_path(args)),
    }
    write_json(sp, state)
    pause_file = Path(args.logs_dir) / "PAUSE"
    while True:
        done = len(read_records(run_path(args)))
        if done >= args.rounds:
            state.update(state="done", rounds=done, updated=now())
            write_json(sp, state)
            print(f"{now()} {args.name}: done ({done} rounds)")
            return 0
        if pause_file.exists():
            state.update(state="paused", rounds=done, updated=now())
            write_json(sp, state)
            print(f"{now()} {args.name}: budget pause file present; stopping at {done} rounds")
            return 3
        if state["no_progress"] >= args.max_stuck or state["attempts"] >= args.max_attempts:
            state.update(state="stuck" if state["no_progress"] >= args.max_stuck else "failed", rounds=done,
                         updated=now())
            write_json(sp, state)
            print(f"{now()} {args.name}: giving up ({state['state']}) after {state['attempts']} attempts at {done} rounds")
            return 2
        cut = repair(run_path(args))
        if cut:
            print(f"{now()} {args.name}: removed {cut} partial line(s) at the end (original kept as .bak)", flush=True)
        state["attempts"] += 1
        state.update(rounds=done, updated=now())
        write_json(sp, state)
        print(f"{now()} {args.name}: attempt {state['attempts']} from round {done}", flush=True)
        sys.stdout.flush()
        rc = subprocess.call(harness_cmd(args), cwd=str(CARTEL), env=env, stdout=sys.stdout,
                             stderr=subprocess.STDOUT, creationflags=NO_WINDOW)
        after = len(read_records(run_path(args)))
        state["no_progress"] = 0 if after > done else state["no_progress"] + 1
        state.update(last_rc=rc, rounds=after, updated=now())
        write_json(sp, state)
        if after < args.rounds:
            wait = min(args.backoff * 2 ** max(state["no_progress"] - 1, 0), 900)
            print(f"{now()} {args.name}: run_game exited rc={rc} at {after} rounds; retrying in {wait:.0f}s", flush=True)
            time.sleep(wait)


def spawn_run(name, models, prefix, rounds, seed, think=False, results_dir=DEFAULT_RESULTS, logs_dir=LOGS,
              base_url=None, max_retries=8) -> int:
    """Start `launch.py run ...` detached from the caller; returns the pid."""
    log = Path(logs_dir) / f"{name}.log"
    cmd = [sys.executable, str(Path(__file__).resolve()), "run", "--name", name, "--models", models,
           "--prefix", prefix, "--rounds", str(rounds), "--seed", str(seed), "--max_retries", str(max_retries),
           "--results_dir", str(results_dir), "--logs_dir", str(logs_dir), "--log", str(log)]
    if think:
        cmd.append("--think")
    if base_url:
        cmd += ["--base_url", base_url]
    return spawn_detached(cmd, Path(logs_dir) / "spawn.log", cwd=Path(__file__).resolve().parents[1],
                          env=child_env({"LLM_BASE_URL": base_url} if base_url else None))


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "run":
        sys.exit(run(args))
    pid = spawn_run(args.name, args.models, args.prefix, args.rounds, args.seed, args.think, args.results_dir,
                    args.logs_dir, args.base_url, args.max_retries)
    print(f"spawned {args.name} pid {pid} log {Path(args.logs_dir) / (args.name + '.log')}")


if __name__ == "__main__":
    main()
