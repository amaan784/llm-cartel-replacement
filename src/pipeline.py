"""Unattended orchestrator for the replacement study (pre-registration in NOTES.md).

    python src/pipeline.py run                  # foreground loop (normally started detached, see below)
    python src/pipeline.py start                # start the loop detached from this shell, log logs/pipeline.log
    python src/pipeline.py plan                 # print what would run and the cost estimate, then exit

What it does, every --tick seconds, idempotently (state in logs/pipeline_state.json, safe to restart):
 1. waits until OPENROUTER_API_KEY is set in compute.env (never reads it out loud);
 2. runs the smoke test (gpt-4.1-mini, 5 rounds);
 3. keeps every base run alive through src/launch.py (Track O: qwen3-14b P0, seeds 0-5, 150 rounds;
    the model screen: gpt-4.1-mini / gpt-4.1 / gemini-2.5-flash, P0 --think, 60 rounds, seeds 0-1);
 4. when the screen is done, applies the pre-registered FAST rule, launches Track F and the B baselines;
 5. for every base run, as soon as it has crossed (find_branch rule) and reached the branch round b,
    snapshots rounds 0..b and queues the arms x sampling seeds (101, 102) through branch_game.py;
 6. runs at most --max_branches branch processes at once (slow track first, seed 101 before 102),
    reruns failed branches from scratch (max 3 reruns), applies the pre-authorized fallbacks;
 7. pauses new launches if spend reaches --budget_pause (USD) and appends progress to NOTES.md.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CARTEL,
    DEFAULT_RESULTS,
    HARNESS_PY,
    LOGS,
    NO_WINDOW,
    NOTES,
    REPO,
    child_env,
    config_path,
    file_cost,
    first_crossing,
    key_is_set,
    load_config,
    median,
    now,
    openrouter_usage,
    kill_tree,
    pid_alive,
    pidx_series,
    proc_start,
    prices,
    read_json,
    read_records,
    redirect_output,
    round_seconds,
    spawn_detached,
    spawn_outside_app,
    write_json,
)
from launch import spawn_run  # noqa: E402

# ------------------------------------------------------------------------------------------------
# pre-registered design (NOTES.md items 6-10)
# ------------------------------------------------------------------------------------------------
TRACK_O_MODEL = "qwen/qwen3-14b"
TRACK_O_SIBLING = "qwen/qwen3-32b"
SCREEN_MODELS = ["openai/gpt-4.1-mini", "openai/gpt-4.1", "google/gemini-2.5-flash"]
SIBLING_F = {
    "openai/gpt-4.1-mini": "openai/gpt-4.1",
    "openai/gpt-4.1": "openai/gpt-4.1-mini",
    "google/gemini-2.5-flash": "google/gemini-2.5-flash-lite",
}
SHORT = {
    "qwen/qwen3-14b": "qwen3-14b",
    "qwen/qwen3-32b": "qwen3-32b",
    "openai/gpt-4.1-mini": "gpt41mini",
    "openai/gpt-4.1": "gpt41",
    "openai/gpt-4.1-nano": "gpt41nano",
    "google/gemini-2.5-flash": "gem25flash",
    "google/gemini-2.5-flash-lite": "gem25flashlite",
    "anthropic/claude-haiku-5.5": "claudehaiku55",
}
BRANCH_SEEDS = [101, 102]
B_SEEDS = [10, 11, 12, 13]
B_ROUNDS = 50
SCREEN_ROUNDS = 60
BASE_ROUNDS = 150
MAX_RERUNS = 3
ARM_ORDER = ["C0", "R1", "R2", "R3", "R4", "R5", "R5f", "RP", "RN", "RNf"]
NEW_MODEL = "anthropic/claude-haiku-5.5"  # exploratory current-generation newcomer (Track F only)
# qwen3-14b reasoning reaches the harness's 4096-token cap (median 2,948, p90 3,802, max 4,064 output tokens in the
# first 132 calls; 23% of calls re-sampled, two runs aborted after 9 cut-offs in a row): qwen runs get 8192.
LONG_TOKENS = 8192
TRACK_PRIORITY = {"O": 0, "OPI": 0, "F": 1}  # slow (qwen) tracks first: they are the critical path

# Output tokens per call used for cost projections until real usage is observed.
DEFAULT_OUT_TOKENS = {
    "qwen/qwen3-14b": 2500, "qwen/qwen3-32b": 2500, "google/gemini-2.5-flash": 900,
    "google/gemini-2.5-flash-lite": 300, "openai/gpt-4.1-mini": 150, "openai/gpt-4.1": 150,
}


def other_provider(model: str) -> str:
    return "google/gemini-2.5-flash" if model.startswith("openai/") else "openai/gpt-4.1"


def run_name(track: str, model: str, prefix: str, think: bool, seed: int) -> str:
    return f"{track}_{SHORT.get(model, model.split('/')[-1])}_{prefix}{'think' if think else ''}_s{seed}"


def safe(s: str) -> str:
    return s.replace("/", "-").replace(":", "-").replace(".", "p")


class Pipeline:
    def __init__(self, args):
        self.a = args
        self.results = Path(args.results_dir)
        self.logs = Path(args.logs_dir)
        self.notes = Path(args.notes)
        self.state_file = self.logs / "pipeline_state.json"
        self.base_url = f"http://127.0.0.1:{args.mock_port}/v1" if args.mock else None
        if self.state_file.exists():
            self.s = read_json(self.state_file, None)
            if not isinstance(self.s, dict):
                sys.exit(f"{self.state_file} exists but cannot be read; refusing to start fresh (that would rerun and "
                         f"overwrite finished branches). Restore it from {self.state_file}.bak or move it away.")
        else:
            self.s = self.fresh_state()
        for k, v in (self.s.get("overrides") or {}).items():  # operator overrides survive restarts
            setattr(args, k, v if v is None or getattr(args, k) is None else type(getattr(args, k))(v))
        if bool(self.s.get("mock")) != bool(args.mock):
            sys.exit(f"{self.state_file} was written with mock={self.s.get('mock')}, but --mock={args.mock}; "
                     "use separate --results_dir/--logs_dir/--notes for mock tests")
        self.deadline = datetime.strptime(args.deadline, "%Y-%m-%d %H:%M") if args.deadline else None
        self.usage_misses = 0

    # ---------------------------------------------------------------- state / notes
    def fresh_state(self) -> dict:
        return {
            "created": now(), "mock": bool(self.a.mock), "usage_at_start": None, "smoke": "pending",
            "screen_decision": None, "tracks": {}, "bases": {}, "branches": {}, "fallbacks": [],
            "deviations": [], "paused": None, "last_progress": 0, "finished": False, "events": [],
        }

    def save(self) -> None:
        try:
            if self.state_file.exists():
                shutil.copyfile(self.state_file, str(self.state_file) + ".bak")
            write_json(self.state_file, self.s)
        except OSError as e:  # e.g. the file is held open by a viewer; try again next tick
            print(f"{now()} could not save state: {e}", flush=True)

    def note(self, msg: str, also_event: bool = True) -> None:
        line = f"- {now()[:16]} [pipeline] {msg}"
        print(line, flush=True)
        try:
            with open(self.notes, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError as e:
            print(f"could not append to {self.notes}: {e}", flush=True)
        if also_event:
            self.s["events"].append(line)

    def deviation(self, msg: str) -> None:
        self.s["deviations"].append(f"{now()} {msg}")
        self.note(f"DEVIATION: {msg}")

    def mid(self, model: str) -> str:
        """Model id as passed to the harness (mock mode routes every id to the local mock)."""
        return f"custom:{model}" if self.a.mock else model

    # ---------------------------------------------------------------- tracks and runs
    def add_track(self, tid, model, prefix, think, seeds, rounds, max_t, sibling, x) -> None:
        if tid in self.s["tracks"]:
            return
        slow_h40 = TRACK_PRIORITY.get(tid) == 0 and "horizon40_slow" in self.s["fallbacks"]
        self.s["tracks"][tid] = {
            "id": tid, "model": model, "prefix": prefix, "think": think, "sibling": sibling, "X": x,
            "horizon": 40 if slow_h40 else self.a.horizon, "max_t": max_t, "added": now(),
        }
        for seed in seeds:
            self.add_base(tid, model, prefix, think, seed, rounds, max_t, role="base")

    def add_base(self, tid, model, prefix, think, seed, rounds, max_t, role) -> str:
        name = run_name(tid, model, prefix, think, seed)
        if name not in self.s["bases"]:
            self.s["bases"][name] = {
                "name": name, "track": tid, "role": role, "model": model, "prefix": prefix, "think": think,
                "seed": seed, "rounds": rounds, "max_t": max_t, "pid": None, "status": "pending",
                "crossing": None, "branch_round": None, "enqueued": False, "spawns": 0,
            }
        return name

    def run_file(self, name: str) -> Path:
        return self.results / "jsonl" / f"{name}.jsonl"

    def ensure_bases(self) -> None:
        for name, b in self.s["bases"].items():
            if b["status"] in ("failed", "held"):
                continue
            n = len(read_records(self.run_file(name)))
            b["done_rounds"] = n
            if n >= b["rounds"]:
                if b["status"] != "complete":
                    b["status"] = "complete"
                continue
            if n and not self.models_ok(name, b):
                continue
            st = read_json(self.logs / f"{name}.state.json", {}) or {}
            alive = pid_alive(b.get("pid"), b.get("pid_start")) or (
                st.get("state") == "running" and pid_alive(st.get("pid"), st.get("start")))
            if alive:
                b["status"] = "running"
                continue
            if st.get("state") in ("failed", "stuck") and st.get("target") == b["rounds"]:
                if b["spawns"] >= 2:
                    b["status"] = "failed"
                    self.note(f"base run {name} FAILED ({st['state']}, last rc {st.get('last_rc')}) at {n} rounds "
                              f"after {b['spawns']} launcher(s); excluded")
                    continue
            if b["spawns"] >= 5:
                b["status"] = "failed"
                self.note(f"base run {name}: {b['spawns']} launchers ended without finishing (last state "
                          f"{st.get('state')}, rc {st.get('last_rc')}); marked failed and excluded")
                continue
            if self.s.get("paused") or self.breaker_active() or time.time() < b.get("not_before", 0):
                continue
            b["not_before"] = time.time() + 60 * (b["spawns"] + 1)
            pid = spawn_run(name, self.mid(b["model"]), b["prefix"], b["rounds"], b["seed"], b["think"],
                            self.results, self.logs, self.base_url,
                            max_tokens=LONG_TOKENS if b["model"].startswith("qwen/") else None)
            b.update(pid=pid, pid_start=proc_start(pid), status="running", spawns=b["spawns"] + 1)
            self.note(f"launched {name} ({b['model']}, {b['prefix']}{', think' if b['think'] else ''}, "
                      f"seed {b['seed']}, to {b['rounds']} rounds) pid {pid}", also_event=False)

    def models_ok(self, name: str, b: dict) -> bool:
        """The resolved model ids in a run's config must be exactly the requested ones (no silent reroute)."""
        cfg = read_json(config_path(self.run_file(name)), {}) or {}
        want = [("custom:" if self.a.mock else "openrouter:") + b["model"]] * 2
        if cfg.get("models") and cfg["models"] != want:
            b["status"] = "failed"
            self.note(f"MODEL MISMATCH in {name}: config has {cfg['models']}, expected {want}; run excluded, "
                      "check compute.env for LLM_MODEL / LLM_BASE_URL / NEBIUS_API_KEY")
            return False
        return True

    # ---------------------------------------------------------------- smoke test
    def smoke(self) -> str:
        """'ok', 'wait' (retry on a later tick) or 'failed' (3 failures in this process: stop)."""
        if self.s["smoke"] == "done":
            return "ok"
        if self.s["smoke"] == "failed":
            return "failed"
        if time.time() < self.s.get("smoke_retry_at", 0):
            return "wait"
        path = self.run_file("smoke")
        if len(read_records(path)) < 5:
            cmd = [str(HARNESS_PY), "run_game.py", "--models", self.mid("openai/gpt-4.1-mini"), "--rounds", "5",
                   "--name", "smoke", "--out_dir", str(self.results / "jsonl"), "--max_retries", "8", "-y"]
            self.note("smoke test: " + " ".join(cmd[1:]))
            env = child_env({"LLM_BASE_URL": self.base_url} if self.base_url else None)
            with open(self.logs / "smoke.log", "a", encoding="utf-8") as log:
                rc = subprocess.call(cmd, cwd=str(CARTEL), env=env, stdout=log, stderr=subprocess.STDOUT,
                                     creationflags=NO_WINDOW)
            if len(read_records(path)) < 5:
                self.s["smoke_fail"] = self.s.get("smoke_fail", 0) + 1
                if self.s["smoke_fail"] >= 3:
                    self.s["smoke"] = "failed"
                    self.note(f"smoke test FAILED 3 times (rc={rc}); see logs/smoke.log. Pipeline stops; restart it "
                              "after fixing the key or endpoint.")
                    return "failed"
                self.s["smoke_retry_at"] = time.time() + 120
                self.note(f"smoke test failed (rc={rc}, attempt {self.s['smoke_fail']}); retrying in 2 min")
                return "wait"
        cfg = load_config(path)
        recs = read_records(path)
        want = [self.mid("openai/gpt-4.1-mini") if self.a.mock else "openrouter:openai/gpt-4.1-mini"] * 2
        if cfg.get("models") != want:
            self.s["smoke"] = "failed"
            self.note(f"smoke test resolved models {cfg.get('models')} instead of {want}; stopping (check compute.env)")
            return "failed"
        self.s["smoke"] = "done"
        self.note(f"smoke test OK: resolved model {cfg.get('models')}, prices {[r['actions'] for r in recs]}")
        return "ok"

    # ---------------------------------------------------------------- screen and FAST
    def screen_names(self):
        return [run_name("S", m, "P0", True, s) for m in SCREEN_MODELS for s in (0, 1)]

    def decide_screen(self) -> None:
        if self.s["screen_decision"] is not None:
            return
        names = self.screen_names()
        if any(self.s["bases"][n]["status"] not in ("complete", "failed") for n in names):
            return
        rows, qual = [], []
        for m in SCREEN_MODELS:
            secs, crosses = [], []
            for s in (0, 1):
                n = run_name("S", m, "P0", True, s)
                recs = read_records(self.run_file(n))
                cfg = read_json(config_path(self.run_file(n)), {}) or {}
                c = first_crossing(pidx_series(recs, cfg), max_t=SCREEN_ROUNDS - 1)
                crosses.append(c)
                secs += [round_seconds(r) for r in recs]
            med = median(secs)
            rows.append({"model": m, "crossings": crosses, "median_round_seconds": med,
                         "resolved": (read_json(config_path(self.run_file(run_name('S', m, 'P0', True, 0))), {}) or {}).get("models")})
            if any(c is not None for c in crosses):
                qual.append((med if med is not None else 1e9, prices().get(m, {}).get("prompt", 1), m))
        if qual:
            # quickest qualifying model; models within 25% of the quickest count as tied and the cheaper one
            # wins (clarification added to the pre-registration at 03:40, before any API call)
            quickest = min(q[0] for q in qual)
            tied = [q for q in qual if q[0] <= 1.25 * quickest]
            fast = sorted(tied, key=lambda q: (q[1], q[0]))[0][2]
            f_model, f_prefix = fast, "P0"
            x_o = fast
        else:
            fast = None
            f_model, f_prefix = "openai/gpt-4.1-mini", "PI"
            x_o = sorted(rows, key=lambda r: (r["median_round_seconds"] or 1e9))[0]["model"]
        x_f = other_provider(f_model)
        dec = {"time": now(), "rows": rows, "FAST": fast, "trackF_model": f_model, "trackF_prefix": f_prefix,
               "X_O": x_o, "X_F": x_f, "sibling_F": SIBLING_F[f_model]}
        self.s["screen_decision"] = dec
        self.note("screen decision: " + json.dumps(dec))
        if fast is None:
            self.deviation("no screened model crossed pidx 0.5 on P0: Track F = gpt-4.1-mini with PI (nudged cartel), "
                           f"Track O X = quickest screened model {x_o}")
        # Track O was added at start without X; fill it now.
        self.s["tracks"]["O"]["X"] = x_o
        self.add_track("F", f_model, f_prefix, True, range(6), BASE_ROUNDS, 149, SIBLING_F[f_model], x_f)
        for seed in B_SEEDS:
            self.add_base("B-O", x_o, "P0", False, seed, B_ROUNDS, B_ROUNDS - 1, role="B")
            self.add_base("B-F", x_f, "P0", True, seed, B_ROUNDS, B_ROUNDS - 1, role="B")
        self.fit_track_f_to_budget()
        self.note(f"queued Track F ({f_model}, {f_prefix}, think) seeds 0-5 and baselines B-O ({x_o}, P0) and "
                  f"B-F ({x_f}, P0, think) seeds {B_SEEDS}; projected cost of the full plan "
                  f"${self.projected_total():.2f}")

    # ---------------------------------------------------------------- branches
    def arms(self, t: dict):
        np_ = "P0" if t["prefix"] != "P0" else None  # newcomers always get P0 in a nudged track
        x, m = t["X"], t["model"]
        out = [
            ("C0", {}, None, None),
            ("R1", {1: t["sibling"]}, None, np_),
            ("R2", {1: x}, None, np_),
            ("R3", {1: x}, 100, np_),
            ("R4", {1: m}, None, "AC"),
            ("R5", {0: x, 1: x}, None, np_),
            ("R5f", {0: x, 1: x}, 100, np_),
        ]
        if t["prefix"] != "P0":
            out.append(("RP", {1: m}, None, "P0"))
        if t["id"] == "F":  # exploratory (added 00:15 on 26 Oct, before any branch result): current-generation newcomer
            out += [("RN", {1: NEW_MODEL}, None, np_), ("RNf", {1: NEW_MODEL}, 100, np_)]
        return out

    def ensure_extras(self) -> None:
        """Exploratory additions for markets queued before they existed: missing arms and the B-N baseline."""
        if self.s["screen_decision"] is None:
            return
        for seed in B_SEEDS:
            self.add_base("B-N", NEW_MODEL, "P0", True, seed, B_ROUNDS, B_ROUNDS - 1, role="B")
        for name, b in self.s["bases"].items():
            if b["role"] != "base" or not b["enqueued"]:
                continue
            t = self.s["tracks"][b["track"]]
            snap = self.results / "snapshots" / f"{name}.jsonl"
            before = len(self.s["branches"])
            for arm, rep, win, pfx in self.arms(t):
                for seed in BRANCH_SEEDS:
                    self.add_branch(name, t, b["branch_round"], snap, arm, rep, win, pfx, seed)
            if len(self.s["branches"]) > before:
                self.note(f"queued {len(self.s['branches']) - before} added exploratory branches for {name}")

    def snapshot(self, name: str, b: int, recs: list) -> Path:
        """Rounds 0..b of a base run (+ its config) as branch_game input, from records already read."""
        snap_dir = self.results / "snapshots"
        snap_dir.mkdir(parents=True, exist_ok=True)
        src = self.run_file(name)
        recs = recs[: b + 1]
        if len(recs) != b + 1:
            raise RuntimeError(f"snapshot of {name} has {len(recs)} rounds, needs {b + 1}")
        dst = snap_dir / f"{name}.jsonl"
        with open(dst, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        shutil.copyfile(config_path(src), config_path(dst))
        return dst

    def scan_bases(self) -> None:
        for name, b in self.s["bases"].items():
            if b["role"] != "base" or b["enqueued"] or b["status"] in ("failed", "held"):
                continue
            t = self.s["tracks"][b["track"]]
            recs = read_records(self.run_file(name))
            cfg = read_json(config_path(self.run_file(name)), None)
            c = first_crossing(pidx_series(recs, cfg), max_t=b["max_t"])
            if c is None:
                b["colluding"] = False if len(recs) >= b["rounds"] else None
                continue
            br = c + 20
            if b["crossing"] is None:
                b.update(crossing=c, branch_round=br, colluding=True)
                self.note(f"{name} crossed pidx 0.5 (20-round mean) at round {c}; branch round {br}")
            if b["rounds"] < br + 1:
                self.note(f"extending {name} to {br + 1} rounds so the branch round exists")
                b["rounds"] = br + 1
            if len(recs) < br + 1 or t.get("X") is None:
                continue
            snap = self.snapshot(name, br, recs)
            jobs = 0
            for arm, rep, win, pfx in self.arms(t):
                for seed in BRANCH_SEEDS:
                    self.add_branch(name, t, br, snap, arm, rep, win, pfx, seed)
                    jobs += 1
            b["enqueued"] = True
            self.note(f"queued {jobs} branches for {name} at round {br} (pre-swap pidx "
                      f"{sum(pidx_series(recs, cfg)[br - 19: br + 1]) / 20:.3f})")

    def add_branch(self, run, t, br, snap, arm, rep, win, pfx, seed) -> None:
        jid = f"{run}__{arm}_s{seed}"
        if jid in self.s["branches"]:
            return
        label = f"{arm}_s{seed}"
        eff = label + (f"_win{win}" if win is not None and rep else "")
        out = self.results / "branches" / f"{run}__r{br}__{safe(eff)}.jsonl"
        self.s["branches"][jid] = {
            "id": jid, "track": t["id"], "run": run, "arm": arm, "seed": seed, "branch_round": br, "label": label,
            "replace": {str(k): v for k, v in rep.items()}, "window": win, "prefix": pfx, "snapshot": str(snap),
            "out": str(out), "state": "pending", "pid": None, "attempts": 0, "not_before": 0,
            "horizon": t["horizon"], "queued": now(),
        }
        if (TRACK_PRIORITY.get(t["id"]) == 0 and "one_seed_slow" in self.s["fallbacks"]
                and seed != BRANCH_SEEDS[0]):
            self.s["branches"][jid]["state"] = "dropped"  # pre-authorized time fallback already in force

    def branch_cmd(self, j: dict) -> list:
        # slow (qwen3) tracks: raise the agents' output cap through src/branch_long.py (see LONG_TOKENS)
        game = ([str(REPO / "src" / "branch_long.py"), "--max_tokens", str(LONG_TOKENS)]
                if TRACK_PRIORITY.get(j["track"]) == 0 else ["branch_game.py"])
        cmd = [str(HARNESS_PY), *game, j["snapshot"], "--at_round", str(j["branch_round"]),
               "--rounds", str(j["horizon"]), "--label", j["label"], "--seed", str(j["seed"]),
               "--out_dir", str(self.results / "branches"), "--max_retries", "8", "-y"]
        if j["replace"]:
            cmd += ["--replace"] + [f"{k}={self.mid(v)}" for k, v in sorted(j["replace"].items())]
        if j["window"] is not None:
            cmd += ["--replacement_window", str(j["window"])]
        if j["prefix"]:
            cmd += ["--replace_prefix", j["prefix"]]
        return cmd

    def branch_complete(self, j: dict) -> bool:
        out = Path(j["out"])
        recs = read_records(out)
        cfg = read_json(config_path(out), {}) or {}
        after = [r for r in recs if r.get("branch_phase") == "after"]
        return len(recs) == j["branch_round"] + 1 + j["horizon"] and len(after) == j["horizon"] and "comparison" in cfg

    def breaker_active(self) -> bool:
        return time.time() < self.s.get("breaker_until", 0)

    def schedule(self) -> None:
        running = 0
        for j in self.s["branches"].values():
            if j["state"] != "running":
                continue
            if pid_alive(j["pid"], j.get("pid_start")):
                running += 1
                continue
            if self.branch_complete(j):
                j.update(state="done", finished=now())
                continue
            # died early: what it paid for is lost on rerun (branch_game rewrites its file)
            self.s["lost_spend"] = self.s.get("lost_spend", 0.0) + file_cost(Path(j["out"]), only_after=True)
            deaths = [t for t in self.s.get("deaths", []) if time.time() - t < 300] + [time.time()]
            self.s["deaths"] = deaths
            if len(deaths) >= 3:
                # several jobs dying together = an API/account problem, not this branch: pause and do not
                # count it against the branch's reruns
                if not self.breaker_active():
                    self.note(f"{len(deaths)} jobs died within 5 min (API outage or credits?); pausing new launches "
                              "for 10 min; these deaths do not count as reruns")
                self.s["breaker_until"] = time.time() + 600
                j["attempts"] = max(j["attempts"] - 1, 0)
                j.update(state="pending", not_before=time.time() + 600)
                continue
            if j["attempts"] > MAX_RERUNS:
                j.update(state="failed", finished=now())
                self.note(f"branch {j['id']} FAILED {j['attempts']} times; excluded (log logs/branches/{j['id']}.log)")
            else:
                wait = 120 * 2 ** (j["attempts"] - 1)
                j.update(state="pending", not_before=time.time() + wait)
                self.note(f"branch {j['id']} exited early (attempt {j['attempts']}); rerun in {wait // 60:.0f} min",
                          also_event=False)
        if self.s.get("paused") or self.breaker_active():
            return
        pend = [j for j in self.s["branches"].values() if j["state"] == "pending" and j["not_before"] <= time.time()]
        pend.sort(key=lambda j: (TRACK_PRIORITY.get(j["track"], 1), j["seed"], self.s["bases"][j["run"]]["seed"],
                                 ARM_ORDER.index(j["arm"])))
        for j in pend:
            if running >= self.a.max_branches:
                break
            cmd = self.branch_cmd(j)
            env = child_env({"LLM_BASE_URL": self.base_url} if self.base_url else None)
            pid = spawn_detached(cmd, self.logs / "branches" / f"{j['id']}.log", cwd=CARTEL, env=env)
            j.update(state="running", pid=pid, pid_start=proc_start(pid), attempts=j["attempts"] + 1,
                     started=now(), cmd=cmd[1:])
            running += 1

    # ---------------------------------------------------------------- fallbacks / time
    def track_o_counts(self, seeds=None):
        bs = [b for b in self.s["bases"].values() if b["track"] == "O" and b["role"] == "base"
              and (seeds is None or b["seed"] in seeds)]
        decided = all(b.get("colluding") is not None or b["status"] == "failed" for b in bs)
        return sum(1 for b in bs if b.get("colluding")), decided, bs

    def fallbacks(self) -> None:
        n, decided, bs = self.track_o_counts()
        if not decided or n >= 4:
            return
        if "extend200" not in self.s["fallbacks"]:
            self.s["fallbacks"].append("extend200")
            self.deviation(f"only {n}/6 Track O runs colluded by round 149: extending non-colluding Track O runs "
                           "to 200 rounds (crossing allowed up to round 179) and adding seeds 6-9 (pre-authorized)")
            for b in bs:
                if not b.get("colluding") and b["status"] != "failed":
                    b.update(rounds=200, max_t=179, colluding=None, status="pending")
            for seed in range(6, 10):
                self.add_base("O", TRACK_O_MODEL, "P0", False, seed, 200, 179, role="base")
            return
        if "trackO_PI" not in self.s["fallbacks"]:
            self.s["fallbacks"].append("trackO_PI")
            self.deviation(f"still only {n} colluding Track O runs after extension: adding Track O-PI "
                           "(qwen3-14b with PI, seeds 0-5, 150 rounds; newcomers get P0; pre-authorized)")
            x = self.s["tracks"]["O"]["X"]
            self.add_track("OPI", TRACK_O_MODEL, "PI", False, range(6), BASE_ROUNDS, 149, TRACK_O_SIBLING, x)

    def time_check(self) -> None:
        if not self.deadline:
            return
        slow = [j for j in self.s["branches"].values() if TRACK_PRIORITY.get(j["track"]) == 0]
        pending = [j for j in slow if j["state"] == "pending"]
        if not pending:
            return
        done = [j for j in slow if j["state"] == "done" and j.get("started") and j.get("finished")]
        if done:
            dur = sum((datetime.strptime(j["finished"], "%Y-%m-%d %H:%M:%S")
                       - datetime.strptime(j["started"], "%Y-%m-%d %H:%M:%S")).total_seconds() for j in done) / len(done)
        else:
            secs = []
            for b in self.s["bases"].values():
                if b["track"] in ("O", "OPI"):
                    secs += [round_seconds(r) for r in read_records(self.run_file(b["name"]))[-20:]]
            med = median(secs)
            if med is None:
                return
            dur = med * self.s["tracks"]["O"]["horizon"] * 1.15
        waves = math.ceil(len(pending) / self.a.max_branches)
        eta = time.time() + dur * (waves + 0.5)
        if eta <= self.deadline.timestamp():
            return
        if "one_seed_slow" not in self.s["fallbacks"]:
            dropped = [j for j in pending if j["seed"] != BRANCH_SEEDS[0]]
            if dropped:
                for j in dropped:
                    j["state"] = "dropped"
                self.s["fallbacks"].append("one_seed_slow")
                self.deviation(f"time: projected slow-track finish {datetime.fromtimestamp(eta):%H:%M} is after the "
                               f"deadline {self.deadline:%H:%M}; dropped {len(dropped)} not-yet-started seed-102 "
                               "branches (pre-authorized: 1 seed per arm)")
                return
        if "horizon40_slow" not in self.s["fallbacks"]:
            self.s["fallbacks"].append("horizon40_slow")
            for tid in ("O", "OPI"):
                if tid in self.s["tracks"]:
                    self.s["tracks"][tid]["horizon"] = 40
            for j in pending:
                j["horizon"] = 40
            self.deviation(f"time: projected slow-track finish {datetime.fromtimestamp(eta):%H:%M} still after "
                           f"the deadline; slow-track horizon set to 40 for branches not yet started; analysis "
                           "uses the first 40 post-swap rounds of every slow-track branch (pre-authorized)")

    # ---------------------------------------------------------------- spend
    def spend(self) -> dict:
        est = 0.0
        for p in (self.results / "jsonl").glob("*.jsonl"):
            est += file_cost(p)
        for p in (self.results / "branches").glob("*.jsonl"):
            est += file_cost(p, only_after=True)
        est += self.s.get("lost_spend", 0.0)  # paid rounds of branches that died and were rerun from scratch
        actual = None
        if not self.a.mock:
            u = openrouter_usage()
            if u is not None:
                self.usage_misses = 0
                if self.s["usage_at_start"] is None:
                    self.s["usage_at_start"] = u
                actual = u - self.s["usage_at_start"]
            else:
                self.usage_misses += 1
        # The account delta is what OpenRouter billed (re-samples and lost branch attempts included); the token
        # estimate ran ~60% high because providers discount cached prompts, so it is only the fallback.
        return {"estimated": est, "actual": actual, "used": actual if actual is not None else est}

    def token_model(self) -> dict:
        """Observed token usage from finished base/screen/baseline rounds (refreshed every 10 min).

        calib: median of observed prompt tokens / (250 + 40 * visible rounds); out: median completion
        tokens per (model, think). Falls back to default_out() until a model has been observed."""
        cache = getattr(self, "_tok", None)
        if cache and time.time() - cache["t"] < 600:
            return cache
        ratios, outs = [], {}
        for p in (self.results / "jsonl").glob("*.jsonl"):
            cfg = read_json(config_path(p), {}) or {}
            think = bool(cfg.get("think"))
            for r in read_records(p)[-60:]:
                hist = min(r["round"] + 1, 100)
                for m in r.get("metrics", []):
                    model = m.get("model", "").split(":", 1)[-1]
                    if m.get("tokens_input"):
                        ratios.append(m["tokens_input"] / (250 + 40 * hist))
                    if m.get("tokens_output") is not None:
                        outs.setdefault((model, think), []).append(m["tokens_output"])
        self._tok = {"t": time.time(), "calib": median(ratios) or 1.0,
                     "out": {k: median(v) for k, v in outs.items() if len(v) >= 10}}
        return self._tok

    @staticmethod
    def default_out(model: str, think: bool) -> float:
        if model.startswith("qwen/"):
            return 2500  # qwen3 thinks by default; thinking is billed as completion tokens
        if model == "google/gemini-2.5-flash":
            return 900 if think else 600
        if model == "google/gemini-2.5-flash-lite":
            return 300
        if model.startswith("anthropic/"):
            return 1700  # measured: claude-haiku-5.5 wrote 1,652 tokens in the format check
        return 150 if think else 12

    def call_cost(self, model: str, hist: int, think: bool = False) -> float:
        p = prices().get(model)
        if not p:
            return 0.0
        tok = self.token_model()
        tin = (250 + 40 * min(max(hist, 0), 100)) * tok["calib"]
        tout = tok["out"].get((model, think)) or self.default_out(model, think)
        return tin * p["prompt"] + tout * p["completion"]

    def projected_job_cost(self, j: dict) -> float:
        t = self.s["tracks"][j["track"]]
        models = [t["model"], t["model"]]
        for k, v in j["replace"].items():
            models[int(k)] = v
        total = 0.0
        for k in range(1, j["horizon"] + 1):
            for i, m in enumerate(models):
                fresh = j["window"] is not None and str(i) in j["replace"]
                total += self.call_cost(m, (k - 1) if fresh else j["branch_round"] + k, t["think"])
        return total

    def projected_base_cost(self, b: dict) -> float:
        done = b.get("done_rounds", 0)
        return sum(2 * self.call_cost(b["model"], r + 1, b["think"]) for r in range(done, b["rounds"]))

    def fit_track_f_to_budget(self) -> None:
        """Pre-registered plan > budget: drop Track F seeds (5, 4, 3) before they start; Track O keeps priority."""
        before = self.projected_total()
        if before <= self.a.budget_limit:
            return
        dropped = []
        for seed in (5, 4, 3):
            name = next((n for n, b in self.s["bases"].items() if b["track"] == "F" and b["seed"] == seed
                         and b["status"] == "pending" and not b["spawns"]), None)
            if name is None:
                continue
            del self.s["bases"][name]
            dropped.append(seed)
            if self.projected_total() <= self.a.budget_limit:
                break
        after = self.projected_total()
        self.deviation(f"budget: projected cost of the full plan ${before:.2f} > ${self.a.budget_limit:.0f}; "
                       f"Track F reduced by seeds {dropped} (projection now ${after:.2f}). ASK the user whether to "
                       "restore them.")
        if after > self.a.budget_limit:
            held = [n for n, b in self.s["bases"].items() if b["track"] in ("F", "B-F") and not b["spawns"]]
            for n in held:
                self.s["bases"][n]["status"] = "held"
            self.note(f"ASK: projection still ${after:.2f} > ${self.a.budget_limit:.0f} with 3 Track F seeds; "
                      f"holding Track F and B-F ({len(held)} runs) so Track O keeps its budget. Release with "
                      "logs/pipeline_control.json {\"version\": N, \"release_held\": true, \"budget_limit\": ...}")

    def apply_control(self) -> None:
        """Operator overrides without a restart: logs/pipeline_control.json with an increasing integer "version"."""
        c = read_json(self.logs / "pipeline_control.json", None)
        try:
            if not isinstance(c, dict) or int(c.get("version", 0)) <= int(self.s.get("control_version", 0)):
                return
            ver = int(c["version"])
            ov = self.s.setdefault("overrides", {})
            for k in ("budget_limit", "budget_pause", "max_branches"):
                if k in c:
                    setattr(self.a, k, type(getattr(self.a, k))(c[k]))
                    ov[k] = getattr(self.a, k)
            if "deadline" in c:
                self.deadline = datetime.strptime(c["deadline"], "%Y-%m-%d %H:%M") if c["deadline"] else None
                self.a.deadline = ov["deadline"] = c["deadline"]
        except (TypeError, ValueError) as e:
            if self.s.get("control_error") != str(e):
                self.s["control_error"] = str(e)
                self.note(f"ignoring logs/pipeline_control.json: {type(e).__name__}: {e}")
            return
        self.s["control_version"] = ver
        if c.get("release_held"):
            for b in self.s["bases"].values():
                if b["status"] == "held":
                    b["status"] = "pending"
        if c.get("early_fallbacks") and "O" in self.s["tracks"]:
            # Start the pre-authorized Track O fallback runs now instead of after round 150 (time only; which runs
            # are used is still decided by the pre-registered rule in fallbacks()).
            for seed in range(6, 10):
                self.add_base("O", TRACK_O_MODEL, "P0", False, seed, 200, 179, role="base")
            self.add_track("OPI", TRACK_O_MODEL, "PI", False, range(6), BASE_ROUNDS, 149, TRACK_O_SIBLING,
                           self.s["tracks"]["O"]["X"])
            self.deviation("started the pre-authorized Track O fallback runs early (seeds 6-9 at P0 to 200 rounds, "
                           "and Track O-PI seeds 0-5) because all 6 Track O runs sat at the competitive price from "
                           "round 2 and the sequential fallback would end after the deadline")
        if c.get("add_track_f_seeds") and "F" in self.s["tracks"]:
            # More Track F markets after the pre-registered ones were analysed: reported as a robustness extension
            # beside the pre-registered sample, never in place of it (src/analyze.py --extended).
            t = self.s["tracks"]["F"]
            new = [int(s) for s in c["add_track_f_seeds"]
                   if run_name("F", t["model"], t["prefix"], t["think"], int(s)) not in self.s["bases"]]
            for seed in new:
                self.add_base("F", t["model"], t["prefix"], t["think"], seed, BASE_ROUNDS, 149, role="base")
            if new:
                self.deviation(f"Track F extended by seeds {new} (user decision after the Track F branches of seeds "
                               "0-4 were analysed): seed 5 was in the pre-registered plan and had been dropped for "
                               "budget; the primary analysis stays seeds 0-4 and all seeds are a robustness check")
        if c.get("pause"):  # operator stop: no new launches or relaunches; running jobs finish
            self.s["paused"] = now()
            (self.logs / "PAUSE").write_text(now(), encoding="utf-8")
        if c.get("unpause"):
            self.s["paused"] = None
            self.s["hard_stopped"] = None
            (self.logs / "PAUSE").unlink(missing_ok=True)
        self.note(f"control v{ver} applied: {json.dumps(c)}")

    def projected_total(self) -> float:
        sp = self.spend()["used"]
        rem = sum(self.projected_base_cost(b) for b in self.s["bases"].values() if b["status"] != "failed")
        rem += sum(self.projected_job_cost(j) for j in self.s["branches"].values() if j["state"] in ("pending", "running"))
        # branches not queued yet: one full arm set per colluding-or-undecided base run
        for b in self.s["bases"].values():
            if b["role"] == "base" and not b["enqueued"] and b.get("colluding") is not False and b["track"] in self.s["tracks"]:
                t = self.s["tracks"][b["track"]]
                if t.get("X") is None:
                    continue
                br = b["branch_round"] or 80
                for arm, rep, win, pfx in self.arms(t):
                    fake = {"track": t["id"], "replace": {str(k): v for k, v in rep.items()}, "window": win,
                            "branch_round": br, "horizon": t["horizon"]}
                    rem += len(BRANCH_SEEDS) * self.projected_job_cost(fake)
        return sp + rem

    def running_cost(self) -> float:
        """Projected cost still to be paid by jobs already running (they continue through a pause)."""
        total = 0.0
        for j in self.s["branches"].values():
            if j["state"] == "running":
                done = sum(1 for r in read_records(Path(j["out"])) if r.get("branch_phase") == "after")
                total += self.projected_job_cost(j) * max(j["horizon"] - done, 0) / max(j["horizon"], 1)
        for b in self.s["bases"].values():
            if b["status"] == "running":
                total += self.projected_base_cost(b)
        return total

    def stop_all(self) -> None:
        for j in self.s["branches"].values():
            if j["state"] == "running" and pid_alive(j["pid"], j.get("pid_start")):
                kill_tree(j["pid"])
                j.update(state="pending", attempts=max(j["attempts"] - 1, 0))
        for b in self.s["bases"].values():
            if b["status"] == "running" and pid_alive(b.get("pid"), b.get("pid_start")):
                kill_tree(b["pid"])

    def budget_check(self) -> None:
        sp = self.spend()
        sp["running"] = self.running_cost()
        self.s["spend"] = sp
        if self.usage_misses == 20 and key_is_set() and not self.a.mock:
            self.note("WARNING: the OpenRouter usage check has failed 20 times in a row; the budget now relies on the "
                      "token-based estimate, which leaves out re-sampled calls")
        how = (f'resume with logs/pipeline_control.json {{"version": N, "budget_pause": X, "budget_limit": Y, '
               f'"unpause": true}} after the user approves')
        if sp["used"] >= self.a.budget_limit and not self.s.get("hard_stopped"):
            self.s["hard_stopped"] = now()
            self.stop_all()
            self.note(f"BUDGET HARD STOP: spend ${sp['used']:.2f} >= ${self.a.budget_limit:.2f}; all running jobs "
                      f"terminated. ASK the user; {how}")
        if not self.s.get("paused") and (sp["used"] >= self.a.budget_pause
                                         or sp["used"] + sp["running"] >= self.a.budget_limit):
            self.s["paused"] = now()
            (self.logs / "PAUSE").write_text(now(), encoding="utf-8")
            self.note(f"BUDGET PAUSE: spend ${sp['used']:.2f} + in-flight ${sp['running']:.2f} vs pause "
                      f"${self.a.budget_pause:.2f} / limit ${self.a.budget_limit:.2f}. No new launches or relaunches; "
                      f"running jobs finish. ASK the user; {how}")

    # ---------------------------------------------------------------- progress / finish
    def progress(self, force: bool = False) -> None:
        if not force and time.time() - self.s["last_progress"] < self.a.progress_every:
            return
        self.s["last_progress"] = time.time()
        bases = self.s["bases"].values()
        bl = ", ".join(f"{b['name']}:{b.get('done_rounds', 0)}/{b['rounds']}" for b in bases if b["status"] != "complete")
        cnt = {}
        for j in self.s["branches"].values():
            cnt[j["state"]] = cnt.get(j["state"], 0) + 1
        sp = self.s.get("spend", {})
        self.note(f"progress: unfinished base runs [{bl or 'none'}]; branches {cnt}; spend ${sp.get('used', 0):.2f} "
                  f"(est ${sp.get('estimated', 0):.2f}, account delta {sp.get('actual')})", also_event=False)

    def finished(self) -> bool:
        if self.s["screen_decision"] is None:
            return False
        if any(b["status"] not in ("complete", "failed", "held") for b in self.s["bases"].values()):
            return False
        if any(b["role"] == "base" and b.get("colluding") and not b["enqueued"] and b["status"] != "failed"
               for b in self.s["bases"].values()):
            return False
        n, decided, _ = self.track_o_counts()
        if decided and n < 4 and "trackO_PI" not in self.s["fallbacks"]:
            return False
        return all(j["state"] in ("done", "failed", "dropped") for j in self.s["branches"].values())

    # ---------------------------------------------------------------- main loop
    def tick(self) -> bool:
        if not self.a.mock and not key_is_set():
            if not self.s.get("waiting_key_logged"):
                self.note("waiting for OPENROUTER_API_KEY in vendor/collusion_hackathon/compute.env")
                self.s["waiting_key_logged"] = True
            return False
        sm = self.smoke()
        if sm == "failed":
            return True
        if sm == "wait":
            return False
        if "O" not in self.s["tracks"]:
            self.add_track("O", TRACK_O_MODEL, "P0", False, range(6), BASE_ROUNDS, 149, TRACK_O_SIBLING, None)
            for m in SCREEN_MODELS:
                for s in (0, 1):
                    self.add_base("S", m, "P0", True, s, SCREEN_ROUNDS, SCREEN_ROUNDS - 1, role="screen")
            self.note("registered Track O (qwen3-14b, P0, seeds 0-5, 150 rounds) and the model screen "
                      f"({', '.join(SCREEN_MODELS)}; P0, think, {SCREEN_ROUNDS} rounds, seeds 0-1)")
        self.apply_control()
        self.budget_check()
        self.ensure_bases()
        self.decide_screen()
        self.ensure_extras()
        self.scan_bases()
        self.fallbacks()
        self.time_check()
        self.schedule()
        self.progress()
        if self.finished():
            self.s["finished"] = now()
            self.progress(force=True)
            self.note("all base runs and branches finished; pipeline exiting")
            return True
        return False

    def loop(self) -> None:
        lock = self.logs / "pipeline.lock"
        other = read_json(lock, {}) or {}
        if other.get("pid") and other["pid"] != os.getpid() and pid_alive(other["pid"], other.get("start")):
            sys.exit(f"pipeline already running (pid {other['pid']})")
        write_json(lock, {"pid": os.getpid(), "start": proc_start(os.getpid()), "started": now()})
        if self.s["smoke"] == "failed":  # a restart means someone looked at it: try again
            self.s.update(smoke="pending", smoke_fail=0, smoke_retry_at=0)
        self.note(f"pipeline started (pid {os.getpid()}, mock={bool(self.a.mock)}, max_branches={self.a.max_branches}, "
                  f"budget_pause=${self.a.budget_pause}, deadline={self.a.deadline})")
        try:
            while True:
                try:
                    stop = self.tick()
                except Exception as e:  # noqa: BLE001 - keep the unattended loop alive, but say so
                    import traceback

                    traceback.print_exc()
                    self.note(f"tick error: {type(e).__name__}: {e}", also_event=False)
                    stop = False
                self.save()
                if stop:
                    break
                time.sleep(self.a.tick)
        finally:
            try:
                lock.unlink()
            except OSError:
                pass


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["run", "start", "plan"])
    ap.add_argument("--results_dir", default=str(DEFAULT_RESULTS))
    ap.add_argument("--logs_dir", default=str(LOGS))
    ap.add_argument("--notes", default=str(NOTES))
    ap.add_argument("--max_branches", type=int, default=12)
    ap.add_argument("--horizon", type=int, default=50)
    ap.add_argument("--budget_pause", type=float, default=47.0, help="USD; no new launches at or above this")
    ap.add_argument("--budget_limit", type=float, default=50.0, help="USD; projected plan above this trims Track F")
    ap.add_argument("--deadline", default="2026-10-25 17:30", help="local time by which slow-track branches should end")
    ap.add_argument("--tick", type=float, default=30.0)
    ap.add_argument("--progress_every", type=float, default=1800.0)
    ap.add_argument("--mock", action="store_true", help="testing: route every model to src/dev/mock_llm.py")
    ap.add_argument("--mock_port", type=int, default=8765)
    ap.add_argument("--log", default=None)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    if args.mock and any(Path(x).resolve() == Path(d).resolve() for x, d in
                         ((args.results_dir, DEFAULT_RESULTS), (args.logs_dir, LOGS), (args.notes, NOTES))):
        sys.exit("--mock needs its own --results_dir, --logs_dir and --notes (never the real ones)")
    Path(args.logs_dir).mkdir(parents=True, exist_ok=True)
    if args.mode == "start":
        argv = [a for a in sys.argv[1:] if a != "start"]
        log = Path(args.logs_dir) / "pipeline.log"
        cmd = [sys.executable, str(Path(__file__).resolve()), "run", *argv, "--log", str(log.resolve())]
        # Plain detached launch (tested end to end). WMI launch is opt-in only: on this machine WMI-started
        # processes could not start the harness venv ("No Python at ...") and were killed after a while.
        pid = spawn_outside_app(cmd, REPO) if os.environ.get("PIPELINE_WMI") == "1" else None
        how = "outside the app (WMI)"
        if pid is None:
            pid = spawn_detached(cmd, Path(args.logs_dir) / "spawn.log", cwd=REPO, env=child_env())
            how = "detached"
        print(f"pipeline started {how}, pid {pid}; log {log}")
        return
    redirect_output(args.log)
    p = Pipeline(args)
    if args.mode == "plan":
        print(json.dumps({"projected_total_usd": round(p.projected_total(), 2), "spend": p.spend(),
                          "tracks": p.s["tracks"], "fallbacks": p.s["fallbacks"]}, indent=2))
        return
    p.loop()


if __name__ == "__main__":
    main()
