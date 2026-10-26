"""One-screen progress view: base runs, branches, spend, pipeline liveness.

    python src/status.py            # runs + branch summary
    python src/status.py --branches # also list every branch
    python src/status.py --live     # also ask OpenRouter for the account's current usage (never prints the key)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DEFAULT_RESULTS,
    LOGS,
    config_path,
    file_cost,
    first_crossing,
    median,
    openrouter_usage,
    pid_alive,
    pidx_series,
    read_json,
    read_records,
    rolling_mean,
    round_seconds,
)


def fmt_eta(sec):
    if sec is None:
        return "-"
    if sec < 90:
        return f"{sec:.0f}s"
    if sec < 5400:
        return f"{sec / 60:.0f}m"
    return f"{sec / 3600:.1f}h"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", default=str(DEFAULT_RESULTS))
    ap.add_argument("--logs_dir", default=str(LOGS))
    ap.add_argument("--branches", action="store_true")
    ap.add_argument("--live", action="store_true")
    a = ap.parse_args()
    results, logs = Path(a.results_dir), Path(a.logs_dir)
    st = read_json(logs / "pipeline_state.json", {}) or {}
    bases = st.get("bases", {})
    lock = read_json(logs / "pipeline.lock", {}) or {}
    alive = pid_alive(lock.get("pid"), lock.get("start"))
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  pipeline: {'RUNNING pid ' + str(lock['pid']) if alive else 'not running'}"
          f"{'  PAUSED since ' + st['paused'] if st.get('paused') else ''}"
          f"{'  finished ' + st['finished'] if st.get('finished') else ''}"
          f"{'  OUTAGE BREAKER ON' if st.get('breaker_until', 0) > time.time() else ''}")
    dec = st.get("screen_decision")
    if dec:
        print(f"screen: FAST={dec['FAST']}  Track F={dec['trackF_model']} {dec['trackF_prefix']}  X_O={dec['X_O']}  "
              f"X_F={dec['X_F']}")
    if st.get("fallbacks"):
        print(f"fallbacks applied: {st['fallbacks']}")

    print(f"\n{'run':34s} {'done':>9s} {'R20':>6s} {'cross':>5s} {'b':>4s} {'s/rnd':>6s} {'ETA':>6s} proc")
    total_est = 0.0
    for p in sorted((results / "jsonl").glob("*.jsonl")):
        name = p.stem
        recs = read_records(p)
        cfg = read_json(config_path(p), {}) or {}
        b = bases.get(name, {})
        target = b.get("rounds") or cfg.get("rounds") or len(recs)
        max_t = b.get("max_t", 149 if target >= 150 else target - 1)
        pidx = pidx_series(recs, cfg)
        roll = rolling_mean(pidx)
        c = first_crossing(pidx, max_t=max_t)
        secs = median([round_seconds(r) for r in recs[-10:]])
        eta = secs * (target - len(recs)) if secs is not None and len(recs) < target else None
        ls = read_json(logs / f"{name}.state.json", {}) or {}
        proc = ("alive" if pid_alive(ls.get("pid"), ls.get("start")) else ls.get("state", "-")) + (
            f" a{ls['attempts']}" if ls.get("attempts", 0) > 1 else "")
        cost = file_cost(p)
        total_est += cost
        r20 = f"{roll[-1]:.3f}" if roll and roll[-1] is not None else "-"
        print(f"{name[:34]:34s} {len(recs):>4d}/{target:<4d} {r20:>6s} {('-' if c is None else c):>5} "
              f"{('-' if c is None else c + 20):>4} {('-' if secs is None else f'{secs:.1f}'):>6s} {fmt_eta(eta):>6s} {proc}")

    br = st.get("branches", {})
    bfiles = list((results / "branches").glob("*.jsonl"))
    for p in bfiles:
        total_est += file_cost(p, only_after=True)
    if br:
        by = {}
        for j in br.values():
            k = (j["track"], j["arm"])
            by.setdefault(k, {}).setdefault(j["state"], 0)
            by[k][j["state"]] += 1
        print("\nbranches (track arm: state counts)")
        for k in sorted(by):
            print(f"  {k[0]:4s} {k[1]:4s} " + "  ".join(f"{s}={n}" for s, n in sorted(by[k].items())))
        running = [j for j in br.values() if j["state"] == "running"]
        if running:
            print(f"running ({len(running)}):")
            for j in sorted(running, key=lambda j: j["id"]):
                n_after = sum(1 for r in read_records(Path(j["out"])) if r.get("branch_phase") == "after")
                print(f"  {j['id'][:60]:60s} {n_after:>3d}/{j['horizon']} attempt {j['attempts']}"
                      f"{'' if pid_alive(j['pid'], j.get('pid_start')) else ' (exited)'}")
        if a.branches:
            for j in sorted(br.values(), key=lambda j: j["id"]):
                print(f"  {j['state']:8s} {j['id']}")
    elif bfiles:
        print(f"\nbranch files: {len(bfiles)} (no pipeline state)")
    sp = st.get("spend", {})
    line = f"\nspend: estimated from token usage ${total_est:.2f}"
    if sp.get("actual") is not None:
        line += f"; account delta at last pipeline tick ${sp['actual']:.2f}"
    if a.live:
        u = openrouter_usage()
        if u is not None and st.get("usage_at_start") is not None:
            line += f"; live account delta ${u - st['usage_at_start']:.2f}"
        elif u is not None:
            line += f"; live account usage (all time) ${u:.2f}"
    print(line)
    for e in (st.get("events") or [])[-5:]:
        print("  " + e)


if __name__ == "__main__":
    main()
