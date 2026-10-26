"""Pre-registered branch point for each base run.

    python src/find_branch.py                      # every run in results/jsonl
    python src/find_branch.py O_ F_ --json out/branch_points.json

Rule (NOTES.md, pre-registration items 1-2): pidx_t = (mean price_t - p_N) / (p_M - p_N);
R_t = mean pidx over rounds t-19..t. A run colludes if R_t > 0.5 for some t <= max_t (149 by default).
The crossing round c is the smallest such t, and the branch round is b = c + 20
(branch_game.py --at_round b, so the first post-swap round is b + 1).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DEFAULT_RESULTS,
    WINDOW,
    first_crossing,
    load_config,
    pidx_series,
    read_records,
    rolling_mean,
    write_json,
)

OFFSET = 20  # branch round = crossing + 20


def branch_point(run_path: Path, max_t: int = 149) -> dict:
    run_path = Path(run_path)
    recs = read_records(run_path)
    try:
        cfg = load_config(run_path)
    except (OSError, ValueError):
        cfg = None
    pidx = pidx_series(recs, cfg)
    roll = rolling_mean(pidx)
    c = first_crossing(pidx, max_t=max_t)
    b = c + OFFSET if c is not None else None
    out = {
        "run": run_path.stem,
        "path": str(run_path),
        "rounds": len(recs),
        "target_rounds": (cfg or {}).get("rounds"),
        "max_t": max_t,
        "crossing": c,
        "branch_round": b,
        "colluding": c is not None,
        "branch_ready": b is not None and len(recs) >= b + 1,
        "rolling_last": roll[-1] if roll else None,
        "pre_swap_pidx": (sum(pidx[b - WINDOW + 1 : b + 1]) / WINDOW) if b is not None and len(recs) >= b + 1 else None,
        "max_rolling": max((r for r in roll if r is not None), default=None),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prefixes", nargs="*", help="only runs whose name starts with one of these")
    ap.add_argument("--results_dir", default=str(DEFAULT_RESULTS))
    ap.add_argument("--max_t", type=int, default=149)
    ap.add_argument("--json", default=None, help="write the table here")
    args = ap.parse_args()
    runs = sorted((Path(args.results_dir) / "jsonl").glob("*.jsonl"))
    if args.prefixes:
        runs = [r for r in runs if any(r.stem.startswith(p) for p in args.prefixes)]
    rows = [branch_point(r, args.max_t) for r in runs]
    print(f"{'run':44s} {'rounds':>6s} {'R_last':>7s} {'maxR':>6s} {'cross':>5s} {'branch':>6s} {'ready':>5s} {'pre':>6s}")
    fmt = lambda v, f: (f % v) if v is not None else "-"
    for r in rows:
        print(
            f"{r['run'][:44]:44s} {r['rounds']:>6d} {fmt(r['rolling_last'], '%.3f'):>7s} "
            f"{fmt(r['max_rolling'], '%.3f'):>6s} {fmt(r['crossing'], '%d'):>5s} {fmt(r['branch_round'], '%d'):>6s} "
            f"{('yes' if r['branch_ready'] else 'no'):>5s} {fmt(r['pre_swap_pidx'], '%.3f'):>6s}"
        )
    if args.json:
        write_json(Path(args.json), rows)


if __name__ == "__main__":
    main()
