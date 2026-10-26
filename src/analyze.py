"""Pre-registered analysis of the replacement branches (NOTES.md, PRE-REGISTRATION items 3-7).

    uv run python src/analyze.py                                   # reads results/ (or the vendor results dir)
    uv run python src/analyze.py --results_dir <dir> --out <dir>   # e.g. the offline mock test

Outputs in out/:
    branch_metrics.csv      one row per branch (every quantity below is computed from these rows)
    run_metrics.csv         one row per base run, screen run and baseline-B run
    table_main.csv / .md    per track x arm: base runs, branches, Delta pidx [95% CI], survival, recovery, ...
    contrasts.csv           H2 (R3 - R2), H3 (R5 - R5f), H4 (R4 - C0), incumbent response
    fig1_trajectories.png   mean price across base runs (min-max band), 30 rounds before to H after the swap
    fig2_effects.png        paired Delta pidx vs control, one dot per base run, bootstrap CI bars
    figS*_*.png             appendix figures
    checks.md               exclusions, failed rounds, price plausibility, resolved model ids
    numbers.json            every reported number with its source file(s)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DEFAULT_RESULTS,
    LOGS,
    OUT,
    REPO,
    benchmarks,
    config_path,
    first_crossing,
    pidx_of,
    read_json,
    read_records,
    rolling_mean,
    round_seconds,
)

ARM_ORDER = ["C0", "R1", "R2", "R3", "R4", "R5", "R5f", "RP", "RN", "RNf"]
ARM_TITLE = {
    "C0": "C0 control (no swap)",
    "R1": "R1 same family",
    "R2": "R2 other provider",
    "R3": "R3 other provider, fresh",
    "R4": "R4 own model, told AC",
    "R5": "R5 both replaced",
    "R5f": "R5f both replaced, fresh",
    "RP": "RP own model, prompt P0",
}
SINGLE = ("R1", "R2", "R3", "R4", "RP", "RN", "RNf")
TRACK_TITLE = {"O": "Track O (qwen3-14b, P0)", "OPI": "Track O-PI (qwen3-14b, PI)", "F": "Track F"}
# Okabe-Ito colour-blind-safe palette
COL = {"O": "#0072B2", "OPI": "#56B4E9", "F": "#D55E00"}
BRANCH_RE = re.compile(r"^(?P<run>.+)__r(?P<b>\d+)__(?P<arm>[A-Za-z0-9]+)_s(?P<seed>\d+)(?:_win\d+)?$")
BOOT = 2000
BOOT_SEED = 12345
CEILING = 2.89  # "No customer would pay more than $2.89" in every prompt


class Numbers:
    """Registry of every reported number and where it comes from (out/numbers.json)."""

    def __init__(self, root: Path):
        self.items = {}
        self.root = root

    def add(self, key, value, source, method=""):
        if isinstance(value, (np.floating, np.integer)):
            value = value.item()
        if isinstance(value, float):
            value = round(value, 4) if np.isfinite(value) else None
        self.items[key] = {"value": value, "source": source, "method": method}
        return value

    def dump(self, path: Path):
        path.write_text(json.dumps(self.items, indent=2), encoding="utf-8")


def rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(REPO)).replace("\\", "/")
    except ValueError:
        return str(p)


def track_of(name: str) -> str:
    return name.split("_", 1)[0]


def boot_ci(values, seed=BOOT_SEED):
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return None, None, None
    if len(v) == 1:
        return float(v[0]), None, None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(BOOT, len(v)))
    means = v[idx].mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def fmt_ci(m, lo, hi, d=2):
    """Same rounding as fig2 (2 decimals, no negative zero)."""
    if m is None:
        return "-"
    if lo is None:
        return _num(m, d)
    return f"{_num(m, d)} [{_num(lo, d)}, {_num(hi, d)}]"


def first_k(series, pred, start=5, w=5):
    """Smallest 1-based k >= start with pred(mean(series[k-w:k])), else None (censored)."""
    for k in range(start, len(series) + 1):
        if pred(float(np.mean(series[k - w : k]))):
            return k
    return None


def median_censored(values):
    """Median where None means 'not within the horizon' (+inf). Returns None if the median is censored."""
    v = sorted(float("inf") if x is None else float(x) for x in values)
    if not v:
        return None
    n = len(v)
    m = v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])
    return None if not np.isfinite(m) else m


# ----------------------------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------------------------
def load_runs(results: Path):
    runs = {}
    for p in sorted((results / "jsonl").glob("*.jsonl")):
        name = p.stem
        if name == "smoke" or name.startswith("mocktest"):
            continue
        recs = read_records(p)
        n_lines = sum(1 for line in open(p, encoding="utf-8") if line.strip())
        cfg = read_json(config_path(p), {}) or {}
        nash, mono = benchmarks(cfg)
        pidx = np.array([pidx_of(r["actions"], nash, mono) for r in recs])
        runs[name] = {"name": name, "path": p, "recs": recs, "cfg": cfg, "pidx": pidx, "lines": n_lines,
                      "nash": nash, "mono": mono, "track": track_of(name)}
    return runs


def branch_metrics(p: Path, H: int):
    m = BRANCH_RE.match(p.stem)
    if not m:
        return None
    recs = read_records(p)
    n_lines = sum(1 for line in open(p, encoding="utf-8") if line.strip())
    cfg = read_json(config_path(p), {}) or {}
    nash, mono = benchmarks(cfg)
    before = [r for r in recs if r.get("branch_phase") == "before"]
    after = [r for r in recs if r.get("branch_phase") == "after"][:H]
    row = {
        "file": rel(p), "run": m["run"], "track": track_of(m["run"]), "branch_round": int(m["b"]),
        "arm": m["arm"], "seed": int(m["seed"]), "horizon": H, "after_rounds": len(after),
        "complete": len(after) >= H and "comparison" in cfg and n_lines == len(recs),
        "failed_lines": n_lines - len(recs), "models": "|".join(cfg.get("models", [])),
        "prefixes": "|".join(cfg.get("prefixes", [])), "replacement_window": cfg.get("replacement_window"),
        "nash": nash, "mono": mono,
    }
    if len(after) < H or len(before) < 30:
        return row
    p_after = np.array([r["actions"] for r in after], dtype=float)
    p_before = np.array([r["actions"] for r in before], dtype=float)
    pid_a = (p_after.mean(axis=1) - nash) / (mono - nash)
    pid_b = (p_before.mean(axis=1) - nash) / (mono - nash)
    g_a = np.array([np.mean(r["gains"]) for r in after])
    g_b = np.array([np.mean(r["gains"]) for r in before])
    win = slice(H - 20, H)
    pre = float(pid_b[-20:].mean())
    row.update({
        "pre_pidx": pre, "pre_gain": float(g_b[-20:].mean()),
        "Y": float(pid_a[win].mean()), "Y_gain": float(g_a[win].mean()),
        "survive": int(pid_a[win].mean() >= 0.5),
        "recover_k": first_k(pid_a, lambda v: v >= 0.9 * pre),
        "persist_k": first_k(pid_a, lambda v: v < 0.5),
        "gap": float((p_after[win, 1] - p_after[win, 0]).mean()),
        "inc10": float(p_after[:10, 0].mean()), "new10": float(p_after[:10, 1].mean()),
        "pre_p0": float(p_before[-10:, 0].mean()), "pre_p1": float(p_before[-10:, 1].mean()),
        "min_pidx_10": float(pid_a[:10].min()),
        "n_below_cost": int((p_after < 1.0).sum()), "n_above_ceiling": int((p_after > CEILING).sum()),
        "traj_before": p_before[-30:].mean(axis=1).tolist(), "traj_after": p_after.mean(axis=1).tolist(),
        "traj0_after": p_after[:, 0].tolist(), "traj1_after": p_after[:, 1].tolist(),
        "traj0_before": p_before[-30:, 0].tolist(), "traj1_before": p_before[-30:, 1].tolist(),
    })
    return row


# ----------------------------------------------------------------------------------------------
# main analysis
# ----------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", default=None, help="default: results/ if it has runs, else the vendor dir")
    ap.add_argument("--logs_dir", default=str(LOGS))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    results = Path(a.results_dir) if a.results_dir else (
        REPO / "results" if any((REPO / "results" / "jsonl").glob("*.jsonl")) else DEFAULT_RESULTS)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    state = read_json(Path(a.logs_dir) / "pipeline_state.json", None) or read_json(
        results / "pipeline_state.json", {}) or {}
    if not state:
        print("WARNING: no pipeline_state.json: horizons default to 50 and crossing limits to round 149")
    tracks_state = state.get("tracks", {})
    for tid, t in tracks_state.items():  # legend names from the pre-registered track definitions
        TRACK_TITLE[tid] = (f"Track {'O-PI' if tid == 'OPI' else tid}: {t['model'].split('/')[-1]}"
                            f"{', nudged prompt' if t['prefix'] != 'P0' else ''}"
                            f"{', reasoning on' if t.get('think') else ''}")
    N = Numbers(REPO)
    print(f"results: {results}\nout: {out}")

    runs = load_runs(results)
    horizon = {tid: t.get("horizon", 50) for tid, t in tracks_state.items()}

    # ---------------- branches
    rows = []
    for p in sorted((results / "branches").glob("*.jsonl")):
        tr = track_of(p.stem.split("__", 1)[0])
        r = branch_metrics(p, horizon.get(tr, 50))
        if r:
            rows.append(r)
    complete = [r for r in rows if r["complete"] and "Y" in r]
    incomplete = [r for r in rows if not (r["complete"] and "Y" in r)]
    tracks = [t for t in ("O", "OPI", "F") if any(r["track"] == t for r in complete)]

    # ---------------- per (track, run, arm): seed averages
    cell = defaultdict(list)
    for r in complete:
        cell[(r["track"], r["run"], r["arm"])].append(r)
    avg = {}
    for k, rs in cell.items():
        avg[k] = {f: float(np.mean([r[f] for r in rs])) for f in ("Y", "Y_gain", "inc10", "gap", "pre_pidx")}
        avg[k]["n"] = len(rs)
        avg[k]["survive"] = int(avg[k]["Y"] >= 0.5)

    def runs_with(track, *arms):
        rs = sorted({k[1] for k in avg if k[0] == track})
        return [r for r in rs if all((track, r, a_) in avg for a_ in arms)]

    def paired(track, arm, ref, field="Y"):
        rs = runs_with(track, arm, ref)
        return rs, [avg[(track, r, arm)][field] - avg[(track, r, ref)][field] for r in rs]

    bfile = lambda track, *arms: f"out/branch_metrics.csv rows track={track}, arm in {{{', '.join(arms)}}} " \
                                 f"(from results/branches/{track}_*__r*__<arm>_s*.jsonl)"

    # ---------------- table
    table = []
    for t in tracks:
        H = horizon.get(t, 50)
        for arm in ARM_ORDER:
            br = [r for r in complete if r["track"] == t and r["arm"] == arm]
            if not br:
                continue
            base_runs = sorted({r["run"] for r in br})
            rs, d = paired(t, arm, "C0")
            m, lo, hi = boot_ci(d) if arm != "C0" else (None, None, None)
            _, dg = paired(t, arm, "C0", "Y_gain")
            mg, glo, ghi = boot_ci(dg) if arm != "C0" else (None, None, None)
            _, dinc = paired(t, arm, "C0", "inc10")
            mi, ilo, ihi = boot_ci(dinc) if arm != "C0" else (None, None, None)
            rec = [r["recover_k"] for r in br]
            med_rec = median_censored(rec)
            if med_rec is not None and float(med_rec).is_integer():
                med_rec = int(med_rec)
            surv_b = sum(r["survive"] for r in br)
            surv_r = sum(avg[(t, run, arm)]["survive"] for run in base_runs)
            row = {
                "track": t, "arm": arm, "horizon": H, "n_base_runs": len(base_runs), "n_branches": len(br),
                "mean_Y": float(np.mean([avg[(t, run, arm)]["Y"] for run in base_runs])),
                "mean_pre": float(np.mean([avg[(t, run, arm)]["pre_pidx"] for run in base_runs])),
                "delta": m, "delta_lo": lo, "delta_hi": hi, "n_paired": len(rs),
                "delta_gain": mg, "delta_gain_lo": glo, "delta_gain_hi": ghi,
                "survive_branches": f"{surv_b}/{len(br)}", "survive_runs": f"{surv_r}/{len(base_runs)}",
                "median_recover": (">" + str(H)) if med_rec is None and rec else med_rec,
                "censored_recover": sum(v is None for v in rec),
                "mean_gap": float(np.mean([avg[(t, run, arm)]["gap"] for run in base_runs])),
                "inc_resp": mi, "inc_resp_lo": ilo, "inc_resp_hi": ihi,
            }
            table.append(row)
            src = bfile(t, arm, "C0")
            pre = f"{t}.{arm}."
            N.add(pre + "n_base_runs", len(base_runs), src)
            N.add(pre + "n_branches", len(br), src)
            N.add(pre + "n_paired_with_C0", len(rs), src, "base runs with both this arm and C0 complete")
            N.add(pre + "mean_Y", row["mean_Y"], src, f"mean pidx over post-swap rounds {H - 19}-{H}, seeds averaged")
            N.add(pre + "survive_branches", row["survive_branches"], src, "branches with Y >= 0.5")
            N.add(pre + "survive_runs", row["survive_runs"], src, "base runs with seed-averaged Y >= 0.5")
            N.add(pre + "median_recover", row["median_recover"], src, "median k (5-round mean >= 0.9 pre)")
            if arm in SINGLE + ("C0",):
                N.add(pre + "mean_gap", row["mean_gap"], src, "mean p1 - p0 over the scoring window")
            if arm != "C0":
                N.add(pre + "delta", m, src, "mean over base runs of Y(arm) - Y(C0)")
                N.add(pre + "delta_ci", [lo, hi], src, f"95% cluster bootstrap over base runs, {BOOT} resamples")
                N.add(pre + "delta_gain", mg, src)
                N.add(pre + "delta_gain_ci", [glo, ghi], src)
                if arm in SINGLE:
                    N.add(pre + "incumbent_response", mi, src, "mean p0 over post-swap rounds 1-10, arm - C0")
                    N.add(pre + "incumbent_response_ci", [ilo, ihi], src)

    # ---------------- contrasts (H2, H3, H4)
    contrasts = []
    for t in tracks:
        for name, arm, ref in (("H2 R3-R2", "R3", "R2"), ("H3 R5-R5f", "R5", "R5f"), ("H4 R4-C0", "R4", "C0"),
                               ("R2-R1", "R2", "R1"), ("RP-C0", "RP", "C0"),
                               ("RNf-RN", "RNf", "RN"), ("RN-R2", "RN", "R2")):  # last two: exploratory
            rs, d = paired(t, arm, ref)
            if not rs:
                continue
            m, lo, hi = boot_ci(d)
            contrasts.append({"track": t, "contrast": name, "n": len(rs), "mean": m, "lo": lo, "hi": hi,
                              "per_run": ";".join(f"{r}:{v:+.3f}" for r, v in zip(rs, d))})
            N.add(f"{t}.contrast.{name.split()[-1]}", m, bfile(t, arm, ref), "paired mean, seeds averaged")
            N.add(f"{t}.contrast.{name.split()[-1]}.n", len(rs), bfile(t, arm, ref), "base runs in the contrast")
            N.add(f"{t}.contrast.{name.split()[-1]}.ci", [lo, hi], bfile(t, arm, ref), "cluster bootstrap 95%")
        r5 = [r["persist_k"] for r in complete if r["track"] == t and r["arm"] == "R5"]
        if r5:
            H = horizon.get(t, 50)
            med = median_censored(r5)
            N.add(f"{t}.R5.median_persistence", med if med is not None else f">{H}", bfile(t, "R5"),
                  "first k with 5-round mean pidx < 0.5; None = never within horizon")
            N.add(f"{t}.R5.persist_censored", f"{sum(v is None for v in r5)}/{len(r5)}", bfile(t, "R5"))

    # ---------------- runs: positive control, screen, baselines
    run_rows = []
    for name, R in runs.items():
        pidx = R["pidx"]
        role = "screen" if name.startswith("S_") else "baseline" if name.startswith("B-") else "base"
        max_t = {"screen": 59, "baseline": 49}.get(role, (state.get("bases", {}).get(name, {}) or {}).get("max_t", 149))
        c = first_crossing(list(pidx), max_t=max_t)
        roll = rolling_mean(list(pidx))
        gains = np.array([np.mean(r["gains"]) for r in R["recs"]]) if R["recs"] else np.array([])
        n = len(pidx)
        row = {
            "run": name, "track": R["track"], "role": role, "rounds": n, "failed_lines": R["lines"] - n,
            "models": "|".join(R["cfg"].get("models", [])), "prefixes": "|".join(R["cfg"].get("prefixes", [])),
            "think": R["cfg"].get("think"), "crossing": c, "branch_round": c + 20 if c is not None and role == "base" else None,
            "max_R20": max((v for v in roll if v is not None), default=None),
            "pidx_last20": float(pidx[-20:].mean()) if n >= 20 else None,
            "pidx_30_49": float(pidx[30:50].mean()) if n >= 50 else None,
            "pidx_40_59": float(pidx[40:60].mean()) if n >= 60 else None,
            "gain_last50": float(gains[-50:].mean()) if n else None,
            "median_round_s": float(np.median([round_seconds(r) or 0 for r in R["recs"]])) if n else None,
            "n_below_cost": int(sum(p < 1.0 for r in R["recs"] for p in r["actions"])),
            "n_above_ceiling": int(sum(p > CEILING for r in R["recs"] for p in r["actions"])),
        }
        run_rows.append(row)
    for t in ("O", "OPI", "F"):
        base = [r for r in run_rows if r["track"] == t and r["role"] == "base"]
        if not base:
            continue
        src = f"out/run_metrics.csv rows track={t} (from results/jsonl/{t}_*.jsonl)"
        N.add(f"{t}.base.n_runs", len(base), src)
        N.add(f"{t}.base.n_colluding", sum(r["crossing"] is not None for r in base), src, "R20 > 0.5 by max_t")
        N.add(f"{t}.base.n_branchable", sum(r["crossing"] is not None and r["rounds"] >= r["crossing"] + 21
                                          for r in base), src, "colluding and reached the branch round")
        cr = [r["crossing"] for r in base if r["crossing"] is not None]
        if cr:
            N.add(f"{t}.base.crossing_median", float(np.median(cr)), src)
            N.add(f"{t}.base.crossing_range", [min(cr), max(cr)], src)
        N.add(f"{t}.base.models", sorted({r["models"] for r in base}), src)
    for t in ("B-O", "B-F", "B-N"):
        all_b = [r for r in run_rows if r["track"] == t]
        bb = [r for r in all_b if r["pidx_30_49"] is not None]  # only complete 50-round baselines count
        if not all_b:
            continue
        src = f"out/run_metrics.csv rows track={t} (from results/jsonl/{t}_*.jsonl)"
        N.add(f"{t}.n", len(bb), src, "complete (50-round) baseline runs")
        N.add(f"{t}.n_incomplete", len(all_b) - len(bb), src)
        N.add(f"{t}.n_colluding_Y", sum(r["pidx_30_49"] >= 0.5 for r in bb), src, "pidx rounds 30-49 >= 0.5")
        if bb:
            N.add(f"{t}.mean_Y", float(np.mean([r["pidx_30_49"] for r in bb])), src)
        N.add(f"{t}.models", sorted({r["models"] for r in bb}), src)
    for r in run_rows:
        if r["role"] == "screen":
            src = f"results/jsonl/{r['run']}.jsonl"
            N.add(f"screen.{r['run']}.crossing", r["crossing"], src)
            N.add(f"screen.{r['run']}.median_round_s", r["median_round_s"], src)
            N.add(f"screen.{r['run']}.pidx_40_59", r["pidx_40_59"], src)

    # ---------------- exploratory: incumbent reaction (single-replacement arms)
    c0_inc = {(t, run): avg[(t, run, "C0")]["inc10"] for (t, run, a_) in avg if a_ == "C0"}
    reaction = []
    for r in complete:
        if r["arm"] not in SINGLE or (r["track"], r["run"]) not in c0_inc:
            continue
        d_inc = r["inc10"] - c0_inc[(r["track"], r["run"])]
        new_rel = r["new10"] - r["pre_p0"]
        if abs(d_inc) <= 0.03 and abs(r["gap"]) <= 0.05:
            kind = "lead"  # incumbent holds, newcomer ends at its price
        elif d_inc < -0.03 and new_rel < -0.03:
            kind = "follow-down"  # incumbent cuts after a lower-priced newcomer (punish or accommodate)
        elif d_inc < -0.03:
            kind = "cut"  # incumbent cuts although the newcomer did not undercut
        elif d_inc > 0.03:
            kind = "raise"
        else:
            kind = "hold, gap persists"
        reaction.append({"track": r["track"], "arm": r["arm"], "run": r["run"], "seed": r["seed"],
                         "d_inc": d_inc, "new_minus_pre_inc": new_rel, "gap": r["gap"], "kind": kind})
    for t in tracks:
        for arm in SINGLE:
            ks = [x["kind"] for x in reaction if x["track"] == t and x["arm"] == arm]
            if ks:
                N.add(f"{t}.{arm}.reaction_counts", {k: ks.count(k) for k in sorted(set(ks))},
                      "out/reaction.csv", "exploratory classification, thresholds 0.03 / 0.05 (see analyze.py)")

    # ---------------- write tables
    def write_csv(path, rows_, fields=None):
        if not rows_:
            path.write_text("", encoding="utf-8")
            return
        fields = fields or [k for k in rows_[0].keys() if not k.startswith("traj")]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for x in rows_:
                w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in x.items() if k in fields})

    bm_fields = ["file", "track", "run", "branch_round", "arm", "seed", "horizon", "after_rounds", "complete",
                 "failed_lines", "models", "prefixes", "replacement_window", "pre_pidx", "pre_gain", "Y", "Y_gain",
                 "survive", "recover_k", "persist_k", "gap", "inc10", "new10", "pre_p0", "pre_p1", "min_pidx_10",
                 "n_below_cost", "n_above_ceiling"]
    write_csv(out / "branch_metrics.csv", rows, bm_fields)
    write_csv(out / "run_metrics.csv", run_rows)
    write_csv(out / "table_main.csv", table)
    write_csv(out / "contrasts.csv", contrasts)
    write_csv(out / "reaction.csv", reaction)

    md = ["| Track | Arm | Markets | Branches | Mean post-swap price index | Change vs no swap [95% CI] | "
          "Survive (branches) | Survive (markets) | Median rounds to recover | Newcomer − incumbent price | "
          "Incumbent response, rounds 1–10 [95% CI] |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in table:
        md.append(f"| {TRACK_TITLE.get(r['track'], r['track'])} | {arm_long(r['arm'])} | {r['n_base_runs']} | "
                  f"{r['n_branches']} | {r['mean_Y']:.2f} | "
                  f"{fmt_ci(r['delta'], r['delta_lo'], r['delta_hi']) if r['arm'] != 'C0' else '(reference)'} | "
                  f"{r['survive_branches']} | {r['survive_runs']} | {r['median_recover']} | "
                  f"{(_num(r['mean_gap']) if r['arm'] in SINGLE + ('C0',) else '-')} | "
                  f"{fmt_ci(r['inc_resp'], r['inc_resp_lo'], r['inc_resp_hi']) if r['arm'] in SINGLE else '-'} |")
    (out / "table_main.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # ---------------- figures
    figures(out, complete, avg, tracks, horizon, runs, run_rows)
    if complete:
        N.add("market.competitive_price", complete[0]["nash"], "run config nash_prices (environment/oligopoly.py)")
        N.add("market.monopoly_price", complete[0]["mono"], "run config monopoly_prices (environment/oligopoly.py)")
    captions(out, complete, avg, tracks, horizon, state)

    # ---------------- checks
    chk = ["# Analysis checks", "", f"results dir: `{rel(results)}`", ""]
    chk.append("## Base runs")
    for r in sorted(run_rows, key=lambda x: x["run"]):
        status = ("colluding (crossing %d)" % r["crossing"]) if r["crossing"] is not None else "not colluding"
        if r["role"] == "base" and r["crossing"] is not None and r["rounds"] < r["crossing"] + 21:
            status += ", but stopped before the branch round (not branchable)"
        chk.append(f"- {r['run']}: {r['rounds']} rounds, {status}, failed lines {r['failed_lines']}, prices <1: "
                   f"{r['n_below_cost']}, >2.89: {r['n_above_ceiling']}, models {r['models']}")
    nb = [r for r in run_rows if r["role"] == "base" and r["crossing"] is None]
    chk += ["", "## Excluded base runs (not colluding, so not branched)"] + [f"- {r['run']}" for r in nb] + (
        ["- none"] if not nb else [])
    chk += ["", "## Incomplete / failed branches (excluded)"] + [
        f"- {r['file']}: after rounds {r['after_rounds']}/{r['horizon']}, failed lines {r['failed_lines']}"
        for r in incomplete] + (["- none"] if not incomplete else [])
    for d in state.get("deviations", []):
        chk.append(f"- deviation: {d}")
    bad = sum(r.get("n_below_cost", 0) + r.get("n_above_ceiling", 0) for r in complete)
    chk += ["", f"Implausible prices in post-swap rounds (below cost or above 2.89): {bad}",
            f"Branches complete: {len(complete)}; incomplete: {len(incomplete)}",
            f"Failed (unparsed) lines in complete branch files: {sum(r['failed_lines'] for r in complete)}"]
    (out / "checks.md").write_text("\n".join(chk) + "\n", encoding="utf-8")
    N.add("checks.branches_complete", len(complete), "out/checks.md")
    N.add("checks.branches_incomplete", len(incomplete), "out/checks.md")
    N.add("checks.implausible_prices_post_swap", bad, "out/branch_metrics.csv n_below_cost + n_above_ceiling")
    N.dump(out / "numbers.json")
    print((out / "table_main.md").read_text(encoding="utf-8"))
    for c in contrasts:
        print(f"{c['track']} {c['contrast']}: {fmt_ci(c['mean'], c['lo'], c['hi'])} (n={c['n']})")
    print(f"complete branches {len(complete)}, incomplete {len(incomplete)}")


ARM_PARTS = {  # (main, detail): one wording for every figure, table and the report
    "C0": ("No swap", "control"),
    "R1": ("Same-family model", "full history"),
    "R2": ("Other-provider model", "full history"),
    "R3": ("Other-provider model", "no history"),
    "R4": ("Own model", "told not to collude"),
    "R5": ("Both firms replaced", "full history"),
    "R5f": ("Both firms replaced", "no history"),
    "RP": ("Own model", "neutral prompt"),
    "RN": ("Newest-generation model", "full history"),
    "RNf": ("Newest-generation model", "no history"),
}
FIG0_TEXT = {
    "C0": "No swap (control)",
    "R1": "Firm 1 → same-family model, full history",
    "R2": "Firm 1 → other-provider model, full history",
    "R3": "Firm 1 → other-provider model, no history",
    "R4": "Firm 1 → own model, told not to collude",
    "R5": "Both firms → other-provider model, full history",
    "R5f": "Both firms → other-provider model, no history",
    "RP": "Firm 1 → own model, neutral prompt",
    "RN": "Firm 1 → newest-generation model, full history",
    "RNf": "Firm 1 → newest-generation model, no history",
}
FIRM_COL = {"inc": "#0072B2", "new": "#D55E00"}  # Okabe-Ito blue / vermillion
LS = ["-", (0, (4, 1.6)), (0, (1, 1.2))]  # line style per track (in addition to colour)
REF = dict(color="0.72", lw=0.7, zorder=1)  # competitive / monopoly reference lines
MAX_PX = 2200  # 7.3 in at 300 dpi; anything wider shrinks below ~7 pt at page width
PANEL_TITLE = 8  # one size for every small-multiple title


def arm_long(a):
    m, d = ARM_PARTS.get(a, (a, ""))
    return f"{a} · {m}, {d}" if d else f"{a} · {m}"


ARM_SHORT = {"C0": "No swap", "R1": "Same family", "R2": "Other provider", "R3": "Other provider", "R4": "Own model",
             "R5": "Both replaced", "R5f": "Both replaced", "RP": "Own model", "RN": "Newest model",
             "RNf": "Newest model"}  # narrow small-multiple titles


def arm_panel(a, short=False):
    m, d = ARM_PARTS.get(a, (a, ""))
    m = ARM_SHORT.get(a, m) if short else m
    return f"{a} · {m}\n({d})" if d else f"{a} · {m}"


def _num(v, d=2):
    v = round(float(v), d) + 0.0  # no negative zero
    return f"{v:+.{d}f}".replace("-", "−")


def _style(plt):
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": PANEL_TITLE, "axes.titleweight": "bold", "axes.labelsize": 9,
        "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": 0.8, "font.family": "DejaVu Sans",
    })


def _save(fig, path):
    fig.savefig(path, dpi=300)
    try:
        import matplotlib.image as mpimg

        w = mpimg.imread(path).shape[1]
        if w > MAX_PX:
            print(f"WARNING: {path.name} is {w / 300:.1f} in wide; text will shrink at page width")
    except Exception:  # noqa: BLE001
        pass


def _traj(complete, t, arm, H, key_before="traj_before", key_after="traj_after", nash=None, mono=None):
    """rows = base runs (seeds averaged), columns = rounds -29..H relative to the swap; price index if nash given."""
    per_run = defaultdict(list)
    for r in complete:
        if r["track"] == t and r["arm"] == arm:
            per_run[r["run"]].append(np.r_[r[key_before], r[key_after][:H]])
    if not per_run:
        return None
    M = np.array([np.mean(v, axis=0) for v in per_run.values()])
    return (M - nash) / (mono - nash) if nash is not None else M


def _index_axis(ax, label=True):
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["0 competitive", "0.5", "1 monopoly"] if label else ["", "", ""])
    ax.axhline(0, **REF)
    ax.axhline(1, **REF)


def _scored(ax, horizons):
    for H in sorted(set(horizons)):
        ax.axvspan(H - 19.5, H + 0.5, color="0.93", zorder=0, lw=0)


def _scored_text(horizon, tracks):
    hs = {t: horizon.get(t, 50) for t in tracks}
    if len(set(hs.values())) == 1:
        H = next(iter(hs.values()))
        return f"rounds {H - 19}–{H}"
    return ", ".join(f"{t}: rounds {H - 19}–{H}" for t, H in hs.items())


def fig0_design(out, plt, arms=None):
    """Schematic of the branching design (no data)."""
    from matplotlib.patches import FancyBboxPatch

    arms = arms or ["C0", "R1", "R2", "R3", "R4", "R5", "R5f"]
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)
    ax.set_xlim(0, 10)
    top = 5.9
    ax.set_ylim(-1.15, top + 0.5)
    ax.axis("off")
    xs = np.linspace(0.3, 4.2, 80)
    ys = 1.3 + 2.4 / (1 + np.exp(-(xs - 2.0) * 2.2)) + 0.06 * np.sin(xs * 9)
    ax.plot(xs, ys, color="0.25", lw=2)  # neutral: blue/orange mean tracks in the other figures
    ax.plot([0.3, 4.2], [1.3, 1.3], **REF)
    ax.plot([0.3, 4.2], [3.75, 3.75], **REF)
    ax.text(0.3, 1.02, "competitive price", fontsize=7.8, color="0.35")
    ax.text(0.3, 3.88, "monopoly price", fontsize=7.8, color="0.35")
    ax.annotate("collusion reached:\n20-round average of the\nprice index above 0.5", xy=(2.25, 2.6),
                xytext=(0.3, 4.55), fontsize=8, arrowprops=dict(arrowstyle="->", color="0.3", lw=0.8))
    ax.plot([4.2, 4.2], [0.9, top + 0.2], color="k", ls=":", lw=1)
    ax.text(4.1, top + 0.1, "swap point:\n20 rounds later", ha="right", va="top", fontsize=8)
    ax.text(2.25, 0.05, "Base market: two LLM firms set prices every round\n(each prompt shows the last 100 rounds)",
            ha="center", fontsize=8)
    step = min(0.78, (top - 0.4) / max(len(arms) - 1, 1))
    for i, code in enumerate(arms):
        y = top - i * step
        ax.annotate("", xy=(5.15, y), xytext=(4.25, ys[-1]),
                    arrowprops=dict(arrowstyle="-|>", color="0.5", lw=0.8, shrinkA=0, shrinkB=2))
        ax.add_patch(FancyBboxPatch((5.2, y - 0.27), 4.75, 0.54, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc="#E3EEF8" if code == "C0" else "#F4F4F4", ec="0.2" if code == "C0" else "0.6",
                                    lw=1.1 if code == "C0" else 0.6))
        ax.text(5.3, y, code, fontsize=8.2, fontweight="bold", va="center")
        ax.text(5.85, y, FIG0_TEXT.get(code, code), fontsize=8.2, va="center")
    ax.text(5.0, -1.1, "Every arm continues for 50 rounds, twice (two random seeds), in every track (model setup). "
            "Outcome: average price index in post-swap\nrounds 31–50 minus that of the no-swap control (C0) from "
            "the same market. Price index: 0 = competitive price, 1 = monopoly price.", ha="center", va="bottom",
            fontsize=7.6, color="0.25")
    _save(fig, out / "fig0_design.png")
    plt.close(fig)


def captions(out, complete, avg, tracks, horizon, state):
    """out/figure_captions.md: how to read each figure, filled with this run's counts and settings."""
    if not complete:
        return
    nash, mono = complete[0]["nash"], complete[0]["mono"]
    n = {t: len({k[1] for k in avg if k[0] == t and k[2] == "C0"}) for t in tracks}
    tr = "; ".join(f"{TRACK_TITLE.get(t, t)}, {n[t]} markets" for t in tracks)
    sc = _scored_text(horizon, tracks)
    H = sorted({horizon.get(t, 50) for t in tracks})
    seeds = sorted({r["seed"] for r in complete})
    st = {t: ("solid" if i == 0 else "dashed" if i == 1 else "dotted") for i, t in enumerate(tracks)}
    keys = ", ".join(f"{TRACK_TITLE.get(t, t).split(':')[0]} {st[t]}" for t in tracks)
    txt = f"""# Figure captions (generated by src/analyze.py)

**Figure 1 (fig0_design.png). Design.** Two LLM firms set prices every round in a repeated Bertrand market;
each prompt shows the last 100 rounds of prices and profits. Once the 20-round average of the price index first
exceeds 0.5 the market runs 20 more rounds and is then copied at that swap point. Each copy continues for
{'/'.join(map(str, H))} rounds under one condition (C0 no swap; R1–R5f replace firm 1 or both firms; RN/RNf,
Track F only and exploratory, bring in a current-generation model, Claude Haiku 5.5), with
{len(seeds)} sampling seeds per condition. The outcome is the mean price index in post-swap {sc} minus that of
C0 from the same market.

**Figure 2 (fig1_trajectories.png). Prices around the swap, one panel per condition.** Price index:
0 = competitive (Nash) price {nash:.3f}, 1 = joint-profit (monopoly) price {mono:.3f}. Thick lines: mean over
markets (each market averaged over its sampling seeds); thin lines: single markets. {tr}. Line style by track:
{keys}. Dotted vertical line: the swap (round 0 is the last round before it). Grey: the rounds scored ({sc}).
C0 shows what the same markets do without a swap.

**Figure 3 (fig2_effects.png). Effect of each swap relative to the no-swap control.** Change in the mean price
index over post-swap {sc}, paired with C0 from the same market. Diamond: mean over markets; bar: 95% percentile
bootstrap interval over markets (2,000 resamples; approximate with this few markets); rings: single markets
(each averaged over its sampling seeds); grey tick: the value at which the market would be exactly back at the
competitive price. Negative values mean prices fell after the swap. {tr}.

**Figure S1 (figS1_firm_prices.png). Incumbent and newcomer prices.** Mean price index of the firm that stays
(solid) and of the replaced firm (dashed; before the swap it is the original agent, after it the newcomer), for
the conditions that replace one firm. Colour = track. This shows who moves first after a swap.

**Figure S2 (figS2_base_runs.png). Base markets, model screen and from-scratch baselines.** 20-round average of
the price index for every base market (top), the screened models (60 rounds), and the newcomer models playing
against themselves from scratch (50 rounds). Dotted line: the 0.5 collusion threshold used to pick the swap point.
"""
    (out / "figure_captions.md").write_text(txt, encoding="utf-8")


def figures(out, complete, avg, tracks, horizon, runs, run_rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style(plt)
    arms = [a for a in ARM_ORDER if any(r["arm"] == a for r in complete)]
    fig0_design(out, plt, arms or None)
    if not arms:
        return
    nash = complete[0]["nash"]
    mono = complete[0]["mono"]
    style = {t: LS[i % len(LS)] for i, t in enumerate(tracks)}
    lw = {t: (2.0 if i == 0 else 1.6) for i, t in enumerate(tracks)}
    scored = _scored_text(horizon, tracks)

    # ---------------- fig1: price index around the swap, one panel per arm
    ncol = 3
    nrow = int(np.ceil((len(arms) + 1) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.2, 1.85 * nrow + 0.55), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    means = [0.0, 1.0]
    n_c0 = {t: len({k[1] for k in avg if k[0] == t and k[2] == "C0"}) for t in tracks}
    for i, arm in enumerate(arms):
        ax = axes[i // ncol, i % ncol]
        odd = []
        for t in tracks:
            H = horizon.get(t, 50)
            M = _traj(complete, t, arm, H, nash=nash, mono=mono)
            if M is None:
                continue
            x = np.arange(-29, H + 1)
            ax.plot(x, M.T, color=COL[t], lw=0.5, alpha=0.3, ls="-", zorder=2)
            ax.plot(x, M.mean(axis=0), color=COL[t], lw=lw[t], ls=style[t], zorder=3,
                    label=textwrap.fill(f"{TRACK_TITLE.get(t, t)} ({n_c0[t]} markets)", 30))
            means += list(M.mean(axis=0))
            if len(M) != n_c0[t]:  # say so only where an arm has fewer markets than the track
                odd.append(f"{t}: {len(M)} mkts")
        _scored(ax, [horizon.get(t, 50) for t in tracks])
        _index_axis(ax)  # shared y: inner panels hide their tick labels by themselves
        ax.axvline(0.5, color="k", ls=":", lw=0.9, zorder=1)
        ax.set_title(arm_panel(arm, short=True) + (f"  [{', '.join(odd)}]" if odd else ""), loc="left")
        ax.set_xticks([-30, 0, 30, 50])
        ax.tick_params(labelbottom=True)
    axes[0, 0].set_ylim(min(-0.12, min(means) - 0.05), max(1.18, max(means) + 0.08))
    for j in range(len(arms), nrow * ncol):
        axes[j // ncol, j % ncol].axis("off")
    handles = []
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in [x[1] for x in handles]:
                handles.append((h, lab))
    note = f"dotted: the swap; grey: scored rounds ({scored.replace('rounds ', '')})"
    bottom = 0.085
    if nrow * ncol - len(arms) >= 2:  # room for an in-grid key (it may spill into the next empty cell)
        key = axes[len(arms) // ncol, len(arms) % ncol]
        key.legend([h for h, _ in handles], [lab for _, lab in handles], loc="upper left", frameon=False,
                   fontsize=8, borderaxespad=0, borderpad=0, handlelength=2.4,
                   title="thick line = mean over markets\nthin lines = single markets", title_fontsize=7.8,
                   alignment="left")
        key.text(0.0, 0.0, note.replace("; ", "\n"), transform=key.transAxes, fontsize=7.8, color="0.3", va="bottom")
    else:  # grid nearly full: key under the figure
        labs = [lab.replace("\n", " ") for _, lab in handles]
        fig.legend([h for h, _ in handles], labs, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2,
                   frameon=False, fontsize=8, handlelength=2.4)
        fig.text(0.5, 0.045, "thick line = mean over markets; thin lines = single markets; " + note, ha="center",
                 fontsize=7.8, color="0.3")
        bottom = 0.15
    fig.supxlabel("round relative to the swap (0 = last round before it)", fontsize=9,
                  y=0.075 if bottom > 0.1 else 0.01)
    fig.supylabel("price index (mean of the two firms)", fontsize=9, x=0.01)
    fig.subplots_adjust(left=0.17, right=0.985, top=0.93, bottom=bottom + 0.035 if bottom > 0.1 else bottom,
                        wspace=0.14, hspace=0.62)
    _save(fig, out / "fig1_trajectories.png")
    plt.close(fig)

    # ---------------- fig2: forest plot of paired effects vs control (final size 6.5 in, labels inside)
    arms2 = [a for a in arms if a != "C0"]
    ypos, y, yt, ylab, c0y = {}, 0.0, [], [], {}
    for a in arms2:
        present = [t for t in tracks if any(k[0] == t and k[2] == a and (t, k[1], "C0") in avg for k in avg)]
        ys_ = []
        for t in present:
            ypos[(a, t)] = y
            ys_.append(y)
            y -= 0.34
        if ys_:
            yt.append(float(np.mean(ys_)))
            ylab.append(arm_panel(a))
        y -= 0.34
    for t in tracks:
        c0 = [v["Y"] for k, v in avg.items() if k[0] == t and k[2] == "C0"]
        c0y[t] = float(np.mean(c0)) if c0 else None
    h_in = 0.24 * len(ypos) + 0.2 * len(yt) + 1.65
    fig, ax = plt.subplots(figsize=(6.5, h_in))
    fig.subplots_adjust(left=0.27, right=0.77, top=1 - 0.25 / h_in, bottom=1.15 / h_in)
    xs_all = [0.0]
    for (a, t), yy in ypos.items():
        runs_ = sorted({k[1] for k in avg if k[0] == t and k[2] == a and (t, k[1], "C0") in avg})
        d = [avg[(t, r, a)]["Y"] - avg[(t, r, "C0")]["Y"] for r in runs_]
        xs_all += d
        m, lo, hi = boot_ci(d)
        if c0y.get(t) is not None:  # where this track would be exactly back at the competitive price
            ax.plot([-c0y[t]] * 2, [yy - 0.2, yy + 0.2], color="0.45", lw=1.1, zorder=7)
            xs_all.append(-c0y[t])
        if lo is not None:
            ax.plot([lo, hi], [yy, yy], color=COL[t], lw=1.6, solid_capstyle="butt", zorder=4)
            xs_all += [lo, hi]
        ax.scatter(d, np.full(len(d), yy - 0.12), s=12, facecolor="none", edgecolor=COL[t], lw=0.7, zorder=5)
        ax.scatter([m], [yy], marker="D", s=30, color=COL[t], edgecolor="k", lw=0.6, zorder=6)
        txt = f"{_num(m)} [{_num(lo)}, {_num(hi)}]" if lo is not None else f"{_num(m)} (n={len(d)}, no CI)"
        ax.text(1.02, yy, txt, transform=ax.get_yaxis_transform(), va="center", fontsize=7.5, color="0.15",
                fontfamily="DejaVu Sans Mono")
    lo_x, hi_x = min(min(xs_all), 0.0), max(max(xs_all), 0.0)
    pad = 0.08 * (hi_x - lo_x) + 0.05
    ax.set_xlim(lo_x - pad, hi_x + pad)
    ax.set_ylim(y + 0.25, 0.62)  # headroom for the direction cues
    ax.axvline(0, color="0.3", lw=0.9)
    ax.set_yticks(yt)
    ax.set_yticklabels(ylab, fontsize=8)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("change in price index vs. no swap\n"
                  f"(post-swap {scored}; 1 unit = the whole competitive-to-monopoly gap)", fontsize=8.2)
    cue = dict(transform=ax.transAxes, va="top", fontsize=8, color="0.3", zorder=8,
               bbox=dict(fc="white", ec="none", pad=1.0))
    ax.text(0.01, 0.995, "◀ prices fell", ha="left", **cue)
    ax.text(0.99, 0.995, "prices rose ▶", ha="right", **cue)
    ax.text(1.02, 1.0, "mean [95% CI]", transform=ax.transAxes, fontsize=7.5, color="0.3", va="bottom",
            fontfamily="DejaVu Sans Mono")
    for t in tracks:
        n = len({k[1] for k in avg if k[0] == t and k[2] == "C0"})
        ax.scatter([], [], marker="D", s=30, color=COL[t], edgecolor="k", lw=0.6,
                   label=f"{TRACK_TITLE.get(t, t)} ({n} markets)")
    ax.scatter([], [], s=12, facecolor="none", edgecolor="0.3", lw=0.7, label="ring = one market")
    ax.plot([], [], color="0.3", lw=1.6, label="bar = 95% bootstrap interval")
    ax.plot([], [], ls="none", marker="|", ms=9, mew=1.2, color="0.45", label="grey tick = fully back at competitive price")
    h_, l_ = ax.get_legend_handles_labels()
    fig.legend(h_, l_, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=7.6,
               handletextpad=0.6, handlelength=1.6, columnspacing=1.2)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color="0.9", lw=0.6)
    ax.set_axisbelow(True)
    _save(fig, out / "fig2_effects.png")
    plt.close(fig)

    # ---------------- figS1: incumbent vs replaced firm (colour = track, line style = firm)
    arms3 = [a for a in arms if a in SINGLE]
    if arms3:
        fig, axes = plt.subplots(len(tracks), len(arms3), figsize=(7.2, 1.75 * len(tracks) + 0.9), sharex=True,
                                 sharey=True, squeeze=False)
        for ti, t in enumerate(tracks):
            H = horizon.get(t, 50)
            for ai, arm in enumerate(arms3):
                ax = axes[ti, ai]
                drew = False
                # incumbent: thin dark line; replaced firm: wide light line (dashes mean "Track F" in fig1)
                for who, k0, kw in (("inc", "0", dict(lw=1.2, alpha=1.0, zorder=4)),
                                    ("new", "1", dict(lw=3.2, alpha=0.45, zorder=3))):
                    M = _traj(complete, t, arm, H, f"traj{k0}_before", f"traj{k0}_after", nash, mono)
                    if M is None:
                        continue
                    drew = True
                    ax.plot(np.arange(-29, H + 1), M.mean(axis=0), color=COL[t], solid_capstyle="butt", **kw)
                if not drew:
                    ax.axis("off")
                    continue
                _scored(ax, [H])
                _index_axis(ax)
                ax.axvline(0.5, color="k", ls=":", lw=0.8)
                ax.set_xticks([-30, 0, 30, 50])
                if ti == 0:
                    title = arm_panel(arm, short=True)
                    if len(arms3) > 4:  # narrow panels: wrap every line
                        title = "\n".join(textwrap.fill(part, 13) for part in title.split("\n"))
                    ax.set_title(title, loc="left", fontsize=PANEL_TITLE)
            head, _, rest = TRACK_TITLE.get(t, t).partition(": ")
            axes[ti, 0].set_ylabel(f"{head}\n{rest.split(',')[0]}", fontsize=8.5, fontweight="bold", color=COL[t])
        for ai in range(len(arms3)):  # x tick labels under the lowest visible panel of every column
            vis = [a for a in axes[:, ai] if a.axison]
            if vis:
                vis[-1].tick_params(labelbottom=True)
        from matplotlib.lines import Line2D

        fig.legend([Line2D([], [], color="0.25", lw=1.2), Line2D([], [], color="0.25", lw=3.2, alpha=0.45)],
                   ["incumbent firm (thin dark line)", "replaced firm: the newcomer after the swap (wide light line)"],
                   loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
        fig.supxlabel("round relative to the swap", fontsize=9)
        fig.supylabel("price index of each firm", fontsize=9, x=0.005)
        fig.subplots_adjust(left=0.19, right=0.99, top=0.74 if len(arms3) > 4 else 0.83, bottom=0.12,
                            wspace=0.12 if len(arms3) <= 4 else 0.25, hspace=0.3)
        _save(fig, out / "figS1_firm_prices.png")
        plt.close(fig)

    # ---------------- figS2: base runs, model screen and from-scratch baselines (rolling price index)
    groups = [g for g in ("O", "OPI", "F", "S", "B-O", "B-F", "B-N") if any(r["track"] == g for r in run_rows)]
    if groups:
        ncol = 3
        nrow = int(np.ceil(len(groups) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(7.2, 1.95 * nrow + 0.3), sharey=True, squeeze=False)
        model_of = {}
        for r in run_rows:
            model_of.setdefault(r["track"], r["models"].split("|")[0].split(":")[-1].split("/")[-1])
        o_tag = "Tracks O/O-PI" if "OPI" in tracks else "Track O"
        titles = {"S": "Model screen (60 rounds)",
                  "B-O": f"{model_of.get('B-O', '')} vs itself,\nfrom scratch\n(newcomer of {o_tag})",
                  "B-F": f"{model_of.get('B-F', '')} vs itself,\nfrom scratch\n(newcomer of Track F)",
                  "B-N": f"{model_of.get('B-N', '')} vs itself,\nfrom scratch\n(newest-model newcomer, F)"}
        screen_cols = ["#009E73", "#CC79A7", "#E69F00", "#000000"]
        for gi, g in enumerate(groups):
            ax = axes[gi // ncol, gi % ncol]
            models_seen = {}
            for name, R in sorted(runs.items()):
                if R["track"] != g:
                    continue
                roll = [np.nan if v is None else v for v in rolling_mean(list(R["pidx"]))]
                if g == "S":
                    mdl = R["cfg"].get("models", ["?"])[0].split(":")[-1].split("/")[-1]
                    first = mdl not in models_seen
                    c = models_seen.setdefault(mdl, screen_cols[len(models_seen) % len(screen_cols)])
                    ax.plot(range(len(roll)), roll, lw=1.1, color=c, alpha=0.9, label=mdl if first else None)
                else:
                    ax.plot(range(len(roll)), roll, lw=1.0, color=COL.get(g, "0.35"), alpha=0.75)
            ax.axhline(0.5, color="k", ls=":", lw=0.8)
            _index_axis(ax)
            head, _, rest = TRACK_TITLE.get(g, g).partition(": ")
            title = titles.get(g, f"{head} base runs" + (f"\n({rest})" if rest else ""))
            if g in ("O", "OPI", "F") and g not in tracks:
                title = f"{head} base runs\n(none reached 0.5; no swaps)"
            ax.set_title(title, loc="left", fontsize=PANEL_TITLE)
            ax.set_xlabel("round", fontsize=8)
            if g == "S":
                ax.legend(frameon=False, fontsize=7.4, loc="lower right", bbox_to_anchor=(1.0, 0.1), labelspacing=0.25,
                          handlelength=1.4, borderpad=0.2)
        axes[0, 0].text(0.98, 0.53, "collusion threshold", transform=axes[0, 0].get_yaxis_transform(), fontsize=7.6,
                        color="0.3", va="bottom", ha="right", bbox=dict(fc="white", ec="none", pad=0.5))
        for j in range(len(groups), nrow * ncol):
            axes[j // ncol, j % ncol].axis("off")
        fig.supylabel("price index, 20-round average", fontsize=9)
        fig.subplots_adjust(left=0.16, right=0.99, top=0.9, bottom=0.1, wspace=0.12, hspace=0.75)
        _save(fig, out / "figS2_base_runs.png")
        plt.close(fig)


if __name__ == "__main__":
    main()
