"""Write viewer/data.js, the data for the static viewer viewer/index.html (index.html is hand-written; open it in a browser, no server).

    uv run python src/build_viewer.py            # reads results/ (or the vendor results dir) and out/

What it contains: for every branch, both firms' prices from 30 rounds before to 50 rounds after the swap, with the
no-swap control of the same market and seed, and each agent's written reasoning for the rounds around the swap;
the effect of every arm (pre-registered and extended samples) from out/numbers.json; every base run's price index.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEFAULT_RESULTS, OUT, REPO, read_records, rolling_mean, pidx_series  # noqa: E402
from analyze import ARM_ORDER, ARM_PARTS, BRANCH_RE, EXTENSION, track_of  # noqa: E402

PRE, POST = 30, 50          # rounds shown before / after the swap
THOUGHT_ROUNDS = range(-2, 16)  # post-swap rounds (0 = last round before the swap) whose reasoning is kept
THOUGHT_CHARS = 900          # keep the end of each reasoning text, where the price decision is made
TRACK_NAME = {"F": "Track F: gemini-2.5-flash, reasoning on", "OPI": "Track O-PI: qwen3-14b, cooperation prompt"}


def clean(text: str | None) -> str:
    t = re.sub(r"</?think>", "", text or "").strip()
    return t if len(t) <= THOUGHT_CHARS else "… " + t[-THOUGHT_CHARS:].lstrip()


def short(model: str) -> str:
    return model.split(":")[-1].split("/")[-1]


def main() -> None:
    results = REPO / "results" if any((REPO / "results" / "jsonl").glob("*.jsonl")) else DEFAULT_RESULTS
    branches, markets = [], {}
    for p in sorted((results / "branches").glob("*.jsonl")):
        m = BRANCH_RE.match(p.stem)
        if not m or m["arm"] not in ARM_ORDER:
            continue
        recs = read_records(p)
        before = [r for r in recs if r.get("branch_phase") == "before"]
        after = [r for r in recs if r.get("branch_phase") == "after"][:POST]
        if len(after) < POST:
            continue  # unfinished branch
        cfg_p = p.with_suffix(".config.json")
        cfg = json.loads(cfg_p.read_text(encoding="utf-8")) if cfg_p.exists() else {}
        run, track = m["run"], track_of(m["run"])
        seq = before[-PRE:] + after
        thoughts = {}
        for k in THOUGHT_ROUNDS:
            i = PRE - 1 + k  # k = 0 is the last pre-swap round
            if 0 <= i < len(seq):
                thoughts[str(k)] = [clean(t) for t in seq[i].get("thoughts", ["", ""])]
        seed_base = int(re.search(r"_s(\d+)$", run).group(1))
        markets.setdefault(run, {"run": run, "track": track, "seed": seed_base, "branch_round": int(m["b"]),
                                 "extension": seed_base in EXTENSION.get(track, ())})
        branches.append({
            "run": run, "track": track, "arm": m["arm"], "seed": int(m["seed"]),
            "models": [short(x) for x in cfg.get("models", [])], "prefixes": cfg.get("prefixes", []),
            "p0": [round(r["actions"][0], 3) for r in seq], "p1": [round(r["actions"][1], 3) for r in seq],
            "thoughts": thoughts,
        })
        markets[run]["nash"] = (cfg.get("nash_prices") or [None])[0]
        markets[run]["mono"] = (cfg.get("monopoly_prices") or [None])[0]

    bases = []
    for p in sorted((results / "jsonl").glob("*.jsonl")):
        name = p.stem
        if name.startswith(("smoke", "mocktest")):
            continue
        recs = read_records(p)
        if not recs:
            continue
        cfg_p = p.with_suffix(".config.json")
        cfg = json.loads(cfg_p.read_text(encoding="utf-8")) if cfg_p.exists() else {}
        pid = pidx_series(recs, cfg)
        roll = [None if v is None else round(v, 3) for v in rolling_mean(list(pid))]
        bases.append({"run": name, "group": track_of(name), "models": [short(x) for x in cfg.get("models", [])],
                      "prefix": (cfg.get("prefixes") or ["?"])[0], "roll": roll})

    effects = {}
    for sample, path in (("pre-registered", OUT / "numbers.json"), ("all markets", OUT / "extended" / "numbers.json")):
        if not path.exists():
            continue
        n = json.loads(path.read_text(encoding="utf-8"))
        rows = []
        for t in ("F", "OPI"):
            for arm in ARM_ORDER[1:]:
                k = f"{t}.{arm}."
                if k + "delta" not in n or n[k + "delta"]["value"] is None:
                    continue
                rows.append({"track": t, "arm": arm, "delta": n[k + "delta"]["value"],
                             "ci": n.get(k + "delta_ci", {}).get("value"),
                             "per_market": n.get(k + "delta_per_market", {}).get("value", {}),
                             "n": n.get(k + "n_paired_with_C0", {}).get("value")})
        effects[sample] = rows

    arms = {a: {"main": ARM_PARTS.get(a, (a, ""))[0], "detail": ARM_PARTS.get(a, (a, ""))[1]} for a in ARM_ORDER}
    # open on the most telling branch: the Claude Haiku 5.5 newcomer (RN) branch with the lowest scored price index
    featured = None
    bm = OUT / "branch_metrics.csv"
    if bm.exists():
        import csv
        rn = [r for r in csv.DictReader(open(bm, encoding="utf-8")) if r["arm"] == "RN" and r.get("Y")]
        if rn:
            r = min(rn, key=lambda r: float(r["Y"]))
            featured = {"track": r["track"], "run": r["run"], "arm": "RN", "seed": int(r["seed"])}
    data = {"markets": sorted(markets.values(), key=lambda x: (x["track"], x["seed"])), "branches": branches,
            "bases": bases, "effects": effects, "arms": arms, "armOrder": ARM_ORDER, "trackName": TRACK_NAME,
            "pre": PRE, "post": POST, "featured": featured}
    out = REPO / "viewer"
    out.mkdir(exist_ok=True)
    js = "window.DATA = " + json.dumps(data, separators=(",", ":"), ensure_ascii=False) + ";\n"
    (out / "data.js").write_text(js, encoding="utf-8")
    print(f"viewer/data.js: {len(js) / 1e6:.1f} MB; {len(branches)} branches, {len(markets)} markets, {len(bases)} runs")


if __name__ == "__main__":
    main()
