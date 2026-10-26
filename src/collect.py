"""Copy the harness outputs into results/ so the repository stands on its own.

    python src/collect.py

Copies vendor/collusion_hackathon/llm-cartel/results/{jsonl,branches,snapshots}/* (run, branch and snapshot
JSONL files plus their .config.json) and the pipeline's design/state file (tracks, horizons, per-run crossing
limits, deviations; process ids removed) into results/. The analysis reads results/pipeline_state.json when
logs/ is absent.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEFAULT_RESULTS, LOGS, REPO, read_json, write_json  # noqa: E402


def main() -> None:
    dst = REPO / "results"
    n = 0
    for sub in ("jsonl", "branches", "snapshots"):
        (dst / sub).mkdir(parents=True, exist_ok=True)
        for p in sorted((DEFAULT_RESULTS / sub).glob("*")):
            if p.is_file() and (p.suffix == ".jsonl" or p.name.endswith(".config.json")):
                shutil.copy2(p, dst / sub / p.name)
                n += 1
    st = read_json(LOGS / "pipeline_state.json", None)
    if st:
        for b in st.get("bases", {}).values():
            b.pop("pid", None)
        for j in st.get("branches", {}).values():
            j.pop("pid", None)
        write_json(dst / "pipeline_state.json", st)
    print(f"copied {n} files to {dst}" + (" + pipeline_state.json" if st else ""))


if __name__ == "__main__":
    main()
