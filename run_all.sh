#!/usr/bin/env bash
# Run the whole study end to end: experiments (in the background, resumable), then collect, analyze and export.
# Options are passed through to src/run_all.py (--no_wait, --skip_runs, --notice).
set -euo pipefail
cd "$(dirname "$0")"
uv run python src/run_all.py "$@"
