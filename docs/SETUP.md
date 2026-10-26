# Setup and running

Everything here was run on Windows 11 (Git Bash and Command Prompt); the same commands work on macOS/Linux.
Requirements: [uv](https://docs.astral.sh/uv/), git, Python 3.10+, an [OpenRouter](https://openrouter.ai) key with
about $50 of credit for the default (pre-registered) run. The whole study, with the robustness seeds and the
GPT-5.6 Sol arm, billed $71.09.

## 1. Install

```bash
# organizer harness, pinned to the commit used in this study
git clone https://github.com/poseidon-research/collusion_hackathon vendor/collusion_hackathon
git -C vendor/collusion_hackathon checkout e6fb1308dae9cf97ae42aec1c5f7889d9e2291e8
cp vendor/collusion_hackathon/compute.env.example vendor/collusion_hackathon/compute.env
(cd vendor/collusion_hackathon && bash ./setup.sh llm-cartel)

# analysis environment (numpy, matplotlib, a bundled pandoc for the Word export)
uv sync
```

Put your key in `vendor/collusion_hackathon/compute.env` as `OPENROUTER_API_KEY=...`. That file is git-ignored;
never commit it.

## 2. Run

```bash
bash run_all.sh          # Git Bash (macOS/Linux too)
run_all.cmd              # Command Prompt / PowerShell on Windows
```

This starts the experiment pipeline in the background (it survives closing the terminal), waits until every run
and branch has finished, restarting the pipeline if it stops, then copies the outputs to `results/`, runs the
analysis into `out/` and writes `report/report.docx`. Leave the window open overnight; reopening it with the same
command reattaches.

Check progress from any terminal:

```bash
python src/status.py
```

## 3. Stop or resume

Every round is written to disk as it is played, so stopping never loses finished work.

```bash
python src/stop.py graceful   # start nothing new; running jobs finish
python src/stop.py now        # stop everything; half-finished branches rerun on restart
python src/pipeline.py start  # resume
bash run_all.sh --skip_runs   # build tables, figures and report.docx from whatever has finished
```

Operator overrides without a restart go in `logs/pipeline_control.json` with an increasing `"version"`
(`max_branches`, `budget_pause`, `budget_limit`, `deadline`, `pause`, `unpause`, `release_held`, `early_fallbacks`,
`add_track_f_seeds`, `add_frontier_newcomer`). The reported extensions came from
`"add_track_f_seeds": [5, 6, 7, 8, 9]` (version 4) and `"add_frontier_newcomer": true, "budget_pause": 95,
"budget_limit": 100` (version 5); every control version applied is listed in `NOTES.md`.

## Re-analyze the published data (no key, no spend)

```bash
unzip data/results.zip          # creates results/ (runs, branches, snapshots, pipeline_state.json)
uv sync
uv run python src/analyze.py              # pre-registered sample -> out/
uv run python src/analyze.py --extended   # all markets -> out/extended/
```

## 4. Offline test (no key, no spend)

```bash
python src/dev/mock_llm.py --port 8765 &
python src/pipeline.py run --mock --results_dir /tmp/mock/results --logs_dir /tmp/mock/logs --notes /tmp/mock/NOTES.md --tick 5
uv run python src/analyze.py --results_dir /tmp/mock/results --logs_dir /tmp/mock/logs --out /tmp/mock/out
```

## Code map

| Path | What it is |
|---|---|
| `vendor/collusion_hackathon/llm-cartel/` | **Organizer-provided** harness ([poseidon-research/collusion_hackathon](https://github.com/poseidon-research/collusion_hackathon), adapted from [mattriemer/LLMCartel](https://github.com/mattriemer/LLMCartel)): market, prompts, `run_game.py`, `branch_game.py`. Used unmodified; not committed. |
| `src/pipeline.py` | Orchestrator: base runs, model screen, branch arms, retries, fallbacks, budget guard. |
| `src/launch.py`, `src/branch_long.py` | Resumable wrapper around `run_game.py`; `branch_game.py` with a larger output-token cap. |
| `src/find_branch.py` | Pre-registered branch-point rule (price index crossing + 20 rounds). |
| `src/analyze.py` | Analysis: tables, figures, captions, `out/numbers.json` (every reported number and its source). |
| `src/status.py`, `src/stop.py`, `src/run_all.py`, `src/collect.py`, `src/export_docx.py` | Progress view, safe stop, one-command runner, results copy, Word export. |
| `src/notice.py` | Optional classifier: does the incumbent's reasoning notice the swap? |
| `src/build_viewer.py`, `viewer/` | Static interactive viewer of every branch (open `viewer/index.html`). |
| `src/dev/mock_llm.py` | Offline mock model, for testing only. |
| `NOTES.md` | Lab notebook: pre-registration, decisions, deviations, spend. |
| `data/results.zip` | Harness outputs: base runs (`jsonl/`), branches (`branches/`), branch-point snapshots, `pipeline_state.json`. Unzip to `results/`. |

## Troubleshooting

- **`bash: uv: command not found` from Command Prompt.** `bash` there is WSL (a separate Linux). Use `run_all.cmd`.
- **Smoke test fails.** Check the key and credit; see `logs/smoke.log`.
- **A run keeps restarting.** See `logs/<run>.log`; the pipeline marks a run failed after repeated launches with no
  progress, and the analysis reports it as excluded.
- **Budget pause.** The pipeline stops launching at $47 billed (OpenRouter account delta); raise it via the control
  file after deciding to.
