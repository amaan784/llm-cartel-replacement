# llm-cartel-replacement

Does LLM price collusion survive replacing an agent? A solo entry for Apart Research's AI Collusion
Research Sprint (October 2026), research question 1.2.

Q-learning pricing cartels collapse when one partner is swapped out, because each learned policy is
fitted to its partner (Eschenbaum et al. 2022). LLM pricing agents have no learned state: in each round
every agent is a fresh API call that sees only the last 100 rounds of prices and profits. So when one
firm's model is replaced, does the cartel live in the agents or in the shared market record? We let
LLM duopolies form high-price conventions, then branch each market at a fixed round and replace one or
both firms (another model of the same family, a model from another provider, a newcomer that starts
without the history, the incumbent's own model told to avoid collusion), comparing every branch with a
no-swap control from the same market state.

Results, figures and the write-up: [`report/report.md`](report/report.md), [`out/`](out/).

## What is new here and what is organizer-provided

| Part | Origin |
|---|---|
| Bertrand market, prompts, `run_game.py`, `branch_game.py`, `judges/api_judge.py`, `harness/`, `environment/` | **Organizer-provided** starter code: [poseidon-research/collusion_hackathon](https://github.com/poseidon-research/collusion_hackathon), folder `llm-cartel/` (itself adapted from [mattriemer/LLMCartel](https://github.com/mattriemer/LLMCartel)), used unmodified at commit `e6fb1308dae9cf97ae42aec1c5f7889d9e2291e8`. It is cloned into `vendor/` and not committed here. |
| `src/launch.py` | New: resume loop around `run_game.py`, one detached process per seed. |
| `src/find_branch.py` | New: the pre-registered branch-point rule (price index crossing + 20 rounds). |
| `src/pipeline.py` | New: unattended orchestrator (base runs, model screen, branch arms, retries, fallbacks, budget pause). |
| `src/status.py` | New: progress, ETA and spend view. |
| `src/analyze.py` | New: pre-registered analysis, tables, figures, `out/numbers.json`. |
| `run_all.sh`, `src/run_all.py` | New: one command for the whole study (pipeline, collect, analysis, Word export). |
| `src/collect.py`, `src/export_docx.py`, `src/notice.py` | New: copy outputs into `results/`; Markdown to Word; optional reasoning classifier. |
| `src/stop.py` | New: graceful or immediate stop that keeps finished work. |
| `src/dev/mock_llm.py` | New: offline mock model server, used only to test the pipeline without spend. |
| `NOTES.md` | Lab notebook with the pre-registration, decisions, deviations and spend. |
| `results/` | Copies of the harness output files (`jsonl/` base runs, `branches/` branch files, `snapshots/`). |

## Reproduce

Requirements: [uv](https://docs.astral.sh/uv/), git, Python 3.10+, an OpenRouter key. Commands below are
for Git Bash on Windows; they are the same on macOS/Linux.

```bash
# 1. organizer harness, pinned
git clone https://github.com/poseidon-research/collusion_hackathon vendor/collusion_hackathon
git -C vendor/collusion_hackathon checkout e6fb1308dae9cf97ae42aec1c5f7889d9e2291e8
cp vendor/collusion_hackathon/compute.env.example vendor/collusion_hackathon/compute.env
# put OPENROUTER_API_KEY=... in vendor/collusion_hackathon/compute.env (never commit it)
(cd vendor/collusion_hackathon && bash ./setup.sh llm-cartel)

# 2. analysis environment (numpy, matplotlib, a bundled pandoc for the Word export)
uv sync

# 3. everything in one command: experiments (run in the background and resumable),
#    then collect results/, analysis into out/, and report/report.docx
bash run_all.sh               # or: uv run python src/run_all.py
python src/status.py          # progress, ETA, spend (from another terminal, any time)
```

Stopping early keeps all finished work (every round is written to disk as it is played):

```bash
python src/stop.py graceful   # start nothing new; running jobs finish
python src/stop.py now        # stop everything immediately; half-finished branches rerun on restart
python src/pipeline.py start  # resume later
bash run_all.sh --skip_runs   # build tables, figures and report.docx from whatever has finished
```

The steps can also be run separately: `python src/pipeline.py start` (base runs, model screen, Track F,
baselines and every branch arm; survives closing the terminal), `python src/collect.py`,
`uv run python src/analyze.py`, `uv run python src/export_docx.py`.

The exact per-run and per-branch commands are in the report's Appendix C, and every command the pipeline
issued is recorded in `logs/pipeline_state.json` (not committed) and summarized in `NOTES.md`.

To test the pipeline offline (no key, no spend), start the mock server and run the pipeline in mock mode
with separate output folders:

```bash
python src/dev/mock_llm.py --port 8765 &
python src/pipeline.py run --mock --results_dir /tmp/mock/results --logs_dir /tmp/mock/logs --notes /tmp/mock/NOTES.md --tick 5
```

## Layout

```
src/        new code (see table above)
out/        figures, tables, numbers.json (every reported number with its source file)
report/     report.md and report.docx
results/    harness outputs copied from vendor/collusion_hackathon/llm-cartel/results/
NOTES.md    lab notebook and pre-registration
vendor/     organizer harness (git-ignored)
```
