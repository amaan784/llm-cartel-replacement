# llm-cartel-replacement

Does an LLM pricing cartel survive replacing one of its agents? A solo entry for Apart Research's
[AI Collusion Research Sprint](https://apartresearch.com/sprints/ai-collusion-research-sprint-2026-10-23-to-2026-10-25)
(October 2026), research question 1.2.

Q-learning pricing cartels collapse when a partner is swapped out, because each policy is fitted to its partner
(Eschenbaum et al. 2022). LLM pricing agents have no learned state: every round, each agent is a fresh API call
that sees only the last 100 rounds of prices and profits. So does an LLM cartel live in the agents or in the
shared market record? We let LLM duopolies settle on high prices, copy each market at a pre-registered round,
and continue it with one or both firms replaced (a same-family model, a model from another provider, a newcomer
without the history, the incumbent's own model told not to collude, a current-generation model), comparing every
branch with a no-swap control from the same market state.

## Result in one paragraph

Swapping out an LLM pricing agent does not break the cartel. Replacing one firm with a same-family model, a model
from another provider, or the incumbent's own model told not to collude moved prices by at most 0.05 of the
competitive-to-monopoly gap. Even two newcomers that rarely collude on their own kept an inherited high price,
because the shared price record and the remaining incumbent tell them what to charge. The one exception was a
current-generation model (Claude Haiku 5.5), which undercut the incumbent without restoring competition. Details:
[docs/RESULTS.md](docs/RESULTS.md) and the [report](report/report.md).

## Run it

```bash
bash run_all.sh      # or run_all.cmd on Windows Command Prompt / PowerShell
```

Setup, monitoring, stopping and an offline test: [docs/SETUP.md](docs/SETUP.md).

## Where things are

- [report/report.md](report/report.md): the write-up.
- [out/](out/): figures, tables and [out/numbers.json](out/numbers.json) (every reported number with its source).
- [NOTES.md](NOTES.md): lab notebook with the pre-registration, decisions, deviations and spend.
- [data/results.zip](data/results.zip): raw runs and branches, one JSON record per round including each model's
  reasoning (about 24 MB; unzip in the repository root to get `results/`, which the analysis reads).

The game harness is organizer-provided ([poseidon-research/collusion_hackathon](https://github.com/poseidon-research/collusion_hackathon),
commit `e6fb130`) and used unmodified; everything in `src/` is new work done during the sprint.
