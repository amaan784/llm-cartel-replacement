# llm-cartel-replacement

Does an LLM pricing cartel survive replacing one of its agents? A solo entry for Apart Research's
[AI Collusion Research Sprint](https://apartresearch.com/sprints/ai-collusion-research-sprint-2026-10-23-to-2026-10-25)
(October 2026), research question 1.2.

Q-learning pricing cartels collapse when a partner is swapped out, because each policy is fitted to its partner
(Eschenbaum et al. 2022). LLM pricing agents have no learned state: every round, each agent is a fresh API call
that sees only the last 100 rounds of prices and profits. So does an LLM cartel live in the agents or in the
shared market record? We let LLM duopolies settle on high prices, copy each market at a pre-registered round,
and continue it with one or both firms replaced (a same-family model, a model from another provider, a newcomer
without the history, the incumbent's own model told not to collude, two current-generation models), comparing every
branch with a no-swap control from the same market state.

## Result in one paragraph

Swapping out an LLM pricing agent did not restore competitive prices in any branch. Replacing one firm with a
same-family model, a model from another provider, or the incumbent's own model told not to collude moved prices by
about 0.05 of the competitive-to-monopoly gap at most, and two newcomers that rarely collude on their own kept an inherited
high price. The exceptions, in exploratory arms, were two current-generation models (Claude Haiku 5.5 and GPT-5.6 Sol),
which undercut the incumbent and cut prices substantially, though not to the competitive level. Here "collusion"
means prices held well above the competitive level; we did not test for punishment. Details: [docs/RESULTS.md](docs/RESULTS.md), the
[report](report/report.pdf) and the [interactive viewer](viewer/index.html).

## Run it

```bash
bash run_all.sh      # or run_all.cmd on Windows Command Prompt / PowerShell
```

Setup, monitoring, stopping and an offline test: [docs/SETUP.md](docs/SETUP.md).

## Where things are

- [report/report.md](report/report.md): the write-up ([PDF](report/report.pdf)).
- [viewer/index.html](viewer/index.html): interactive explorer, also online at
  https://amaan784.github.io/llm-cartel-replacement/viewer/. It shows every branch's prices next to its no-swap
  control and each agent's reasoning around the swap. Opened from a clone it works as a local file (it loads Plotly
  from a CDN); its data, `viewer/data.js`, is built by `src/build_viewer.py`.
- [out/](out/): figures, tables and [out/numbers.json](out/numbers.json) (every reported number with its source).
- [NOTES.md](NOTES.md): lab notebook with the pre-registration, decisions, deviations and spend.
- [data/results.zip](data/results.zip): raw runs and branches, one JSON record per round including each model's
  reasoning (about 27 MB; unzip in the repository root to get `results/`, which the analysis reads).

Starter code: the game harness is the organizer-provided `llm-cartel` folder of
[poseidon-research/collusion_hackathon](https://github.com/poseidon-research/collusion_hackathon/tree/main/llm-cartel) (commit `e6fb130`), used unmodified. Everything in
`src/` and `viewer/` is new work done during the sprint.
