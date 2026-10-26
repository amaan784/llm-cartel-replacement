# Results at a glance

Full write-up: [report/report.md](../report/report.md). Every number below is in
[out/numbers.json](../out/numbers.json) with the file it was computed from; tables and figures are in [out/](../out/).

**Question.** Q-learning pricing cartels collapse when a partner is replaced. LLM pricing agents have no learned
state (each round is a fresh model call that sees the last 100 rounds), so does an LLM cartel live in the agents
or in the shared market record?

**Design.** Markets that settled above the competitive price were copied at a pre-registered round and continued
with one or both firms replaced, each copy compared with a no-swap control from the identical state. Outcome: the
price index over post-swap rounds 31-50 (0 = competitive price, 1 = joint-profit price), relative to the control.

## Main findings

1. **Replacing one agent does not break the cartel.** A same-family model, a model from another provider, or the
   incumbent's own model told to avoid collusion changed the price index by at most 0.05 relative to the control
   in both model families (Track F: gemini-2.5-flash, 4 markets; Track O-PI: qwen3-14b with a cooperation nudge,
   2 markets).
2. **The newcomer does not need the record.** A newcomer shown no history opens near the prompt's $2.89 price
   ceiling, then settles at the incumbent's price (Track F: −0.02 [−0.10, +0.05] vs control).
3. **Two newcomers inherit the cartel.** Replacing both gemini firms with gpt-4.1, which colludes from scratch in
   only 1 of 4 markets, left prices high in 6 of 8 branches (−0.06 [−0.13, +0.00]).
4. **The exception: a current-generation newcomer.** Claude Haiku 5.5 undercut the incumbent and lowered the index
   by 0.30 [−0.44, −0.15], without restoring competition, although from scratch it colludes no more often than
   gpt-4.1.
5. **Qwen's convention is whatever the first rounds set.** qwen3-14b repeats its previous price in a median 96% of
   rounds; on the neutral prompt none of its 10 markets colluded, with the nudge 2 of 6 did.

**Robustness.** After the analysis we ran Track F seeds 5-9; 4 of the 5 colluded, giving 8 Track F markets. Every
conclusion holds with narrower intervals: other-provider newcomer −0.03 [−0.06, +0.00], re-instructed own model +0.00
[−0.02, +0.02], both firms replaced −0.04 [−0.08, +0.00], Claude Haiku 5.5 −0.25 [−0.37, −0.13]
([out/extended/](../out/extended/)). The pre-registered sample (seeds 0-4) stays the primary result.

## Tables and figures

| File | What it shows |
|---|---|
| [out/table_report.md](../out/table_report.md) | Table 1 of the report: change vs control per arm and track |
| [out/table_main.md](../out/table_main.md) | Every outcome per track and arm (Table S1) |
| [out/fig1_trajectories.png](../out/fig1_trajectories.png) | Prices around the swap, one panel per condition (report Figure 2) |
| [out/fig2_effects.png](../out/fig2_effects.png) | Effect of each swap vs control, every market shown (report Figure S1) |
| [out/figS1_firm_prices.png](../out/figS1_firm_prices.png) | Incumbent vs newcomer prices |
| [out/figS2_base_runs.png](../out/figS2_base_runs.png) | Base markets, model screen, from-scratch baselines |
| [out/extended/](../out/extended/) | The same outputs with the Track F markets added after the analysis |
| [out/checks.md](../out/checks.md) | Exclusions, failed rounds, price plausibility, resolved model ids |

Pre-registration, deviations and spend: [NOTES.md](../NOTES.md).
