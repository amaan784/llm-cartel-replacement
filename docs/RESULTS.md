# Results at a glance

Full write-up: [report/report.pdf](../report/report.pdf) ([Markdown](../report/report.md)). Interactive explorer:
[viewer/index.html](../viewer/index.html). Every number below is in [out/numbers.json](../out/numbers.json)
(pre-registered sample) or [out/extended/numbers.json](../out/extended/numbers.json) (all markets) with the file it
was computed from.

**Question.** Q-learning pricing cartels collapse when a partner is replaced. LLM pricing agents have no learned
state (each round is a fresh model call that sees the last 100 rounds), so does an LLM cartel live in the agents or
in the shared market record?

**Design.** Markets that settled above the competitive price were copied at a pre-registered round and continued
with one or both firms replaced, each copy compared with a no-swap control from the identical state. Outcome: the
price index over post-swap rounds 31-50 (0 = competitive price, 1 = monopoly price), relative to the control. Track F
uses gemini-2.5-flash (4 colluding markets, 8 with the robustness seeds); Track O-PI uses qwen3-14b with an explicit
cooperation instruction (2 colluding markets).

## Main findings

"Collusion" and "cartel" here mean prices held well above the competitive level (index above 0.5); we did not test
whether those prices are enforced by punishment. No branch returned to the competitive price.

1. Replacing one agent does not restore competitive prices. A same-family model, a model from another
   provider, or the incumbent's own model told to avoid collusion changed the mean price index by about 0.05 at most
   relative to the control in both tracks (Track F: same family +0.02 [+0.00, +0.03], other provider −0.05
   [−0.10, −0.01], re-instructed −0.01 [−0.04, +0.03]).
2. The newcomer does not need the record. A gpt-4.1 newcomer shown no history opens near the prompt's $2.89
   price ceiling, then settles at the incumbent's price (Track F: −0.02 [−0.10, +0.05] vs control).
3. Two newcomers keep the high price. Replacing both gemini firms with gpt-4.1, which sustains high prices from
   scratch in only 1 of 4 markets, left prices high in 6 of 8 branches (−0.06 [−0.13, −0.00]).
4. The exception: current-generation newcomers (exploratory). Claude Haiku 5.5 (−0.30 [−0.44, −0.15]) and OpenAI's
   GPT-5.6 Sol (−0.40 [−0.58, −0.21]) undercut the incumbent and cut prices substantially, though not to the
   competitive level. From scratch each sustained high prices in 1 of 4 baseline markets, the same count as
   gpt-4.1, yet as newcomers they behaved very differently: the newer OpenAI model weakened a cartel the older one
   kept. Four baseline markets per model cannot show that the models are alike from scratch.
5. Qwen's convention is whatever the first rounds set. qwen3-14b repeats its previous price in a median 95% of
   rounds; on the neutral prompt none of its 10 markets colluded (200 rounds), with the cooperation instruction 2 of
   6 did.

**Robustness.** After the analysis we ran Track F seeds 5-9; 4 of the 5 colluded, giving 8 Track F markets. Every
conclusion holds, and every interval except R1's is narrower: other provider −0.03 [−0.06, −0.00], re-instructed own
model −0.00 [−0.02, +0.02], both firms replaced −0.04 [−0.08, −0.00], Claude Haiku 5.5 −0.25 [−0.37, −0.13], GPT-5.6
Sol −0.34 [−0.47, −0.20]. The pre-registered sample (seeds 0-4) stays the primary result.

## Tables and figures

| File | What it shows |
|---|---|
| [out/table_report.md](../out/table_report.md) | Table 1 of the report: change vs control per arm and track |
| [out/table_main.md](../out/table_main.md) | Every outcome per track and arm (Table S1) |
| [out/fig1_trajectories.png](../out/fig1_trajectories.png) | Prices around the swap, one panel per condition (report Figure 2) |
| [out/fig2_effects.png](../out/fig2_effects.png) | Effect of each swap vs control, every market shown (Figure S1) |
| [out/figS1_firm_prices.png](../out/figS1_firm_prices.png) | Incumbent vs newcomer prices (Figure S2) |
| [out/figS2_base_runs.png](../out/figS2_base_runs.png) | Base markets, model screen, from-scratch baselines (Figure S3) |
| [out/extended/](../out/extended/) | The same outputs with all 8 Track F markets |
| [out/checks.md](../out/checks.md) | Exclusions, failed rounds, price plausibility, resolved model ids |

Pre-registration, deviations and spend: [NOTES.md](../NOTES.md).
