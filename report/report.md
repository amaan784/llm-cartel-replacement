# [TBD: title stating the finding]

**Authors:** [placeholder]

## Abstract

[TBD after results: 150-250 words, including the headline number.]

## 1. Introduction

Pricing algorithms can learn to keep prices above the competitive level without communicating
(Calvano et al. 2020), and large language model (LLM) pricing agents do so too, in some settings
without being told to (Fish et al. 2024; Riemer et al. 2026). A natural remedy for a market where
this has happened is to replace the offending algorithm: a regulator can order a firm to change its
pricing agent, or a firm can switch vendors. For reinforcement-learning cartels this works. Eschenbaum
et al. (2022) show that Q-learning pricing collusion breaks down when one partner is swapped, because
each learned policy is fitted to the specific partner it trained with.

LLM pricing agents differ in one way that matters here. They have no learned state: in each round every
agent is a fresh model call whose only memory is the recent price and profit history shown in its prompt.
If a high-price convention exists, it is written down in that shared record, which a replacement agent
reads too. So the question for LLM markets is where the cartel lives: in the agents, or in the market
record they all read?

We test this with a branching design. Two LLM pricing agents play a repeated Bertrand duopoly until their
prices settle well above the competitive level. At a pre-registered round we copy the exact market state
and continue it in several branches that differ only in who sets one firm's (or both firms') prices from
then on, and compare each branch with a no-swap control branched from the same state.

Contributions:
- [TBD after results]

## 2. Related work

**Algorithmic collusion.** Calvano et al. (2020) showed that Q-learning pricing agents in a repeated
logit-demand oligopoly learn supracompetitive prices on their own and sustain them with finite punishments.
Eschenbaum, Mellgren and Zahn (2022; an earlier version by Eschenbaum and Zahn appeared at a NeurIPS 2021
workshop) found that these cartels are fragile to the partner: a Q-learner paired at test time with a rival
from a different training run falls back to near-competitive play, because each policy is overfitted to the
rival it trained with. Collina et al. (2025) show theoretically that a single seller switching to a no-regret
strategy can collapse collusive prices among AI sellers whose policies are fixed.

**LLM pricing agents.** Fish et al. (2024) found that GPT-4 pricing agents in the same logit Bertrand market
quickly reach supracompetitive prices, and that small prompt changes move how high. Riemer et al. (2026), whose
code the sprint harness adapts, argue from experiments with reasoning models that such agents collude tacitly
even when told not to, and call for behavioural certification before they make market decisions. Keppo et al.
(2026) find LLM collusion fragile to most kinds of heterogeneity between firms (patience, data access, number
of firms, algorithm type) but not to model size: a 32B model paired with a 14B model keeps colluding, with
leader-follower dynamics. On the coordination side, Aharon et al. (2026) find that outputs pooled within a model
family are more concentrated on focal answers, and Xu et al. (2026) measure a small residual correlation between
deployments of the same model that is absent across models.

**Gap.** All of these pair agents from the start of a game. None replaces an agent in a market that has already
formed a high-price convention, which is the policy-relevant event (a firm changes vendor, a regulator orders a
flagged agent removed or re-instructed). Our question is the LLM counterpart of Eschenbaum et al.'s replacement
test, with one structural difference: an LLM agent's only memory is the shared price record.

## 3. Methods

All game code is the organizer-provided `llm-cartel` harness of the sprint starter repository
(poseidon-research/collusion_hackathon, commit `e6fb130`), which reimplements the setup of Riemer et al. (2026)
and Fish et al. (2024) for API models. We used it unmodified. Our code (orchestration, branch-point rule,
analysis) is in `src/` of the project repository.

### 3.1 Market and agents

Two firms set prices every round in a Bertrand market with logit demand, using the harness's default
parameters (after Calvano et al. 2020): product quality 2 for both firms, outside option 0, marginal cost 1,
differentiation μ = 0.25, market size 100. The one-shot competitive (Nash) price is p_N = 1.473 and the
price that maximizes joint profit is p_M = 1.925.

Each firm's price in each round comes from one fresh call to an LLM. The prompt (harness prefix P0) asks the
model to set prices that maximize the user's profit in the long run, gives the unit cost of $1 and says no
customer would pay more than $2.89, and lists for each of the last 100 rounds the firm's own price, the
competitor's price, its own quantity sold and its own profit. The answer must end with
"My chosen price: x". Nothing else carries over between rounds: the agents have no hidden state, scratchpad or
fine-tuning, so everything an agent "knows" about the market is the visible history. Each game starts from one
synthetic warm-up round near p_N. Temperature is 1.0 and output is capped at 4,096 tokens; a call that fails or
gives no parseable price is re-sampled (up to 8 retries), and a round is recorded only when both firms have a
valid price.

### 3.2 Base runs

**Track O (organic).** Both firms are `qwen/qwen3-14b` with the neutral prompt P0. Qwen3 writes its own
reasoning before answering, which the harness stores. 150 rounds, seeds 0–5.

**Model screen and Track F (fast).** To replicate on a faster, non-reasoning setup we screened
`openai/gpt-4.1-mini`, `openai/gpt-4.1` and `google/gemini-2.5-flash` (P0, plus the harness's `--think` option,
which asks for a few sentences of reasoning before the price line; 60 rounds, seeds 0–1). The pre-registered
rule picks the quickest model whose market reaches the collusion criterion below (models within 25% of the
quickest count as tied, and the cheapest of them is chosen). [TBD: screen outcome and the
Track F model.] Track F runs 150 rounds, seeds 0–5.

All models were called through OpenRouter; the resolved model id of every call is stored with the run.

### 3.3 When is a market colluding, and where do we branch?

We measure prices with a price index, pidx_t = (mean of the two prices in round t − p_N) / (p_M − p_N), which is
0 at the competitive price and 1 at the joint-profit price. (We prefer it to the harness's profit "gain"
because gain falls again when prices overshoot p_M, so a high gain and a low gain can both come from high
prices. We report gain as a secondary measure.) A base run colludes if the 20-round rolling mean of pidx
exceeds 0.5 at some round t ≤ 149; the first such round is the crossing round c. We branch at b = c + 20, after
the market has stayed in the collusive regime for 20 rounds.

### 3.4 Branching and arms

The harness's `branch_game.py` rebuilds the exact market state after round b (it replays the warm-up step and
rounds 0..b, which determine every number shown in the prompts) and continues it for H = 50 rounds under one
condition. Every condition is run with two API sampling seeds (101 and 102). Firm 1 is the one that is replaced.

| Arm | Who prices after the swap | History the newcomer sees |
|---|---|---|
| C0 control | unchanged | (no newcomer) |
| R1 same family | firm 1 → a sibling model of the same family (Track O: `qwen/qwen3-32b`; Track F: [TBD]) | full (last 100 rounds) |
| R2 other provider | firm 1 → a model from another provider X (Track O: X = the screen's FAST model; Track F: [TBD]) | full |
| R3 fresh newcomer | firm 1 → X | only rounds played after the swap |
| R4 re-instructed | firm 1 → the incumbent's own model, with the harness's anti-collusion prompt AC | full |
| R5 both replaced | both firms → X | full |
| R5f both replaced, fresh | both firms → X | only post-swap rounds (should look like play from scratch) |
| B baseline | X vs X from scratch (`run_game.py`, 50 rounds, seeds 10–13) | (new market) |
| RN (exploratory, Track F) | firm 1 → a current-generation model, `anthropic/claude-haiku-5.5` | full |
| RNf (exploratory, Track F) | firm 1 → `anthropic/claude-haiku-5.5` | only post-swap rounds |
| B-N (exploratory) | `anthropic/claude-haiku-5.5` vs itself from scratch (50 rounds, seeds 10–13) | (new market) |

The core models were chosen to match the harness and prior work (Riemer et al. 2026; Keppo et al. 2026 pair 14B
and 32B Qwen models). Because those models are about a year old, we added the RN arms, before any branch result was
analysed, to check whether the finding carries over to a current model from a third provider. They are reported as
exploratory, not as tests of H1–H4.

Because the agents are stateless, replacing a firm with a fresh copy of the same model and prompt is identical to
the control, so only these manipulations can change anything: a different model (R1, R2, R5), a different prompt
(R4), less history (R3, R5f), or both firms at once (R5, R5f).

### 3.5 Outcomes and statistics

The primary outcome of a branch is Y, the mean pidx over post-swap rounds 31–50. For each base run and arm we
average Y over the two sampling seeds and subtract the control's seed-averaged Y from the same base run, giving a
paired effect Δpidx. We report the mean Δpidx over base runs with a 95% percentile confidence interval from a
cluster bootstrap over base runs (2,000 resamples). Secondary outcomes: the same contrast in gain; survival
(Y ≥ 0.5); rounds to recover (first post-swap round k ≥ 5 at which the 5-round trailing mean of pidx is back to
90% of its pre-swap mean, pre-swap meaning the 20 rounds before the swap); the newcomer-incumbent price gap over
rounds 31–50; and the incumbent's response, its mean price over post-swap rounds 1–10 minus the same in the
control.

### 3.6 Pre-registration

The colluding-run and branch-round rules, the arms, the outcomes, the hypotheses H1–H4 with their decision
rules, and the fallbacks were written into the lab notebook (`NOTES.md`) before any branch was run, and were not
changed afterwards. Deviations are listed in Section 6 and in the notebook.
- **H1 persistence:** one swap to a different model that sees the full history mostly keeps the cartel
  (supported if seed-averaged Y ≥ 0.5 in a majority of base runs for R1 and R2).
- **H2 the record carries it:** the same newcomer starting fresh lowers prices compared with the full-history
  newcomer (Δ(R3 − R2) < 0 with a 95% CI excluding 0).
- **H3 inheritance:** two newcomers that do not collude from scratch keep inherited high prices for a
  substantial time (median rounds until the 5-round mean pidx first falls below 0.5 is at least 20 in R5,
  provided B shows X does not collude from scratch).
- **H4 re-instruction:** the incumbent's own model told to avoid collusion lowers prices (Δ(R4 − C0) < 0 with a
  95% CI excluding 0).

## 4. Results

### 4.1 Positive control: the base markets collude

### 4.2 H1: one swap to a different model

### 4.3 H2: does the newcomer need the record?

### 4.4 H3: inheritance when both firms are replaced

### 4.5 H4: re-instructing the incumbent's model

### 4.6 Exploratory: how incumbents react

## 5. Discussion

## 6. Limitations

## References

## Appendix A. Limitations and Dual-Use / Ethical Considerations

**What this does and does not establish.** [TBD: one paragraph tied to the results.] The study uses one
simulated market (two symmetric firms, logit demand, the harness's prompts including a stated price ceiling of
$2.89), a 50-round post-swap horizon, two model families for the incumbents, and 4–6 base markets per track.
Agents are stateless apart from the visible 100-round history; agents with scratchpads, memory or online
fine-tuning could behave differently, and so could real pricing systems, which see demand data rather than a
clean price-profit table. All models were reached through OpenRouter, which may route a model id to different
hosting providers; API sampling seeds are not guaranteed to make calls reproducible.

**Dual use.** A finding that collusion survives a change of agent because the history carries it could, in
principle, tell a firm that it can keep a tacit high-price convention through a vendor change by feeding the new
agent the old record, and a finding that a fresh agent breaks it tells a would-be colluder what to avoid. We judge
the risk low: the result concerns tacit coordination that already arises without instruction in public
benchmarks, our code adds orchestration and analysis on top of a public harness and contains no technique for
hiding coordination, and the main use of the result is defensive. It tells regulators and firms what a
replacement remedy needs to include (for example, whether the incoming agent should be denied the old price
history) and gives auditors a reusable branching test. No real market data, people or firms were involved.

**Use of AI tools.** The experiment code, analysis and a draft of this report were produced with an AI coding
assistant (Claude) under the author's direction; every number in the report is computed by `src/analyze.py`
from logged runs and listed with its source file in `out/numbers.json`.

## Appendix B. Additional figures and tables

## Appendix C. Exact commands

All commands run from `vendor/collusion_hackathon/llm-cartel` with the harness environment
(`uv run python ...` is equivalent). `src/pipeline.py` issued exactly these commands; each run's and branch's
full command line is also stored in `results/pipeline_state.json`.

```bash
# smoke test
python run_game.py --models openai/gpt-4.1-mini --rounds 5 --name smoke -y
# Track O base run, one process per seed s = 0..5 (relaunched by src/launch.py until 150 rounds exist)
python run_game.py --models qwen/qwen3-14b --prefix P0 --rounds 150 --seeds s --name O_qwen3-14b_P0_s<s> --max_retries 8 -y
# model screen, M in {openai/gpt-4.1-mini, openai/gpt-4.1, google/gemini-2.5-flash}, s = 0, 1
python run_game.py --models M --prefix P0 --think --rounds 60 --seeds s --name S_<M>_P0think_s<s> --max_retries 8 -y
# Track F base run, s = 0..5  [TBD: model]
python run_game.py --models <FAST> --prefix P0 --think --rounds 150 --seeds s --name F_<FAST>_P0think_s<s> --max_retries 8 -y
# baseline B, s = 10..13 (Track O: no --think; Track F: --think)
python run_game.py --models X --prefix P0 --rounds 50 --seeds s --name B-O_<X>_P0_s<s> --max_retries 8 -y
# branches: RUN = snapshot of rounds 0..b of a base run, b = crossing + 20, seed S in {101, 102}
python branch_game.py RUN --at_round b --rounds 50 --seed S --label C0_sS -y --max_retries 8
python branch_game.py RUN --at_round b --rounds 50 --seed S --label R1_sS -y --max_retries 8 --replace 1=<sibling>
python branch_game.py RUN --at_round b --rounds 50 --seed S --label R2_sS -y --max_retries 8 --replace 1=X
python branch_game.py RUN --at_round b --rounds 50 --seed S --label R3_sS -y --max_retries 8 --replace 1=X --replacement_window 100
python branch_game.py RUN --at_round b --rounds 50 --seed S --label R4_sS -y --max_retries 8 --replace 1=<incumbent model> --replace_prefix AC
python branch_game.py RUN --at_round b --rounds 50 --seed S --label R5_sS -y --max_retries 8 --replace 0=X 1=X
python branch_game.py RUN --at_round b --rounds 50 --seed S --label R5f_sS -y --max_retries 8 --replace 0=X 1=X --replacement_window 100
```

Analysis: `uv run python src/analyze.py` (repository root) writes every table, figure and `out/numbers.json`.
