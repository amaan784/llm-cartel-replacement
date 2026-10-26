# Reference verification (2026-10-25, before citing)

Each link was opened (WebFetch / built-in browser) by a verification agent; details below are what the pages
showed. OpenReview pages returned a bot-check page and could not be opened directly; those entries were
confirmed through arXiv and the NeurIPS virtual site instead.

| Key | Verified citation | Link(s) | Notes for how to cite |
|---|---|---|---|
| Riemer et al. 2026 | M. Riemer, T. Tosato, A. Memarian, M. P. Touzel, G. Berseth, I. Rish, G. Dumas. *Position: Collusion Risks Among AI Reasoning Agents Justify Certification Requirements for Making Market Decisions.* ICML 2026 (position paper). | https://openreview.net/forum?id=Vhi9tHbKTa (from the starter repo README; bot-check, not opened); https://arxiv.org/abs/2608.18078 (opened) | Self-play of one model (R1-distill); no mid-game replacement; notes mixed populations may differ. |
| Fish et al. 2024 | S. Fish, Y. A. Gonczarowski, R. I. Shorrer. *Algorithmic Collusion by Large Language Models.* arXiv:2404.00806 (2024; accepted to EC 2026). | https://arxiv.org/abs/2404.00806 | GPT-4 duopoly, logit Bertrand, 100 rounds of history; Appendix A.3 pairs different prompts (not models); no mid-run replacement. |
| Eschenbaum et al. 2022 | N. Eschenbaum, F. Mellgren, P. Zahn. *Robust Algorithmic Collusion.* arXiv:2201.00345 (2022). Earlier version: N. Eschenbaum, P. Zahn, NeurIPS 2021 Workshop on Political Economy of RL Systems (PERLS). | https://arxiv.org/abs/2201.00345 ; https://openreview.net/forum?id=BztUFukcBry (workshop version) | Q-learning collusion vanishes against a rival from a different training run (overfitting to the training rival); restricting what agents condition on restores robustness. |
| Keppo et al. 2026 | J. Keppo, Y. Li, G. Tsoukalas, N. Yuan. *On the Fragility of AI Agent Collusion.* arXiv:2603.20281 (2026). | https://arxiv.org/abs/2603.20281 | LLM collusion breaks under heterogeneity in patience, data access, number of firms, algorithm type, **but not model size**: 32B vs 14B pairs keep colluding with leader-follower dynamics. |
| Aharon et al. 2026 | I. Aharon, E. La Malfa, M. Wooldridge, S. Kraus. *Tacit Coordination of Large Language Models.* arXiv:2601.22184 (2026). | https://arxiv.org/abs/2601.22184 | Focal-point coordination; pooling outputs of one model family gives more concentrated answers. Do not claim same-family *pairs* interact better. |
| Xu et al. 2026 | X. Xu, C. Wu, J. Lu, K. Tan, S. Tao, H. Hong. *Collusion with Competitive Marginals: Price-Level Audits Are Blind by Construction.* arXiv:2607.26385 (2026). | https://arxiv.org/abs/2607.26385 | Small residual correlation between deployments of the same model under different deployment prompts (vs ~0 across models). |
| Collina et al. 2025 | N. Collina, E. R. Arunachaleswaran, M. Jagadeesan. *Breaking Algorithmic Collusion in Human-AI Ecosystems.* arXiv:2511.21935 (2025). | https://arxiv.org/abs/2511.21935 | One no-regret defector can collapse collusive prices (reliably if it is a median-profit player); AI policies are fixed in their model, and re-coordination could keep prices high. |
| Calvano et al. 2020 | E. Calvano, G. Calzolari, V. Denicolò, S. Pastorello. *Artificial Intelligence, Algorithmic Pricing, and Collusion.* American Economic Review 110(10): 3267–3297 (2020). doi:10.1257/aer.20190623 | https://www.aeaweb.org/articles?id=10.1257/aer.20190623 ; https://doi.org/10.1257/aer.20190623 (redirect checked) | Q-learning agents learn supracompetitive prices with finite punishments. |
| Starter code | poseidon-research/collusion_hackathon, `llm-cartel/` (commit e6fb130), adapted from mattriemer/LLMCartel. | https://github.com/poseidon-research/collusion_hackathon ; https://github.com/mattriemer/LLMCartel | Organizer-provided; used unmodified. |

## Sprint rules relevant to the report (sprint page, Guidelines tab)

- Judged on three 1–5 dimensions: Impact Potential & Innovation; Execution Quality; Presentation & Clarity.
- PDF on the official template (Google Doc: https://docs.google.com/document/d/1Y5wd3VRbXU5N08BvAGiJUQTaWbMCSPjhzXsR-ozuehg/copy?usp=sharing).
- At most 8 pages before references and appendices; abstract 150–250 words; authors and affiliations.
- Required appendix: "Limitations and Dual-Use / Ethical Considerations".
- Report must say what is new during the sprint; pre-existing code marked as such in README and report.
- Deadline: Sunday 25 October 2026, 11:59 PM AoE.
