# Paper Research Journey and Validation Audit

Date: 2026-05-27

This document converts the post-benchmark experiments into a paper-safe research narrative. It also records the validation status of each experiment so the thesis does not overclaim exploratory results.

## Validation Summary

| Experiment | Artifact status | Validation result | Paper-safe claim |
| --- | --- | --- | --- |
| Historical model plus sentiment | Complete | Metrics and deltas are internally consistent. External repo commit is recorded. | Sentiment conditionally improved some historical-model variants, but not uniformly. |
| Sentiment-return association | Complete | CSVs now explicitly store the one-calendar-day sentiment lag. No main lagged Spearman association meets the threshold. | Sentiment has weak descriptive associations, not stable causal or monotonic return impact. |
| Leakage and overfitting audit | Complete | Leakage status is PASS across 45 panel checks and 43,880 validation/test rows. | Final sentiment-only results are leakage-safe but weak. |
| AraBERT financial adaptation | Complete as feasibility | MLM adaptation completed for 20 CPU steps. A fixed-mask evaluation lowered validation MLM loss from 0.9624 to 0.8468 versus base AraBERT. Supervised fine-tuning correctly blocked without human labels. | Report as domain-adaptation protocol with MLM improvement, not an Arabic FinBERT equivalent. |
| Social-media extension | Complete as feasibility | Reddit and X were not collected. Only official/ethical routes were evaluated. | Keep social media as future work unless approved official access is obtained. |

The latest consistency audit returned `issues: 0`.

## Research Journey Narrative

The thesis should be written as a progression rather than as a list of disconnected experiments.

1. The project began with an industrial motivation from Robin/Beltone: test whether sentiment could improve an EGX historical direction-forecasting model.
2. The historical integration experiment showed conditional gains, especially for OIH with the MLP, but also losses for other ticker-model pairs.
3. Because the historical model already contained strong price-derived features, the research question was narrowed to a sentiment-only benchmark.
4. The core thesis pipeline then focused on bilingual Mubasher news, language-aware sentiment routing, relevance weighting, event-aware variants, and leakage-safe walk-forward evaluation.
5. The final accepted sentiment-only result was deliberately modest: OIH 5-day all-news bilingual passed validity checks but had balanced accuracy close to random.
6. The correlation/event study was added to inspect whether raw daily sentiment aligns with stock returns. It did not find a stable monotonic relation.
7. The AraBERT adaptation experiment tested the Arabic-finance NLP path. It showed the corpus can support domain-adaptive MLM and improves held-out MLM loss over base AraBERT, but it still cannot support supervised Arabic financial sentiment fine-tuning without human labels.
8. The social-media feasibility study closed the scope ethically: Reddit is possible only with official approval, and X/Twitter broad search is paid under official access.

## Required Paper Wording

Use:

- "conditional gains" for the historical-model experiment.
- "weak descriptive association" for sentiment-return correlations.
- "leakage-safe but modest" for the final sentiment-only benchmark.
- "domain-adaptive MLM feasibility" for AraBERT adaptation.
- "future work under approved official access" for social media.

Avoid:

- "sentiment improves the model" without qualifying ticker/model dependence.
- "sentiment impacts the stock" as a causal claim.
- "fine-tuned Arabic FinBERT" for the AraBERT experiment.
- "social-media experiment" if no posts were collected.
- selecting the OIH 5-day event-weighted row as the final headline result, because it was flagged for overfitting.

## Thesis Integration Points

Already updated:

- `Bachelor_Thesis_Template/results.tex`: added the auxiliary research-journey experiments and corrected coverage wording.
- `Bachelor_Thesis_Template/literature_review.tex`: added EGX Arabic tweets and recent Arabic financial NLP references.
- `Bachelor_Thesis_Template/conclusion.tex`: replaced placeholders with conclusion and future-work text.

Recommended final cleanup before submission:

- Re-run the resumable scraper for the one missing CIB Arabic listing page if time allows.
- Decide whether to include the full AraBERT checkpoint as an external artifact; it is large and should not be committed casually.
- Compile the LaTeX thesis and inspect table/page breaks after the added results text.
