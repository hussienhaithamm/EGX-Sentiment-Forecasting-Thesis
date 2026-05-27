# Experiment Log

This file records the additional experiments run after the main sentiment-only benchmark. It is meant to provide paper-ready traceability: what was tested, where the outputs are stored, and how strongly the results should be claimed.

## Experiment 1: Historical Model With Sentiment Features

**Question.** Does the thesis sentiment pipeline improve a historical price-based EGX direction model from the Robin/Beltone internship codebase?

**External codebase.** `https://github.com/anastamerr/EGX-Direction-Forecasting`, clean local checkout at commit `4a3652bda8cfd5e05809fe7c38345964c93e688c`.

**Script.** `scripts/run_historical_sentiment_feature_experiment.py`

**Outputs.**

- `outputs/historical_sentiment_feature_experiment/historical_sentiment_summary.md`
- `outputs/historical_sentiment_feature_experiment/historical_sentiment_metrics.csv`
- `outputs/historical_sentiment_feature_experiment/historical_sentiment_deltas.csv`
- `outputs/historical_sentiment_feature_experiment/historical_sentiment_predictions.csv`
- `outputs/historical_sentiment_feature_experiment/historical_sentiment_feature_importance.csv`

**Design.** The experiment compares the external repo's historical features against the same features augmented with one-day-lagged bilingual sentiment features. The target is 5-trading-day direction. Tickers are CIB, TMG, and OIH. Models are logistic regression, XGBoost, and MLP. Baseline and augmented variants share the same chronological split.

**Headline result.** Sentiment improves raw test accuracy in 5 of 9 ticker/model comparisons. The strongest gain is OIH with MLP: accuracy increases by `+0.143`, balanced accuracy by `+0.106`, and F1 by `+0.309`. The strongest loss is CIB with MLP: accuracy changes by `-0.105`. Logistic regression is mostly flat or worse. XGBoost shows small gains for CIB and OIH but a small loss for TMG.

**Paper interpretation.** This supports a conditional integration claim: sentiment can help when added to a stronger historical model, but it is not uniformly beneficial across tickers or models.

## Experiment 2: Sentiment Score and Stock Impact

**Question.** Are daily bilingual sentiment scores associated with contemporaneous or future stock returns?

**Script.** `scripts/run_sentiment_stock_impact_analysis.py`

**Outputs.**

- `outputs/sentiment_stock_impact_analysis/sentiment_stock_impact_report.md`
- `outputs/sentiment_stock_impact_analysis/sentiment_return_correlations.csv`
- `outputs/sentiment_stock_impact_analysis/sentiment_event_study.csv`
- `outputs/sentiment_stock_impact_analysis/sentiment_return_panel.csv`

**Design.** Daily sentiment is shifted by one calendar day before being aligned to trading dates. Pearson and Spearman correlations are computed for same-day, 1-day, 3-day, and 5-day returns. P-values are estimated by block permutation with five-day blocks. An event-style grouping compares top and bottom sentiment quartiles against all-news and no-prior-news days.

**Headline result.** No strong, stable monotonic sentiment-return association appears under the pre-specified threshold of absolute Spearman correlation >= `0.10` and block-permutation p-value <= `0.05`. The largest displayed Spearman values are OIH 3-day `-0.127` with p-value `0.155`, OIH 5-day `-0.109` with p-value `0.186`, and TMG same-day `-0.102` with p-value `0.120`.

**Paper interpretation.** The correlation analysis supports the thesis's conservative conclusion. Sentiment has weak descriptive associations with returns, but correlation alone does not justify a strong causal or standalone predictive claim.

**Relevant literature.** Tetlock's media sentiment study is the main foundational reference for testing whether media tone relates to market outcomes. Liapis and Kotsiantis provide a recent benchmark-style example of sentiment/emotion features with correlation-aware feature selection in forecasting.

## Experiment 3: Leakage Audit

**Question.** Are the final sentiment forecasting results contaminated by look-ahead leakage?

**Script.** `scripts/run_phase7_validity_checks.py`

**Outputs.**

- `outputs/final_sentiment_experiments/leakage_check_report.md`
- `outputs/final_sentiment_experiments/leakage_check_report.json`
- `outputs/final_sentiment_experiments/coverage_analysis.md`
- `outputs/final_sentiment_experiments/overfitting_report.md`

**Design.** The audit checks target construction, feature alignment, rolling features, scaling, validation-only hyperparameter selection, validation-only threshold tuning, test-label exclusion, split embargoes, and walk-forward history limits.

**Headline result.** Leakage status is `PASS`. The audit ran 45 panel checks and checked 43,880 validation/test prediction rows. Direction history violations: `0`. Return history violations: `0`. No leakage violations were detected.

**Additional validity result.** Overfitting analysis flagged 15 of 45 rows. The high-accuracy OIH 5-day event-weighted candidate is therefore exploratory, not accepted. The accepted final candidate remains OIH 5-day all-news bilingual with test accuracy `0.532`, balanced accuracy `0.503`, F1 `0.630`, and sign accuracy `0.585`.

**Coverage note.** Live page-count checks indicate that one additional Mubasher Arabic listing page may be available for CIB. TMG and OIH appear locally complete by live page count. OIH remains the sparsest article history.

**Paper interpretation.** The final sentiment-only result is weak but leakage-safe. The audit should be reported to explain why the thesis accepts a modest result instead of selecting the highest raw-accuracy row.

## Experiment 4: AraBERT Financial Adaptation

**Question.** Can AraBERT be fine-tuned on Arabic financial terminology to move toward an Arabic analogue of FinBERT?

**Script.** `scripts/run_arabert_financial_adaptation_experiment.py`

**Outputs.**

- `outputs/arabert_financial_adaptation/arabert_financial_adaptation_report.md`
- `outputs/arabert_financial_adaptation/arabic_financial_term_coverage.csv`
- `outputs/arabert_financial_adaptation/arabic_financial_adaptation_articles.csv`
- `outputs/arabert_financial_adaptation/corpus/arabic_financial_mlm_train.txt`
- `outputs/arabert_financial_adaptation/corpus/arabic_financial_mlm_validation.txt`
- `outputs/arabert_financial_adaptation/arabert_financial_mlm/`

**Design.** The valid FinBERT analogue has two stages: domain-adaptive masked-language-model pre-training on financial text, then supervised sentiment fine-tuning on human-labeled Arabic financial sentiment examples. The thesis dataset supports the first stage but not the second, because its sentiment labels are outputs from the current AraBERT route rather than independent human labels.

**Headline result.** The Arabic corpus contains 1,327 Arabic financial articles, split chronologically into 1,061 MLM training articles and 266 validation articles. A short CPU-based AraBERT MLM adaptation run completed for 20 steps. Train loss was `0.9676`; validation loss was `0.8218`. A follow-up fixed-mask evaluation compared the adapted checkpoint against base AraBERT on the same corpus. On the held-out validation split, MLM loss decreased from `0.9624` to `0.8468`, and perplexity decreased from `2.618` to `2.332`, a relative loss reduction of `12.01%`. Frequent financial terms include `استثمار` in 59.2% of Arabic articles, `ارباح` in 58.5%, and `البورصه` in 54.3%.

**Supervised fine-tuning status.** Not run. The current dataset does not provide independent human sentiment labels. Training on the current model's own predictions would be pseudo-label distillation, not valid evidence of improved Arabic financial sentiment classification.

**Paper interpretation.** Report this as a feasibility/domain-adaptation protocol with an MLM improvement over base AraBERT, not as a completed Arabic FinBERT equivalent. A proper future-work experiment should collect or adopt human-labeled Arabic financial sentiment data and evaluate against the current AraBERT route.

**Relevant literature.** FinBERT motivates domain-adaptive financial language modeling for sentiment classification. AraBERT, ARBERT/MARBERT, and recent Arabic financial NLP resources such as SAHM and Arabic Saudi-market financial sentiment work support the need for Arabic finance-specific evaluation data.

## Recommended References For These Experiments

- Tetlock, P. C. (2007). Giving Content to Investor Sentiment: The Role of Media in the Stock Market. Journal of Finance. https://ideas.repec.org/a/bla/jfinan/v62y2007i3p1139-1168.html
- Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-trained Language Models. https://arxiv.org/abs/1908.10063
- Antoun, W., Baly, F., and Hajj, H. (2020). AraBERT: Transformer-based Model for Arabic Language Understanding. https://arxiv.org/abs/2003.00104
- Abdul-Mageed, M., Elmadany, A., and Nagoudi, E. M. B. (2021). ARBERT and MARBERT: Deep Bidirectional Transformers for Arabic. https://arxiv.org/abs/2101.01785
- Elbadry et al. (2026). SAHM: A Benchmark for Arabic Financial and Shari'ah-Compliant Reasoning. https://arxiv.org/abs/2604.19098
- Albaqawi et al. (2026). LLM-Based Financial Sentiment Analysis in Arabic: Evidence from Saudi Markets. https://arxiv.org/abs/2605.19714

## Experiment 5: Social Media Source Feasibility

**Question.** Can social media be added to the thesis ethically, for free, and through officially approved access?

**Script.** `scripts/run_social_media_source_feasibility.py`

**Outputs.**

- `outputs/social_media_feasibility/social_media_feasibility_report.md`
- `outputs/social_media_feasibility/run_metadata.json`

**Platform findings.**

- Reddit has an official route: Data API access for approved non-commercial use, plus a research access path. Reddit explicitly allows academic research use only under restrictions: no redistribution of Reddit data or derivative products, and no model training on Reddit data without explicit consent.
- X/Twitter is not a free broad-read option. Current official X API documentation describes pay-per-use pricing for public reads, while cheaper owned reads cover only the authenticated user's own account data.
- Unofficial scraping or API bypasses should not be used in the thesis.

**EGX-specific feasibility.** Reddit is ethically possible only after API approval, but likely sparse for EGX firm-level discussion. X is the stronger platform for Egyptian market chatter, but it is not free for broad public search. Therefore, a full social-media comparison should remain future work unless institutional X credits or approved Reddit research access become available.

**Proposed future design.** Collect posts officially, map Arabic and English aliases for CIB, TMG, OIH, and EGX30, score sentiment with the same bilingual pipeline, then compare news-only, social-only, news+social, and ensemble models under the existing chronological leakage-safe framework.

**Relevant literature.**

- AlKhatib et al. (2016) specifically studied Arabic tweets and Arab stock indices, including EGX30, and reported Granger-causality and correlation evidence between Arabic tweet volume/sentiment and market changes.
- Bollen et al. (2011), Nguyen et al. (2015), and Xu and Cohen (2018) remain relevant broader social-media forecasting references already used in the thesis.
