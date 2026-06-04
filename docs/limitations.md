# Limitations

- Mubasher is a single news provider.
- Off-the-shelf sentiment classifiers are comparative baselines, not EGX-specific human-validated sentiment models.
- Corporate-action transition states are documented and validated, but archival exchange metadata remains a research maintenance burden.
- The pipeline deliberately excludes price-derived predictors, so results quantify sentiment-only value rather than trading-system performance.
- News coverage is still uneven across firms and sectors, even after point-in-time scoping.
- Raw accuracy can be inflated by majority-class behavior; balanced accuracy and baseline deltas remain the primary evidence.
- Historical EGX30 constituent weights are not available in the reproducible dataset, so index-weighted sentiment is not reported.
- Abstention/high-signal results depend on coverage and threshold design and should not be compared to full-session accuracy without the reported coverage.
