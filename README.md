# Market Mood in Two Languages

This repository implements a leakage-safe financial NLP benchmark for predicting the direction of the Egyptian Exchange `EGX30` index from Arabic and English news sentiment only.

The production study uses a survivorship-bias-free point-in-time registry for `2021-04-21` through `2026-04-20`. It reconstructs 12 EGX30 membership intervals and a 57-company historical union. EGX30 closing prices are restricted to target construction; no price-derived feature enters the model matrix.

## Pipeline

1. Build and validate the point-in-time historical EGX30 company registry.
2. Resumably scrape bilingual Mubasher news for the full historical company union.
3. Score English articles with FinBERT and compare Arabic scoring routes using AraBERT and CAMeLBERT.
4. Deduplicate stories across company pages and compute equal-weight active-constituent market mood features.
5. Predict 1, 3, and 5 trading-day EGX30 direction under embargoed expanding-window walk-forward validation.
6. Run independent target, lag, membership, walk-forward-history, and metric-parity audits.
7. Generate easy-win robustness reports: corpus coverage, stronger naive baselines, class-imbalance diagnostics, sector aggregation, weighting availability, and abstention/high-signal tests.

## Production Results

| Sentiment source | Horizon | Accuracy | Balanced accuracy | F1 |
| --- | --- | --- | --- | --- |
| finbert_arabert | 1 | 0.522 | 0.512 | 0.591 |
| finbert_arabert | 3 | 0.404 | 0.416 | 0.447 |
| finbert_arabert | 5 | 0.639 | 0.536 | 0.754 |
| finbert_camelbert | 1 | 0.621 | 0.545 | 0.737 |
| finbert_camelbert | 3 | 0.522 | 0.495 | 0.614 |
| finbert_camelbert | 5 | 0.381 | 0.489 | 0.378 |

The strongest validity-gated bilingual row by balanced accuracy is `finbert_camelbert` at horizon `1`, with accuracy `0.621`, balanced accuracy `0.545`, and F1 `0.737`. These results should be described as promising but limited evidence of weak directional signal. Balanced accuracy is prioritized over raw accuracy because raw accuracy is partially explained by class imbalance and majority-class behavior.

## Robustness and Easy-Win Validity Extensions

The additive easy-win reports live under `reports/`:

- `reports/egx30_coverage_audit.md` checks company, yearly, and daily news coverage.
- `reports/egx30_baseline_comparison.md` compares sentiment models with always-Up, always-Down, training-majority, rolling-majority, article-count-only, and shuffled-sentiment baselines.
- `reports/egx30_abstention_experiment.md` reports high-signal predictions with coverage and abstained sample counts.
- `reports/egx30_sector_feature_ablation.md` summarizes compact sector-aware aggregation from a static reviewable map.
- `reports/egx30_weighting_ablation.md` reports equal-weighted/article-count-weighted sentiment and marks true historical index weights unavailable.

## Reproduce

```bash
PYTHONPATH=src python3 scripts/run_egx30_5yr_pipeline.py --skip-scrape
```

Remove `--skip-scrape` to resume or refresh the full archive scrape.

## Artifact Storage

Raw scrape checkpoints, scored article tables, and large panel/prediction materializations are reproducible local artifacts and are excluded from Git. Commit the compact registries, metrics, validity report, summaries, source code, and thesis files. Publish large research artifacts through versioned external storage when they need to accompany a thesis or research release.

## Limitations

The benchmark is intentionally sentiment-only. It tests whether bilingual constituent news contains weak directional information before any future integration with price-derived predictors. The result is not a deployable trading strategy, and true historical EGX30 constituent weights are unavailable in the reproducible dataset.

## Important Artifacts

- `data/egx30_historical_constituents_5yr.csv`
- `data/egx30_rebalance_events_5yr.csv`
- `data/processed_news_5yr.csv`
- `data/processed_news_5yr_camelbert.csv`
- `data/egx30_5yr_universe_metrics.csv`
- `data/egx30_5yr_validity_report.json`
- `outputs/egx30_5yr_universe_experiments/research_summary.md`
- `reports/egx30_coverage_audit.md`
- `reports/egx30_baseline_comparison.md`
- `reports/egx30_abstention_experiment.md`

## Tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The earlier CIB/TMG/OIH experiments remain useful as a three-company ablation. They are not the production EGX30 benchmark.
