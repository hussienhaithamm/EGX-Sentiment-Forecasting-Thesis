# Market Mood in Two Languages

A reproducible research repository for testing whether Arabic and English financial news sentiment contains short-horizon directional information for the Egyptian Exchange 30-stock benchmark index (`EGX30`).

The project builds a point-in-time EGX30 universe, collects bilingual Mubasher company news, scores article sentiment with language-specific transformer routes, and evaluates market-direction forecasts under embargoed walk-forward validation. The main benchmark is intentionally sentiment-only: EGX30 prices are used to build target labels, not model features.

## What This Repository Provides

- A five-year point-in-time EGX30 membership registry from `2021-04-21` to `2026-04-20`.
- A 57-company historical constituent union with 12 validated membership intervals.
- Arabic and English news preprocessing, deduplication, sentiment scoring, and daily market-mood aggregation.
- Sentiment-only direction models for 1-, 3-, and 5-trading-day horizons.
- Baseline, leakage, lag-alignment, membership, metric-parity, and robustness checks.
- A supplementary integration experiment that adds lagged sentiment to an EGX30 historical XGBoost model.
- A compiled thesis PDF under [`Bachelor_Thesis_Template/bachelor.pdf`](Bachelor_Thesis_Template/bachelor.pdf).

## Headline Result

The strongest validity-gated sentiment-only row is the bilingual `finbert_camelbert` route at the 1-trading-day horizon.

| Sentiment route | Horizon | Accuracy | Balanced accuracy | F1 | MCC |
| --- | ---: | ---: | ---: | ---: | ---: |
| `finbert_arabert` | 1 | 0.522 | 0.512 | 0.591 | 0.033 |
| `finbert_arabert` | 3 | 0.404 | 0.416 | 0.447 | -0.151 |
| `finbert_arabert` | 5 | 0.639 | 0.536 | 0.754 | 0.095 |
| `finbert_camelbert` | 1 | 0.621 | 0.545 | 0.737 | 0.115 |
| `finbert_camelbert` | 3 | 0.522 | 0.495 | 0.614 | -0.012 |
| `finbert_camelbert` | 5 | 0.381 | 0.489 | 0.378 | -0.023 |

The result is best interpreted as weak but measurable directional evidence. Balanced accuracy is the primary metric because raw accuracy is partly affected by upward class imbalance and majority-class behavior.

The supplementary historical-model experiment shows a stronger applied result. Adding lagged bilingual sentiment features to the EGX30 historical XGBoost model improves test accuracy from `0.739` to `0.802`, balanced accuracy from `0.757` to `0.793`, and F1 from `0.789` to `0.851`.

## Repository Layout

```text
Bachelor_Thesis_Template/   LaTeX thesis source and compiled PDF
configs/                    Ticker and company configuration
data/                       Compact registries, metrics, and prepared research data
docs/                       Methodology, experiment logs, audits, and thesis support notes
outputs/                    Generated experiment outputs
reports/                    Compact robustness and validity reports
scripts/                    Reproducible pipeline, evaluation, audit, and thesis sync scripts
src/                        Shared data-loading and pipeline utilities
tests/                      Unit tests for pipeline and validation behavior
```

Large scrape checkpoints and full intermediate materializations are treated as reproducible local artifacts. The repository keeps compact data, metrics, reports, source code, and thesis files under version control.

## Reproduce the Main Pipeline

Create an isolated environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the full reproducible pipeline using the local scrape cache:

```bash
PYTHONPATH=src python3 scripts/run_egx30_5yr_pipeline.py --skip-scrape
```

Remove `--skip-scrape` only when the archive scrape needs to resume or refresh.

## Validation

Run the test suite:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Rebuild the thesis PDF:

```bash
python3 scripts/compile_thesis_clean.py
```

Core validation artifacts:

- [`data/egx30_5yr_validity_report.json`](data/egx30_5yr_validity_report.json)
- [`data/egx30_5yr_universe_metrics.csv`](data/egx30_5yr_universe_metrics.csv)
- [`reports/egx30_baseline_comparison.md`](reports/egx30_baseline_comparison.md)
- [`reports/egx30_coverage_audit.md`](reports/egx30_coverage_audit.md)
- [`reports/egx30_abstention_experiment.md`](reports/egx30_abstention_experiment.md)
- [`reports/egx30_sector_feature_ablation.md`](reports/egx30_sector_feature_ablation.md)
- [`reports/egx30_weighting_ablation.md`](reports/egx30_weighting_ablation.md)

## Method Summary

1. Reconstruct the active EGX30 universe for each date.
2. Collect Arabic and English Mubasher articles for the full historical constituent union.
3. Normalize timestamps, clean text, deduplicate repeated stories, and enforce active-membership filtering.
4. Score English articles with FinBERT and Arabic articles with AraBERT or CAMeLBERT.
5. Aggregate company-day sentiment into equal-weight daily EGX30 market mood.
6. Shift sentiment features before trading-date alignment to prevent same-day and future leakage.
7. Evaluate 1-, 3-, and 5-trading-day direction labels with expanding-window walk-forward validation.
8. Compare against always-class, majority, article-count-only, shuffled-sentiment, and historical-model baselines.

## Supplementary Historical Model

The historical-model integration uses the public [`anastamerr/EGX-Direction-Forecasting`](https://github.com/anastamerr/EGX-Direction-Forecasting) implementation as the reference design and evaluates its XGBoost classifier with and without lagged sentiment features.

This experiment is separate from the sentiment-only benchmark. It tests whether the same market-mood signal improves an existing price-history model, while the primary thesis question tests whether sentiment alone carries directional information.

## Scope and Limitations

This repository is a research benchmark, not a trading system. The main model excludes price-derived predictors by design, the news source is limited to Mubasher, and the Arabic sentiment route relies on general Arabic encoders rather than expert-labeled Arabic financial sentiment data. Historical EGX30 constituent weights are not available in the reproducible dataset, so market mood is aggregated with equal active-constituent weighting.

The three-company CIB/TMG/OIH experiments remain useful as ablations, but the production benchmark is the full point-in-time EGX30 study.
