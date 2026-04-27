# EGX Sentiment Stock Forecasting Thesis

This repository contains the clean reproducible project for the bachelor thesis **"Stock Market Forecasting with Sentiment Analysis"**.

The study tests whether sentiment extracted from Mubasher financial news can predict Egyptian stock market direction and returns for:

- CIB / COMI.CA / COMI
- TMG / TMGH.CA / TMGH
- OIH / OIH.CA / OIH

The main benchmark is sentiment-only: no OHLCV or technical indicators are used as model features.

## Repository Structure

```text
src/
  egx_direction/          Sentiment-only forecasting, targets, event features
  news_pipeline/          Mubasher scraper, cleaning helpers, FinBERT/AraBERT routing
scripts/
  run_sentiment_only_pipeline.py
  run_sentiment_only_evaluation_from_dataset.py
  run_multi_horizon_event_evaluation.py
data/
  raw/news/mubasher_sentiment_only/
  processed/news/sentiment_ready/
outputs/final_sentiment_experiments/
tests/
docs/
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

If you do not install the package, run scripts with:

```bash
PYTHONPATH=src python3 ...
```

## Reproduce The Pipeline

Run the full scrape, cleaning, sentiment labeling, and 5-day evaluation:

```bash
PYTHONPATH=src python3 scripts/run_sentiment_only_pipeline.py \
  --tickers CIB TMG OIH \
  --news-data-dir data/raw/news/mubasher_sentiment_only \
  --processed-news-dir data/processed/news/sentiment_ready \
  --output-dir outputs/sentiment_only_pipeline \
  --max-pages 100 \
  --horizon 5
```

Run the no-scrape enhanced benchmark from the included labeled dataset:

```bash
PYTHONPATH=src python3 scripts/run_sentiment_only_evaluation_from_dataset.py \
  --feature-set enhanced \
  --sentiment-labeled-path data/processed/news/sentiment_ready/sentiment_labeled_articles.csv \
  --output-dir outputs/sentiment_only_enhanced_features
```

Run the final multi-horizon and event-aware experiments:

```bash
PYTHONPATH=src python3 scripts/run_multi_horizon_event_evaluation.py \
  --refit-every-rows 250 \
  --classification-c-grid 0.05 0.1 0.5 \
  --regression-alpha-grid 10.0
```

Final outputs are saved in `outputs/final_sentiment_experiments/`.

## Key Final Outputs

- `horizon_comparison.csv`
- `direction_metrics_by_horizon.csv`
- `return_metrics_by_horizon.csv`
- `predictions_by_horizon.csv`
- `summary_by_horizon.json`
- `event_labeled_articles.csv`
- `event_feature_panel.csv`
- `event_model_metrics.csv`
- `event_vs_all_news_comparison.csv`

## Tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

The current test suite covers sentiment article preparation, lagged feature alignment, split embargo behavior, event keyword labeling, and event feature panel lagging.

## Documentation

- `docs/methodology.md`
- `docs/data_pipeline.md`
- `docs/evaluation.md`
- `docs/results.md`
- `docs/limitations.md`
