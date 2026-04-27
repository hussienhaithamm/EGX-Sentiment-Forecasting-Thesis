# Methodology

This thesis evaluates whether financial-news sentiment alone can forecast EGX stock movements.

The pipeline:

1. Scrapes Mubasher Arabic and English news for CIB, TMG, and OIH.
2. Cleans article title/body text, normalizes ticker aliases, removes duplicates, and assigns local Cairo dates.
3. Routes English articles to FinBERT and Arabic articles to AraBERT.
4. Converts article sentiment into signed sentiment scores and probabilities.
5. Aggregates article sentiment to ticker-date features.
6. Aligns sentiment features to later trading dates using a previous-calendar-day lag.
7. Builds future return and direction targets for 1, 3, and 5 trading-day horizons.
8. Evaluates with chronological validation and test windows using walk-forward refits.

The main benchmark is intentionally sentiment-only. Price history is used only to construct the future target, not as a predictive feature.

## Models

Classification uses logistic regression with balanced class weights. Return regression uses ridge regression. Classification thresholds are tuned only on validation predictions and then applied to test predictions.

## Event Features

Event-aware features are built from English and Arabic keywords for earnings, dividends, M&A, contracts/projects, expansion/investment, financing, and governance/disclosure. The experiments compare:

- all-news sentiment features
- event-only sentiment features
- event-weighted sentiment features

No future article text or future price movement is used in feature construction.
