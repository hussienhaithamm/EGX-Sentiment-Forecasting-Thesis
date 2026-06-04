# Historical Model Sentiment Feature Experiment

This experiment re-runs the historical EGX direction model from `anastamerr/EGX-Direction-Forecasting` and compares a historical-only feature set against the same feature set augmented with one-day-lagged bilingual sentiment features from the thesis pipeline.

The target and chronological train/test split are shared between both variants for each ticker/model pair.

Rows whose future-window target cannot be observed from the available close series are dropped before splitting. In this run, `5` terminal rows were removed, leaving `2026-04-08` as the latest valid target date.

## Test Metrics

| Ticker | Model | Variant | Features | Accuracy | Balanced Acc. | F1 | AUC | Test period |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| EGX30 | xgboost | historical_only | 18 | 0.739 | 0.757 | 0.789 | 0.848 | 2024-05-16 to 2026-04-08 |
| EGX30 | xgboost | historical_sentiment | 67 | 0.802 | 0.793 | 0.851 | 0.839 | 2024-05-16 to 2026-04-08 |

## Sentiment Delta

| Ticker | Model | Delta Accuracy | Delta Balanced Acc. | Delta F1 | Delta AUC |
| --- | --- | ---: | ---: | ---: | ---: |
| EGX30 | xgboost | +0.064 | +0.036 | +0.062 | -0.008 |

## Aggregate Mean Delta

| Model | Accuracy | Balanced Acc. | F1 | AUC |
| --- | ---: | ---: | ---: | ---: |
| xgboost | +0.064 | +0.036 | +0.062 | -0.008 |

## Interpretation

Sentiment changed raw test accuracy by `+0.064` for `EGX30` using `xgboost`.

For this scoped XGBoost integration, the sentiment-augmented model improves thresholded classification accuracy, balanced accuracy, and F1. Mean AUC declines overall.
