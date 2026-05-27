# Historical Model Sentiment Feature Experiment

This experiment re-runs the historical EGX direction model from `anastamerr/EGX-Direction-Forecasting` and compares a historical-only feature set against the same feature set augmented with one-day-lagged bilingual sentiment features from the thesis pipeline.

The target and chronological train/test split are shared between both variants for each ticker/model pair.

## Test Metrics

| Ticker | Model | Variant | Features | Accuracy | Balanced Acc. | F1 | AUC | Test period |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| CIB | logistic | historical_only | 19 | 0.755 | 0.762 | 0.777 | 0.826 | 2024-04-28 to 2026-04-20 |
| CIB | logistic | historical_sentiment | 68 | 0.753 | 0.758 | 0.777 | 0.817 | 2024-04-28 to 2026-04-20 |
| OIH | logistic | historical_only | 19 | 0.557 | 0.636 | 0.466 | 0.793 | 2024-04-29 to 2026-04-20 |
| OIH | logistic | historical_sentiment | 68 | 0.563 | 0.643 | 0.472 | 0.784 | 2024-04-29 to 2026-04-20 |
| TMG | logistic | historical_only | 19 | 0.775 | 0.781 | 0.767 | 0.844 | 2024-04-29 to 2026-04-20 |
| TMG | logistic | historical_sentiment | 68 | 0.748 | 0.757 | 0.724 | 0.830 | 2024-04-29 to 2026-04-20 |
| CIB | mlp | historical_only | 19 | 0.763 | 0.741 | 0.811 | 0.810 | 2024-04-28 to 2026-04-20 |
| CIB | mlp | historical_sentiment | 68 | 0.658 | 0.583 | 0.774 | 0.731 | 2024-04-28 to 2026-04-20 |
| OIH | mlp | historical_only | 19 | 0.468 | 0.573 | 0.254 | 0.690 | 2024-04-29 to 2026-04-20 |
| OIH | mlp | historical_sentiment | 68 | 0.611 | 0.679 | 0.563 | 0.804 | 2024-04-29 to 2026-04-20 |
| TMG | mlp | historical_only | 19 | 0.515 | 0.544 | 0.195 | 0.757 | 2024-04-29 to 2026-04-20 |
| TMG | mlp | historical_sentiment | 68 | 0.639 | 0.653 | 0.563 | 0.756 | 2024-04-29 to 2026-04-20 |
| CIB | xgboost | historical_only | 19 | 0.681 | 0.708 | 0.678 | 0.818 | 2024-04-28 to 2026-04-20 |
| CIB | xgboost | historical_sentiment | 68 | 0.694 | 0.718 | 0.696 | 0.807 | 2024-04-28 to 2026-04-20 |
| OIH | xgboost | historical_only | 19 | 0.725 | 0.752 | 0.744 | 0.829 | 2024-04-29 to 2026-04-20 |
| OIH | xgboost | historical_sentiment | 68 | 0.746 | 0.764 | 0.771 | 0.820 | 2024-04-29 to 2026-04-20 |
| TMG | xgboost | historical_only | 19 | 0.758 | 0.761 | 0.763 | 0.819 | 2024-04-29 to 2026-04-20 |
| TMG | xgboost | historical_sentiment | 68 | 0.752 | 0.754 | 0.756 | 0.820 | 2024-04-29 to 2026-04-20 |

## Sentiment Delta

| Ticker | Model | Delta Accuracy | Delta Balanced Acc. | Delta F1 | Delta AUC |
| --- | --- | ---: | ---: | ---: | ---: |
| CIB | logistic | -0.002 | -0.004 | +0.000 | -0.009 |
| OIH | logistic | +0.006 | +0.007 | +0.006 | -0.009 |
| TMG | logistic | -0.027 | -0.024 | -0.043 | -0.014 |
| CIB | mlp | -0.105 | -0.158 | -0.037 | -0.079 |
| OIH | mlp | +0.143 | +0.106 | +0.309 | +0.115 |
| TMG | mlp | +0.124 | +0.109 | +0.368 | -0.000 |
| CIB | xgboost | +0.013 | +0.010 | +0.018 | -0.011 |
| OIH | xgboost | +0.021 | +0.013 | +0.028 | -0.010 |
| TMG | xgboost | -0.006 | -0.006 | -0.007 | +0.001 |

## Aggregate Mean Delta

| Model | Accuracy | Balanced Acc. | F1 | AUC |
| --- | ---: | ---: | ---: | ---: |
| logistic | -0.008 | -0.007 | -0.012 | -0.011 |
| mlp | +0.054 | +0.019 | +0.213 | +0.012 |
| xgboost | +0.009 | +0.005 | +0.013 | -0.006 |

## Interpretation

Sentiment improved raw test accuracy in 5 of 9 ticker/model comparisons. The largest accuracy gain was `+0.143` for `OIH` using `mlp`. The largest accuracy loss was `-0.105` for `CIB` using `mlp`.

The result should be described as conditional rather than uniformly positive. The sentiment signal adds small gains for several model/ticker pairs, but it does not consistently improve every ticker or every metric. Mean AUC declines overall.
