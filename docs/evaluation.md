# Evaluation

The evaluation is chronological and leakage-safe.

## Splits

Each ticker panel is sorted by trading date and split into train, validation, and test windows. An embargo equal to at least the forecast horizon is inserted between train/validation/test regions.

## Walk-Forward Logic

For each evaluation row, the model trains only on rows whose target horizon would already be known by that date. The final experiment uses `--refit-every-rows 250` for tractability while keeping an expanding walk-forward training window.

## Metrics

Classification:

- accuracy
- balanced accuracy
- precision
- recall
- F1
- confusion matrix (`tn`, `fp`, `fn`, `tp`)

Regression:

- MAE
- RMSE
- return sign accuracy
- return correlation

## Final Experiment Command

```bash
PYTHONPATH=src python3 scripts/run_multi_horizon_event_evaluation.py \
  --refit-every-rows 250 \
  --classification-c-grid 0.05 0.1 0.5 \
  --regression-alpha-grid 10.0
```
