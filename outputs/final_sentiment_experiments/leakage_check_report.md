# Leakage Check Report

Overall status: **PASS**

## Automated Checks

- Panel checks run: 45
- Walk-forward rows checked: 43880
- Direction history violations: 0
- Return history violations: 0

## Static Code Checks

- `target_construction`: build_sentiment_targets uses Close.shift(-horizon) and drops rows without future return.
- `feature_alignment`: all-news and event features are calendar reindexed then shifted by at least one day before merging to trading dates.
- `rolling_features`: rolling statistics are computed on shifted daily feature series, so the latest source date is t-1.
- `scaling`: StandardScaler is inside each sklearn Pipeline and is fit only inside model.fit() on the current training frame.
- `hyperparameter_selection`: C/alpha are selected from validation walk-forward predictions before final test predictions are generated.
- `threshold_tuning`: classification threshold is selected only from validation predictions by choose_validation_threshold().
- `test_exclusion`: No test labels are passed into hyperparameter or threshold selection functions.

## Violations

No leakage violations were detected by the automated checks.
