# EGX30 Baseline Comparison

## Interpretation

Raw accuracy is not enough for this task because EGX30 direction labels can be imbalanced. Balanced accuracy, F1, MCC, predicted Up share, and deltas versus the training-majority baseline are reported alongside accuracy.

The sentiment model beats the training-majority baseline on balanced accuracy in `8` of `18` test rows. Rows that do not beat this baseline should be treated as weak or non-evidential even when raw accuracy is high.

## Best Sentiment-Model Test Rows

| horizon | sentiment_source | view | accuracy | balanced_accuracy | f1 | mcc | predicted_up_share | actual_up_share | delta_balanced_accuracy_vs_training_majority |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | finbert_camelbert | arabic_only | 0.611 | 0.580 | 0.693 | 0.163 | 0.645 | 0.621 | 0.080 |
| 1 | finbert_arabert | arabic_only | 0.601 | 0.565 | 0.690 | 0.134 | 0.665 | 0.621 | 0.065 |
| 1 | finbert_camelbert | bilingual | 0.621 | 0.545 | 0.737 | 0.115 | 0.823 | 0.621 | 0.045 |
| 3 | finbert_arabert | arabic_only | 0.660 | 0.536 | 0.785 | 0.143 | 0.936 | 0.645 | 0.036 |
| 5 | finbert_arabert | bilingual | 0.639 | 0.536 | 0.754 | 0.072 | 0.738 | 0.733 | 0.036 |
| 3 | finbert_camelbert | arabic_only | 0.478 | 0.533 | 0.459 | 0.067 | 0.320 | 0.645 | 0.033 |
| 5 | finbert_camelbert | arabic_only | 0.332 | 0.526 | 0.192 | 0.080 | 0.094 | 0.733 | 0.026 |
| 1 | finbert_arabert | bilingual | 0.522 | 0.512 | 0.591 | 0.023 | 0.547 | 0.621 | 0.012 |
| 3 | finbert_camelbert | bilingual | 0.522 | 0.495 | 0.614 | -0.009 | 0.591 | 0.645 | -0.005 |
| 3 | finbert_arabert | english_only | 0.517 | 0.495 | 0.605 | -0.010 | 0.576 | 0.645 | -0.005 |

## Class-Imbalance Warnings

| horizon | sentiment_source | view | accuracy | balanced_accuracy | f1 | predicted_up_share | actual_up_share | warning |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | finbert_arabert | arabic_only | 0.660 | 0.536 | 0.785 | 0.936 | 0.645 | extreme predicted Up share |
| 5 | finbert_camelbert | arabic_only | 0.332 | 0.526 | 0.192 | 0.094 | 0.733 | extreme predicted Up share |

## Baseline Notes

- `always_up` and `always_down` expose label skew.
- `training_majority` uses only the initial training split.
- `rolling_majority` uses only prior labels available before each predicted row.
- `article_count_only_logit` tests whether lagged news volume and coverage explain performance without sentiment polarity.
- `sentiment_shuffled_logit` is deterministic and breaks sentiment timing while preserving the model path.
