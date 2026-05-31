# Overfitting Report

Rows checked: 45
Rows flagged for train/test overfitting: 14

A row is flagged when train-test accuracy or balanced-accuracy gap exceeds 0.10, train-test F1 gap exceeds 0.20, or test RMSE is more than 1.5x train RMSE.

## Previous Headline Accuracy Candidate

OIH 5-day event-weighted bilingual test accuracy is `0.563` with balanced accuracy `0.502`.
Train accuracy is `0.625` and validation accuracy is `0.505`.
Overall overfit flag for this row: `True`. This row should not be treated as the accepted best model.

## Accepted Validity-Passing Candidate

The strongest candidate that passes the explicit validity filter is `CIB` horizon `5` `event_only` `bilingual`.
Validation accuracy/balanced accuracy: `0.546` / `0.506`.
Test accuracy/balanced accuracy/F1/sign accuracy: `0.566` / `0.517` / `0.702` / `0.551`.
Overall overfit flag for this row: `False`.

## Highest Train-Test Accuracy Gaps

| Horizon | Strategy | Ticker | View | Train Acc. | Val Acc. | Test Acc. | Train-Test Gap | Flag |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 5 | event_only | OIH | bilingual | 0.590 | 0.528 | 0.423 | 0.167 | True |
| 1 | event_weighted | OIH | bilingual | 0.619 | 0.453 | 0.453 | 0.166 | True |
| 5 | all_news | OIH | bilingual | 0.585 | 0.524 | 0.425 | 0.160 | True |
| 1 | event_only | OIH | bilingual | 0.600 | 0.449 | 0.443 | 0.157 | True |
| 1 | all_news | OIH | bilingual | 0.588 | 0.457 | 0.449 | 0.140 | True |
| 1 | event_weighted | CIB | bilingual | 0.619 | 0.533 | 0.485 | 0.135 | True |
| 3 | all_news | OIH | english_only | 0.581 | 0.524 | 0.461 | 0.120 | True |
| 3 | all_news | OIH | bilingual | 0.586 | 0.524 | 0.467 | 0.119 | True |
| 5 | event_weighted | TMG | bilingual | 0.611 | 0.544 | 0.493 | 0.118 | True |
| 1 | all_news | CIB | bilingual | 0.589 | 0.518 | 0.487 | 0.102 | True |

## Interpretation

Several higher-dimensional feature sets show large in-sample training performance that does not carry to validation/test. Final claims should therefore be based on validation/test metrics only. The OIH 5-day event-weighted accuracy improvement is measured on test data but is flagged for overfitting, so it is exploratory. The accepted candidate is still weak because its balanced accuracy remains close to random.
