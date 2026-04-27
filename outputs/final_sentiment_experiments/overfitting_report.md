# Overfitting Report

Rows checked: 45
Rows flagged for train/test overfitting: 15

A row is flagged when train-test accuracy or balanced-accuracy gap exceeds 0.10, train-test F1 gap exceeds 0.20, or test RMSE is more than 1.5x train RMSE.

## Previous Headline Accuracy Candidate

OIH 5-day event-weighted bilingual test accuracy is `0.571` with balanced accuracy `0.511`.
Train accuracy is `0.609` and validation accuracy is `0.485`.
Overall overfit flag for this row: `True`. This row should not be treated as the accepted best model.

## Accepted Validity-Passing Candidate

The strongest candidate that passes the explicit validity filter is `OIH` horizon `5` `all_news` `bilingual`.
Validation accuracy/balanced accuracy: `0.522` / `0.525`.
Test accuracy/balanced accuracy/F1/sign accuracy: `0.532` / `0.503` / `0.630` / `0.585`.
Overall overfit flag for this row: `False`.

## Highest Train-Test Accuracy Gaps

| Horizon | Strategy | Ticker | View | Train Acc. | Val Acc. | Test Acc. | Train-Test Gap | Flag |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | event_only | OIH | bilingual | 0.605 | 0.457 | 0.432 | 0.172 | True |
| 1 | event_weighted | OIH | bilingual | 0.609 | 0.459 | 0.441 | 0.168 | True |
| 3 | event_weighted | OIH | bilingual | 0.625 | 0.509 | 0.461 | 0.164 | True |
| 3 | event_only | OIH | bilingual | 0.630 | 0.497 | 0.473 | 0.157 | True |
| 5 | event_weighted | CIB | bilingual | 0.592 | 0.476 | 0.459 | 0.133 | True |
| 1 | all_news | OIH | bilingual | 0.591 | 0.471 | 0.461 | 0.130 | True |
| 3 | all_news | OIH | bilingual | 0.600 | 0.530 | 0.471 | 0.129 | True |
| 3 | all_news | OIH | english_only | 0.585 | 0.524 | 0.461 | 0.124 | True |
| 5 | event_weighted | TMG | bilingual | 0.611 | 0.542 | 0.495 | 0.116 | True |
| 5 | event_only | TMG | bilingual | 0.615 | 0.542 | 0.499 | 0.116 | True |

## Interpretation

Several higher-dimensional feature sets show large in-sample training performance that does not carry to validation/test. Final claims should therefore be based on validation/test metrics only. The OIH 5-day event-weighted accuracy improvement is measured on test data but is flagged for overfitting, so it is exploratory. The accepted candidate is still weak because its balanced accuracy remains close to random.
