# EGX30 Abstention Experiment

## Interpretation

This experiment asks whether sentiment-only predictions are more useful on high-signal days. Thresholds are selected from validation rows only, then applied to the test rows. Accuracy is always reported with coverage because high accuracy on limited coverage is not sufficient evidence.

## Best Covered Test Rows

| horizon | sentiment_source | view | target_coverage | coverage_pct | covered_test_samples | accuracy | balanced_accuracy | f1 | mcc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | finbert_arabert | bilingual | 0.200 | 0.356 | 72 | 0.681 | 0.580 | 0.785 | 0.164 |
| 1 | finbert_camelbert | arabic_only | 0.300 | 0.340 | 69 | 0.580 | 0.571 | 0.651 | 0.136 |
| 1 | finbert_camelbert | arabic_only | 0.200 | 0.340 | 69 | 0.580 | 0.571 | 0.651 | 0.136 |
| 3 | finbert_camelbert | bilingual | 0.200 | 0.355 | 72 | 0.569 | 0.558 | 0.644 | 0.111 |
| 3 | finbert_camelbert | arabic_only | 0.300 | 0.345 | 70 | 0.457 | 0.551 | 0.406 | 0.112 |
| 3 | finbert_camelbert | arabic_only | 0.200 | 0.345 | 70 | 0.457 | 0.551 | 0.406 | 0.112 |
| 1 | finbert_camelbert | arabic_only | 0.500 | 0.522 | 106 | 0.594 | 0.544 | 0.695 | 0.090 |
| 3 | finbert_camelbert | arabic_only | 0.500 | 0.527 | 107 | 0.467 | 0.544 | 0.436 | 0.092 |
| 3 | finbert_camelbert | bilingual | 0.300 | 0.453 | 92 | 0.554 | 0.537 | 0.637 | 0.071 |
| 5 | finbert_arabert | bilingual | 0.500 | 0.619 | 125 | 0.656 | 0.537 | 0.772 | 0.070 |

## Limitation

Abstention changes the decision problem: it evaluates fewer market sessions. These results are promising only when balanced accuracy improves while retaining enough covered samples to be practically meaningful.
