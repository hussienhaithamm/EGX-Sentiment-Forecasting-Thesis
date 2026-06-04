# EGX30 Full-Universe Overfitting Report

Rows checked: `18`

Rows flagged: `9`

A row is flagged when the absolute train-test accuracy or balanced-accuracy gap exceeds `0.10`, or the absolute F1 gap exceeds `0.20`.

## Largest Accuracy Gaps

| Source | View | Horizon | Train Accuracy | Test Accuracy | Accuracy Gap | Balanced Gap | F1 Gap | Flag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| finbert_arabert | bilingual | 3 | 0.622 | 0.404 | 0.218 | 0.225 | 0.148 | True |
| finbert_camelbert | bilingual | 5 | 0.575 | 0.381 | 0.194 | 0.124 | 0.104 | True |
| finbert_arabert | english_only | 5 | 0.535 | 0.356 | 0.178 | 0.109 | 0.070 | True |
| finbert_camelbert | english_only | 5 | 0.535 | 0.356 | 0.178 | 0.109 | 0.070 | True |
| finbert_arabert | arabic_only | 5 | 0.626 | 0.460 | 0.166 | 0.185 | 0.078 | True |
| finbert_camelbert | bilingual | 3 | 0.678 | 0.522 | 0.156 | 0.184 | 0.090 | True |
| finbert_arabert | bilingual | 1 | 0.661 | 0.522 | 0.139 | 0.149 | 0.087 | True |
| finbert_camelbert | arabic_only | 5 | 0.464 | 0.332 | 0.132 | -0.001 | -0.022 | True |
| finbert_arabert | english_only | 1 | 0.579 | 0.493 | 0.087 | 0.090 | 0.051 | False |
| finbert_camelbert | english_only | 1 | 0.579 | 0.493 | 0.087 | 0.090 | 0.051 | False |
| finbert_arabert | bilingual | 5 | 0.713 | 0.639 | 0.075 | 0.155 | 0.018 | True |
| finbert_arabert | english_only | 3 | 0.590 | 0.517 | 0.073 | 0.084 | 0.042 | False |
| finbert_camelbert | english_only | 3 | 0.590 | 0.517 | 0.073 | 0.084 | 0.042 | False |
| finbert_arabert | arabic_only | 3 | 0.595 | 0.660 | -0.065 | 0.004 | -0.057 | False |
| finbert_camelbert | arabic_only | 3 | 0.542 | 0.478 | 0.064 | 0.045 | -0.035 | False |
