# Results

## Dataset

Clean sentiment-labeled article counts:

| Ticker | Articles |
| --- | ---: |
| CIB | 1,013 |
| TMG | 597 |
| OIH | 302 |
| Total | 1,912 |

Event keyword matches in the final event-labeling pass:

| Ticker | Event-labeled articles |
| --- | ---: |
| CIB | 874 |
| TMG | 564 |
| OIH | 285 |

The high event count shows that Mubasher articles are often explicitly event/news driven; event filtering is therefore not a narrow filter for this dataset.

## Previous 5-Day Baseline Bilingual Metrics

| Ticker | Accuracy | Balanced Acc. | F1 | Sign Acc. | Return Corr. |
| --- | ---: | ---: | ---: | ---: | ---: |
| OIH | 0.554 | 0.513 | 0.669 | 0.587 | 0.036 |
| TMG | 0.507 | 0.507 | 0.524 | 0.534 | 0.008 |
| CIB | 0.502 | 0.497 | 0.549 | 0.541 | 0.072 |

## Previous Enhanced Feature Metrics

| Ticker | Accuracy | Balanced Acc. | F1 | Sign Acc. | Return Corr. |
| --- | ---: | ---: | ---: | ---: | ---: |
| CIB | 0.523 | 0.499 | 0.619 | 0.537 | -0.003 |
| TMG | 0.478 | 0.479 | 0.466 | 0.534 | -0.019 |
| OIH | 0.573 | 0.511 | 0.710 | 0.575 | 0.010 |

## Final Multi-Horizon All-News Bilingual Results

| Horizon | Ticker | Accuracy | Balanced Acc. | Precision | Recall | F1 | Sign Acc. | Return Corr. |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | CIB | 0.491 | 0.505 | 0.534 | 0.184 | 0.274 | 0.474 | -0.013 |
| 1 | TMG | 0.520 | 0.507 | 0.714 | 0.021 | 0.041 | 0.469 | -0.013 |
| 1 | OIH | 0.461 | 0.500 | 0.444 | 0.848 | 0.583 | 0.426 | -0.015 |
| 3 | CIB | 0.514 | 0.482 | 0.529 | 0.901 | 0.667 | 0.516 | -0.018 |
| 3 | TMG | 0.511 | 0.518 | 0.500 | 0.798 | 0.615 | 0.493 | -0.103 |
| 3 | OIH | 0.471 | 0.508 | 0.692 | 0.034 | 0.065 | 0.518 | -0.035 |
| 5 | CIB | 0.543 | 0.487 | 0.555 | 0.942 | 0.698 | 0.510 | -0.052 |
| 5 | TMG | 0.493 | 0.490 | 0.498 | 0.939 | 0.651 | 0.532 | -0.024 |
| 5 | OIH | 0.532 | 0.503 | 0.581 | 0.688 | 0.630 | 0.585 | 0.034 |

## Event Filtering Summary

Event filtering did not produce a consistent accuracy improvement.

- Mean event-only accuracy delta vs all-news: `-0.013`
- Best event-only accuracy delta: `+0.025`
- Mean event-weighted accuracy delta vs all-news: `-0.012`
- Best event-weighted accuracy delta: `+0.039`

Notable positives:

- CIB 1-day event-only improved accuracy from 0.491 to 0.507 and return correlation from -0.013 to 0.050.
- CIB 3-day event-only improved accuracy from 0.514 to 0.525 and return correlation from -0.018 to 0.045.
- OIH 5-day event-weighted improved accuracy from 0.532 to 0.571 and F1 from 0.630 to 0.706, but balanced accuracy remained weak at 0.511.
- TMG 5-day event-only improved return correlation from -0.024 to 0.073, but accuracy stayed near random.

## Best Final Results

Before Phase 7 overfitting filtering, the strongest raw accuracy result in the final multi-horizon/event run was:

- OIH, 5-day, event-weighted bilingual: accuracy `0.571`, balanced accuracy `0.511`, F1 `0.706`.

Phase 7 flagged that row for overfitting:

- train accuracy `0.609`
- validation accuracy `0.485`
- test accuracy `0.571`
- overfit flag: `True`

Therefore it is exploratory and should not be presented as the accepted final model.

The accepted validity-passing candidate is:

- OIH, 5-day, all-news bilingual
- validation accuracy/balanced accuracy/F1: `0.522` / `0.525` / `0.546`
- test accuracy/balanced accuracy/F1/sign accuracy: `0.532` / `0.503` / `0.630` / `0.585`
- overfit flag: `False`

Other final-run bests:

- Best balanced accuracy: TMG, 3-day, all-news bilingual, balanced accuracy `0.518`.
- Best return sign accuracy: OIH, 5-day, all-news bilingual, sign accuracy `0.585`, return correlation `0.034`.

Across all observed project runs, the previous enhanced OIH 5-day result still has the highest raw accuracy (`0.573`) and F1 (`0.710`), but Phase 7 shows why raw accuracy alone should not be used as the final acceptance criterion. Balanced accuracy remains weak at about `0.50` to `0.52`.

## Phase 7 Validity Results

- Leakage checks: `PASS`; 45 panel checks, 43,880 validation/test prediction rows checked, 0 history violations.
- Coverage: local scrape matches live Mubasher listing pages for all ticker/language pairs checked.
- Sparse coverage remains material: prior-7-day article coverage is CIB `0.569`, TMG `0.387`, OIH `0.305`.
- Overfitting: 15 of 45 experiment rows were flagged for train/test overfitting.
- The final accepted model is OIH 5-day all-news bilingual, not the higher raw-accuracy event-weighted candidate.

## Interpretation

The sentiment-only signal is weak and inconsistent. Some horizons and tickers produce accuracy above 0.53, but balanced accuracy is usually close to 0.50. High F1 is often caused by models predicting the majority upward class, not by robust discrimination between up and down days.

The thesis should present this as an honest negative/limited-signal result: financial-news sentiment alone contains some directional information, but it is not reliable enough as a standalone EGX forecasting model in this dataset.
