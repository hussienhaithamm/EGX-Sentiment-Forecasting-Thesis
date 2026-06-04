# EGX30 Full-Universe Results

## Dataset

- Timeline: `2021-04-21` through `2026-04-20`
- Historical company union: `57`
- Membership intervals: `12`
- Prepared and scored base articles: `9635`
- Arabic articles: `7418`
- English articles: `2217`

## Point-in-Time Scoped Articles

| Sentiment source | PIT scoped articles |
| --- | --- |
| finbert_arabert | 4782 |
| finbert_camelbert | 4782 |

## Bilingual Test Metrics

| Sentiment source | Horizon | Accuracy | Balanced accuracy | F1 |
| --- | --- | --- | --- | --- |
| finbert_arabert | 1 | 0.522 | 0.512 | 0.591 |
| finbert_arabert | 3 | 0.404 | 0.416 | 0.447 |
| finbert_arabert | 5 | 0.639 | 0.536 | 0.754 |
| finbert_camelbert | 1 | 0.621 | 0.545 | 0.737 |
| finbert_camelbert | 3 | 0.522 | 0.495 | 0.614 |
| finbert_camelbert | 5 | 0.381 | 0.489 | 0.378 |

## Validity

The independent audit status is `PASS`. It checked `18` panels, `18666` target rows, `18666` lag rows, `7290` validation/test predictions, and `36` saved metric rows.

## Interpretation

The reported comparison is index-level and sentiment-only. Accuracy must be interpreted with balanced accuracy, F1, MCC, actual Up share, and predicted Up share. EGX30 closes are used only to create labels. The earlier CIB/TMG/OIH benchmark is retained as an ablation rather than mixed into the production claims.

## Robustness and Easy-Win Validity Extensions

The easy-win audit pass adds coverage, baseline, class-imbalance, sector, weighting, and abstention reports under `reports/`.

- Coverage audit: the point-in-time scoped corpus contains 4,782 articles; the top five companies account for about 30.0% of articles and the top ten for about 48.6%, below the pre-set concentration warning threshold.
- Baselines: only 8 of 18 sentiment-model test rows beat the training-majority baseline on balanced accuracy. Rows that do not beat that baseline are treated as weak or non-evidential even when raw accuracy appears acceptable.
- Abstention: the best high-signal row reaches balanced accuracy around 0.580 at about 35.6% coverage, so it is reported as a conditional result rather than as full-session forecasting performance.
- Sector aggregation: a compact static sector map is added and `UNKNOWN` sectors are tolerated.
- Weighting: equal-weighted and article-count-weighted sentiment are reported; true historical index weights are unavailable in the reproducible dataset and are not faked with price or market-cap proxies.

These findings support careful language: promising but limited, weak directional signal, and balanced accuracy as the primary decision metric due to class imbalance.
