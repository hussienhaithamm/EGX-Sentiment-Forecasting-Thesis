# Sentiment Score and Stock Impact Analysis

This experiment tests whether daily bilingual news sentiment is associated with same-day and future stock returns. Sentiment is shifted by one calendar day before being aligned to trading dates, so predictive correlations use only prior news.

Permutation p-values shuffle sentiment in five-day blocks to reduce the chance that serial dependence alone creates an apparently significant correlation.

## Main Spearman Correlations

| Ticker | Return horizon | Rows | Correlation | Permutation p-value |
| --- | ---: | ---: | ---: | ---: |
| OIH | 3 | 135 | -0.127 | 0.155 |
| OIH | 5 | 135 | -0.109 | 0.186 |
| TMG | 0 | 235 | -0.102 | 0.120 |
| OIH | 1 | 135 | -0.089 | 0.355 |
| CIB | 3 | 345 | 0.064 | 0.245 |
| CIB | 0 | 345 | 0.055 | 0.314 |

## Event-Style Return Groups

| Ticker | Group | Horizon | Rows | Mean return | Median return | Positive return share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CIB | all_news_days | 1 | 345 | 0.0019 | 0.0007 | 0.519 |
| CIB | bottom_sentiment_quartile | 1 | 87 | -0.0015 | -0.0005 | 0.437 |
| CIB | no_prior_news_days | 1 | 2121 | 0.0009 | 0.0000 | 0.465 |
| CIB | top_sentiment_quartile | 1 | 87 | 0.0004 | 0.0000 | 0.471 |
| CIB | all_news_days | 3 | 345 | 0.0040 | 0.0002 | 0.501 |
| CIB | bottom_sentiment_quartile | 3 | 87 | -0.0029 | -0.0025 | 0.448 |
| CIB | no_prior_news_days | 3 | 2119 | 0.0030 | 0.0010 | 0.516 |
| CIB | top_sentiment_quartile | 3 | 87 | 0.0030 | 0.0000 | 0.494 |
| CIB | all_news_days | 5 | 345 | 0.0044 | 0.0036 | 0.536 |
| CIB | bottom_sentiment_quartile | 5 | 87 | 0.0002 | -0.0010 | 0.483 |
| CIB | no_prior_news_days | 5 | 2117 | 0.0052 | 0.0014 | 0.517 |
| CIB | top_sentiment_quartile | 5 | 87 | 0.0039 | 0.0011 | 0.529 |
| OIH | all_news_days | 1 | 135 | 0.0065 | 0.0021 | 0.504 |
| OIH | bottom_sentiment_quartile | 1 | 34 | 0.0151 | 0.0076 | 0.559 |
| OIH | no_prior_news_days | 1 | 2328 | 0.0009 | 0.0000 | 0.384 |
| OIH | top_sentiment_quartile | 1 | 34 | -0.0021 | 0.0000 | 0.471 |
| OIH | all_news_days | 3 | 135 | 0.0082 | 0.0000 | 0.496 |
| OIH | bottom_sentiment_quartile | 3 | 34 | 0.0199 | 0.0070 | 0.559 |
| OIH | no_prior_news_days | 3 | 2326 | 0.0036 | 0.0000 | 0.457 |
| OIH | top_sentiment_quartile | 3 | 34 | -0.0049 | -0.0065 | 0.382 |
| OIH | all_news_days | 5 | 135 | 0.0125 | 0.0000 | 0.467 |
| OIH | bottom_sentiment_quartile | 5 | 34 | 0.0361 | 0.0000 | 0.471 |
| OIH | no_prior_news_days | 5 | 2324 | 0.0061 | 0.0000 | 0.471 |
| OIH | top_sentiment_quartile | 5 | 34 | -0.0072 | -0.0031 | 0.441 |
| TMG | all_news_days | 1 | 234 | 0.0024 | 0.0000 | 0.457 |
| TMG | bottom_sentiment_quartile | 1 | 59 | 0.0050 | 0.0000 | 0.475 |
| TMG | no_prior_news_days | 1 | 2228 | 0.0013 | 0.0000 | 0.441 |
| TMG | top_sentiment_quartile | 1 | 59 | 0.0006 | 0.0000 | 0.424 |
| TMG | all_news_days | 3 | 233 | 0.0050 | -0.0013 | 0.468 |
| TMG | bottom_sentiment_quartile | 3 | 59 | 0.0068 | 0.0000 | 0.492 |
| TMG | no_prior_news_days | 3 | 2227 | 0.0041 | 0.0000 | 0.488 |
| TMG | top_sentiment_quartile | 3 | 59 | 0.0014 | 0.0000 | 0.475 |
| TMG | all_news_days | 5 | 233 | 0.0092 | -0.0013 | 0.485 |
| TMG | bottom_sentiment_quartile | 5 | 59 | 0.0116 | -0.0044 | 0.441 |
| TMG | no_prior_news_days | 5 | 2225 | 0.0068 | 0.0003 | 0.500 |
| TMG | top_sentiment_quartile | 5 | 59 | 0.0064 | 0.0018 | 0.525 |

## Interpretation

No strong, stable monotonic sentiment-return association appears under the pre-specified threshold of absolute Spearman correlation >= 0.10 and permutation p-value <= 0.05.
The event-style grouping is useful for interpretation, but it remains sensitive to sparse coverage and market regime effects.
