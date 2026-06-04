# EGX30 Sector-Aware Aggregation

## Result

A compact sector map was added and sector-level daily aggregates were generated. This pass reports the sector feature layer as an audit-ready ablation scaffold; the main sentiment model remains unchanged.

## Sector Coverage Summary

| sector | trading_days_with_articles | article_count | avg_sector_sentiment |
| --- | --- | --- | --- |
| Real Estate | 525 | 882 | 0.116 |
| Financial Services | 508 | 865 | 0.109 |
| Materials | 391 | 636 | 0.135 |
| Healthcare | 329 | 486 | 0.098 |
| Financial Technology | 290 | 416 | 0.174 |
| Consumer Staples | 269 | 400 | 0.114 |
| Consumer Discretionary | 194 | 297 | 0.187 |
| Telecommunications | 209 | 289 | 0.148 |
| Banks | 191 | 267 | 0.186 |
| Energy | 80 | 109 | 0.210 |
| Transportation | 56 | 76 | 0.271 |
| Education | 41 | 59 | 0.186 |

## Limitation

Sector labels are manually reviewable metadata, and `UNKNOWN` is tolerated by the feature builder. No sector label is used to create target labels or price-derived predictors.
