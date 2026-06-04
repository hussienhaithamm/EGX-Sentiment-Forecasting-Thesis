# EGX30 Coverage Audit

## Interpretation

This audit checks whether the point-in-time EGX30 news corpus is broadly distributed across companies and trading dates. It is an audit artifact only; EGX30 prices are used only for trading-day calendars and target labels elsewhere, not as sentiment-model features.

## Concentration Diagnostics

- Total scoped articles: `4782`
- Top 5 company article share: `0.300`
- Top 10 company article share: `0.486`
- Article-count HHI: `0.037`
- Covered trading days with only one company represented: `0.155`

## Coverage Warnings

- No concentration warning crossed the pre-set thresholds.

## Top Companies By Article Count

| ticker | company_name | article_count | arabic_article_count | english_article_count | articles_per_active_trading_day |
| --- | --- | --- | --- | --- | --- |
| HRHO | EFG Holding | 381 | 251 | 130 | 0.292 |
| ETEL | Telecom Egypt | 289 | 231 | 58 | 0.222 |
| MASR | Madinet Masr for Housing and Development | 284 | 215 | 69 | 0.228 |
| PHDC | Palm Hills Developments | 266 | 217 | 49 | 0.204 |
| FWRY | Fawry for Banking Technology and Electronic Payment | 216 | 160 | 56 | 0.166 |
| EFIH | E-Finance for Digital and Financial Investments | 200 | 163 | 37 | 0.182 |
| EAST | Eastern Company | 187 | 148 | 39 | 0.143 |
| ABUK | Abu Qir Fertilizers and Chemicals | 179 | 139 | 40 | 0.137 |
| HELI | Heliopolis Company for Housing and Development | 163 | 145 | 18 | 0.156 |
| CLHO | Cleopatra Hospitals Group | 158 | 138 | 20 | 0.218 |
| CCAP | Qalaa Holdings | 158 | 132 | 26 | 0.135 |
| GBCO | GB Corp | 156 | 109 | 47 | 0.120 |
| EKHO | Egypt Kuwait Holding | 152 | 135 | 17 | 0.117 |
| MFPC | Misr Fertilizers Production Company Mopco | 142 | 109 | 33 | 0.121 |
| SKPC | Sidi Kerir Petrochemicals | 137 | 101 | 36 | 0.110 |

## Yearly Coverage

| year | article_count | arabic_article_count | english_article_count | trading_days | trading_days_with_at_least_1_article | trading_days_with_at_least_3_articles | trading_days_with_at_least_5_articles | covered_trading_day_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2021 | 823 | 561 | 262 | 131 | 124 | 98 | 63 | 0.947 |
| 2022 | 1089 | 880 | 209 | 212 | 202 | 162 | 99 | 0.953 |
| 2023 | 829 | 669 | 160 | 188 | 178 | 117 | 45 | 0.947 |
| 2024 | 876 | 691 | 185 | 217 | 199 | 139 | 60 | 0.917 |
| 2025 | 907 | 713 | 194 | 227 | 207 | 136 | 75 | 0.912 |
| 2026 | 258 | 181 | 77 | 65 | 58 | 41 | 19 | 0.892 |

## Limitations

Coverage remains uneven when a small group of firms or sectors receives disproportionate news attention. Trading-day coverage does not establish that articles were equally informative across firms, and low-coverage days should be interpreted cautiously in forecasting results.
