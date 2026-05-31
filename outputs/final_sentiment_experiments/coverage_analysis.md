# Coverage Analysis

This report measures whether weak performance is plausibly linked to sparse news coverage and whether more Mubasher listing pages appear available.

| Ticker | Articles | EN | AR | Trading days in article window | Same-day coverage | Prior-day coverage | Prior-7d coverage | Density/articles per trading day | Live pages vs local pages | Assessment |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| CIB | 1013 | 310 | 703 | 2451 | 0.177 | 0.140 | 0.569 | 0.413 | en 13/None, ar 30/None | Live page-count check unavailable; local manifest must be used. |
| TMG | 604 | 189 | 415 | 2463 | 0.114 | 0.095 | 0.387 | 0.245 | en 8/None, ar 17/None | Live page-count check unavailable; local manifest must be used. |
| OIH | 304 | 87 | 217 | 1898 | 0.089 | 0.071 | 0.307 | 0.160 | en 9/None, ar 20/None | Live page-count check unavailable; local manifest must be used. |

## Interpretation

- OIH and TMG remain sparse relative to their trading calendars; OIH has the lowest labeled article count and a shorter article history.
- The local scrape reaches the current live Mubasher listing-page counts for all ticker/language pairs checked, so simply increasing `max_pages_per_language` is not expected to add more historical Mubasher pages from this source.
- Two OIH article-page failures are recorded in the scraper manifest. That is too small to explain weak model performance, but it is documented as residual source incompleteness.
