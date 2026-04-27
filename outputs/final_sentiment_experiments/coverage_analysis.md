# Coverage Analysis

This report measures whether weak performance is plausibly linked to sparse news coverage and whether more Mubasher listing pages appear available.

| Ticker | Articles | EN | AR | Trading days in article window | Same-day coverage | Prior-day coverage | Prior-7d coverage | Density/articles per trading day | Live pages vs local pages | Assessment |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| CIB | 1013 | 310 | 703 | 2451 | 0.177 | 0.140 | 0.569 | 0.413 | en 13/13, ar 30/30 | Local scrape covers all currently available Mubasher listing pages for this ticker/language set. |
| TMG | 597 | 188 | 409 | 2463 | 0.114 | 0.095 | 0.387 | 0.242 | en 8/8, ar 17/17 | Local scrape covers all currently available Mubasher listing pages for this ticker/language set. |
| OIH | 302 | 87 | 215 | 1898 | 0.088 | 0.071 | 0.305 | 0.159 | en 9/9, ar 20/20 | Local scrape covers all currently available Mubasher listing pages for this ticker/language set. |

## Interpretation

- OIH and TMG remain sparse relative to their trading calendars; OIH has the lowest labeled article count and a shorter article history.
- The local scrape reaches the current live Mubasher listing-page counts for all ticker/language pairs checked, so simply increasing `max_pages_per_language` is not expected to add more historical Mubasher pages from this source.
- Two OIH article-page failures are recorded in the scraper manifest. That is too small to explain weak model performance, but it is documented as residual source incompleteness.
