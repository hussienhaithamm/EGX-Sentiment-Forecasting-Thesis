# EGX30 Data Pipeline

The production pipeline forecasts EGX30 direction from sentiment only. `data/egx30_historical_constituents_5yr.csv` is the point-in-time universe registry. `data/scraped_cache/egx30_5yr/` stores resumable raw bilingual Mubasher records. `data/processed_news_5yr.csv` and `data/processed_news_5yr_camelbert.csv` store scored article tables. `data/EGX30 Index Historical Data.csv` is restricted to label construction.

Raw scrape checkpoints, scored article tables, and large experiment panels are reproducible local materializations excluded from Git. The reviewable repository surface is the compact registry, metrics, validity report, summaries, code, and thesis source. Large thesis-release artifacts should be published through versioned external storage.
