# EGX30 Full-Universe Experiment Log

## Phase 1: Point-in-Time Universe

Built `data/egx30_historical_constituents_5yr.csv` for 12 continuous intervals and 57 historical companies. The validator allows only two documented corporate-action transition states: OIH/OFH and Pioneers Holding.

## Phase 2: Bilingual Collection

Scraped Mubasher Arabic and English company pages with per-company JSONL checkpoints, randomized `0.5-1.5s` delays, retry backoff, and exact timeline filtering.

## Phase 3: Sentiment Baselines

The primary route uses English FinBERT plus Arabic AraBERT. The comparative modern baseline uses English FinBERT plus Arabic CAMeLBERT. A GPT zero-shot cache integration remains available, but no API experiment is reported because `OPENAI_API_KEY` was not configured.

## Phase 4: Strict Evaluation

Built equal-weight active-constituent market mood features after point-in-time membership filtering and cross-company article deduplication. Ran logistic direction classification and ridge return regression at horizons 1, 3, and 5 with chronological expanding-window walk-forward validation and embargoes.

## Phase 5: Validity

Audit status: `PASS`.

| Sentiment source | Horizon | Accuracy | Balanced accuracy | F1 |
| --- | --- | --- | --- | --- |
| finbert_arabert | 1 | 0.522 | 0.512 | 0.591 |
| finbert_arabert | 3 | 0.404 | 0.416 | 0.447 |
| finbert_arabert | 5 | 0.639 | 0.536 | 0.754 |
| finbert_camelbert | 1 | 0.621 | 0.545 | 0.737 |
| finbert_camelbert | 3 | 0.522 | 0.495 | 0.614 |
| finbert_camelbert | 5 | 0.381 | 0.489 | 0.378 |
