# Data Pipeline

## Inputs

Included local data:

- `data/raw/news/mubasher_sentiment_only/`: full Mubasher scrape for CIB, TMG, and OIH.
- `data/processed/news/sentiment_ready/sentiment_labeled_articles.csv`: cleaned sentiment-labeled dataset.
- `data/Commercial Int Bank Stock Price History.csv`
- `data/TMG Holding Stock Price History.csv`
- `data/Orascom Investment Holding.csv`

Clean sentiment-labeled article counts:

| Ticker | Articles |
| --- | ---: |
| CIB | 1,013 |
| TMG | 597 |
| OIH | 302 |
| Total | 1,912 |

## Sentiment Dataset Columns

The processed sentiment dataset includes canonical ticker, detected language, timestamps, cleaned text, relevance scores, sentiment label, positive/negative/neutral probabilities, confidence, and continuous sentiment score.

## Leakage Controls

- Article timestamps are converted to Africa/Cairo local dates.
- Daily sentiment is shifted before it is aligned to a trading date.
- Targets are future returns over the selected horizon.
- Walk-forward training excludes rows whose future target window would overlap the evaluation row.
