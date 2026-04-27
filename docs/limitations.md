# Limitations

- Sentiment-only features are weak predictors of short-horizon EGX returns.
- Balanced accuracy remains close to random for most ticker/horizon combinations.
- OIH has only 302 labeled articles, which limits signal stability.
- Event keywords are broad; many Mubasher articles match at least one event term, so event filtering is not very selective.
- AraBERT and FinBERT may not fully capture Egyptian market-specific language, abbreviations, or sarcasm.
- The project intentionally excludes OHLCV and technical indicators from the main benchmark; this isolates sentiment, but it also makes the forecasting task harder.
- The final multi-horizon run uses a 250-row refit cadence for runtime practicality. It remains chronological, but exact metrics may differ from a more frequent refit cadence.
- Return correlations are small and unstable, so regression outputs should not be interpreted as economically strong forecasts.

Recommended thesis framing: sentiment-only forecasting provides limited evidence of predictive value, but the results do not support using news sentiment as a standalone trading signal for the selected EGX stocks.
