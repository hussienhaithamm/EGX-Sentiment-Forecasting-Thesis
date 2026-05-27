# Citation Audit

Date: 2026-05-15

Scope: this audit covers all papers currently cited in `/Users/mohamed/EGX-Sentiment-Forecasting-Thesis/Bachelor_Thesis_Template/introduction.tex`, `/Users/mohamed/EGX-Sentiment-Forecasting-Thesis/Bachelor_Thesis_Template/background.tex`, `/Users/mohamed/EGX-Sentiment-Forecasting-Thesis/Bachelor_Thesis_Template/literature_review.tex`, and `/Users/mohamed/EGX-Sentiment-Forecasting-Thesis/Bachelor_Thesis_Template/methodology.tex`.

Legend:
- `Direct fit`: directly supports bilingual financial-news sentiment forecasting, stock-movement prediction from text, domain-specific financial sentiment modeling, or evaluation discipline central to this thesis.
- `Background fit`: valid background or theory, but not direct evidence for the thesis core claim.
- `Peripheral / caveat needed`: real and relevant paper, but the setting is materially different from this thesis, so the citation should be used only with explicit caveats.

## Market Theory and Forecasting Background

- `fama1970` — *Efficient Capital Markets: A Review of Theory and Empirical Work* — `Background fit`
  Supports the motivation that public information is absorbed into prices. It does not support bilingual sentiment modeling or EGX-specific forecasting.

- `lo1988randomwalk` — *Stock Market Prices Do Not Follow Random Walks: Evidence from a Simple Specification Test* — `Background fit`
  Supports the claim that strict random-walk assumptions are empirically too strong. Use it only for market predictability background.

- `atsalakis2009survey` — *Surveying Stock Market Forecasting Techniques—Part II: Soft Computing Methods* — `Background fit`
  A forecasting survey. Useful for broad field context, not for text sentiment specifically.

- `sezer2020dlreview` — *Financial Time Series Forecasting with Deep Learning: A Systematic Literature Review: 2005–2019* — `Background fit`
  Strong review for horizons, targets, architectures, and evaluation variation. It is not a sentiment paper, so it should not carry sentiment-specific claims by itself.

- `fischer2018lstm` — *Deep Learning with Long Short-Term Memory Networks for Financial Market Predictions* — `Background fit`
  Relevant as a price-series baseline reference. Not a text or sentiment study.

- `nelson2017lstm` — *Stock Market's Price Movement Prediction with LSTM Neural Networks* — `Background fit`
  Relevant as price-only deep learning background. Not direct support for text-based forecasting claims.

- `vaswani2017attention` — *Attention Is All You Need* — `Background fit`
  Valid transformer architecture background only.

- `devlin2019bert` — *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* — `Background fit`
  Valid pretraining background only. It does not justify finance-specific sentiment quality without a domain adaptation paper such as FinBERT.

## Textual Sentiment in Finance

- `tetlock2007media` — *Giving Content to Investor Sentiment: The Role of Media in the Stock Market* — `Direct fit`
  Strong foundational paper for media tone and market effects. It fits the thesis topic well.

- `kearney2014survey` — *Textual Sentiment in Finance: A Survey of Methods and Models* — `Direct fit`
  Strong survey for finance text, sentiment sources, and methodology. One of the best support papers in the thesis.

- `das2007yahoo` — *Yahoo! for Amazon: Sentiment Extraction from Small Talk on the Web* — `Direct fit`
  Relevant early sentiment-in-finance paper. It is not financial news, but it is still directly about market-related sentiment extraction.

- `loughran2011liability` — *When Is a Liability Not a Liability? Textual Analysis, Dictionaries, and 10-Ks* — `Direct fit`
  Strong support for the claim that generic sentiment lexicons misread financial vocabulary.

- `mudinas2019market` — *Market Trend Prediction using Sentiment Analysis: Lessons Learned and Paths Forward* — `Direct fit`
  Good support for the warning that sentiment alone is not universally sufficient. Better for methodological caution than for benchmark numbers.

## Social Media and Other Non-News Sentiment Papers

- `bollen2011twitter` — *Twitter Mood Predicts the Stock Market* — `Peripheral / caveat needed`
  Real and relevant, but index-level and social-mood oriented. It should not be used as firm-level evidence without explicitly noting that limitation.

- `nguyen2015social` — *A Novel Method for Predicting Stock Price Movement Using the Sentiment from Social Media* — `Peripheral / caveat needed`
  Relevant and closer to firm-level forecasting than Bollen, but still social-media based and not bilingual financial news.

- `mao2011predicting` — *Predicting Financial Markets: Comparing Survey, News, Twitter and Search Engine Data* — `Peripheral / caveat needed`
  Useful for comparing signal sources and timing differences. It is not a direct benchmark for firm-linked financial-news prediction.

- `albahli2022twitter` — *A Machine Learning Method for Prediction of Stock Market Using Real-Time Twitter Data* — `Peripheral / caveat needed`
  Relevant to social-media market prediction, but too far from the thesis core setting to support detailed claims about firm-level news quality or bilingual routing.

- `xu2018stock` — *Stock Movement Prediction from Tweets and Historical Prices* — `Peripheral / caveat needed`
  Strong multimodal forecasting paper, but it is about tweets and prices, not financial news. It is appropriate only as a social-text fusion reference.

- `huang2021candlestick` — *Stock Price Movement Prediction Using Sentiment Analysis and Candlestick Chart Representation* — `Peripheral / caveat needed`
  Real and relevant, but its external text source is social-media sentiment rather than firm-linked financial news. It should be framed as a multimodal fusion example, not a news benchmark.

## Financial News and News-Oriented Stock Prediction

- `schumaker2009azfin` — *Textual Analysis of Stock Market Prediction Using Breaking Financial News: The AZFin Text System* — `Direct fit`
  Strong direct fit for short-horizon financial-news prediction with explicit entity linkage.

- `hagenau2013automated` — *Automated News Reading: Stock Price Prediction Based on Financial News Using Context-Capturing Features* — `Direct fit`
  Strong direct fit for financial-news-based stock prediction.

- `ding2015event` — *Deep Learning for Event-Driven Stock Prediction* — `Direct fit`
  Strong direct fit, although event-driven rather than pure sentiment-only. Still highly relevant to text-based market prediction.

- `hu2018chaotic` — *Listening to Chaotic Whispers: A Deep Learning Framework for News-Oriented Stock Trend Prediction* — `Direct fit`
  Strong direct fit for news-oriented stock trend prediction.

- `mohan2019news` — *Stock Price Prediction Using News Sentiment Analysis* — `Direct fit`
  Strong direct fit for news sentiment and stock prediction at scale.

- `dahal2023comparative` — *A Comparative Study on Effect of News Sentiment on Stock Price Prediction with Deep Learning Architecture* — `Direct fit`
  Strong direct fit for isolating the effect of adding financial news sentiment to LSTM and GRU models. It is regression-oriented, so it should not be presented as a directional classification benchmark.

- `cristescu2022marketnews` — *Using Market News Sentiment Analysis for Stock Market Prediction* — `Direct fit`
  Directly relevant because it uses market-news sentiment as an exogenous forecasting factor. The limitation is that it focuses on regression fit, not robust directional classification.

- `zhao2023integrated` — *Deep Learning-Based Integrated Framework for Stock Price Movement Prediction* — `Peripheral / caveat needed`
  Relevant architecture paper, but the reported target is the Hang Seng Index rather than firm-level news-to-stock relevance.

- `liapis2023deep` — *Investigating Deep Stock Market Forecasting with Sentiment Analysis* — `Direct fit`
  One of the strongest benchmark-style references in the thesis. It directly supports the claims that performance depends on horizon and feature setup.

- `shi2021deeplearning` — *Stock Movement Prediction with Sentiment Analysis Based on Deep Learning Networks* — `Direct fit`
  Relevant stock-movement and sentiment paper. It is reasonable as a general forecasting reference, but it is a weaker source for the very specific class-imbalance claim unless the text is phrased cautiously.

## Financial Sentiment Language Models and Arabic NLP

- `araci2019finbert` — *FinBERT: Financial Sentiment Analysis with Pre-trained Language Models* — `Direct fit`
  Strong direct fit for the English sentiment-model choice. It supports domain-specific financial sentiment extraction, not market prediction by itself.

- `antoun2020arabert` — *AraBERT: Transformer-Based Model for Arabic Language Understanding* — `Direct fit`
  Strong support for using Arabic-specialized encoders. It is not finance-specific, so its role is language-model justification rather than stock-prediction evidence.

- `abdulmageed2021arbert` — *ARBERT & MARBERT: Deep Bidirectional Transformers for Arabic* — `Direct fit`
  Strong support for Arabic-specialized modeling and benchmark strength. Like AraBERT, it supports model selection rather than direct forecasting evidence.

- `inoue2021camelbert` — *The Interplay of Variant, Size, and Task Type in Arabic Pre-trained Language Models* — `Direct fit`
  Strong support for the thesis argument that language variant and task proximity matter when routing Arabic text.

## Sentence-Level Notes After Cleanup

- `/Users/mohamed/EGX-Sentiment-Forecasting-Thesis/Bachelor_Thesis_Template/introduction.tex:10`
  The paragraph has been tightened so the citations now support language-aware modeling and evaluation discipline rather than the local EGX or Robin/Beltone context itself. The remaining limitation is that the bilingual EGX news environment is still stated as project context rather than being backed by a separate market or dataset citation.

- `/Users/mohamed/EGX-Sentiment-Forecasting-Thesis/Bachelor_Thesis_Template/introduction.tex:18`
  The paragraph has been tightened so the citations now support source choice and language-model selection rather than the industrial workflow itself.

- `/Users/mohamed/EGX-Sentiment-Forecasting-Thesis/Bachelor_Thesis_Template/background.tex:18`
  The wording has been softened so the citations support evaluation caution and reporting practice, while the class-imbalance point is presented as a methodological concern of this thesis.

- `/Users/mohamed/EGX-Sentiment-Forecasting-Thesis/Bachelor_Thesis_Template/literature_review.tex:26`
  The wording has been corrected to `Ho and Huang`, and the paper is now described consistently as social-media-plus-candlestick fusion rather than a financial-news benchmark.

## Metadata Corrections Applied During This Audit

- Corrected `dahal2023comparative` author list and issue number in `/Users/mohamed/EGX-Sentiment-Forecasting-Thesis/Bachelor_Thesis_Template/references.bib`.
- Corrected `huang2021candlestick` author list in `/Users/mohamed/EGX-Sentiment-Forecasting-Thesis/Bachelor_Thesis_Template/references.bib`.
- Corrected `cristescu2022marketnews` author list in `/Users/mohamed/EGX-Sentiment-Forecasting-Thesis/Bachelor_Thesis_Template/references.bib`.
- Corrected `albahli2022twitter` author list in `/Users/mohamed/EGX-Sentiment-Forecasting-Thesis/Bachelor_Thesis_Template/references.bib`.
