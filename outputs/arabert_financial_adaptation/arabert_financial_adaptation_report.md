# AraBERT Financial Adaptation Experiment

This experiment prepares the Arabic side of the thesis corpus for finance-domain adaptation of AraBERT. The valid FinBERT analogue is domain-adaptive masked-language-model pre-training followed by supervised sentiment fine-tuning on human-labeled Arabic financial sentiment examples.

The current thesis dataset does not contain independent human sentiment labels. Existing article labels are model outputs from the current AraBERT route, so using them as ground truth would only distill the same model and would not be valid evidence of improved Arabic financial sentiment classification.

## Corpus

- Arabic financial articles: `1327`
- Chronological MLM train articles: `1061`
- Chronological MLM validation articles: `266`
- Date range: `2010-08-11T03:00:00+00:00` to `2026-04-22T16:24:26+00:00`

## Frequent Financial Terms

| Category | Term | Articles | Share |
| --- | --- | ---: | ---: |
| investment_projects | استثمار | 786 | 0.592 |
| earnings | ارباح | 776 | 0.585 |
| market_terms | البورصه | 720 | 0.543 |
| investment_projects | مشروع | 328 | 0.247 |
| financing | تمويل | 267 | 0.201 |
| disclosure_governance | القوائم الماليه | 252 | 0.190 |
| disclosure_governance | مجلس الاداره | 185 | 0.139 |
| disclosure_governance | الجمعيه العامه | 177 | 0.133 |
| investment_projects | عقد | 163 | 0.123 |
| market_terms | المؤشر | 156 | 0.118 |
| market_terms | التداول | 152 | 0.115 |
| market_terms | السهم | 152 | 0.115 |
| disclosure_governance | افصاح | 134 | 0.101 |
| dividends | كوبون | 110 | 0.083 |
| financing | سندات | 96 | 0.072 |

## MLM Adaptation Status

- Status: `COMPLETED`
- Model directory: `outputs/arabert_financial_adaptation/arabert_financial_mlm`
- Train loss: `0.9676`
- Evaluation loss: `0.8218`
- Max steps: `20`

## Held-Out MLM Evaluation Against Base AraBERT

The saved financial-domain checkpoint was evaluated against the original `aubmindlab/bert-base-arabertv02` model on the same Arabic financial corpus using identical masked-token positions.

| Split | Model | Examples | Masked tokens | Loss | Perplexity |
| --- | --- | ---: | ---: | ---: | ---: |
| train | base_arabert | 1061 | 40372 | 0.9413 | 2.563 |
| train | financial_adapted_arabert | 1061 | 40372 | 0.8202 | 2.271 |
| validation | base_arabert | 266 | 10292 | 0.9624 | 2.618 |
| validation | financial_adapted_arabert | 266 | 10292 | 0.8468 | 2.332 |

The adapted checkpoint lowers held-out validation MLM loss by `0.1156`, a relative reduction of `12.01%` compared with base AraBERT. This supports the claim that short domain adaptation made the model better at reconstructing masked Arabic financial text.

## Supervised Sentiment Fine-Tuning Status

Not run. A valid supervised experiment requires human-labeled Arabic financial sentiment examples with positive, negative, and neutral labels. Pseudo-labels from the current AraBERT classifier are not accepted as ground truth.

## Paper Interpretation

This should be reported as a prepared domain-adaptation protocol and MLM feasibility result, not as a completed Arabic FinBERT equivalent. The available corpus is useful for financial terminology exposure, and the adapted checkpoint improves held-out MLM loss against base AraBERT. The thesis still needs an independently labeled Arabic financial sentiment set before making supervised sentiment performance claims.
