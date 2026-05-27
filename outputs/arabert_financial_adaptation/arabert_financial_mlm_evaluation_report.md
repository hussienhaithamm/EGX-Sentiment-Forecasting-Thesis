# AraBERT Financial MLM Evaluation

This evaluates the saved finance-adapted AraBERT checkpoint as a masked-language model on the Arabic financial article corpus.

This is not supervised sentiment accuracy. The checkpoint has no sentiment-classification head, and the thesis dataset does not contain independent human sentiment labels.

## Configuration

- Base model: `aubmindlab/bert-base-arabertv02`
- Adapted model: `outputs/arabert_financial_adaptation/arabert_financial_mlm`
- MLM probability: `0.15`
- Max sequence length: `256`
- Batch size: `4`
- Seed: `123`

## Results

| Split | Model | Examples | Masked tokens | Loss | Perplexity |
| --- | --- | ---: | ---: | ---: | ---: |
| train | base_arabert | 1061 | 40372 | 0.9413 | 2.563 |
| train | financial_adapted_arabert | 1061 | 40372 | 0.8202 | 2.271 |
| validation | base_arabert | 266 | 10292 | 0.9624 | 2.618 |
| validation | financial_adapted_arabert | 266 | 10292 | 0.8468 | 2.332 |

## Interpretation

The adapted checkpoint lowers validation MLM loss by 0.1156 relative to the base AraBERT model, a relative reduction of 12.01%.
This result can support a domain-adaptation feasibility claim only. It does not prove better stock forecasting or better sentiment classification.
