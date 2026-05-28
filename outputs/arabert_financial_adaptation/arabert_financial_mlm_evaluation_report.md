# AraBERT Financial MLM Evaluation

This evaluates the saved finance-adapted AraBERT checkpoint as a masked-language model on the Arabic financial article corpus.

This is not supervised sentiment accuracy. The checkpoint has no sentiment-classification head, and the thesis dataset does not contain independent human sentiment labels.

## Configuration

- Base model: `aubmindlab/bert-base-arabertv02`
- Adapted model: `outputs/arabert_financial_adaptation/arabert_financial_mlm`
- MLM probability: `0.15`
- Max sequence length: `256`
- Batch size: `4`
- Seed: `42`

## Results

| Split | Model | Examples | Masked tokens | Loss | Perplexity |
| --- | --- | ---: | ---: | ---: | ---: |
| train | base_arabert | 1061 | 40403 | 0.9131 | 2.492 |
| train | financial_adapted_arabert | 1061 | 40403 | 0.7882 | 2.199 |
| validation | base_arabert | 266 | 10080 | 0.9414 | 2.564 |
| validation | financial_adapted_arabert | 266 | 10080 | 0.8238 | 2.279 |

## Interpretation

The adapted checkpoint lowers validation MLM loss by 0.1175 relative to the base AraBERT model, a relative reduction of 12.49%.
This result can support a domain-adaptation feasibility claim only. It does not prove better stock forecasting or better sentiment classification.
