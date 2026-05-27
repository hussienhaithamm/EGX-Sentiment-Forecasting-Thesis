# EGX Sentiment Forecasting Thesis

Repository: <https://github.com/hussienhaithamm/EGX-Sentiment-Forecasting-Thesis>

This repository contains the reproducible implementation, experiment outputs, and thesis source for **Sentiment-Enhanced Deep Stock Market Forecasting Using Bilingual Financial News: An Egyptian Market Case Study**.

The project studies whether Arabic and English financial-news sentiment carries useful short-horizon forecasting information for selected Egyptian Exchange equities:

- Commercial International Bank: `CIB`, `COMI`
- Talaat Moustafa Group: `TMG`, `TMGH`
- Orascom Investment Holding: `OIH`

The core benchmark is intentionally conservative. It tests sentiment-only features, without open-high-low-close-volume features or technical indicators, under chronological splits, walk-forward evaluation, embargoes, and explicit leakage checks.

## What Is Included

```text
src/
  egx_direction/          Sentiment-only targets, features, event labels, evaluation helpers
  news_pipeline/          Mubasher collection, cleaning, ticker matching, sentiment routing
scripts/
  run_multi_horizon_event_evaluation.py
  run_phase7_validity_checks.py
  run_historical_sentiment_feature_experiment.py
  run_sentiment_stock_impact_analysis.py
  run_arabert_financial_adaptation_experiment.py
  evaluate_arabert_financial_adaptation.py
  run_social_media_source_feasibility.py
data/
  raw/news/mubasher_sentiment_only/
  processed/news/sentiment_ready/
outputs/
  final_sentiment_experiments/
  historical_sentiment_feature_experiment/
  sentiment_stock_impact_analysis/
  arabert_financial_adaptation/
  social_media_feasibility/
Bachelor_Thesis_Template/
  bachelor.tex            Main LaTeX thesis entry point
  bachelor.pdf            Compiled thesis PDF
docs/
  experiment_log.md
  paper_research_journey_plan.md
  master_thesis_checklist_audit.md
tests/
```

Large model weights are intentionally excluded from Git. The AraBERT financial-adaptation checkpoint directory is ignored; the corpus, metadata, evaluation CSVs, and reports are tracked.

## Main Findings

- The final sentiment-only benchmark is leakage-safe but weak as a standalone forecasting model.
- The strongest validity-passing sentiment-only result is `OIH`, 5-day horizon, all-news bilingual view:
  - test accuracy `0.532`
  - balanced accuracy `0.503`
  - F1 `0.630`
  - return sign accuracy `0.585`
- Event filtering and event weighting produce isolated gains but no stable aggregate improvement.
- Adding sentiment to a historical price-based EGX model improves raw accuracy in 5 of 9 ticker/model comparisons, but the effect is conditional.
- Direct sentiment-return correlation analysis finds no stable monotonic association under the pre-specified threshold.
- Short Arabic financial-domain adaptation of AraBERT reduces held-out masked-language-model loss from `0.9624` to `0.8468` versus base AraBERT.
- Social media remains future work unless official approved access is available.

## Setup

Use Python 3.9 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

If the package is not installed, run scripts with:

```bash
PYTHONPATH=src python3 <script>
```

## Reproducing the Core Experiments

Run the final multi-horizon event-aware benchmark:

```bash
PYTHONPATH=src python3 scripts/run_multi_horizon_event_evaluation.py \
  --refit-every-rows 250 \
  --classification-c-grid 0.05 0.1 0.5 \
  --regression-alpha-grid 10.0
```

Run leakage, coverage, and overfitting checks:

```bash
PYTHONPATH=src python3 scripts/run_phase7_validity_checks.py
```

Run the historical-model sentiment feature experiment:

```bash
python3 scripts/run_historical_sentiment_feature_experiment.py
```

Run the direct sentiment-return association analysis:

```bash
python3 scripts/run_sentiment_stock_impact_analysis.py
```

Prepare and run the AraBERT financial-domain adaptation feasibility experiment:

```bash
python3 scripts/run_arabert_financial_adaptation_experiment.py --run-mlm-training --max-steps 20 --batch-size 2
python3 scripts/evaluate_arabert_financial_adaptation.py --batch-size 4
```

Run the social-media official-access feasibility report:

```bash
python3 scripts/run_social_media_source_feasibility.py
```

## Important Outputs

Final benchmark:

- `outputs/final_sentiment_experiments/horizon_comparison.csv`
- `outputs/final_sentiment_experiments/direction_metrics_by_horizon.csv`
- `outputs/final_sentiment_experiments/return_metrics_by_horizon.csv`
- `outputs/final_sentiment_experiments/event_model_metrics.csv`
- `outputs/final_sentiment_experiments/event_vs_all_news_comparison.csv`
- `outputs/final_sentiment_experiments/leakage_check_report.md`
- `outputs/final_sentiment_experiments/coverage_analysis.md`
- `outputs/final_sentiment_experiments/overfitting_report.md`

Auxiliary experiments:

- `outputs/historical_sentiment_feature_experiment/historical_sentiment_summary.md`
- `outputs/sentiment_stock_impact_analysis/sentiment_stock_impact_report.md`
- `outputs/arabert_financial_adaptation/arabert_financial_adaptation_report.md`
- `outputs/arabert_financial_adaptation/arabert_financial_mlm_evaluation_report.md`
- `outputs/social_media_feasibility/social_media_feasibility_report.md`

Paper and audit documents:

- `Bachelor_Thesis_Template/bachelor.pdf`
- `Bachelor_Thesis_Template/bachelor.tex`
- `docs/experiment_log.md`
- `docs/paper_research_journey_plan.md`
- `docs/master_thesis_checklist_audit.md`

## Testing

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The current tests cover sentiment article preparation, lagged feature alignment, split embargo behavior, event keyword labeling, and event feature-panel lagging.

## Thesis Compilation

From `Bachelor_Thesis_Template/`:

```bash
pdflatex -interaction=nonstopmode -halt-on-error bachelor.tex
bibtex bachelor
pdflatex -interaction=nonstopmode -halt-on-error bachelor.tex
pdflatex -interaction=nonstopmode -halt-on-error bachelor.tex
```

The compiled PDF is tracked as `Bachelor_Thesis_Template/bachelor.pdf`.

## Notes on Scope

This repository is a research artifact, not a trading system. Results should be interpreted as evidence about bilingual financial-news sentiment under the stated data, alignment, and validation constraints. The thesis deliberately prefers leakage-safe modest results over optimistic results produced by looser evaluation.
