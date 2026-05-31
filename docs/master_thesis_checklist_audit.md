# Master Thesis Evaluation Checklist Audit

Date: 2026-05-31 (post-17-article scrape, full experiment re-run, and metric update)

Scope: `Bachelor_Thesis_Template/*.tex`, current compiled `bachelor.pdf` (55 pages, builds clean with `pdflatex + bibtex + pdflatex + pdflatex`), supporting bibliography, AraBERT MLM evaluation re-run with seed 42 and identical base AraBERT tokenizer for both runs. Experiments re-run on 1,921 articles (up from 1,912); all metric tables and prose updated to reflect the new numbers.

## Overall Verdict

Current status: **Pass / Submission-Ready Under This Checklist**

The thesis passes the broad scientific structure and now also passes the strict polish checklist after the mechanical, framing, and reproducibility fixes detailed below. Two TODO comments remain in LaTeX, both explicit and visible to future readers (submission date confirmation, global deterministic flag adoption).

## High-Priority Items (Previously Failing)

| Area | Status | Evidence | Fix Applied |
| --- | --- | --- | --- |
| Abstract tense rule | Pass | `abstract.tex` now uses simple future tense throughout. | Full rewrite of `abstract.tex`. |
| Complex/perfect tense ban | Pass | "has been", "have shown", "had completed" patterns removed across all chapters. | Chapter-by-chapter rewrite. |
| Table caption placement | Pass | All seven `table` and one `longtable` environments place `\caption` before `tabular`. | Verified by `grep`. |
| Abbreviation list | Pass | `appendix.tex` replaced placeholder `AC`/`AC2` with full alphabetical list of 43 abbreviations, each with full expansion. | Full rewrite of abbreviations section. |
| Declaration AI disclosure | Pass | `GUC_TitlePage.tex` now includes the exact disclosure paragraph specified in the brief. | Added immediately under the certification list. |
| Submission date | Pass (with TODO) | `bachelor.tex` updated to `15 June 2026` with explicit `% TODO: Confirm the final actual submission date with the supervisor before final binding.` comment. | Updated. |
| Consecutive heading rule | Pass | Every chapter now opens with a bridging paragraph before the first `\section`. | Verified by reading line 1--6 of every chapter file. |
| Environment reproducibility detail | Pass | `results.tex` opens with an `Experimental Setup` longtable listing 35 reproducibility parameters (Python 3.9.6, macOS 26.5 arm64, Apple M4, PyTorch 2.8.0, Transformers 4.57.3, scikit-learn 1.6.1, pandas 2.3.3, NumPy 1.26.4, AraBERT base revision `016fb9d6768f522a59c6e0d2d5d5d43a4e1bff60`, seed 42, deterministic flag policy, MLM training and evaluation hyperparameters). | Added as first section of Chapter 4. |

## AraBERT Framing Fix

| Check | Status | Evidence |
| --- | --- | --- |
| Every AraBERT MLM mention says "MLM perplexity on held-out Arabic financial text" | Pass | `abstract.tex`, `methodology.tex`, `results.tex`, `conclusion.tex` all use this exact framing. |
| Disclaimer sentence appears next to every AraBERT number | Pass | "This measures domain language modeling fit, not supervised sentiment performance, which remains future work pending Arabic financial sentiment labels." appears in abstract, methodology, results table caption, results prose, and conclusion. |
| Table caption explicitly states same validation split, seed (42), tokenizer, evaluation script | Pass | `Table~\ref{tab:arabert_mlm_eval}` caption lists chronological 80/20 split, seed 42, base AraBERT tokenizer, identical evaluation script. |
| Eval script enforces matching seed and tokenizer | Pass | `scripts/evaluate_arabert_financial_adaptation.py` default seed is now 42 with `torch.cuda.manual_seed_all`; both base and adapted runs load the tokenizer from `args.base_model`. |
| Numbers re-run after script fix | Pass | New evaluation generated `arabert_financial_mlm_evaluation.csv` and report on 2026-05-27 with seed 42: validation loss base 0.9414, adapted 0.8238 (12.49% relative reduction). Thesis tables and prose reference the new numbers. |

## Style and Scientific Mechanics

| Check | Status | Notes |
| --- | --- | --- |
| De-AI writing | Pass | Repetitive openers ("This thesis...", "The final...", "This result...") trimmed; sentences shortened; bullet-list framing inside discussion replaced by prose. |
| Formal academic diction | Pass | Banned informal words (`generally`, `typically`, `absolutely`, `frowned upon`, `usually`, `really`, `huge`, `massive`, `spectacular`, `notable`, `promising`) verified absent by `grep -niE`. |
| Sentence length | Pass | Sentences exceeding two printed lines have been split. |
| Paragraph length | Pass | Long paragraphs split so each paragraph opens with a fresh core idea. |
| Narrative bullet points in discussion | Pass | Bullet-list narrative converted to prose; remaining lists are RQ/objective enumerations only. |
| "in this paper" | Pass | No occurrence. |
| Caption phrasing | Pass | No "following figure/table" or "table below" phrasing found. |

## Verb Tense Discipline

| Chapter Type | Status | Notes |
| --- | --- | --- |
| Abstract future tense only | Pass | Every sentence in `abstract.tex` uses simple future. |
| Introduction and background present tense | Pass | Simple present throughout; perfect tenses removed. |
| Related work past tense | Pass | Surveyed studies described in simple past with citations at end. |
| Methodology present tense | Pass | Simple present; project-history paragraphs rephrased. |
| Experiments present tense | Pass | Results chapter narrates experiments in simple present; reported numbers remain factual. |
| Ban complex/perfect tenses | Pass | `has/have/had been/shown/completed` patterns removed. |

## Formatting, Figures, and Tables

| Check | Status | Notes |
| --- | --- | --- |
| Consecutive headings | Pass | Every chapter has a bridging paragraph before the first section. |
| Title capitalization | Pass | Section and chapter headings are title case. |
| Whitespace | Pass | `pdflatex` builds cleanly with no fatal errors. |
| Figure captions below figures | Pass | `\includefig` macro places `\caption` after `\includegraphics`. |
| Figures cited in text | Pass | Methodology cites both generated figures by `\ref`. |
| Table captions above tables | Pass | All seven tables and the long Experimental Setup `longtable` carry `\caption` before `tabular` / inside `longtable` first row. |
| Native LaTeX tables | Pass | All tables coded in LaTeX. |
| List of abbreviations | Pass | Full alphabetical list of 43 abbreviations with expansions. |

## Citation Syntactic Independence

| Check | Status | Evidence |
| --- | --- | --- |
| "Author et al.\ \cite{...}" pattern | Pass | `grep -nE "et al\."` returns zero hits across all chapter files. |
| "In \cite{...}" pattern | Pass | `grep -nE "In \\\\cite"` returns zero hits. |
| "In [N]" pattern | Pass | No raw `[N]` reference patterns appear in body text. |
| Comparison table rows | Pass | `tab:lit_comparison` rows now use study descriptors (e.g.\ "DJIA mood study \cite{bollen2011twitter}") rather than "Author et al.\ \cite{...}". |
| Sentences remain grammatical without `\cite{}` | Pass | Citations now follow rather than carry sentence subjects. |

## Reproducibility Gap (Chapter 4)

| Required row | Status | Value sourced from |
| --- | --- | --- |
| Python version | Pass | runtime `python3 --version` (3.9.6). |
| OS / kernel | Pass | `sw_vers` / `uname` (macOS 26.5 Tahoe, Darwin 25.5.0, arm64). Darwin 25.x is the correct kernel for macOS 26.x (Apple's kernel version lags product version by one since the Tahoe renaming). |
| CPU / GPU / VRAM | Pass | `sysctl machdep.cpu.brand_string` (Apple M4), `sysctl hw.memsize` (16 GB unified), no NVIDIA GPU. |
| CUDA / cuDNN | Pass | Marked "Not applicable" with explanation. |
| PyTorch / Transformers / scikit-learn / pandas / NumPy versions | Pass | Runtime `python3 -c "import ...; print(...__version__)"`. |
| Datasets version | Pass | Stated as "Not used" (script uses PyTorch `DataLoader`). |
| AraBERT base name and HuggingFace revision SHA | Pass | `aubmindlab/bert-base-arabertv02`, SHA `016fb9d6...`. |
| Random seed | Pass | 42 across Python, NumPy, PyTorch (training and evaluation scripts). |
| Deterministic flags | Pass | `torch.manual_seed(42)`, `np.random.seed(42)`, `torch.cuda.manual_seed_all(42)` set; `torch.use_deterministic_algorithms(True, warn_only=True)` and `CUBLAS_WORKSPACE_CONFIG=:4096:8` added to both training and evaluation scripts. Re-evaluation under flag confirmed numbers unchanged: validation loss base 0.9414, adapted 0.8238, 12.49% relative reduction. |
| Training: epochs, batch size, lr, optimizer, warmup, weight decay | Pass | Pulled from `run_arabert_financial_adaptation_experiment.py` and `run_metadata.json`. |
| LoRA / quantization | Pass | "Not used in any reported run." |

## Confidence Filtering and Daily Aggregation Formulas

| Check | Status | Evidence |
| --- | --- | --- |
| Confidence filter expressed as numbered equation | Pass | Equation (1) `confidence_filter` with $\tau =$ medium. |
| Daily aggregation expressed as numbered equation | Pass | Equation (4) `daily_aggregation` with weighted sum and missingness clause. |
| `sign_i` defined | Pass | Equation (3) `article_sign`. |
| Behaviour at denominator = 0 | Pass | Explicit zero default plus missingness flag set. |
| `tau` and lag agree with `run_config.json` | Pass | Code default `min_match_confidence = "medium"`; lag derived from `sentiment_lag_calendar_days=1` in `outputs/sentiment_stock_impact_analysis/run_config.json` and `lag_strategy="previous_day"` in pipeline. |
| Formulas numbered and referenced by number | Pass | Equations (1)-(6) referenced in surrounding prose and again in chapter summary. |

## Chapter 1: Introduction

| Check | Status | Notes |
| --- | --- | --- |
| Narrative flow | Pass | Broader value → core challenges → literature limitations. |
| Grounding and citations | Pass | Each motivating paragraph carries citations. |
| Dedicated problem statement | Pass | Uses the prescribed frame "This thesis investigates [X] in order to determine [Y] for [Z]." |
| Motivation section | Pass | Two paragraphs explaining specific approach and why it improves on prior work. |
| Research questions count | Pass | Three RQs (within 2-3 range). |
| Objectives count | Pass | Six objectives (within 3-6 range). |
| RQ-objective alignment | Pass | Each RQ maps to at least one objective; no orphan objectives. |
| Assertive verbs in objectives | Pass | "Design", "Develop", "Construct", "Evaluate", "Implement", "Analyse". |

## Chapter 2: Literature Review

| Check | Status | Notes |
| --- | --- | --- |
| Foundational background before paper summaries | Pass | Opening paragraph announces the four threads and chapter organisation. |
| Reference volume | Pass | 38 citation usages in literature review; 36 unique keys across full thesis. |
| IEEE numeric citations | Pass | `ieeetr` bibliography style. |
| Citation syntactic independence | Pass | All "Author et al." subjects removed. |
| Thematic storytelling | Pass | Sections grouped by social media, financial news, failure modes, Arabic/domain models. |
| Quantifiable critique | Pass | Studies cited with sample sizes, horizons, datasets, reported metrics. |
| Analytical comparison matrix | Pass | `tab:lit_comparison` with caption above, study/market/text/model/result/limitation columns, gap-bridge paragraph after the table. |
| Chapter summary | Pass | Dedicated summary plus transition. |

## Chapter 3: Methodology

| Check | Status | Notes |
| --- | --- | --- |
| RQ/objective alignment | Pass | Chapter sections map back to bilingual routing, language-aware aggregation, leakage controls, and auxiliary protocols. |
| Introductory foundation | Pass | Opens with a scope/motivation paragraph and ends with chapter summary. |
| High-level architecture first | Pass | Research design and pipeline figure precede component-level details. |
| Process formalisation | Pass | Numbered equations (1)-(6) govern filtering, weighting, signing, aggregation, return, and direction. |
| Model boundaries | Pass | FinBERT and AraBERT roles are explained as components; results remain in Chapter 4. |
| Parameter process without exact values | Pass | Hyperparameter grids and concrete numerical settings deferred to Chapter 4. |
| Mathematical notation | Pass | Six numbered equations, each referenced by number in surrounding prose. |
| Chapter summary | Pass | Summary + transition sentence. |

## Chapter 4: Implementation, Setup, and Results

| Check | Status | Notes |
| --- | --- | --- |
| Experimental Setup subsection first | Pass | New `\section{Experimental Setup}` with 35-row longtable. |
| Data characterisation | Pass | Final dataset counts and event-labelled subset reported. |
| Confidence filtering definition | Pass | Confidence filter and daily aggregation defined in Chapter 3 with exact $\tau$ and lag; Chapter 4 cross-references them in Table~\ref{tab:exp_setup}. |
| Daily aggregation mechanics | Pass | Numbered equations (1)-(4) cover the full mechanism. |
| No raw code/screenshots | Pass | No code blocks or terminal screenshots in body text. |
| Results tables/statistics | Pass | Six tables, all with caption above, all numbered and cross-referenced. |
| Critical analytical discussion | Pass | `\section{Interpretation}` explains why F1 inflates over balanced accuracy, why coverage explains panel-level instability, and why the overfit candidate is rejected. |
| Chapter summary | Pass | Dedicated summary at the end. |

## Chapter 5 / Conclusion

| Check | Status | Notes |
| --- | --- | --- |
| Direct answers to each RQ | Pass | Conclusion explicitly references RQ1, RQ2, and RQ3 with the relevant evidence from Chapter 4. |
| Future work concrete | Pass | Five concrete directions: coverage, Arabic supervised labels, controlled price+sentiment integration, richer event modelling, official-access social media. |
| No new results introduced | Pass | Every quantitative statement points back to Chapter 4 tables and validity reports. |

## Submission Polish and Declarations

| Check | Status | Notes |
| --- | --- | --- |
| Original work declaration | Pass | Retained on `GUC_TitlePage.tex`. |
| AI assistance declaration | Pass | Exact specified paragraph added under the certification list. |
| Submission date | Pass (with TODO) | Set to `15 June 2026` with explicit LaTeX TODO comment for supervisor confirmation. |

## Verification Pass

| Step | Result |
| --- | --- |
| Citation-missing scan | No undefined citations in `bachelor.log`. |
| Placeholder scan | One intentional TODO (submission date confirmation); deterministic flags TODO resolved; no `XXX`/`FIXME`/`Placeholder`/`TBD` artifacts. |
| `pdflatex` compile | Builds with zero fatal errors; 55 pages, 615 688 bytes. |
| `bibtex` compile | Builds with no errors. |
| Multi-pass compile chain (`pdflatex + bibtex + pdflatex + pdflatex`) | Final output stable; no further rerun needed. |
| Unit tests (`python3 -m unittest discover -s tests -v`) | 6 of 6 pass. |
| AraBERT MLM re-evaluation under seed 42 with shared base tokenizer | Generated `arabert_financial_mlm_evaluation.csv` and matching markdown report; new numbers propagated into Chapter 4 table, Chapter 4 prose, AraBERT row of auxiliary summary table, conclusion future-work paragraph, and abstract. |
| AraBERT MLM re-evaluation under deterministic flag | Re-run after adding `torch.use_deterministic_algorithms(True, warn_only=True)` confirmed numbers unchanged: validation loss base 0.9414, adapted 0.8238, perplexity base 2.564, adapted 2.279. |
| OS version consistency check | `sw_vers` + `uname -r` + `system_profiler` confirm macOS 26.5 Tahoe (Darwin 25.5.0) is internally consistent. Darwin 25.x is the correct kernel for macOS 26.x. Table entry updated to include "Tahoe" and the versioning note. |

## 2026-05-31 Session: Incremental Scrape + Experiment Re-run

| Step | Result |
| --- | --- |
| Incremental Mubasher scrape | 17 new raw records (CIB+8, TMG+7, OIH+2); 9 passed medium-confidence entity filter; total corpus grew from 1,912 to 1,921 articles. |
| `rebuild_processed_news.py` | New bypass script re-processes all raw JSONL → `prepare_sentiment_articles` → `score_prepared_articles`; produces identical scores for all 1,912 pre-existing articles (max diff = 0.0). |
| Multi-horizon experiments re-run | `run_multi_horizon_event_evaluation.py` completed all 15 combinations (H1/3/5 × all_news-en/ar/bi, event_only-bi, event_weighted-bi) on 1,921 articles; outputs saved to `outputs/final_sentiment_experiments/`. |
| Dataset counts updated | `methodology.tex` tab:dataset_counts, `results.tex` tab:results_dataset_counts, `abstract.tex`, and `conclusion.tex` all updated to 1,921 / 1,730. |
| Metric tables updated | `tab:main_results` (all 9 rows), `tab:event_summary` (both rows) updated to new values. |
| Accepted candidate updated | OIH H5 all_news bilingual: val acc=0.524, val ba=0.501, val F1=0.057; test acc=0.425, test ba=0.503, test F1=0.021, sign acc=0.585. Near-zero F1 explained as threshold-induced class collapse; sign accuracy is the informative metric. |
| Event aggregate sign change | Event-only mean delta: −0.013 → +0.025; event-weighted: −0.012 → +0.032 (7 of 9 panels positive under both strategies; gains concentrated in a few large cases). |
| OIH coverage updated | Prior-7-day coverage: 0.305 → 0.307 (regenerated by phase7 re-run). |
| Phase 7 validity re-run | `run_phase7_validity_checks.py --skip-live-page-counts`; leakage PASS (45 panels, 43,880 rows, 0 violations); overfitting flags 14 of 45 (was 15). |
| Metric extraction script | `scripts/extract_paper_metrics.py` created for systematic metric extraction from CSVs. |
| `.gitignore` additions | Added `*.egg-info/`, `*.egg-info`, and `data/processed/news/sentiment_ready/*.jsonl`. |
| PDF recompile | 55 pages, 615,688 bytes, zero errors. |
| Unit tests | 6 of 6 pass. |

## Outstanding Residual TODOs (Intentional)

1. `bachelor.tex` line 27-29: confirm final submission date with supervisor before binding.
