# Master Thesis Evaluation Checklist Audit

Date: 2026-05-27

Scope: `Bachelor_Thesis_Template/*.tex`, current compiled `bachelor.pdf`, and supporting bibliography metadata.

## Overall Verdict

Current status: **Partial Pass / Not Submission-Ready Under This Checklist**

The thesis passes the broad scientific structure: it has a motivated introduction, extensive background and related work, a reproducible methodology, reported experiments, critical interpretation, and future work. It does not yet pass the stricter polish checklist because several mechanical and institutional requirements remain unresolved.

## High-Priority Failures

| Area | Status | Evidence | Required Fix |
| --- | --- | --- | --- |
| Abstract tense rule | Fail | `abstract.tex` uses present and past-tense result reporting. | Rewrite abstract in future tense if this checklist is mandatory. |
| Complex/perfect tense ban | Fail | Examples include `has been made`, `has long treated`, `has evolved`, `were executed`, and other non-simple forms. | Rewrite tense across chapters or relax this rule. |
| Table caption placement | Fail | All native tables currently place `\caption` below `tabular`; checklist requires captions above tables. | Move every table caption before the `tabular` environment. |
| Abbreviation list | Fail | `appendix.tex` still contains placeholder acronyms `AC` and `AC2`; many real acronyms are missing. | Replace with full acronym list: EGX, CIB, TMG, OIH, NLP, BERT, MLM, etc. |
| Declaration AI disclosure | Fail | `GUC_TitlePage.tex` states original work and acknowledgement only; no generative-AI disclosure exists. | Add a transparent statement describing AI assistance scope. |
| Submission date | Fail / Needs confirmation | `bachelor.tex` sets submission date to `31 March 2026`; current audit date is `27 May 2026`. | Update to the final actual submission date. |
| Consecutive heading rule | Fail | `chapter` titles are followed immediately by first `section` headings in multiple files. | Add a short chapter-opening paragraph before the first section. |
| Environment reproducibility detail | Partial / Fail | `results.tex` lists package families and Python 3.9+, but not exact installed versions, IDE, hardware, or deterministic seeds. | Add exact Python version, package versions, CPU/GPU details, and seed policy. |

## Style and Scientific Mechanics

| Check | Status | Notes |
| --- | --- | --- |
| De-AI writing | Partial | The text is coherent and academic, but repeated patterns remain: "This thesis...", "The final...", "This result..." and repeated conservative-result phrasing. |
| Formal academic diction | Partial | No severe informal terms dominate, but words such as `useful`, `strong`, `weak`, and `cleaner` appear frequently. They should be replaced or tied to metrics. |
| Sentence length | Fail | Many sentences exceed 35 words, including abstract, introduction, methodology, results, and conclusion. |
| Paragraph length | Mostly Pass | Paragraphs are generally manageable; the abstract and some results paragraphs are dense. |
| Narrative bullet points | Mostly Pass | Primary chapters mostly avoid bullets. Introduction uses RQ/objective list format, which is acceptable if the template permits it. |
| Phrase "in this paper" | Pass | No occurrence found. |
| Caption phrasing | Pass | No "following figure/table" or "table below" phrasing found. |

## Verb Tense Discipline

| Chapter Type | Status | Notes |
| --- | --- | --- |
| Abstract future tense only | Fail | Current abstract summarizes completed work and results. |
| Introduction and background present tense | Partial | Mostly present, but contains past/perfect forms where discussing project emergence and prior shifts. |
| Related work past tense | Partial | Many prior-work sentences use past tense, but some current-tense interpretation remains. |
| Methodology present tense | Partial | Mostly present, but project-history paragraphs use past tense. |
| Experiments present tense | Partial | Results chapter naturally uses past tense for completed experiments; this conflicts with the checklist rule. |
| Ban complex/perfect tenses | Fail | Multiple `has/have/were/was` constructions remain. |

## Formatting, Figures, and Tables

| Check | Status | Notes |
| --- | --- | --- |
| Consecutive headings | Fail | Detected in `background.tex`, `literature_review.tex`, `methodology.tex`, `results.tex`, and `conclusion.tex`. |
| Title capitalization | Pass | Section and chapter headings are title case. |
| Whitespace | Partial | LaTeX compiles. Full visual PDF inspection in Overleaf is still needed. |
| Figure captions below figures | Pass | Custom `\includefig` macro places captions below images. |
| Figures cited in text | Pass | Methodology cites both generated figures before inclusion. |
| Table captions above tables | Fail | All table captions are below. |
| Native LaTeX tables | Pass | Tables are coded in LaTeX; no table screenshot detected. |
| List of abbreviations | Fail | Placeholder abbreviation list only. |

## Chapter 1: Introduction

| Check | Status | Notes |
| --- | --- | --- |
| Narrative flow | Pass | The chapter moves from market value to information asymmetry, prior research, bilingual EGX gap, and project scope. |
| Grounding and citations | Pass | Introduction includes citations across the motivating paragraphs. |
| Dedicated problem statement | Pass | The problem statement closely matches the target template. |
| Motivation section | Pass | Motivation explains industrial context and sentiment-only design. |
| Research questions count | Partial | Checklist asks 2 to 3 RQs; current thesis has 4 RQs. |
| Objectives count | Pass | Current thesis has 6 objectives, within the requested 3 to 6 range. |
| RQ-objective alignment | Pass | All RQs map to one or more objectives. |

## Chapter 2: Literature Review and Related Work

| Check | Status | Notes |
| --- | --- | --- |
| Background foundation before papers | Pass | `background.tex` provides foundations before `literature_review.tex`. If the evaluator expects one combined chapter, chapter naming may need adjustment. |
| Reference volume | Pass | Bibliography has 38 entries; literature/background cite 27 distinct sources each. |
| IEEE numeric citations | Pass after compile | `ieeetr` outputs numeric bracket citations. |
| Citation syntactic independence | Partial / Fail | The comparison table uses `Bollen et al.\ \cite{...}` style, which violates the strict wording in the checklist. |
| Thematic storytelling | Pass | Sections are grouped by social media, financial news, failure modes, and Arabic/domain models. |
| Quantifiable critique | Pass | Many studies include sample sizes, horizons, datasets, and reported metrics. |
| Analytical comparison matrix | Partial | A comparison table exists, but it is study-oriented rather than a strict technical-dimension matrix. |
| Chapter summary | Pass | Dedicated summary exists and transitions to methodology. |

## Chapter 3: Methodology

| Check | Status | Notes |
| --- | --- | --- |
| RQ/objective alignment | Pass | Methodology maps to data, sentiment routing, aggregation, evaluation, and auxiliary protocols. |
| Introductory foundation | Partial | Starts with project context, but chapter heading is followed immediately by section heading. |
| High-level architecture first | Pass | Research design and pipeline figure appear before component details. |
| Evaluation design section | Not Applicable | No direct user-interaction study. |
| Process formalization | Partial / Fail | Processes are prose-based. No pseudocode algorithms are presented. |
| Model boundaries | Pass | FinBERT/AraBERT roles are explained as components; results stay in Chapter 5. |
| Parameter process without exact values | Partial | Some dataset counts appear in methodology tables; hyperparameter details mainly stay in results. |
| Prompt engineering | Not Applicable | No prompt-based LLM component is core to the thesis pipeline. |
| Mathematical notation | Pass | Return and direction formulas are numbered and explained. |
| Chapter summary | Pass | Summary and transition exist. |

## Chapter 4: Implementation, Setup, and Results

| Check | Status | Notes |
| --- | --- | --- |
| Exact environment | Fail | Exact package versions, IDE, hardware, GPU/CPU allocations, and deterministic seeds are missing. |
| Data characterization | Pass | Dataset counts and experiment settings are reported. |
| Confidence filtering definition | Fail | Confidence filtering is mentioned but no exact mathematical boundary appears. |
| Daily aggregation mechanics | Partial | Aggregation is described in methodology, but not with implementation-level formulas and conflict rules. |
| No raw code/screenshots | Pass | No source code snippets or terminal screenshots appear in the thesis text. |
| Results tables/statistics | Pass | Results, deltas, validity checks, and MLM evaluation appear in tables/prose. |
| Critical analytical discussion | Pass | Interpretation discusses class skew, coverage sparsity, overfitting, and weak signal. |
| Chapter summary | Pass | Dedicated summary exists. |

## Submission Polish and Declarations

| Check | Status | Notes |
| --- | --- | --- |
| Original work declaration | Pass | Present in `GUC_TitlePage.tex`. |
| AI assistance declaration | Fail | Missing. |
| Submission date | Fail / Needs confirmation | Currently set to `31 March 2026`, likely not final. |

## Recommended Fix Order

1. Replace the placeholder abbreviation appendix with the real abbreviation list.
2. Add AI-assistance disclosure to the declaration page.
3. Update the submission date.
4. Move all table captions above tables.
5. Add chapter-opening paragraphs before the first section in each chapter.
6. Add exact reproducibility environment details and seed policy to the results chapter.
7. Add confidence-filtering and daily-aggregation formulas or algorithms.
8. Decide whether to obey the strict tense rule. If yes, rewrite abstract and tense across chapters.
9. Shorten long sentences and reduce repeated phrasing.
10. Convert the literature comparison table into a stronger technical matrix if the evaluator expects a dimension-by-method table.
