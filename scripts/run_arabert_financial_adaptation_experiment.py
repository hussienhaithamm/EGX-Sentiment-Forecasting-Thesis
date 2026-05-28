"""Prepare and optionally run AraBERT financial-domain adaptation.

The valid analogue to FinBERT has two stages:
1. Domain-adaptive pre-training on Arabic financial text with masked language modeling.
2. Supervised sentiment fine-tuning on human-labeled Arabic financial sentiment data.

This repository has Arabic financial news text, but not human sentiment labels.
The script therefore always prepares the DAPT corpus and refuses supervised
fine-tuning unless a human-labeled CSV is explicitly supplied.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd


ARABIC_DIACRITICS_RE = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
ARABIC_NORMALIZATION_MAP = str.maketrans(
    {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ى": "ي",
        "ؤ": "و",
        "ئ": "ي",
        "ة": "ه",
        "ـ": "",
    }
)

FINANCIAL_TERMS: Dict[str, Sequence[str]] = {
    "earnings": ("ارباح", "صافي الربح", "نتائج الاعمال", "الايرادات", "المبيعات", "الخسائر"),
    "dividends": ("توزيعات", "كوبون", "ارباح نقديه", "توزيع ارباح"),
    "disclosure_governance": ("افصاح", "مجلس الاداره", "الجمعيه العامه", "القوائم الماليه"),
    "financing": ("تمويل", "قرض", "تسهيلات ائتمانيه", "سندات", "اسهم خزينة"),
    "investment_projects": ("استثمار", "توسع", "مشروع", "عقد", "اتفاقيه"),
    "market_terms": ("البورصه", "التداول", "السهم", "المؤشر", "القيمه السوقيه"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare AraBERT Arabic financial adaptation corpus.")
    parser.add_argument(
        "--sentiment-labeled-path",
        default="data/processed/news/sentiment_ready/sentiment_labeled_articles.csv",
    )
    parser.add_argument("--output-dir", default="outputs/arabert_financial_adaptation")
    parser.add_argument("--base-model", default="aubmindlab/bert-base-arabertv02")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--mlm-probability", type=float, default=0.15)
    parser.add_argument("--run-mlm-training", action="store_true")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--human-labeled-csv",
        default=None,
        help="Optional CSV with Arabic financial text and human labels. Without this, supervised sentiment fine-tuning is not run.",
    )
    return parser.parse_args()


def normalize_arabic(text: object) -> str:
    value = "" if text is None or (isinstance(text, float) and math.isnan(text)) else str(text)
    value = ARABIC_DIACRITICS_RE.sub("", value)
    value = value.translate(ARABIC_NORMALIZATION_MAP)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def load_arabic_articles(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    language = frame.get("detected_language", frame.get("article_language", "")).fillna("").astype(str).str.lower()
    arabic = frame[language.eq("ar")].copy()
    arabic["published_timestamp"] = pd.to_datetime(arabic["published_timestamp"], errors="coerce", utc=True)
    text_columns = ["clean_article_text", "cleaned_title", "cleaned_body_text", "title", "body_text"]
    texts: List[str] = []
    for _, row in arabic.iterrows():
        parts: List[str] = []
        for column in text_columns:
            if column not in arabic.columns:
                continue
            text = normalize_arabic(row.get(column, ""))
            if text and text not in parts:
                parts.append(text)
        texts.append("\n\n".join(parts))
    arabic["adaptation_text"] = texts
    arabic = arabic[arabic["adaptation_text"].str.len() >= 40].copy()
    return arabic.sort_values("published_timestamp").reset_index(drop=True)


def term_coverage(arabic: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    normalized_texts = arabic["adaptation_text"].map(normalize_arabic)
    for category, terms in FINANCIAL_TERMS.items():
        for term in terms:
            normalized_term = normalize_arabic(term)
            contains = normalized_texts.str.contains(re.escape(normalized_term), regex=True, na=False)
            rows.append(
                {
                    "category": category,
                    "term": term,
                    "article_count": int(contains.sum()),
                    "article_share": float(contains.mean()) if len(contains) else 0.0,
                }
            )
    return pd.DataFrame(rows).sort_values(["article_count", "category"], ascending=[False, True])


def write_corpus_split(arabic: pd.DataFrame, output_dir: Path) -> Dict[str, object]:
    corpus_dir = output_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    split_idx = int(len(arabic) * 0.8)
    train = arabic.iloc[:split_idx].copy()
    validation = arabic.iloc[split_idx:].copy()
    train_path = corpus_dir / "arabic_financial_mlm_train.txt"
    val_path = corpus_dir / "arabic_financial_mlm_validation.txt"
    train_path.write_text("\n\n".join(train["adaptation_text"].tolist()) + "\n", encoding="utf-8")
    val_path.write_text("\n\n".join(validation["adaptation_text"].tolist()) + "\n", encoding="utf-8")
    return {
        "train_path": str(train_path),
        "validation_path": str(val_path),
        "train_articles": int(len(train)),
        "validation_articles": int(len(validation)),
        "first_article": arabic["published_timestamp"].min().isoformat() if len(arabic) else "",
        "last_article": arabic["published_timestamp"].max().isoformat() if len(arabic) else "",
    }


def _run_mlm_training(arabic: pd.DataFrame, args: argparse.Namespace, output_dir: Path) -> Dict[str, object]:
    try:
        import numpy as np
        import torch
        from torch.utils.data import Dataset
        from torch.utils.data import DataLoader
        from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling
    except Exception as exc:
        return {"status": "SKIPPED", "reason": f"Training dependencies unavailable: {type(exc).__name__}: {exc}"}

    class TextDataset(Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __len__(self):
            return len(self.encodings["input_ids"])

        def __getitem__(self, idx):
            return {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}

    torch.manual_seed(42)
    np.random.seed(42)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)
    split_idx = int(len(arabic) * 0.8)
    train_texts = arabic.iloc[:split_idx]["adaptation_text"].tolist()
    val_texts = arabic.iloc[split_idx:]["adaptation_text"].tolist()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForMaskedLM.from_pretrained(args.base_model)
    train_enc = tokenizer(train_texts, truncation=True, padding="max_length", max_length=args.max_length)
    val_enc = tokenizer(val_texts, truncation=True, padding="max_length", max_length=args.max_length)
    train_dataset = TextDataset(train_enc)
    val_dataset = TextDataset(val_enc)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=args.mlm_probability)

    model_dir = output_dir / "arabert_financial_mlm"
    model_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)

    losses: List[float] = []
    step = 0
    while step < args.max_steps:
        for batch in train_loader:
            if step >= args.max_steps:
                break
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))
            step += 1

    model.eval()
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    eval_losses: List[float] = []
    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            if idx >= 25:
                break
            batch = {key: value.to(device) for key, value in batch.items()}
            eval_losses.append(float(model(**batch).loss.detach().cpu()))

    model.save_pretrained(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    return {
        "status": "COMPLETED",
        "model_dir": str(model_dir),
        "train_loss": float(np.mean(losses)) if losses else float("nan"),
        "eval_loss": float(np.mean(eval_losses)) if eval_losses else float("nan"),
        "max_steps": int(args.max_steps),
        "device": str(device),
    }


def write_report(
    arabic: pd.DataFrame,
    coverage: pd.DataFrame,
    corpus_meta: Dict[str, object],
    mlm_result: Dict[str, object],
    args: argparse.Namespace,
    output_path: Path,
) -> None:
    top_terms = coverage.head(15)
    lines = [
        "# AraBERT Financial Adaptation Experiment",
        "",
        "This experiment prepares the Arabic side of the thesis corpus for finance-domain adaptation of AraBERT. The valid FinBERT analogue is domain-adaptive masked-language-model pre-training followed by supervised sentiment fine-tuning on human-labeled Arabic financial sentiment examples.",
        "",
        "The current thesis dataset does not contain independent human sentiment labels. Existing article labels are model outputs from the current AraBERT route, so using them as ground truth would only distill the same model and would not be valid evidence of improved Arabic financial sentiment classification.",
        "",
        "## Corpus",
        "",
        f"- Arabic financial articles: `{len(arabic)}`",
        f"- Chronological MLM train articles: `{corpus_meta['train_articles']}`",
        f"- Chronological MLM validation articles: `{corpus_meta['validation_articles']}`",
        f"- Date range: `{corpus_meta['first_article']}` to `{corpus_meta['last_article']}`",
        "",
        "## Frequent Financial Terms",
        "",
        "| Category | Term | Articles | Share |",
        "| --- | --- | ---: | ---: |",
    ]
    for _, row in top_terms.iterrows():
        lines.append(f"| {row['category']} | {row['term']} | {int(row['article_count'])} | {row['article_share']:.3f} |")

    lines.extend(
        [
            "",
            "## MLM Adaptation Status",
            "",
            f"- Status: `{mlm_result.get('status')}`",
        ]
    )
    if mlm_result.get("status") == "COMPLETED":
        lines.extend(
            [
                f"- Model directory: `{mlm_result.get('model_dir')}`",
                f"- Train loss: `{mlm_result.get('train_loss'):.4f}`",
                f"- Evaluation loss: `{mlm_result.get('eval_loss'):.4f}`",
                f"- Max steps: `{mlm_result.get('max_steps')}`",
            ]
        )
    else:
        lines.append(f"- Reason: `{mlm_result.get('reason', 'MLM training was not requested.')}`")

    lines.extend(["", "## Supervised Sentiment Fine-Tuning Status", ""])
    if args.human_labeled_csv:
        lines.append(
            f"A human-labeled CSV was supplied at `{args.human_labeled_csv}`. The next step is to train a three-class classifier with chronological or source-separated validation."
        )
    else:
        lines.append(
            "Not run. A valid supervised experiment requires human-labeled Arabic financial sentiment examples with positive, negative, and neutral labels. Pseudo-labels from the current AraBERT classifier are not accepted as ground truth."
        )

    lines.extend(
        [
            "",
            "## Paper Interpretation",
            "",
            "This should be reported as a prepared domain-adaptation protocol and feasibility result, not as a completed Arabic FinBERT equivalent. The available corpus is useful for financial terminology exposure, but the thesis still needs an independently labeled Arabic financial sentiment set before making supervised performance claims.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    arabic = load_arabic_articles(Path(args.sentiment_labeled_path))
    coverage = term_coverage(arabic)
    corpus_meta = write_corpus_split(arabic, output_dir)
    coverage.to_csv(output_dir / "arabic_financial_term_coverage.csv", index=False)
    arabic[["canonical_ticker", "published_timestamp", "adaptation_text"]].to_csv(
        output_dir / "arabic_financial_adaptation_articles.csv",
        index=False,
    )

    if args.run_mlm_training:
        mlm_result = _run_mlm_training(arabic, args, output_dir)
    else:
        mlm_result = {"status": "SKIPPED", "reason": "MLM training was not requested."}

    metadata = {
        "args": vars(args),
        "corpus": corpus_meta,
        "mlm_result": mlm_result,
        "supervised_finetuning_status": (
            "requires_human_labels" if args.human_labeled_csv is None else "human_labeled_csv_supplied"
        ),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    write_report(arabic, coverage, corpus_meta, mlm_result, args, output_dir / "arabert_financial_adaptation_report.md")
    print("Saved AraBERT financial adaptation artifacts to", output_dir)


if __name__ == "__main__":
    main()
