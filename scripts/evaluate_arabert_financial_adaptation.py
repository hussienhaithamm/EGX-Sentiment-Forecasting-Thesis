"""Evaluate the AraBERT financial-domain MLM checkpoint.

This is not a supervised sentiment evaluation. The adapted checkpoint was
trained with masked language modeling, so the valid direct test is masked-token
loss/perplexity on held-out Arabic financial news text. A supervised sentiment
test requires independent human labels and a classification head.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate AraBERT financial MLM adaptation.")
    parser.add_argument(
        "--articles-path",
        default="outputs/arabert_financial_adaptation/arabic_financial_adaptation_articles.csv",
    )
    parser.add_argument("--adapted-model-dir", default="outputs/arabert_financial_adaptation/arabert_financial_mlm")
    parser.add_argument("--base-model", default="aubmindlab/bert-base-arabertv02")
    parser.add_argument("--output-dir", default="outputs/arabert_financial_adaptation")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--mlm-probability", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=0, help="0 means evaluate the full split.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed shared across python, numpy, and torch. Must match the training-script seed (42) so the masked-token positions are identical between base and adapted runs.",
    )
    return parser.parse_args()


def load_splits(path: Path) -> Dict[str, List[str]]:
    frame = pd.read_csv(path)
    if "published_timestamp" in frame.columns:
        frame["published_timestamp"] = pd.to_datetime(frame["published_timestamp"], errors="coerce", utc=True)
        frame = frame.sort_values("published_timestamp")
    texts = frame["adaptation_text"].fillna("").astype(str)
    texts = texts[texts.str.len() >= 40].tolist()
    split_idx = int(len(texts) * 0.8)
    return {"train": texts[:split_idx], "validation": texts[split_idx:]}


def evaluate_model(
    *,
    model_name_or_path: str,
    tokenizer_name_or_path: str,
    texts: Iterable[str],
    max_length: int,
    mlm_probability: float,
    batch_size: int,
    max_batches: int,
    seed: int,
) -> Dict[str, float]:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling

    class TextDataset(Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __len__(self) -> int:
            return len(self.encodings["input_ids"])

        def __getitem__(self, idx: int):
            return {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}

    text_list = list(texts)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    model = AutoModelForMaskedLM.from_pretrained(model_name_or_path)
    encodings = tokenizer(text_list, truncation=True, padding="max_length", max_length=max_length)
    dataset = TextDataset(encodings)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=mlm_probability)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    losses: List[float] = []
    masked_tokens = 0
    examples = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches and batch_idx >= max_batches:
                break
            labels = batch["labels"]
            masked_tokens += int((labels != -100).sum().item())
            examples += int(labels.shape[0])
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss
            losses.append(float(loss.detach().cpu()))

    mean_loss = float(np.mean(losses)) if losses else float("nan")
    return {
        "examples": float(examples),
        "masked_tokens": float(masked_tokens),
        "loss": mean_loss,
        "perplexity": float(math.exp(mean_loss)) if math.isfinite(mean_loss) else float("nan"),
        "device": str(device),
    }


def write_report(results: pd.DataFrame, output_path: Path, metadata: Dict[str, object]) -> None:
    lines = [
        "# AraBERT Financial MLM Evaluation",
        "",
        "This evaluates the saved finance-adapted AraBERT checkpoint as a masked-language model on the Arabic financial article corpus.",
        "",
        "This is not supervised sentiment accuracy. The checkpoint has no sentiment-classification head, and the thesis dataset does not contain independent human sentiment labels.",
        "",
        "## Configuration",
        "",
        f"- Base model: `{metadata['base_model']}`",
        f"- Adapted model: `{metadata['adapted_model_dir']}`",
        f"- MLM probability: `{metadata['mlm_probability']}`",
        f"- Max sequence length: `{metadata['max_length']}`",
        f"- Batch size: `{metadata['batch_size']}`",
        f"- Seed: `{metadata['seed']}`",
        "",
        "## Results",
        "",
        "| Split | Model | Examples | Masked tokens | Loss | Perplexity |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in results.sort_values(["split", "model"]).iterrows():
        lines.append(
            f"| {row['split']} | {row['model']} | {int(row['examples'])} | {int(row['masked_tokens'])} | "
            f"{row['loss']:.4f} | {row['perplexity']:.3f} |"
        )

    validation = results[results["split"].eq("validation")]
    base = validation[validation["model"].eq("base_arabert")]
    adapted = validation[validation["model"].eq("financial_adapted_arabert")]
    lines.extend(["", "## Interpretation", ""])
    if not base.empty and not adapted.empty:
        base_loss = float(base.iloc[0]["loss"])
        adapted_loss = float(adapted.iloc[0]["loss"])
        delta = adapted_loss - base_loss
        rel = (base_loss - adapted_loss) / base_loss if base_loss else float("nan")
        if delta < 0:
            lines.append(
                f"The adapted checkpoint lowers validation MLM loss by {-delta:.4f} relative to the base AraBERT model, a relative reduction of {rel:.2%}."
            )
        else:
            lines.append(
                f"The adapted checkpoint does not lower validation MLM loss in this run; validation loss changes by {delta:.4f} relative to the base AraBERT model."
            )
    lines.append(
        "This result can support a domain-adaptation feasibility claim only. It does not prove better stock forecasting or better sentiment classification."
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = load_splits(Path(args.articles_path))

    rows: List[Dict[str, object]] = []
    for split, texts in splits.items():
        for model_label, model_path, tokenizer_path in (
            ("base_arabert", args.base_model, args.base_model),
            ("financial_adapted_arabert", args.adapted_model_dir, args.base_model),
        ):
            result = evaluate_model(
                model_name_or_path=model_path,
                tokenizer_name_or_path=tokenizer_path,
                texts=texts,
                max_length=args.max_length,
                mlm_probability=args.mlm_probability,
                batch_size=args.batch_size,
                max_batches=args.max_batches,
                seed=args.seed,
            )
            rows.append({"split": split, "model": model_label, **result})

    results = pd.DataFrame(rows)
    results_path = output_dir / "arabert_financial_mlm_evaluation.csv"
    report_path = output_dir / "arabert_financial_mlm_evaluation_report.md"
    metadata = vars(args).copy()
    (output_dir / "arabert_financial_mlm_evaluation_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
    results.to_csv(results_path, index=False)
    write_report(results, report_path, metadata)
    print(f"Saved MLM evaluation to {report_path}")


if __name__ == "__main__":
    main()
