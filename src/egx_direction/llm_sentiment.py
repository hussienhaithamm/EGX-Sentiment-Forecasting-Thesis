"""Cached zero-shot LLM sentiment scoring for the EGX30 market-mood baseline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import pandas as pd


DEFAULT_LLM_MODEL = "gpt-4o-mini"
DEFAULT_PROMPT_VERSION = "egx30_market_sentiment_v1"
LLM_CACHE_COLUMNS = (
    "article_id",
    "llm_input_sha256",
    "llm_prompt_version",
    "llm_provider",
    "llm_model",
    "llm_sentiment_label",
    "llm_positive_probability",
    "llm_negative_probability",
    "llm_neutral_probability",
    "llm_sentiment_score",
    "llm_confidence",
    "llm_generated_at",
    "llm_response_id",
)


@dataclass(frozen=True)
class LlmSentimentResult:
    label: str
    positive_probability: float
    negative_probability: float
    neutral_probability: float
    response_id: str = ""


def build_llm_article_text(row: Mapping[str, object], *, max_characters: int = 6000) -> str:
    """Build deterministic article text for hashing and zero-shot inference."""

    title = str(row.get("cleaned_title", row.get("title", "")) or "").strip()
    body = str(
        row.get(
            "clean_article_text",
            row.get("cleaned_body_text", row.get("body_text", "")),
        )
        or ""
    ).strip()
    text = "\n\n".join(value for value in (title, body) if value)
    return text[:max_characters]


def llm_input_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_llm_sentiment_result(payload: Mapping[str, object], *, response_id: str = "") -> LlmSentimentResult:
    """Validate and normalize one structured zero-shot sentiment response."""

    probabilities = {
        "positive": float(payload["positive_probability"]),
        "negative": float(payload["negative_probability"]),
        "neutral": float(payload["neutral_probability"]),
    }
    if any(value < 0.0 or value > 1.0 for value in probabilities.values()):
        raise ValueError(f"LLM sentiment probabilities must be between 0 and 1: {probabilities}")
    total = sum(probabilities.values())
    if total <= 0.0:
        raise ValueError("LLM sentiment probabilities must have a positive sum.")
    probabilities = {label: value / total for label, value in probabilities.items()}
    label = max(probabilities, key=probabilities.get)
    return LlmSentimentResult(
        label=label,
        positive_probability=probabilities["positive"],
        negative_probability=probabilities["negative"],
        neutral_probability=probabilities["neutral"],
        response_id=response_id,
    )


class OpenAIZeroShotSentimentClassifier:
    """Classify bilingual financial articles with OpenAI structured outputs."""

    def __init__(self, *, model: str = DEFAULT_LLM_MODEL, client: object = None) -> None:
        self.model = model
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self.client = client

    def classify(self, article_text: str) -> LlmSentimentResult:
        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "Classify the likely market sentiment expressed by the supplied Egyptian financial-news "
                        "article. Use only the article text. Return probabilities for positive, negative, and "
                        "neutral sentiment. Do not infer later market outcomes."
                    ),
                },
                {"role": "user", "content": article_text},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "egx_market_sentiment",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "positive_probability": {"type": "number"},
                            "negative_probability": {"type": "number"},
                            "neutral_probability": {"type": "number"},
                        },
                        "required": [
                            "positive_probability",
                            "negative_probability",
                            "neutral_probability",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
        )
        return normalize_llm_sentiment_result(
            json.loads(response.output_text),
            response_id=str(getattr(response, "id", "") or ""),
        )


def load_llm_sentiment_cache(path: Path) -> pd.DataFrame:
    """Load and validate an article-level LLM sentiment cache."""

    if not path.exists():
        return pd.DataFrame(columns=LLM_CACHE_COLUMNS)
    cache = pd.read_csv(path, dtype={"article_id": str})
    missing = set(LLM_CACHE_COLUMNS) - set(cache.columns)
    if missing:
        raise ValueError(f"LLM sentiment cache is missing columns: {sorted(missing)}")
    cache = cache[list(LLM_CACHE_COLUMNS)].copy()
    cache["article_id"] = cache["article_id"].astype(str)
    if cache["article_id"].duplicated().any():
        raise ValueError("LLM sentiment cache contains duplicate article_id values.")
    return cache


def _cache_row(
    *,
    article_id: str,
    input_hash: str,
    result: LlmSentimentResult,
    model: str,
    prompt_version: str,
) -> Dict[str, object]:
    return {
        "article_id": article_id,
        "llm_input_sha256": input_hash,
        "llm_prompt_version": prompt_version,
        "llm_provider": "openai",
        "llm_model": model,
        "llm_sentiment_label": result.label,
        "llm_positive_probability": result.positive_probability,
        "llm_negative_probability": result.negative_probability,
        "llm_neutral_probability": result.neutral_probability,
        "llm_sentiment_score": result.positive_probability - result.negative_probability,
        "llm_confidence": max(
            result.positive_probability,
            result.negative_probability,
            result.neutral_probability,
        ),
        "llm_generated_at": datetime.now(timezone.utc).isoformat(),
        "llm_response_id": result.response_id,
    }


def generate_llm_sentiment_cache(
    articles: pd.DataFrame,
    *,
    output_path: Path,
    classifier: OpenAIZeroShotSentimentClassifier,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    max_articles: Optional[int] = None,
) -> pd.DataFrame:
    """Incrementally classify articles and persist each completed response."""

    if "article_id" not in articles.columns:
        raise ValueError("Article frame must contain article_id.")
    cache = load_llm_sentiment_cache(output_path)
    existing = cache.set_index("article_id", drop=False)
    rows: List[Dict[str, object]] = cache.to_dict(orient="records")
    generated = 0

    for _, article in articles.iterrows():
        article_id = str(article["article_id"])
        text = build_llm_article_text(article)
        input_hash = llm_input_sha256(text)
        if article_id in existing.index:
            cached = existing.loc[article_id]
            if (
                cached["llm_input_sha256"] != input_hash
                or cached["llm_prompt_version"] != prompt_version
                or cached["llm_model"] != classifier.model
            ):
                raise ValueError(f"Cached LLM sentiment row is stale for article_id={article_id}.")
            continue
        if not text:
            raise ValueError(f"Article text is empty for article_id={article_id}.")
        if max_articles is not None and generated >= max_articles:
            break
        result = classifier.classify(text)
        rows.append(
            _cache_row(
                article_id=article_id,
                input_hash=input_hash,
                result=result,
                model=classifier.model,
                prompt_version=prompt_version,
            )
        )
        generated += 1
        cache = pd.DataFrame(rows, columns=LLM_CACHE_COLUMNS)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cache.to_csv(output_path, index=False)
    return pd.DataFrame(rows, columns=LLM_CACHE_COLUMNS)


def attach_llm_sentiment_scores(
    scoped_articles: pd.DataFrame,
    cache: pd.DataFrame,
    *,
    require_complete: bool = True,
) -> pd.DataFrame:
    """Replace routed sentiment outputs with cached LLM scores for comparison."""

    frame = scoped_articles.copy()
    frame["article_id"] = frame["article_id"].astype(str)
    llm_cache = cache.copy()
    llm_cache["article_id"] = llm_cache["article_id"].astype(str)
    merged = frame.merge(llm_cache, on="article_id", how="left", validate="many_to_one")
    missing = merged["llm_sentiment_label"].isna()
    if require_complete and missing.any():
        raise ValueError(f"LLM sentiment cache is incomplete: {int(missing.sum())} market articles are missing.")
    merged = merged[~missing].copy()
    merged["sentiment_label"] = merged["llm_sentiment_label"]
    merged["positive_prob"] = pd.to_numeric(merged["llm_positive_probability"], errors="raise")
    merged["negative_prob"] = pd.to_numeric(merged["llm_negative_probability"], errors="raise")
    merged["neutral_prob"] = pd.to_numeric(merged["llm_neutral_probability"], errors="raise")
    merged["sentiment_score"] = pd.to_numeric(merged["llm_sentiment_score"], errors="raise")
    merged["sentiment_score_standardized"] = merged["sentiment_score"]
    merged["confidence"] = pd.to_numeric(merged["llm_confidence"], errors="raise")
    merged["is_scored"] = 1
    relevance = pd.to_numeric(merged["entity_relevance_weight"], errors="coerce").fillna(0.0)
    merged["sentiment_article_weight"] = relevance * merged["confidence"]
    return merged
