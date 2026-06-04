from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from egx_direction.llm_sentiment import (
    LlmSentimentResult,
    OpenAIZeroShotSentimentClassifier,
    attach_llm_sentiment_scores,
    generate_llm_sentiment_cache,
    llm_input_sha256,
    normalize_llm_sentiment_result,
)


class FakeClassifier:
    model = "test-zero-shot-model"

    def __init__(self) -> None:
        self.calls = []

    def classify(self, article_text: str) -> LlmSentimentResult:
        self.calls.append(article_text)
        return LlmSentimentResult(
            label="positive",
            positive_probability=0.7,
            negative_probability=0.1,
            neutral_probability=0.2,
            response_id=f"response-{len(self.calls)}",
        )


class LlmSentimentTests(unittest.TestCase):
    def test_openai_classifier_requests_strict_structured_output(self) -> None:
        class FakeResponse:
            id = "response-1"
            output_text = (
                '{"positive_probability": 0.6, "negative_probability": 0.1, '
                '"neutral_probability": 0.3}'
            )

        class FakeResponses:
            def __init__(self) -> None:
                self.kwargs = {}

            def create(self, **kwargs) -> FakeResponse:
                self.kwargs = kwargs
                return FakeResponse()

        class FakeClient:
            def __init__(self) -> None:
                self.responses = FakeResponses()

        client = FakeClient()
        classifier = OpenAIZeroShotSentimentClassifier(model="gpt-test", client=client)

        result = classifier.classify("Market article")

        self.assertEqual(result.label, "positive")
        self.assertEqual(client.responses.kwargs["model"], "gpt-test")
        response_format = client.responses.kwargs["text"]["format"]
        self.assertEqual(response_format["type"], "json_schema")
        self.assertTrue(response_format["strict"])

    def test_normalize_llm_sentiment_result_normalizes_probabilities_and_label(self) -> None:
        result = normalize_llm_sentiment_result(
            {
                "positive_probability": 0.4,
                "negative_probability": 0.2,
                "neutral_probability": 0.2,
            }
        )

        self.assertEqual(result.label, "positive")
        self.assertEqual(result.positive_probability, 0.5)
        self.assertEqual(result.negative_probability, 0.25)
        self.assertEqual(result.neutral_probability, 0.25)

    def test_generate_cache_is_resumable_and_rejects_stale_rows(self) -> None:
        articles = pd.DataFrame(
            [
                {"article_id": "1", "cleaned_title": "First", "clean_article_text": "Body one"},
                {"article_id": "2", "cleaned_title": "Second", "clean_article_text": "Body two"},
            ]
        )
        classifier = FakeClassifier()
        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "cache.csv"

            cache = generate_llm_sentiment_cache(articles, output_path=output_path, classifier=classifier)
            resumed = generate_llm_sentiment_cache(articles, output_path=output_path, classifier=classifier)

            self.assertEqual(len(cache), 2)
            self.assertEqual(len(resumed), 2)
            self.assertEqual(len(classifier.calls), 2)
            changed = articles.copy()
            changed.loc[0, "clean_article_text"] = "Changed body"
            with self.assertRaisesRegex(ValueError, "stale"):
                generate_llm_sentiment_cache(changed, output_path=output_path, classifier=classifier)

    def test_attach_llm_scores_replaces_routed_scores_and_requires_complete_cache(self) -> None:
        articles = pd.DataFrame(
            [
                {
                    "article_id": "1",
                    "Ticker": "EGX30",
                    "entity_relevance_weight": 0.8,
                    "sentiment_label": "negative",
                    "sentiment_score_standardized": -0.9,
                },
                {
                    "article_id": "2",
                    "Ticker": "EGX30",
                    "entity_relevance_weight": 1.0,
                    "sentiment_label": "neutral",
                    "sentiment_score_standardized": 0.0,
                },
            ]
        )
        cache = pd.DataFrame(
            [
                {
                    "article_id": "1",
                    "llm_sentiment_label": "positive",
                    "llm_positive_probability": 0.7,
                    "llm_negative_probability": 0.1,
                    "llm_neutral_probability": 0.2,
                    "llm_sentiment_score": 0.6,
                    "llm_confidence": 0.7,
                },
                {
                    "article_id": "2",
                    "llm_sentiment_label": "negative",
                    "llm_positive_probability": 0.1,
                    "llm_negative_probability": 0.8,
                    "llm_neutral_probability": 0.1,
                    "llm_sentiment_score": -0.7,
                    "llm_confidence": 0.8,
                },
            ]
        )

        scored = attach_llm_sentiment_scores(articles, cache)

        self.assertEqual(scored["sentiment_label"].tolist(), ["positive", "negative"])
        self.assertEqual(scored["sentiment_score_standardized"].tolist(), [0.6, -0.7])
        self.assertAlmostEqual(scored.iloc[0]["sentiment_article_weight"], 0.56)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            attach_llm_sentiment_scores(articles, cache.iloc[:1])


if __name__ == "__main__":
    unittest.main()
