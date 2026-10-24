"""
JSONSchemaOnlyExtractor (system "JSONSchemaOnly" in the paper).

The LLM is told to emit JSON of the BatchLock shape (review_id, aspects[].l1,
l2, sentiment, verbatim_span) but is NOT given any closed label list. It sits
between Vanilla (free-form) and ClosedLabel (schema + closed vocabulary) and
separates the effect of output structure from that of vocabulary closure.
"""
from __future__ import annotations
from typing import List

from batchlock_extractor import (
    BatchLockExtractor, BatchLockValidator, ValidationResult, LLMClient,
    AnalyticalDimensions,
)
from batchlock_extractor_v2 import StandingDeskDimensions


JSON_SCHEMA_ONLY_SYSTEM_PROMPT = """You are extracting product aspects from a batch of customer reviews.

Output a JSON array. Each element must follow this shape:
  {{"review_id": "<the review id provided>",
    "aspects": [{{"l1_category": "<your chosen L1 category name>",
                  "l2_category": "<your chosen L2 sub-category name>",
                  "sentiment": "positive" | "negative" | "neutral",
                  "verbatim_span": "<a literal substring of the review>"}}]
  }}

Use your own judgement for category names. Do NOT use extra fields. Do NOT
explain your output. If a review mentions no product aspects, return an empty
aspects array."""


class _JSONOnlyValidator(BatchLockValidator):
    """Same structural checks as BatchLockValidator but DISABLE closed-set membership checks."""
    def __init__(self, dims):
        super().__init__(dims)
        # neutralise L1/L2 vocab sets — anything goes
        self._l1_set = None
        self._l2_set = None

    def validate(self, output, original_reviews):
        # Bypass label-set membership check; just structural + verbatim
        errors = []
        if not isinstance(output, list):
            return ValidationResult(False, ["Output is not a JSON array"])
        for i, item in enumerate(output):
            if not isinstance(item, dict):
                errors.append(f"item[{i}] not a dict"); continue
            if "review_id" not in item:
                errors.append(f"item[{i}] missing review_id"); continue
            aspects = item.get("aspects", [])
            if not isinstance(aspects, list):
                errors.append(f"item[{i}] aspects is not list"); continue
            rid = item.get("review_id")
            text = original_reviews.get(rid, "")
            for j, asp in enumerate(aspects):
                if not isinstance(asp, dict):
                    errors.append(f"item[{i}].aspects[{j}] not a dict"); continue
                if "l1_category" not in asp or not asp["l1_category"]:
                    errors.append(f"item[{i}].aspects[{j}] missing l1_category")
                if "sentiment" not in asp or asp["sentiment"] not in {"positive","negative","neutral"}:
                    errors.append(f"item[{i}].aspects[{j}] sentiment missing/invalid")
                verb = asp.get("verbatim_span")
                if not isinstance(verb, str):
                    errors.append(f"item[{i}].aspects[{j}] verbatim_span not a string")
                elif text and not self._is_substring(verb, text):
                    errors.append(f"item[{i}].aspects[{j}] verbatim '{verb[:40]}...' not in source")
        return ValidationResult(len(errors) == 0, errors)


class JSONSchemaOnlyExtractor(BatchLockExtractor):
    """JSON-schema-only baseline: enforces structured output shape, leaves label vocab open."""
    def __init__(self, dimensions: AnalyticalDimensions, llm_client: LLMClient, **kwargs):
        # Allow 2 retries on structural violations (not 0 like ClosedLabel — schema-shape errors
        # are recoverable; label-vocab compliance is what we're measuring, so retry on shape is fair)
        kwargs.setdefault("max_retries", 2)
        super().__init__(dimensions, llm_client, **kwargs)
        # Swap in the relaxed validator
        self.validator = _JSONOnlyValidator(dimensions)

    def _build_system_prompt(self) -> str:
        return JSON_SCHEMA_ONLY_SYSTEM_PROMPT

    def _build_user_prompt(self, batch_reviews: List[dict]) -> str:
        lines = []
        for r in batch_reviews:
            rid = r.get("review_id", "?")
            text = r.get("text", "")
            lines.append(f"review_id: {rid}\n{text}\n---")
        return "\n".join(lines)


def make_json_schema_only_extractor(llm_client: LLMClient, batch_size: int = 30):
    dims = StandingDeskDimensions()   # vocab is loaded but unused at validation time
    return JSONSchemaOnlyExtractor(dims, llm_client, batch_size=batch_size)
