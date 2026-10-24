"""
ClosedLabelExtractor (system "ClosedLabel" in the paper).

Lists the same closed L1/L2 inventory as BatchLock in a short prompt, with no
domain check, no schema block (Modules B/C) and no validator retries. It
isolates the contribution of the BatchLock pipeline beyond "show the LLM the
list of allowed labels".
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List

from batchlock_extractor import (
    BatchLockExtractor, BatchLockValidator, ValidationResult, LLMClient,
    AnalyticalDimensions,
)
from batchlock_extractor_v2 import StandingDeskDimensions

CLOSED_LABEL_SYSTEM_PROMPT = """You are extracting product aspects from a batch of customer reviews.

The ONLY allowed L1 categories are:
{l1_labels}

The ONLY allowed L2 sub-categories are:
{l2_labels}

For each review, output a JSON array. Each element must be:
  {{"review_id": "<the review id provided>",
    "aspects": [{{"l1_category": "<one of the L1 labels above>",
                  "l2_category": "<one of the L2 labels above, or 'general'>",
                  "sentiment": "positive" | "negative" | "neutral",
                  "verbatim_span": "<a literal substring of the review>"}}]
  }}

If the review does not mention any of the listed categories, return an empty
aspects array. Do NOT invent new category names. Do NOT explain your output."""


class ClosedLabelExtractor(BatchLockExtractor):
    """Closed-label baseline: same vocabulary as BatchLock, but no Module B/C/validator-retry."""

    def __init__(self, dimensions: AnalyticalDimensions, llm_client: LLMClient, **kwargs):
        # Force max_retries to 0 — this baseline has NO retry mechanism
        kwargs.setdefault("max_retries", 0)
        super().__init__(dimensions, llm_client, **kwargs)

    def _build_system_prompt(self) -> str:
        l1 = ", ".join(self.dims.l1_labels)
        l2 = ", ".join(self.dims.l2_labels)
        return CLOSED_LABEL_SYSTEM_PROMPT.format(l1_labels=l1, l2_labels=l2)

    def _build_user_prompt(self, batch_reviews: List[dict]) -> str:
        """Plain enumeration of reviews — no Module C structure."""
        lines = []
        for r in batch_reviews:
            rid = r.get("review_id", "?")
            text = r.get("text", "")
            lines.append(f"review_id: {rid}\n{text}\n---")
        return "\n".join(lines)


def make_closed_label_extractor(llm_client: LLMClient, batch_size: int = 30) -> ClosedLabelExtractor:
    """Closed-label baseline with the same dimensions BatchLock uses for standing desks."""
    dims = StandingDeskDimensions()  # same closed L1/L2 vocab as BatchLock_v2
    return ClosedLabelExtractor(dims, llm_client, batch_size=batch_size)
