"""
Vanilla baseline (system "Vanilla" in the paper).

A free-form analyst prompt: per review it asks for an issue category of the
model's own choosing, a sentiment and a quote, plus a short summary report.
No closed label inventory, no typed schema and no validator retries.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List

from batchlock_extractor import (
    BatchLockExtractor,
    AnalyticalDimensions,
    LLMClient,
)

# ============================================================
# Vanilla Module A — generic ABSA prompt, no schema lock
# ============================================================

VANILLA_MODULE_A = """You are a customer review analyst. Read the batch of product reviews and produce a structured analytical report.

For each review, extract:
- One or more **issue categories** (use your own intuitive grouping)
- Sentiment polarity (positive / negative / neutral / mixed)
- A representative quote (you may shorten or paraphrase for clarity)

After analyzing all reviews, write a 200-300 word summary report covering:
- Top issues by frequency
- Overall sentiment distribution
- Notable customer feedback themes
- Recommendations for the product team

Use Markdown formatting with headers, bullet points, and percentages.

OUTPUT JSON:
{
  "per_review": [
    {"review_id": <id>, "category": <name>, "sentiment": <polarity>, "quote": <text>}
  ],
  "summary_report": <markdown string>
}

Use natural categories that make sense to you. Don't overthink the taxonomy."""


# ============================================================
# Vanilla "dimensions" — no closed label set, freer prompt
# ============================================================

@dataclass
class VanillaDimensions(AnalyticalDimensions):
    """For Vanilla: no closed L1/L2 set; LLM is free to invent labels."""
    domain: str = "product_review"
    # Override with empty (LLM uses own categories)
    l1_labels: List[str] = field(default_factory=lambda: [])
    l2_labels: List[str] = field(default_factory=lambda: [])

    def to_prompt_block(self) -> str:
        return f"""USER DOMAIN: {self.domain}
(No predefined categories. Use natural grouping.)"""


# ============================================================
# VanillaBaselineExtractor — same API, looser prompt
# ============================================================

class VanillaBaselineExtractor(BatchLockExtractor):
    """Vanilla extractor with no schema lock — for A/B comparison.

    Important: This is a benchmark Vanilla, not a 'crippled' Vanilla.
    It still uses good output structure (tables, %), but without the
    methodology discipline that BatchLock provides.
    """

    def _build_system_prompt(self) -> str:
        # No L1/L2 substitution since vanilla doesn't use them
        return VANILLA_MODULE_A + "\n\n" + self.dims.to_prompt_block()

    def _validator_should_check_label_set(self) -> bool:
        # Vanilla doesn't enforce closed set
        return False


# ============================================================
# Vanilla validator (loose — doesn't enforce closed sets)
# ============================================================

class VanillaValidator:
    """Vanilla validator: only checks JSON structure, not label vocab."""

    def __init__(self):
        pass

    def validate(self, output: any, original_reviews: dict):
        """Returns 'valid' if output is a list of per-review records.

        After the parent extractor's dict-to-list unwrap (which now also
        unwraps `per_review`), Vanilla's output reaches this validator as a
        list of {review_id, category, sentiment, quote} records.
        """
        from batchlock_extractor import ValidationResult
        errors = []
        if not isinstance(output, list):
            return ValidationResult(False, [f"Output is not a JSON list (got {type(output).__name__})"])
        if not output:
            return ValidationResult(False, ["Output list is empty"])
        for i, item in enumerate(output):
            if not isinstance(item, dict):
                errors.append(f"item[{i}] not a dict")
                continue
            if "review_id" not in item:
                errors.append(f"item[{i}] missing review_id")
            if "category" not in item and "aspects" not in item:
                errors.append(f"item[{i}] missing category (or aspects)")
        return ValidationResult(len(errors) == 0, errors)


# ============================================================
# Make-helper
# ============================================================

def make_vanilla_baseline_extractor(llm_client: LLMClient, **kwargs) -> VanillaBaselineExtractor:
    """One-liner: returns a Vanilla baseline extractor configured for product reviews."""
    dims = VanillaDimensions()
    extractor = VanillaBaselineExtractor(dims, llm_client, **kwargs)
    # Swap validator to Vanilla (loose)
    extractor.validator = VanillaValidator()
    return extractor


# ============================================================
# Sample Vanilla output (what we expect the prompt to produce)
# ============================================================

SAMPLE_VANILLA_OUTPUT = """
{
  "per_review": [
    {"review_id": "r1", "category": "Setup Issues", "sentiment": "negative", "quote": "instructions terrible"},
    {"review_id": "r2", "category": "Product Quality", "sentiment": "positive", "quote": "great desk"},
    ...
  ],
  "summary_report": "## Customer Review Analysis Summary

  ### Top Issues
  - **Setup Issues**: around 40% of reviews mentioned this
  - **Quality Problems**: about 30% of reviews

  ### Sentiment Distribution
  - Positive: ~25%
  - Negative: ~60%
  - Neutral: ~15%

  ### Key Themes
  Many customers reported difficulty with assembly and durability concerns.
  Customer service responsiveness was mixed.

  ### Recommendations
  - Improve assembly instructions
  - Enhance customer support response times
  - Address quality control issues

  ### Purchase Motivations
  (Many customers bought for work from home or back pain relief.)
  "
}
"""

if __name__ == "__main__":
    print(__doc__)
    print("\nSample expected output:")
    print(SAMPLE_VANILLA_OUTPUT)
