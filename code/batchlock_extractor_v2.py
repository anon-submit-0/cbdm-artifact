"""
BatchLock v2: the deployed map prompt (system "BatchLock" in the paper).

Adds to v1 (batchlock_extractor.py):
  * a domain check (off-topic reviews return aspects: [] with out_of_domain=true),
  * scope and provenance rules,
  * the standing-desk L1/L2 inventory used in the deployment domain.

The prompt string MODULE_A_PROMPT_V2 is kept byte-identical to the one used in
the reported runs (including its internal section headings); see Appendix A of
the paper.
"""

from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from batchlock_extractor import (
    BatchLockExtractor,
    BatchLockValidator,
    AnalyticalDimensions,
    LLMClient,
    BatchResult,
)

# ============================================================
# Module A v2 — adds domain check + scope discipline + provenance
# ============================================================

MODULE_A_PROMPT_V2 = """You are an aspect-based sentiment analysis (ABSA) system for {domain}.

============ DOMAIN CHECK (CRITICAL — read first) ============

Before extracting aspects, verify: Is this review about {domain}?

Signals it IS about {domain}: {domain_keywords}
Signals it is NOT (e.g., laptop / phone / book): {non_domain_keywords}

If review is clearly out-of-domain, return:
  {{
    "review_id": "<id>",
    "aspects": [],
    "out_of_domain": true,
    "reason": "<short explanation>"
  }}

DO NOT shoehorn out-of-domain content into the {domain} schema.

============ OUTPUT SCHEMA (locked, do not deviate) ============

For in-domain reviews, output a JSON array of objects matching this exact schema:

{{
  "review_id": "<id from input>",
  "out_of_domain": false,
  "aspects": [
    {{
      "l1_category": "<one of: %(l1_labels)s>",
      "l2_category": "<one of: %(l2_labels)s | 'general'>",
      "sentiment": "<positive | negative | neutral>",
      "verbatim_span": "<exact contiguous substring from review>",
      "confidence": <float 0-1>
    }}
  ]
}}

============ SCOPE DISCIPLINE (Critical Issue #3 from E1 review) ============

Stay strictly within the schema above. DO NOT:
- Add purchase motivation analysis unless explicitly requested
- Add "executive summary" sections unless requested
- Make cross-product comparisons unless input contains both products
- Estimate metrics like NPS that require survey data (NPS ≠ sentiment ratio)
- Cite ratios (e.g., "2.3x more") without showing the numerator and denominator

============ PROVENANCE REQUIREMENT (Critical Issue #1) ============

If you must emit any percentage or ratio in free-text reasoning fields, include:
- numerator (integer count)
- denominator (integer count)
- methodology (one-line description of how the count was derived)

============ CONSTRAINTS ============

1. l1_category MUST be from closed set. No new labels.
2. l2_category MUST be from closed set OR literal "general".
3. verbatim_span MUST be a contiguous substring of the review text (character-exact).
4. Multiple aspects per review → multiple aspect entries.
5. No aspect identified → aspects: [].
6. Do NOT paraphrase. Do NOT invent.

OUTPUT ONLY the JSON array. No prose."""


# ============================================================
# AnalyticalDimensions v2 — standing-desk-aligned L2 labels
# ============================================================

STANDING_DESK_L1 = [
    "Assembly", "Build Quality", "Lift Mechanism", "Stability",
    "Size & Fit", "Appearance", "Customer Service", "Value for Money",
    "Ergonomics", "Cable Management", "Comfort", "General",
]

STANDING_DESK_L2 = [
    # Assembly
    "instructions", "parts_completeness", "difficulty", "assembly_time", "tools", "screws",
    # Build Quality
    "durability", "frame_material", "desktop_material", "packaging", "workmanship", "accessories",
    # Lift Mechanism
    "motor_speed", "controller_keypad", "smooth_operation", "noise",
    "weight_capacity", "height_range", "memory_preset",
    # Stability
    "wobble", "frame_rigidity", "tipping_risk",
    # Size & Fit
    "desktop_dimensions", "leg_room",
    # Appearance
    "design", "color", "general_appearance",
    # Customer Service
    "support_response", "after_sales", "shipping_delivery", "warranty",
    # Value for Money
    "price", "discount_perception",
    # Ergonomics
    "back_relief", "posture_support", "standing_fatigue",
    # Cable Management
    "general_cable_mgmt",
    # Comfort
    "edge_finish", "surface_texture",
    # Catch-all
    "general",
]

STANDING_DESK_KEYWORDS = (
    "desk, standing, sit-stand, sit stand, lift, height adjust, crossbar, "
    "motor, wobble, cable tray, monitor arm, ergonomic"
)
NON_DOMAIN_KEYWORDS = (
    "laptop, notebook, macbook, chromebook, RAM, processor, "
    "phone, smartphone, tablet, book, novel, kindle"
)


@dataclass
class StandingDeskDimensions(AnalyticalDimensions):
    """Default dimensions for standing desk reviews (overrides v1 chair-oriented defaults)."""
    domain: str = "standing_desk"
    domain_keywords: str = STANDING_DESK_KEYWORDS
    non_domain_keywords: str = NON_DOMAIN_KEYWORDS
    l1_labels: List[str] = field(default_factory=lambda: STANDING_DESK_L1.copy())
    l2_labels: List[str] = field(default_factory=lambda: STANDING_DESK_L2.copy())


# ============================================================
# FindingEmitter — wraps reduce output with provenance metadata
# ============================================================

@dataclass
class Finding:
    """A reportable finding with provenance — replaces raw aggregate dicts."""
    l1: str
    l2: Optional[str]
    sentiment: str
    numerator: int          # count contributing to this finding
    denominator: int        # total reviews analyzed (sample size)
    percentage: float       # numerator / denominator, computed
    methodology: str        # one-line explanation
    verbatims: List[Dict]   # top-K supporting quotes
    out_of_domain_excluded: int = 0  # count of OOD reviews excluded

    def to_report_line(self) -> str:
        """Format for human-readable report (forced provenance display)."""
        pct = self.percentage * 100
        return (
            f"[{self.l1}/{self.l2 or 'general'} {self.sentiment}] "
            f"{self.numerator}/{self.denominator} = {pct:.1f}% "
            f"(method: {self.methodology})"
        )


class FindingEmitter:
    """Converts BatchLock reduce output to provenance-attached Findings."""

    def __init__(self, total_reviews_in_corpus: int):
        self.n_total = total_reviews_in_corpus

    def emit(self, reduce_output: Dict[str, Any]) -> List[Finding]:
        findings = []
        n_ood = reduce_output.get("out_of_domain_count", 0)
        n_valid = self.n_total - n_ood

        for cat in reduce_output.get("categories", []):
            for sent in ("positive", "negative", "neutral"):
                count = cat.get(sent, 0)
                if count == 0:
                    continue
                findings.append(Finding(
                    l1=cat["l1"],
                    l2=cat["l2"],
                    sentiment=sent,
                    numerator=count,
                    denominator=n_valid,
                    percentage=count / max(1, n_valid),
                    methodology=f"Counted reviews where BatchLock extracted ({cat['l1']}/{cat['l2']}, {sent}) with confidence >= 0.7, n_total_valid={n_valid}, n_excluded_ood={n_ood}",
                    verbatims=cat.get("verbatims", []),
                    out_of_domain_excluded=n_ood,
                ))
        return findings


# ============================================================
# BatchLockExtractorV2 — uses new Module A
# ============================================================

class BatchLockExtractorV2(BatchLockExtractor):
    """Drop-in replacement for v1, uses Module A v2."""

    def _build_system_prompt(self) -> str:
        l1_str = ", ".join(self.dims.l1_labels)
        l2_str = ", ".join(self.dims.l2_labels + ["general"])
        domain = getattr(self.dims, "domain", "general_product_review")
        dom_kw = getattr(self.dims, "domain_keywords", "(unspecified)")
        non_dom_kw = getattr(self.dims, "non_domain_keywords", "(unspecified)")
        return (
            MODULE_A_PROMPT_V2.format(
                domain=domain,
                domain_keywords=dom_kw,
                non_domain_keywords=non_dom_kw,
            ) % {
                "l1_labels": l1_str,
                "l2_labels": l2_str,
            }
            + "\n\n"
            + self.dims.to_prompt_block()
        )

    def reduce_v2(
        self,
        batch_results: List[BatchResult],
        total_corpus_size: int,
        verbatims_per_label: int = 10,
    ) -> Dict[str, Any]:
        """Reduce + provenance-emit findings.

        Returns:
            {
              "findings": List[Finding],
              "raw_aggregates": legacy_reduce_output,
              "out_of_domain_count": int,
              "scope_violations": List[str],  # tasks where Compiled added scope-out content
            }
        """
        raw = super().reduce(batch_results, verbatims_per_label=verbatims_per_label)

        # Count out-of-domain (scan batch outputs for items with out_of_domain=True)
        n_ood = 0
        for br in batch_results:
            for item in br.output:
                if item.get("out_of_domain") is True:
                    n_ood += 1

        raw["out_of_domain_count"] = n_ood

        emitter = FindingEmitter(total_corpus_size)
        findings = emitter.emit(raw)

        return {
            "findings": findings,
            "raw_aggregates": raw,
            "out_of_domain_count": n_ood,
            "report_lines": [f.to_report_line() for f in findings],
        }


# ============================================================
# Convenience: build an extractor with all v2 fixes in one line
# ============================================================

def make_standing_desk_extractor(llm_client: LLMClient, **kwargs) -> BatchLockExtractorV2:
    """One-liner: returns a v2 extractor configured for standing-desk reviews."""
    dims = StandingDeskDimensions()
    return BatchLockExtractorV2(dims, llm_client, **kwargs)


# ============================================================
# Smoke test
# ============================================================

async def _smoke_v2():
    """Smoke test domain check by mixing laptop + standing desk reviews."""
    test_reviews = [
        # In-domain
        {"review_id": "r1", "text": "The assembly was hard but the desk is sturdy."},
        {"review_id": "r2", "text": "Wobbly at full height. Motor is noisy."},
        # Out-of-domain (laptop)
        {"review_id": "r3", "text": "Great laptop. Plenty fast for browsing."},
        {"review_id": "r4", "text": "Battery life is only 4 hours."},
    ]
    import os
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("DEEPSEEK_API_KEY"):
        print("⚠️  No API key set, skipping smoke test")
        return
    if os.environ.get("DEEPSEEK_API_KEY"):
        llm = LLMClient(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url=os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1"),
            model=os.environ.get("LLM_MODEL", "deepseek-v4-flash"),
        )
    else:
        llm = LLMClient(model=os.environ.get("BATCHLOCK_MODEL", "gpt-4o-mini"))
    extractor = make_standing_desk_extractor(llm, batch_size=4)
    results = await extractor.extract_all(test_reviews)

    n_ood = sum(
        1 for br in results for item in br.output
        if item.get("out_of_domain") is True
    )
    print(f"\nResults:")
    print(f"  Total: 4 reviews")
    print(f"  Out-of-domain detected: {n_ood} (expected: 2)")

    summary_v2 = extractor.reduce_v2(results, total_corpus_size=4)
    print(f"\nFindings with provenance:")
    for line in summary_v2["report_lines"]:
        print(f"  {line}")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_smoke_v2())
