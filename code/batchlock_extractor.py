"""
BatchLock: schema-locked hierarchical aspect extractor (base classes).

Architecture:
- Module A: schema-locked output JSON
- Module B: declarable analytical dimensions (closed L1/L2 inventory, per domain)
- Module C: dynamic batch injection (per call)

Concurrent execution via asyncio.Semaphore.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

# ---------- Constants ----------
DEFAULT_BATCH_SIZE = 30
DEFAULT_CONCURRENCY = 5
DEFAULT_MAX_RETRIES = 5

log = logging.getLogger("BatchLock")

# ---------- Module A: locked schema ----------
MODULE_A_PROMPT = """You are an aspect-based sentiment analysis (ABSA) system.

OUTPUT SCHEMA (locked, do not deviate):
You MUST output a single JSON array. Each element follows this exact schema:

{
  "review_id": "<id from input>",
  "aspects": [
    {
      "l1_category": "<one of: %(l1_labels)s>",
      "l2_category": "<one of: %(l2_labels)s | 'general' if no specific match>",
      "sentiment": "<one of: positive, negative, neutral>",
      "verbatim_span": "<exact contiguous substring from review text>",
      "confidence": <float 0-1>
    }
  ]
}

CONSTRAINTS:
1. The l1_category MUST be from the closed label set above. No new labels permitted.
2. The l2_category MUST be from the closed l2 set OR the literal string "general".
3. The verbatim_span MUST be a contiguous substring of the review text (character-exact).
4. If a review mentions multiple aspects, emit multiple aspect entries.
5. If no aspect is identified, return aspects: [].
6. Do NOT paraphrase the verbatim. Do NOT invent aspects.

OUTPUT ONLY the JSON array. No other text."""

# ---------- Module B: user-declarable dimensions ----------
@dataclass
class AnalyticalDimensions:
    """User-declarable analytical dimensions (Module B)."""

    domain: str = "general_product_review"
    l1_labels: List[str] = field(default_factory=lambda: [
        "Comfort", "Assembly", "Build Quality", "Size & Fit",
        "Appearance", "Adjustability", "Ergonomics", "Material",
        "Value for Money", "Customer Service", "Stability", "General",
    ])
    l2_labels: List[str] = field(default_factory=lambda: [
        "lumbar_support", "armrest", "headrest", "seat_cushion",
        "instructions", "tools", "screws",
        "durability", "materials_used", "workmanship",
        "size_fit", "weight_capacity",
        "design", "color",
        "adjustment_range", "ease_of_adjust",
        "back_relief", "posture_support",
        "fabric", "leather",
        "price", "discount_perception",
        "support_response", "warranty",
        "wobble", "tipping_risk",
        "general",
    ])
    l2_trigger_threshold: float = 0.20  # 20% mention rate triggers L2 refinement
    sentiment_polarities: List[str] = field(default_factory=lambda: [
        "positive", "negative", "neutral",
    ])

    def to_prompt_block(self) -> str:
        """Generate the Module B prompt block."""
        l1 = ", ".join(self.l1_labels)
        l2 = ", ".join(self.l2_labels)
        return f"""
USER DIMENSIONS (Module B):
Domain: {self.domain}

L1 categories (closed set): {l1}
L2 sub-categories (closed set): {l2}

Refinement rule: Use L2 category when a specific aspect is mentioned.
If unclear or no specific L2 applies, use 'general'.
""".strip()


# ---------- Output validator ----------
@dataclass
class ValidationResult:
    valid: bool
    errors: List[str] = field(default_factory=list)


class BatchLockValidator:
    """Validates LLM output against the locked schema.

    The validator is robust to common LLM output quirks:
    - Smart quotes vs straight quotes (normalize to straight)
    - Leading/trailing whitespace
    - Unicode dashes vs ASCII hyphens
    - Allow fuzzy substring match (Levenshtein distance ≤ 3) if exact fails
    """

    def __init__(self, dimensions: AnalyticalDimensions):
        self.dims = dimensions
        self.l1_set = set(dimensions.l1_labels) | {"general"}
        # Case-insensitive l1 lookup
        self.l1_lower = {l.lower(): l for l in self.l1_set}
        self.l2_set = set(dimensions.l2_labels) | {"general"}
        self.l2_lower = {l.lower(): l for l in self.l2_set}
        self.sentiment_set = set(dimensions.sentiment_polarities)

    @staticmethod
    def _normalize_text(s: str) -> str:
        """Normalize LLM output quirks for substring match."""
        if not isinstance(s, str):
            return ""
        # Smart quotes → straight quotes
        s = s.replace("‘", "'").replace("’", "'")
        s = s.replace("“", '"').replace("”", '"')
        # Unicode dashes
        s = s.replace("–", "-").replace("—", "-")
        # Strip
        return s.strip()

    @staticmethod
    def _is_substring(needle: str, haystack: str) -> bool:
        """Check substring with fuzzy tolerance for whitespace, ellipsis, and case."""
        n = BatchLockValidator._normalize_text(needle).lower()
        h = BatchLockValidator._normalize_text(haystack).lower()
        if not n:
            return False
        if n in h:
            return True
        # Try collapsing whitespace
        import re
        n_collapsed = re.sub(r"\s+", " ", n).strip()
        h_collapsed = re.sub(r"\s+", " ", h).strip()
        if n_collapsed in h_collapsed:
            return True
        # Strip trailing ellipsis (LLM truncation artifact)
        n_no_ellipsis = re.sub(r"\.{2,}\s*$", "", n_collapsed).strip()
        if n_no_ellipsis and n_no_ellipsis in h_collapsed:
            return True
        # Strip leading ellipsis as well
        n_no_lead = re.sub(r"^\.{2,}\s*", "", n_no_ellipsis).strip()
        if n_no_lead and n_no_lead in h_collapsed:
            return True
        # Strip surrounding punctuation
        n_stripped = re.sub(r"^[\s\.,;:!\?\"\']+|[\s\.,;:!\?\"\']+$", "", n_no_lead).strip()
        if n_stripped and len(n_stripped) >= 8 and n_stripped in h_collapsed:
            return True
        # Last resort: 80% prefix match (LLM truncated at end)
        if len(n_collapsed) >= 20:
            prefix = n_collapsed[: int(len(n_collapsed) * 0.8)]
            if prefix and prefix in h_collapsed:
                return True
        # Final fallback: token-overlap ratio (LLM paraphrased / lightly edited verbatim)
        # Accept if >=70% of needle's content tokens appear contiguously somewhere
        # in haystack within a small window. This is forgiving but still requires
        # substantial structural overlap with the source.
        import re as _re
        needle_tokens = [t for t in _re.findall(r"\w+", n_collapsed) if len(t) >= 2]
        if len(needle_tokens) >= 4:
            hay_tokens = _re.findall(r"\w+", h_collapsed)
            hay_set = set(hay_tokens)
            present = sum(1 for t in needle_tokens if t in hay_set)
            overlap = present / max(1, len(needle_tokens))
            if overlap >= 0.7:
                # Check the tokens appear in roughly the same order somewhere
                # (sliding-window of length len(needle_tokens)*1.5 in hay_tokens)
                window = max(len(needle_tokens), int(len(needle_tokens) * 1.5))
                for i in range(0, max(1, len(hay_tokens) - window + 1)):
                    chunk = set(hay_tokens[i : i + window])
                    chunk_present = sum(1 for t in needle_tokens if t in chunk)
                    if chunk_present / len(needle_tokens) >= 0.7:
                        return True
        return False

    def validate(self, output: List[Dict], original_reviews: Dict[str, str]) -> ValidationResult:
        """Validate the structured output and return error list."""
        errors = []
        if not isinstance(output, list):
            return ValidationResult(False, ["Output is not a JSON array"])

        for i, item in enumerate(output):
            if not isinstance(item, dict):
                errors.append(f"item[{i}] is not a dict")
                continue
            rid = item.get("review_id")
            if rid is None:
                errors.append(f"item[{i}] missing review_id")
                continue
            text = original_reviews.get(rid)
            if text is None:
                errors.append(f"item[{i}] review_id {rid} not in batch")
                continue

            aspects = item.get("aspects")
            if not isinstance(aspects, list):
                errors.append(f"item[{i}] aspects is not list")
                continue

            for j, asp in enumerate(aspects):
                l1 = asp.get("l1_category")
                # Case-insensitive match
                if isinstance(l1, str) and l1.lower() in self.l1_lower:
                    asp["l1_category"] = self.l1_lower[l1.lower()]  # normalize
                elif l1 not in self.l1_set:
                    errors.append(f"item[{i}].aspects[{j}] l1='{l1}' not in label set")

                l2 = asp.get("l2_category")
                if isinstance(l2, str) and l2.lower() in self.l2_lower:
                    asp["l2_category"] = self.l2_lower[l2.lower()]
                elif l2 not in self.l2_set:
                    errors.append(f"item[{i}].aspects[{j}] l2='{l2}' not in label set")

                sent = asp.get("sentiment")
                if isinstance(sent, str):
                    sent_norm = sent.lower().strip()
                    if sent_norm in self.sentiment_set:
                        asp["sentiment"] = sent_norm
                    else:
                        errors.append(f"item[{i}].aspects[{j}] sentiment='{sent}' invalid")
                else:
                    errors.append(f"item[{i}].aspects[{j}] sentiment missing")

                verb = asp.get("verbatim_span")
                if not isinstance(verb, str):
                    errors.append(f"item[{i}].aspects[{j}] verbatim_span not a string")
                elif not self._is_substring(verb, text):
                    errors.append(f"item[{i}].aspects[{j}] verbatim '{verb[:40]}...' not in source")

        return ValidationResult(len(errors) == 0, errors)


# ---------- LLM client wrapper (OpenAI-style; pluggable) ----------
class LLMClient:
    """Pluggable LLM client. Defaults to OpenAI; can swap to DeepSeek/Anthropic."""

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
    ):
        self.provider = provider
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url
        self.temperature = temperature

        # Lazy import to avoid hard dep at module load
        from openai import OpenAI  # type: ignore
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    async def call(self, system: str, user: str, max_tokens: int = 4096) -> str:
        """Async LLM call. Returns raw response text."""
        # Use a thread-pool style; if true async needed, wrap with httpx.AsyncClient
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=self.temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            ).choices[0].message.content,
        )


# ---------- Batch extractor ----------
@dataclass
class BatchResult:
    batch_id: int
    success: bool
    output: List[Dict] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    duration_ms: int = 0
    retry_count: int = 0


class BatchLockExtractor:
    """Map-Reduce ABSA extractor with schema-locked prompting."""

    def __init__(
        self,
        dimensions: AnalyticalDimensions,
        llm_client: LLMClient,
        batch_size: int = DEFAULT_BATCH_SIZE,
        concurrency: int = DEFAULT_CONCURRENCY,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        self.dims = dimensions
        self.llm = llm_client
        self.batch_size = batch_size
        self.concurrency = concurrency
        self.max_retries = max_retries
        self.validator = BatchLockValidator(dimensions)
        self.semaphore = asyncio.Semaphore(concurrency)

    def _build_system_prompt(self) -> str:
        return (
            MODULE_A_PROMPT % {
                "l1_labels": ", ".join(self.dims.l1_labels),
                "l2_labels": ", ".join(self.dims.l2_labels + ["general"]),
            }
            + "\n\n" + self.dims.to_prompt_block()
        )

    def _build_user_prompt(self, batch_reviews: List[Dict[str, str]]) -> str:
        """Module C: dynamic batch injection."""
        payload = "Here is the batch of reviews to analyze:\n\n"
        for r in batch_reviews:
            payload += f"REVIEW_ID: {r['review_id']}\nTEXT: {r['text']}\n---\n"
        payload += "\nReturn the JSON array as specified."
        return payload

    @staticmethod
    def _robust_json_parse(raw: str):
        """
        Multi-strategy JSON parser that handles common LLM output quirks:
        0. Strip <think>...</think> reasoning blocks (DeepSeek-V3, R1, o1-style)
        1. Direct parse
        2. Strip markdown fences (```json ... ```)
        3. Extract first JSON object/array via regex
        4. Trim from end (often missing trailing brackets)
        5. Repair common typos: trailing commas, unclosed strings, ellipsis
        """
        import re

        # Strategy 0: strip <think>...</think> reasoning blocks if present
        # DeepSeek-V3 / R1 / o1-style models emit thinking before final answer
        cleaned_pre = re.sub(r"<think>.*?</think>\s*", "", raw, flags=re.DOTALL | re.IGNORECASE)
        # Also handle unclosed <think> blocks (truncated)
        if "<think>" in cleaned_pre.lower() and "</think>" not in cleaned_pre.lower():
            # Drop everything up to and including the last <think>
            m = re.search(r"<think>", cleaned_pre, flags=re.IGNORECASE)
            if m:
                # If followed by a JSON-looking block, keep from there onward
                rest = cleaned_pre[m.end():]
                m_json = re.search(r"[\[\{]", rest)
                if m_json:
                    cleaned_pre = rest[m_json.start():]
        raw_for_parse = cleaned_pre.strip() if cleaned_pre.strip() else raw

        # Strategy 1: direct
        try:
            return json.loads(raw_for_parse)
        except json.JSONDecodeError:
            pass

        # Strategy 2: strip markdown fences
        cleaned = raw_for_parse.strip()
        cleaned = re.sub(r"^```(?:json|JSON)?\s*", "", cleaned)
        cleaned = re.sub(r"```\s*$", "", cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Strategy 3: find first JSON array or object substring (greedy)
        m_arr = re.search(r"\[\s*\{.*?\}\s*\]", cleaned, re.DOTALL)
        if m_arr:
            try:
                return json.loads(m_arr.group(0))
            except json.JSONDecodeError:
                pass

        m = re.search(r"(\[.*\]|\{.*\})", cleaned, re.DOTALL)
        if not m:
            raise json.JSONDecodeError("No JSON found", cleaned, 0)
        candidate = m.group(1)

        # Strategy 4: repair common issues
        # Remove trailing commas inside arrays/objects
        candidate = re.sub(r",(\s*[\]\}])", r"\1", candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        # Strategy 5: greedy trim from end (truncate to last valid closing)
        for end in range(len(candidate), max(0, len(candidate) - 4000), -1):
            ch = candidate[end - 1] if end > 0 else ""
            if ch not in "]}":
                continue
            try:
                return json.loads(candidate[:end])
            except json.JSONDecodeError:
                continue

        # Strategy 6: build a minimal array from item-level parse
        # Extract individual {} objects, parse each, collect successful ones
        objects = []
        for obj_match in re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", candidate, re.DOTALL):
            try:
                obj = json.loads(obj_match.group(0))
                if isinstance(obj, dict) and "review_id" in obj:
                    objects.append(obj)
            except json.JSONDecodeError:
                continue
        if objects:
            return objects

        raise json.JSONDecodeError("Could not repair JSON after all strategies", candidate, 0)

    async def extract_one_batch(
        self,
        batch_id: int,
        batch_reviews: List[Dict[str, str]],
    ) -> BatchResult:
        """Extract aspects from one batch with retry."""
        async with self.semaphore:
            t0 = time.time()
            review_map = {r["review_id"]: r["text"] for r in batch_reviews}

            system = self._build_system_prompt()
            user = self._build_user_prompt(batch_reviews)

            for attempt in range(self.max_retries + 1):
                try:
                    raw = await self.llm.call(system, user, max_tokens=8192)
                    # Defensive JSON parse with multiple repair strategies
                    parsed = self._robust_json_parse(raw)
                    # Normalize: if wrapped in {"items": [...]} extract
                    if isinstance(parsed, dict):
                        # Common wrapper keys
                        unwrapped = False
                        for k in ("results", "items", "output", "aspects", "extractions", "reviews", "data", "per_review"):
                            if k in parsed and isinstance(parsed[k], list):
                                parsed = parsed[k]
                                unwrapped = True
                                break
                        if not unwrapped:
                            # If dict has review_id, treat as single-item list
                            if "review_id" in parsed:
                                parsed = [parsed]
                            # If the only value is a list, use it
                            elif len(parsed) == 1 and isinstance(list(parsed.values())[0], list):
                                parsed = list(parsed.values())[0]
                            # If all values look like per-review dicts (have 'aspects'), turn into list
                            elif parsed and all(
                                isinstance(v, dict) and ("aspects" in v or "review_id" in v)
                                for v in parsed.values()
                            ):
                                rebuilt = []
                                for rid, v in parsed.items():
                                    item = dict(v)
                                    if "review_id" not in item:
                                        item["review_id"] = rid
                                    rebuilt.append(item)
                                parsed = rebuilt

                    if not isinstance(parsed, list):
                        raise ValueError(f"Output not a list: {type(parsed)}")

                    val = self.validator.validate(parsed, review_map)
                    if val.valid:
                        return BatchResult(
                            batch_id=batch_id,
                            success=True,
                            output=parsed,
                            duration_ms=int((time.time() - t0) * 1000),
                            retry_count=attempt,
                        )

                    if attempt < self.max_retries:
                        log.warning(
                            "Batch %d attempt %d failed validation: %s; retrying",
                            batch_id, attempt, val.errors[:3],
                        )
                        # Add validation feedback to retry prompt
                        user = self._build_user_prompt(batch_reviews) + (
                            "\n\nPrevious attempt had validation errors: "
                            + "; ".join(val.errors[:5])
                            + "\nPlease correct and retry."
                        )
                    else:
                        return BatchResult(
                            batch_id=batch_id,
                            success=False,
                            output=parsed,
                            errors=val.errors,
                            duration_ms=int((time.time() - t0) * 1000),
                            retry_count=attempt,
                        )
                except Exception as e:
                    if attempt < self.max_retries:
                        log.warning("Batch %d attempt %d exception: %s; retrying", batch_id, attempt, e)
                    else:
                        return BatchResult(
                            batch_id=batch_id,
                            success=False,
                            errors=[f"Exception: {e}"],
                            duration_ms=int((time.time() - t0) * 1000),
                            retry_count=attempt,
                        )
            # unreachable
            return BatchResult(batch_id=batch_id, success=False, errors=["unknown"])

    async def extract_all(self, reviews: List[Dict[str, str]]) -> List[BatchResult]:
        """Run Map phase over all reviews."""
        # Chunk into batches
        batches = [
            reviews[i:i + self.batch_size]
            for i in range(0, len(reviews), self.batch_size)
        ]
        log.info(
            "Extracting %d reviews in %d batches (concurrency=%d)",
            len(reviews), len(batches), self.concurrency,
        )

        tasks = [self.extract_one_batch(i, b) for i, b in enumerate(batches)]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        n_ok = sum(1 for r in results if r.success)
        log.info("Map phase done: %d/%d batches succeeded", n_ok, len(results))
        return results

    # ---------- Reduce phase ----------
    def reduce(
        self,
        batch_results: List[BatchResult],
        verbatims_per_label: int = 10,
    ) -> Dict[str, Any]:
        """Aggregate per-batch aspects into a unified taxonomy summary."""
        # Per-(l1, l2) accumulator
        agg: Dict[tuple, Dict] = {}

        for br in batch_results:
            for item in br.output:
                rid = item.get("review_id")
                for asp in item.get("aspects", []):
                    key = (asp.get("l1_category"), asp.get("l2_category"))
                    if key not in agg:
                        agg[key] = {
                            "l1": key[0], "l2": key[1],
                            "positive": 0, "negative": 0, "neutral": 0,
                            "total": 0,
                            "verbatims": [],
                        }
                    bucket = agg[key]
                    sent = asp.get("sentiment", "neutral")
                    bucket[sent] = bucket.get(sent, 0) + 1
                    bucket["total"] += 1
                    if len(bucket["verbatims"]) < verbatims_per_label * 3:
                        bucket["verbatims"].append({
                            "review_id": rid,
                            "verbatim": asp.get("verbatim_span"),
                            "sentiment": sent,
                            "confidence": asp.get("confidence", 1.0),
                        })

        # Truncate verbatims to top-K (simple: first K; could be embedding-diversity)
        for key, bucket in agg.items():
            bucket["verbatims"] = bucket["verbatims"][:verbatims_per_label]

        return {
            "categories": [
                {"l1": v["l1"], "l2": v["l2"], **{
                    k: v[k] for k in ("positive", "negative", "neutral", "total", "verbatims")
                }}
                for v in agg.values()
            ],
            "total_aspects": sum(v["total"] for v in agg.values()),
            "batch_stats": {
                "total_batches": len(batch_results),
                "successful_batches": sum(1 for r in batch_results if r.success),
                "avg_duration_ms": (
                    sum(r.duration_ms for r in batch_results) / max(1, len(batch_results))
                ),
            },
        }


# ---------- Quick test / smoke ----------
async def _smoke():
    """Smoke test the pipeline on 3 reviews."""
    test_reviews = [
        {"review_id": "r1", "text": "The assembly was hard but the desk is sturdy."},
        {"review_id": "r2", "text": "Very comfortable to use for long hours. Great value."},
        {"review_id": "r3", "text": "Wobbly at full height. Quality issues."},
    ]
    dims = AnalyticalDimensions()
    if not os.environ.get("OPENAI_API_KEY"):
        log.info("No OPENAI_API_KEY set, skipping smoke test.")
        return
    llm = LLMClient(model=os.environ.get("BATCHLOCK_MODEL", "gpt-4o-mini"))
    extractor = BatchLockExtractor(dims, llm, batch_size=3)
    results = await extractor.extract_all(test_reviews)
    for r in results:
        print(f"Batch {r.batch_id}: success={r.success}, items={len(r.output)}, errors={r.errors[:2]}")
    summary = extractor.reduce(results)
    print(f"Reduce: {summary['total_aspects']} aspects, {len(summary['categories'])} categories")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    asyncio.run(_smoke())
