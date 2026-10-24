"""
CBDM runner: Cross-Batch Drift Measurement.

Extends the original controlled cross-batch drift protocol
(10 anchors x 5 batches x 3 seeds, batch = 10 anchors + 20 fillers) to:
  * 3 disjoint anchor sets per domain (30 anchors total, same batch shape),
  * multiple LLM back-ends (DeepSeek-V4-Flash, Claude Opus 4.6),
  * two public SemEval-2016 Task 5 domains (restaurants, laptops) with gold
    aspect categories, enabling per-batch aspect-category F1.

Every LLM call result is cached as one JSON line, so the run is resumable.

Usage:
  python cbdm_run.py --domain desk --provider deepseek_v4_flash \
      --systems vanilla,json_schema_only,closed_label,batchlock
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import random
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from batchlock_extractor import LLMClient  # noqa: E402
from batchlock_extractor_v2 import (  # noqa: E402
    BatchLockExtractorV2, StandingDeskDimensions,
)
from closed_label_extractor import ClosedLabelExtractor  # noqa: E402
from json_schema_only_extractor import JSONSchemaOnlyExtractor  # noqa: E402
from vanilla_baseline_extractor import make_vanilla_baseline_extractor  # noqa: E402

SEEDS = [0, 42, 1337]
N_SETS = 3
N_ANCHORS = 10
N_FILLERS = 20
N_BATCHES = 5

# Provider registry: a local Python file defining PROVIDERS = {name: obj with .model,
# .api_key, .base_url}. It holds API keys and is NOT part of this release; see
# providers_example.py and the README.
CFG_PATH = Path(os.environ.get("CBDM_PROVIDERS", HERE / "providers.py"))


def load_provider(name: str):
    spec = importlib.util.spec_from_file_location("llm_cfg", CFG_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PROVIDERS[name]


class ProviderLLMClient(LLMClient):
    def __init__(self, provider: str):
        cfg = load_provider(provider)
        self.provider = "openai-compatible"
        self.model = cfg.model
        self.api_key = cfg.api_key
        self.base_url = cfg.base_url
        self.temperature = 0.0
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=600)
        # DeepSeek-V4 endpoints now default to thinking mode, which exhausts the output
        # budget on reasoning tokens; disable it to match the original no-CoT setting.
        self.extra_body = {"thinking": {"type": "disabled"}} if "deepseek.com" in self.base_url else None

    max_tokens_override = None  # set by --max-tokens; None keeps the extractor's own budget

    async def call(self, system: str, user: str, max_tokens: int = 4096) -> str:
        loop = asyncio.get_event_loop()
        max_tokens = self.max_tokens_override or max_tokens
        kwargs = dict(model=self.model, temperature=self.temperature, max_tokens=max_tokens,
                      messages=[{"role": "system", "content": system},
                                {"role": "user", "content": user}],
                      response_format={"type": "json_object"})
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body

        def _stream():
            # Streaming keeps the gateway connection alive on long JSON outputs.
            parts = []
            for chunk in self.client.chat.completions.create(stream=True, **kwargs):
                if chunk.choices and chunk.choices[0].delta.content:
                    parts.append(chunk.choices[0].delta.content)
            return "".join(parts)

        raw = await loop.run_in_executor(None, _stream)
        return unwrap_envelope(raw)


def _find_review_list(obj):
    """Depth-first search for the first list of dicts that carry a review_id."""
    if isinstance(obj, list):
        if obj and all(isinstance(x, dict) for x in obj) and any("review_id" in x for x in obj):
            return obj
        for x in obj:
            found = _find_review_list(x)
            if found is not None:
                return found
    elif isinstance(obj, dict):
        for v in obj.values():
            found = _find_review_list(v)
            if found is not None:
                return found
    return None


def unwrap_envelope(raw: str) -> str:
    """JSON mode forces a top-level object, and models pick arbitrary envelope keys
    ({"reviews": [...]}, {"type": ..., "data": {...}}). Return the per-review list if one
    exists, else the raw text unchanged. Applied identically to every system."""
    try:
        obj = json.loads(raw)
    except Exception:
        obj = _parse_concatenated(raw)
        if obj is None:
            return raw
    found = _find_review_list(obj)
    return json.dumps(found, ensure_ascii=False) if found is not None else raw


def _parse_concatenated(raw: str):
    """Parse a stream of back-to-back JSON values (JSON-lines style output)."""
    dec, i, vals = json.JSONDecoder(), 0, []
    s = raw.strip()
    while i < len(s):
        while i < len(s) and s[i] in " \t\r\n,":
            i += 1
        if i >= len(s):
            break
        try:
            v, i = dec.raw_decode(s, i)
        except Exception:
            break
        vals.append(v)
    return vals if len(vals) > 1 else None


# ------------------------------------------------------------------ domains
@dataclass
class SemEvalDimensions(StandingDeskDimensions):
    """Closed vocabulary = SemEval-2016 entity (L1) and attribute (L2) inventories."""
    domain: str = "restaurant"
    domain_keywords: str = ""
    non_domain_keywords: str = ""
    l1_labels: List[str] = field(default_factory=list)
    l2_labels: List[str] = field(default_factory=list)


SEMEVAL = {
    "restaurants": dict(
        domain="restaurant",
        domain_keywords="food, dish, meal, waiter, staff, service, ambience, menu, drinks, wine, price, location",
        non_domain_keywords="laptop, phone, furniture, book, hotel room booking",
        l1=["FOOD", "DRINKS", "SERVICE", "AMBIENCE", "LOCATION", "RESTAURANT"],
        l2=["GENERAL", "QUALITY", "PRICES", "STYLE_OPTIONS", "MISCELLANEOUS"],
    ),
    "laptops": dict(
        domain="laptop",
        domain_keywords="laptop, notebook, screen, keyboard, battery, OS, software, CPU, memory, hard drive, support",
        non_domain_keywords="restaurant, food, furniture, desk, book",
        l1=["LAPTOP", "DISPLAY", "KEYBOARD", "MOUSE", "MOTHERBOARD", "CPU", "FANS_COOLING", "PORTS",
            "MEMORY", "POWER_SUPPLY", "OPTICAL_DRIVES", "BATTERY", "GRAPHICS", "HARD_DISC",
            "MULTIMEDIA_DEVICES", "HARDWARE", "SOFTWARE", "OS", "WARRANTY", "SHIPPING", "SUPPORT", "COMPANY"],
        l2=["GENERAL", "PRICE", "QUALITY", "OPERATION_PERFORMANCE", "USABILITY", "DESIGN_FEATURES",
            "PORTABILITY", "CONNECTIVITY", "MISCELLANEOUS"],
    ),
}


def make_dims(domain: str):
    if domain == "desk":
        return StandingDeskDimensions()
    s = SEMEVAL[domain]
    return SemEvalDimensions(domain=s["domain"], domain_keywords=s["domain_keywords"],
                             non_domain_keywords=s["non_domain_keywords"],
                             l1_labels=list(s["l1"]), l2_labels=list(s["l2"]))


def load_pool(domain: str) -> List[Dict]:
    """Return candidate reviews ordered longest-first (anchors are taken from the head)."""
    if domain == "desk":
        conn = sqlite3.connect(ROOT / "reviews.db")
        rows = conn.execute("""
            SELECT r.review_id, r.text FROM raw_reviews r
            LEFT JOIN products p ON r.product_id = p.product_id
            WHERE r.text_lang = 'en' AND length(r.text) BETWEEN 100 AND 1500
              AND (p.category_l1 IN ('standing_desk', 'sit_stand_desk') OR p.category_l1 IS NULL)
              AND COALESCE(p.product_name, '') != 'Documented test SKU'  -- internal test listing
            ORDER BY length(r.text) DESC""").fetchall()
        conn.close()
        return [{"review_id": a, "text": b, "gold": None} for a, b in rows]
    revs = json.load(open(ROOT / f"data_semeval16/{domain}_reviews.json"))
    # Same length filter as the desk corpus; gold must be non-empty.
    revs = [r for r in revs if 100 <= len(r["text"]) <= 1500 and r["gold"]]
    revs.sort(key=lambda r: (-len(r["text"]), r["review_id"]))
    return revs


def anchor_sets(pool: List[Dict]):
    """Set k = reviews [10k, 10k+10) of the longest-first ordering; fillers = the rest."""
    anchors = [pool[k * N_ANCHORS:(k + 1) * N_ANCHORS] for k in range(N_SETS)]
    fillers = pool[N_SETS * N_ANCHORS:]
    return anchors, fillers


def make_batches(anchors, fillers, rng):
    """Identical to the original protocol: shuffle fillers, take disjoint slices, shuffle positions."""
    pool = fillers.copy()
    rng.shuffle(pool)
    batches = []
    for k in range(N_BATCHES):
        batch = anchors + pool[k * N_FILLERS:(k + 1) * N_FILLERS]
        rng.shuffle(batch)
        batches.append(batch)
    return batches


def make_extractor(system: str, dims, llm):
    if system == "vanilla":
        return make_vanilla_baseline_extractor(llm, batch_size=30)
    if system == "json_schema_only":
        return JSONSchemaOnlyExtractor(dims, llm, batch_size=30)
    if system == "closed_label":
        return ClosedLabelExtractor(dims, llm, batch_size=30)
    if system == "batchlock":
        return BatchLockExtractorV2(dims, llm, batch_size=30)
    raise ValueError(system)


def labels_from_output(output: List[Dict]) -> Dict[str, List[list]]:
    """review_id -> [(l1, l2, sentiment), ...]; handles BatchLock-style and Vanilla-style items."""
    out: Dict[str, List[list]] = {}
    for item in output:
        if not isinstance(item, dict):
            continue
        rid = item.get("review_id")
        if rid is None:
            continue
        rid = str(rid)
        if isinstance(item.get("aspects"), list):
            out[rid] = [[a.get("l1_category"), a.get("l2_category"), a.get("sentiment")]
                        for a in item["aspects"] if isinstance(a, dict)]
        elif "category" in item:
            out.setdefault(rid, []).append([item.get("category"), "general", item.get("sentiment")])
        else:
            out.setdefault(rid, [])
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["desk", "restaurants", "laptops"])
    ap.add_argument("--provider", required=True)
    ap.add_argument("--systems", default="vanilla,json_schema_only,closed_label,batchlock")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--sets", default="0,1,2")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="Override the Map-call output budget (default: extractor's 8,192)")
    args = ap.parse_args()

    tag = f"@{args.max_tokens // 1024}k" if args.max_tokens else ""
    out_dir = ROOT / "results" / f"{args.domain}__{args.provider}{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "calls.jsonl"
    done = set()
    if cache_path.exists():
        for line in cache_path.open():
            r = json.loads(line)
            if r.get("ok"):
                done.add((r["system"], r["set"], r["seed"], r["batch"]))

    pool = load_pool(args.domain)
    anchors_by_set, fillers = anchor_sets(pool)
    meta = {"domain": args.domain, "provider": args.provider,
            "anchor_sets": [[a["review_id"] for a in s] for s in anchors_by_set],
            "n_pool": len(pool), "n_fillers_pool": len(fillers)}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))

    dims = make_dims(args.domain)
    llm = ProviderLLMClient(args.provider)
    llm.max_tokens_override = args.max_tokens
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()

    jobs = []
    for system in args.systems.split(","):
        for s in [int(x) for x in args.sets.split(",")]:
            for seed in SEEDS:
                # Seed offset per set keeps set 0 identical to the original seed schedule.
                rng = random.Random(seed + 1000 * s)
                for b, batch in enumerate(make_batches(anchors_by_set[s], fillers, rng)):
                    if (system, s, seed, b) not in done:
                        jobs.append((system, s, seed, b, batch))
    print(f"[{args.domain}/{args.provider}] {len(jobs)} pending calls ({len(done)} cached)", flush=True)

    async def run(job):
        system, s, seed, b, batch = job
        async with sem:
            ex = make_extractor(system, dims, llm)
            ex.semaphore = asyncio.Semaphore(1)
            t0 = time.time()
            res = await ex.extract_one_batch(b, [{"review_id": r["review_id"], "text": r["text"]} for r in batch])
            # A call "succeeds" for caching purposes if the model returned parseable items;
            # validator failures are kept (same semantics as the original protocol).
            ok = bool(res.output) or not any(e.startswith("Exception") for e in res.errors)
            rec = {"system": system, "set": s, "seed": seed, "batch": b, "ok": ok,
                   "valid": res.success, "retries": res.retry_count,
                   "elapsed": round(time.time() - t0, 1), "errors": res.errors[:3],
                   "batch_ids": [r["review_id"] for r in batch],
                   "labels": labels_from_output(res.output)}
            async with lock:
                with cache_path.open("a") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"  {system} set{s} seed{seed} b{b}: ok={ok} valid={res.success} "
                  f"n={len(rec['labels'])} {rec['elapsed']}s", flush=True)

    await asyncio.gather(*(run(j) for j in jobs))
    print("DONE", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
