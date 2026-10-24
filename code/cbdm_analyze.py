"""
Analysis for CBDM v2 runs (results/<domain>__<provider>/calls.jsonl).

Per (domain, provider, system):
  * CBLSC kappa_J: per anchor and seed, mean pairwise Jaccard of the anchor's L1 label
    sets over the batches in which it received >=1 label (>=2 such batches required);
    anchor score = mean over seeds; system score = mean over anchors with 95% percentile
    bootstrap CI (10,000 resamples). Identical to the original protocol.
  * Fleiss kappa / Krippendorff alpha: each anchor = subject, each batch = rater, code =
    sorted L1 set ('NONE' if empty); computed per (anchor set, seed) and averaged.
  * |V_L1|: distinct L1 strings emitted over all reviews of all batches (case-folded).
  * OOV rate: share of emitted aspects whose L1 is outside the closed inventory.
  * SemEval only: micro-F1 of predicted vs. gold aspect categories at entity (L1) level
    and entity#attribute level, computed per batch over all gold-annotated reviews.
Paired one-sided Wilcoxon signed-rank tests over anchors (two-sided for BL vs CL).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats
from statsmodels.stats.inter_rater import fleiss_kappa
import krippendorff

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from cbdm_run import SEMEVAL, make_dims  # noqa: E402

SYSTEMS = ["vanilla", "json_schema_only", "closed_label", "batchlock"]
NAMES = {"vanilla": "Vanilla", "json_schema_only": "JSONSchemaOnly",
         "closed_label": "ClosedLabel", "batchlock": "BatchLock"}


def norm(x):
    return str(x).strip().lower() if x is not None else None


def l1_set(labels):
    return frozenset(norm(a[0]) for a in labels if a and a[0] not in (None, ""))


def bootstrap(vals, n=10000, seed=42):
    rng = np.random.default_rng(seed)
    v = np.asarray(vals, float)
    means = rng.choice(v, size=(n, len(v)), replace=True).mean(1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def load(domain, provider):
    p = ROOT / "results" / f"{domain}__{provider}"
    meta = json.loads((p / "meta.json").read_text())
    calls = [json.loads(l) for l in (p / "calls.jsonl").open()]
    # keep the last record per key (resumed runs append)
    last = {}
    for c in calls:
        last[(c["system"], c["set"], c["seed"], c["batch"])] = c
    return meta, list(last.values())


def gold_map(domain):
    if domain == "desk":
        return {}
    revs = json.load(open(ROOT / f"data_semeval16/{domain}_reviews.json"))
    return {r["review_id"]: r["gold"] for r in revs}


def analyze(domain, provider):
    meta, calls = load(domain, provider)  # provider may carry an '@16k' budget tag
    anchors = {a: s for s, ids in enumerate(meta["anchor_sets"]) for a in ids}
    dims = make_dims(domain)
    closed_l1 = {norm(x) for x in dims.l1_labels} | {"general"}
    gold = gold_map(domain)
    out = {"domain": domain, "provider": provider, "systems": {}}

    for system in SYSTEMS:
        cs = [c for c in calls if c["system"] == system]
        if not cs:
            continue
        # anchor -> seed -> list of L1 sets (one per batch)
        per = defaultdict(lambda: defaultdict(list))
        per_strict = defaultdict(lambda: defaultdict(list))  # includes empty sets (abstentions)
        codes = defaultdict(lambda: defaultdict(dict))  # (set,seed) -> anchor -> batch -> code
        vocab, n_asp, n_oov = set(), 0, 0
        f1_batches = {"l1": [], "ea": []}
        anchor_f1 = defaultdict(lambda: defaultdict(list))  # anchor -> seed -> per-batch entity F1
        n_calls, n_valid, n_empty = len(cs), sum(c["valid"] for c in cs), 0
        n_items = sum(len(c["batch_ids"]) for c in cs)
        n_covered = sum(len(set(map(str, c["labels"])) & set(c["batch_ids"])) for c in cs)
        for c in cs:
            labels = {str(k): v for k, v in c["labels"].items()}
            if not labels:
                n_empty += 1
            for rid, labs in labels.items():
                for a in labs:
                    if a and a[0] not in (None, ""):
                        vocab.add(norm(a[0]))
                        n_asp += 1
                        n_oov += norm(a[0]) not in closed_l1
            for a, s in anchors.items():
                if s != c["set"]:
                    continue
                st = l1_set(labels.get(a, []))
                if st:
                    per[a][c["seed"]].append(st)
                per_strict[a][c["seed"]].append(st)
                codes[(c["set"], c["seed"])][a][c["batch"]] = ",".join(sorted(st)) if st else "NONE"
            if gold and system in ("closed_label", "batchlock"):
                tp = {"l1": 0, "ea": 0}; fp = dict(tp); fn = dict(tp)
                for rid in c["batch_ids"]:
                    g = gold.get(rid)
                    if not g:
                        continue
                    g_l1 = {x.split("#")[0] for x in g}
                    g_ea = set(g)
                    labs = labels.get(rid, [])
                    p_l1 = {str(a[0]).upper() for a in labs if a and a[0]} - {"GENERAL"}
                    p_ea = {f"{str(a[0]).upper()}#{str(a[1]).upper()}" for a in labs if a and a[0] and a[1]}
                    for k, P, G in (("l1", p_l1, g_l1), ("ea", p_ea, g_ea)):
                        tp[k] += len(P & G); fp[k] += len(P - G); fn[k] += len(G - P)
                    if rid in anchors and anchors[rid] == c["set"] and rid in labels:
                        d = len(p_l1) + len(g_l1)
                        anchor_f1[rid][c["seed"]].append(2 * len(p_l1 & g_l1) / d if d else 1.0)
                for k in tp:
                    d = 2 * tp[k] + fp[k] + fn[k]
                    f1_batches[k].append(2 * tp[k] / d if d else 0.0)

        anchor_scores = {}
        for a, seeds in per.items():
            vals = []
            for sd, sets in seeds.items():
                if len(sets) >= 2:
                    vals.append(np.mean([len(x & y) / len(x | y) for x, y in combinations(sets, 2)]))
            if vals:
                anchor_scores[a] = float(np.mean(vals))
        m, lo, hi = bootstrap(list(anchor_scores.values())) if anchor_scores else (None, None, None)
        strict_scores = {}
        for a, seeds in per_strict.items():
            vals = [np.mean([(len(x & y) / len(x | y)) if (x | y) else 1.0 for x, y in combinations(sets, 2)])
                    for sets in seeds.values() if len(sets) >= 2]
            if vals:
                strict_scores[a] = float(np.mean(vals))
        sm, slo, shi = bootstrap(list(strict_scores.values())) if strict_scores else (None, None, None)

        fk, ka = [], []
        for key, amap in codes.items():
            alist = sorted(amap)
            cats = sorted({v for a in alist for v in amap[a].values()})
            idx = {v: i for i, v in enumerate(cats)}
            nb = 5
            rows = []
            for a in alist:
                if len(amap[a]) < nb:  # (set, seed) not finished yet
                    continue
                row = [0] * len(cats)
                for b in range(nb):
                    row[idx[amap[a][b]]] += 1
                rows.append(row)
            if len(rows) >= 2 and len(cats) >= 2:
                fk.append(float(fleiss_kappa(np.array(rows))))
            rel = [[idx[amap[a][b]] if b in amap[a] else np.nan for a in alist] for b in range(nb)]
            try:
                ka.append(float(krippendorff.alpha(reliability_data=rel, level_of_measurement="nominal")))
            except Exception:
                pass

        # Gold-based view of drift: how much does the *same* anchor's F1 move across batches?
        f1_range = [max(v) - min(v) for seeds in anchor_f1.values() for v in seeds.values() if len(v) >= 2]
        corr = None
        common = [a for a in anchor_scores if a in anchor_f1]
        if len(common) >= 5:
            mf1 = [np.mean([x for v in anchor_f1[a].values() for x in v]) for a in common]
            rho = stats.spearmanr([anchor_scores[a] for a in common], mf1)
            corr = [float(rho.statistic), float(rho.pvalue), len(common)]

        out["systems"][system] = {
            "anchor_f1_range": [float(np.mean(f1_range)), float(np.mean(np.array(f1_range) > 0))] if f1_range else None,
            "cblsc_f1_spearman": corr,
            "cblsc": m, "ci": [lo, hi], "n_anchors": len(anchor_scores),
            "per_anchor": anchor_scores,
            "strict": sm, "strict_ci": [slo, shi], "per_anchor_strict": strict_scores,
            "fleiss": float(np.mean(fk)) if fk else None,
            "kripp": float(np.mean(ka)) if ka else None,
            "vocab_l1": len(vocab), "oov_rate": n_oov / n_asp if n_asp else None,
            "coverage": n_covered / n_items, "n_calls": n_calls, "valid_rate": n_valid / n_calls, "empty_calls": n_empty,
            "f1_l1": [float(np.mean(f1_batches["l1"])), float(np.std(f1_batches["l1"]))] if f1_batches["l1"] else None,
            "f1_ea": [float(np.mean(f1_batches["ea"])), float(np.std(f1_batches["ea"]))] if f1_batches["ea"] else None,
        }

    tests = {}
    S = out["systems"]
    for key, suffix in (("per_anchor", ""), ("per_anchor_strict", "_strict")):
      for a_sys, b_sys, alt in (("closed_label", "vanilla", "greater"), ("batchlock", "vanilla", "greater"),
                              ("json_schema_only", "vanilla", "greater"),
                              ("closed_label", "json_schema_only", "greater"),
                              ("batchlock", "closed_label", "two-sided")):
        if a_sys in S and b_sys in S:
            common = sorted(set(S[a_sys][key]) & set(S[b_sys][key]))
            x = [S[a_sys][key][k] for k in common]
            y = [S[b_sys][key][k] for k in common]
            d = np.array(x) - np.array(y)
            if len(common) >= 5 and np.any(d != 0):
                w = stats.wilcoxon(x, y, alternative=alt)
                tests[f"{a_sys}_vs_{b_sys}{suffix}"] = {"n": len(common), "W": float(w.statistic),
                                                "p": float(w.pvalue), "alt": alt,
                                                "mean_diff": float(d.mean()),
                                                "diff_ci": list(bootstrap(list(d))[1:])}
    out["tests"] = tests
    return out


def main():
    runs = sorted(p.name for p in (ROOT / "results").iterdir() if (p / "calls.jsonl").exists())
    allres = []
    for r in runs:
        domain, provider = r.split("__")
        provider_cfg = provider.split("@")[0]
        res = analyze(domain, provider)
        allres.append(res)
        print(f"\n=== {domain} / {provider}")
        for s, v in res["systems"].items():
            f1 = f" F1(E)={v['f1_l1'][0]:.3f}±{v['f1_l1'][1]:.3f} F1(E#A)={v['f1_ea'][0]:.3f}±{v['f1_ea'][1]:.3f} anchorF1range={v['anchor_f1_range']} rho={v['cblsc_f1_spearman']}" if v["f1_l1"] else ""
            print(f"  {NAMES[s]:15s} kJ={v['cblsc']:.3f} [{v['ci'][0]:.3f},{v['ci'][1]:.3f}] strict={v['strict']:.3f} [{v['strict_ci'][0]:.3f},{v['strict_ci'][1]:.3f}] n={v['n_anchors']} "
                  f"FK={v['fleiss'] if v['fleiss'] is None else round(v['fleiss'],3)} "
                  f"KA={v['kripp'] if v['kripp'] is None else round(v['kripp'],3)} |V|={v['vocab_l1']} "
                  f"oov={v['oov_rate']:.3f} cov={v['coverage']:.3f} calls={v['n_calls']} valid={v['valid_rate']:.2f} empty={v['empty_calls']}{f1}")
        for k, t in res["tests"].items():
            print(f"  {k:32s} n={t['n']} W={t['W']:.1f} p={t['p']:.2g} Δ={t['mean_diff']:+.3f} [{t['diff_ci'][0]:+.3f},{t['diff_ci'][1]:+.3f}]")
    (ROOT / "results" / "summary.json").write_text(json.dumps(allres, indent=1))


if __name__ == "__main__":
    main()
