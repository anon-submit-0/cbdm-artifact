"""
Emit LaTeX tables and number macros from results/summary.json (written by cbdm_analyze.py),
so every number quoted in the paper is generated from the raw call logs.

Outputs (in ../generated/, or $CBDM_TABLE_DIR):
  tab_main.tex    CBLSC (kappa_J) with 95% CI, 4 systems x 3 domains x 2 LLMs
  tab_gold.tex    coverage / strict CBLSC / SemEval gold F1 for ClosedLabel vs BatchLock
  macros.tex      \newcommand macros used in running text
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("CBDM_TABLE_DIR", ROOT / "generated"))
OUT.mkdir(parents=True, exist_ok=True)

S = {(r["domain"], r["provider"]): r for r in json.load(open(ROOT / "results" / "summary.json"))}
DOM = [("desk", "Desk", "Desks"), ("restaurants", "Rest", "Restaurants"), ("laptops", "Lap", "Laptops")]
LLM = [("deepseek_v4_flash", "DS", "DS-V4-Flash"), ("claude_opus_4_6", "Op", "Opus-4.6")]
SYS = [("vanilla", "V", "Vanilla"), ("json_schema_only", "J", "JSONSchemaOnly"),
       ("closed_label", "C", "ClosedLabel"), ("batchlock", "B", "BatchLock")]


def f2(x):
    return f"{x:.2f}"


def pval(p):
    if p < 1e-3:
        e = f"{p:.0e}".split("e")
        return f"{e[0]}\\!\\times\\!10^{{{int(e[1])}}}"
    return f"{p:.3f}" if p < 0.1 else f"{p:.2f}"


def get(d, l, s):
    return S.get((d, l), {}).get("systems", {}).get(s)


macros = []


def m(name, val):
    macros.append(f"\\newcommand{{\\{name}}}{{{val}}}")


# ---- main table
rows = []
for s, sk, sname in SYS:
    cells = []
    for d, dk, _ in DOM:
        for l, lk, _ in LLM:
            v = get(d, l, s)
            if v is None or v["cblsc"] is None:
                cells.append("--")
                continue
            best = max((get(d, l, t)["cblsc"] for t, _, _ in SYS if get(d, l, t)), default=0)
            num = f2(v["cblsc"])
            num = f"\\textbf{{{num}}}" if abs(v["cblsc"] - best) < 1e-9 else num
            cells.append(f"{num}\\,{{\\scriptsize[{f2(v['ci'][0])},{f2(v['ci'][1])}]}}")
            m(f"kj{dk}{lk}{sk}", f2(v["cblsc"]))
            m(f"kjlo{dk}{lk}{sk}", f2(v["ci"][0]))
            m(f"kjhi{dk}{lk}{sk}", f2(v["ci"][1]))
            m(f"strict{dk}{lk}{sk}", f2(v["strict"]))
            m(f"cov{dk}{lk}{sk}", f"{100 * v['coverage']:.1f}")
            m(f"drop{dk}{lk}{sk}", f"{100 - 100 * v['coverage']:.0f}")
            m(f"voc{dk}{lk}{sk}", str(v["vocab_l1"]))
            m(f"oov{dk}{lk}{sk}", f"{100 * v['oov_rate']:.0f}")
            m(f"nanc{dk}{lk}{sk}", str(v["n_anchors"]))
            if v["fleiss"] is not None:
                m(f"fk{dk}{lk}{sk}", f2(v["fleiss"]))
            if v["kripp"] is not None:
                m(f"ka{dk}{lk}{sk}", f2(v["kripp"]))
            if v.get("f1_l1"):
                m(f"fe{dk}{lk}{sk}", f2(v["f1_l1"][0]))
                m(f"fesd{dk}{lk}{sk}", f2(v["f1_l1"][1]))
                m(f"fea{dk}{lk}{sk}", f2(v["f1_ea"][0]))
                m(f"feasd{dk}{lk}{sk}", f2(v["f1_ea"][1]))
                if v.get("anchor_f1_range"):
                    m(f"frng{dk}{lk}{sk}", f2(v["anchor_f1_range"][0]))
                    m(f"frngshare{dk}{lk}{sk}", f"{100 * v['anchor_f1_range'][1]:.0f}")
                if v.get("cblsc_f1_spearman"):
                    m(f"rho{dk}{lk}{sk}", f2(v["cblsc_f1_spearman"][0]))
                    m(f"rhop{dk}{lk}{sk}", pval(v["cblsc_f1_spearman"][1]))
    rows.append(f"{sname} & " + " & ".join(cells) + r" \\")

vocrow = []
for s, sk, sname in SYS:
    cells = []
    for d, dk, _ in DOM:
        for l, lk, _ in LLM:
            v = get(d, l, s)
            cells.append(str(v["vocab_l1"]) if v else "--")
    vocrow.append(f"{sname} & " + " & ".join(cells) + r" \\")

(OUT / "tab_main.tex").write_text(r"""\begin{table*}[t]
\centering\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lcccccc}
\toprule
& \multicolumn{2}{c}{Desks (deployment)} & \multicolumn{2}{c}{Restaurants (SemEval-16)} & \multicolumn{2}{c}{Laptops (SemEval-16)} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
System & DS-V4-Flash & Opus-4.6 & DS-V4-Flash & Opus-4.6 & DS-V4-Flash & Opus-4.6 \\
\midrule
\multicolumn{7}{l}{\emph{CBLSC $\kappa_J$ (mean over anchors, 95\% bootstrap CI)}} \\
""" + "\n".join(rows) + r"""
\midrule
\multicolumn{7}{l}{\emph{Distinct L1 strings emitted over all 15 batches $\times$ 3 anchor sets ($|V_{\mathrm{L1}}|$)}} \\
""" + "\n".join(vocrow) + r"""
\bottomrule
\end{tabular}
\caption{Cross-batch label-set consistency under CBDM ($3$ anchor sets $\times 10$ anchors $=30$ anchors per cell; $5$ batches $\times 3$ seeds per anchor set; $T{=}0$). All systems in a column share the LLM, anchors, fillers and batch order; only the prompt differs. The closed inventory has 12 (desks), 6 (restaurants) and 22 (laptops) L1 labels.}
\label{tab:main}
\end{table*}
""")

# ---- gold / coverage table (ClosedLabel vs BatchLock)
g = []
for d, dk, dname in DOM:
    for l, lk, lname in LLM:
        c, b = get(d, l, "closed_label"), get(d, l, "batchlock")
        if not c or not b:
            continue
        fe = (lambda v: f"{f2(v['f1_l1'][0])}" if v.get("f1_l1") else "--")
        fea = (lambda v: f"{f2(v['f1_ea'][0])}" if v.get("f1_ea") else "--")
        g.append(f"{dname} & {lname} & {100*c['coverage']:.1f} / {100*b['coverage']:.1f} & "
                 f"{f2(c['cblsc'])} / {f2(b['cblsc'])} & {f2(c['strict'])} / {f2(b['strict'])} & "
                 f"{fe(c)} / {fe(b)} & {fea(c)} / {fea(b)} \\\\")
(OUT / "tab_gold.tex").write_text(r"""\begin{table*}[t]
\centering\small
\setlength{\tabcolsep}{5pt}
\begin{tabular}{llccccc}
\toprule
Domain & LLM & Coverage (\%) & $\kappa_J$ & $\kappa_J^{\mathrm{strict}}$ & Entity F1 & Entity\#Attr.\ F1 \\
\midrule
""" + "\n".join(g) + r"""
\bottomrule
\end{tabular}
\caption{ClosedLabel / BatchLock on the axes that lenient CBLSC does not see. Coverage: share of input reviews that receive an output record. $\kappa_J^{\mathrm{strict}}$ counts a missing record as an empty label set. F1: micro-F1 against SemEval-2016 gold categories, averaged over the 45 batches (desks have no gold).}
\label{tab:gold}
\end{table*}
""")

# ---- tests
for d, dk, _ in DOM:
    for l, lk, _ in LLM:
        t = S.get((d, l), {}).get("tests", {})
        for key, name in (("batchlock_vs_closed_label", "BvC"), ("closed_label_vs_vanilla", "CvV"),
                          ("closed_label_vs_json_schema_only", "CvJ"), ("json_schema_only_vs_vanilla", "JvV"),
                          ("batchlock_vs_closed_label_strict", "BvCstrict")):
            if key in t:
                m(f"p{name}{dk}{lk}", pval(t[key]["p"]))
                m(f"d{name}{dk}{lk}", f"{t[key]['mean_diff']:+.2f}")
                m(f"n{name}{dk}{lk}", str(t[key]["n"]))

# ---- budget re-run (16k) macros and table
BUDGET_LLM = [("deepseek_v4_flash", "", "DS-V4-Flash"), ("claude_opus_4_6", "Op", "Opus-4.6")]
for d, dk, dname in DOM:
    for l, lsuf, _ in BUDGET_LLM:
        for s_, sk, _ in SYS[2:]:
            v = get(d, l + "@16k", s_)
            if not v:
                continue
            m(f"kj{dk}BigK{lsuf}{sk}", f2(v["cblsc"])); m(f"strict{dk}BigK{lsuf}{sk}", f2(v["strict"]))
            m(f"cov{dk}BigK{lsuf}{sk}", f"{100 * v['coverage']:.1f}")
            if v.get("f1_l1"):
                m(f"fe{dk}BigK{lsuf}{sk}", f2(v["f1_l1"][0])); m(f"fea{dk}BigK{lsuf}{sk}", f2(v["f1_ea"][0]))
                m(f"fesd{dk}BigK{lsuf}{sk}", f2(v["f1_l1"][1]))
        t = S.get((d, l + "@16k"), {}).get("tests", {})
        for key, name in (("batchlock_vs_closed_label", "BvC"), ("batchlock_vs_closed_label_strict", "BvCstrict")):
            if key in t:
                m(f"p{name}{dk}BigK{lsuf}", pval(t[key]["p"])); m(f"d{name}{dk}BigK{lsuf}", f"{t[key]['mean_diff']:+.2f}")
bud = []
for d, dk, dname in DOM:
    for l, _, lname in BUDGET_LLM:
        for budget, lab in ((l, "8,192"), (l + "@16k", "16,384")):
            c, b = get(d, budget, "closed_label"), get(d, budget, "batchlock")
            if not c or not b or not get(d, l + "@16k", "batchlock"):
                continue
            t = S[(d, budget)]["tests"]
            fe = (lambda v: f2(v["f1_l1"][0]) if v.get("f1_l1") else "--")
            bud.append(f"{dname} & {lname} & {lab} & {100*c['coverage']:.1f} / {100*b['coverage']:.1f} & "
                       f"{f2(c['cblsc'])} / {f2(b['cblsc'])} & {t['batchlock_vs_closed_label']['mean_diff']:+.2f} ($p{{=}}{pval(t['batchlock_vs_closed_label']['p'])}$) & "
                       f"{f2(c['strict'])} / {f2(b['strict'])} & {fe(c)} / {fe(b)} \\\\")
(OUT / "tab_budget.tex").write_text(r"""\begin{table*}[h]
\centering\small
\setlength{\tabcolsep}{5pt}
\begin{tabular}{llcccccc}
\toprule
& LLM & Budget & Cov.\ (\%) & $\kappa_J$ & B$-$C & $\kappa_J^{\mathrm{strict}}$ & Ent.\ F1 \\
\midrule
""" + "\n".join(bud) + r"""
\bottomrule
\end{tabular}
\caption{\textsc{ClosedLabel} / \textsc{BatchLock} at the deployment output budget and at twice that budget (same anchors, fillers and seeds). B$-$C: paired mean difference in $\kappa_J$ with two-sided Wilcoxon $p$ over 30 anchors.}
\label{tab:budget}
\end{table*}
""")

# ---- multi-rater / coverage / vocabulary statistics for every cell
st = []
for d, dk, dname in DOM:
    for l, lk, lname in LLM:
        for s_, sk, sname in SYS:
            v = get(d, l, s_)
            if not v:
                continue
            fk = f2(v["fleiss"]) if v["fleiss"] is not None else "--"
            ka = f2(v["kripp"]) if v["kripp"] is not None else "--"
            st.append(f"{dname} & {lname} & {sname} & {f2(v['cblsc'])} & {f2(v['strict'])} & {fk} & {ka} & "
                      f"{100*v['coverage']:.1f} & {v['vocab_l1']} & {100*v['oov_rate']:.0f} \\\\")
(OUT / "tab_stats.tex").write_text(r"""\begin{table*}[h]
\centering\footnotesize
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{lllccccccc}
\toprule
Domain & LLM & System & $\kappa_J$ & $\kappa_J^{\mathrm{strict}}$ & Fleiss $\kappa$ & Kripp.\ $\alpha$ & Cov.\ (\%) & $|V_{\mathrm{L1}}|$ & OOV (\%) \\
\midrule
""" + "\n".join(st) + r"""
\bottomrule
\end{tabular}
\caption{All statistics per cell. Fleiss' $\kappa$ and Krippendorff's $\alpha$ treat the five batches of one (anchor set, seed) as raters and the sorted L1 set of an anchor as a nominal code, and are averaged over the nine (anchor set, seed) groups; Fleiss' $\kappa$ uses only anchors with output in all five batches. OOV: share of emitted aspects whose L1 label is outside the closed inventory.}
\label{tab:stats}
\end{table*}
""")

# ---- ranges over all main-grid cells (both LLMs, three domains)
for s_, sk, _ in SYS:
    vals = [get(d, l, s_)["cblsc"] for d, _, _ in DOM for l, _, _ in LLM if get(d, l, s_)]
    if vals:
        m(f"kjmin{sk}", f2(min(vals))); m(f"kjmax{sk}", f2(max(vals))); m(f"ncells{sk}", str(len(vals)))
    covs = [get(d, l, s_)["coverage"] for d, _, _ in DOM for l, _, _ in LLM if get(d, l, s_)]
    if covs:
        m(f"dropmax{sk}", f"{100 - 100 * min(covs):.0f}")
ncalls = sum(get(d, l, s_)["n_calls"] for d, _, _ in DOM for l, _, _ in LLM for s_, _, _ in SYS if get(d, l, s_))
m("ncallsmain", f"{ncalls:,}".replace(",", "{,}"))

(OUT / "macros.tex").write_text("\n".join(macros) + "\n")
print(f"wrote {len(macros)} macros, {len(rows)} main rows, {len(g)} gold rows -> {OUT}")
