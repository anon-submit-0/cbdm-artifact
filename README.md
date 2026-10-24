# Cross-Batch Drift Measurement (CBDM): code, prompts and run logs

Anonymous artifact accompanying the submission *"Evaluating Deployed LLM Extractors
Without Gold Labels: Cross-Batch Label Consistency in Production Review Analytics"*.

CBDM embeds the same anchor reviews in several batches of different filler reviews and
scores how well each anchor's label sets agree across batches (CBLSC, mean pairwise
Jaccard, `kappa_J`). It needs no gold labels.

## Contents

| Path | What |
|---|---|
| `code/cbdm_run.py` | Runs the protocol: 3 anchor sets x 10 anchors, 5 batches x 3 seeds, 30 reviews per batch, one LLM call per batch, every call cached as one JSON line |
| `code/vanilla_baseline_extractor.py` | **Vanilla**: free-form analyst prompt |
| `code/json_schema_only_extractor.py` | **JSONSchemaOnly**: typed JSON output, no label list |
| `code/closed_label_extractor.py` | **ClosedLabel**: short prompt listing the closed L1/L2 inventory |
| `code/batchlock_extractor.py`, `code/batchlock_extractor_v2.py` | **BatchLock**: the deployed map prompt and validator (v2 is the one evaluated) |
| `code/cbdm_analyze.py` | CBLSC with bootstrap CIs, strict variant, coverage, Fleiss' kappa, Krippendorff's alpha, Wilcoxon tests, SemEval gold F1 |
| `code/make_tables.py` | Writes every table and every number quoted in the paper (LaTeX macros) from `results/summary.json` |
| `code/prepare_semeval.py` | Builds the SemEval-2016 review files from the official XML |
| `code/providers_example.py` | Template for LLM endpoint configuration |
| `results/<domain>__<llm>[@16k]/calls.jsonl` | Raw per-batch outputs of every call reported in the paper (1,260 calls; 31 additional records are failed JSONSchemaOnly attempts that were retried) |
| `results/<domain>__<llm>/meta.json` | Anchor sets |
| `results/summary.json` | Output of `cbdm_analyze.py` |

The prompts are reproduced verbatim in Appendix A of the paper and live in the extractor
files above. They were written for the deployment before this study and are used unchanged.

## Reproducing the paper's numbers from the logs (no API calls)

```bash
pip install -r requirements.txt
# 1. SemEval-2016 Task 5 (English, review level, restaurants + laptops, train + test):
#    obtain the four official XML files and put them in data_semeval16/
python code/prepare_semeval.py
# 2. Recompute all statistics from results/*/calls.jsonl
python code/cbdm_analyze.py
# 3. Regenerate the paper's tables and number macros into generated/
python code/make_tables.py
```

The tables and macros produced in step 3 are identical to those in the paper. Bootstrap
confidence intervals of paired differences printed by step 2 can differ in the third
decimal because desk review IDs are hashed (see below), which changes the iteration
order of the resampling; none of these intervals is quoted in the paper.

## Re-running the LLM calls

```bash
cp code/providers_example.py code/providers.py   # fill in endpoints and keys
python code/cbdm_run.py --domain restaurants --provider deepseek_v4_flash
python code/cbdm_run.py --domain laptops --provider deepseek_v4_flash \
    --systems closed_label,batchlock --max-tokens 16384     # output-budget re-run
```

Runs are resumable: completed calls in `calls.jsonl` are skipped.

## Deployment-domain (desk) data

The desk corpus consists of publicly visible reviews and comments that we do not
redistribute. In this release:

* review IDs are replaced by `desk::<first 16 hex digits of SHA-256(original id)>`;
* the 14 emitted label strings that name a brand or product model (all produced by the
  free-form systems Vanilla and JSONSchemaOnly) are replaced by `[REDACTED-LABEL-k]`,
  one placeholder per distinct string, so every metric is unchanged;
* validator messages that quote review text are withheld.

The desk L1/L2 inventory is in `code/batchlock_extractor_v2.py`
(`STANDING_DESK_L1`, `STANDING_DESK_L2`). `cbdm_run.py --domain desk` needs the corpus
and is included for completeness.

## License

Code: Apache-2.0. Prompts, inventories and run logs: CC BY 4.0. SemEval-2016 data are
subject to their original terms and are not included.
