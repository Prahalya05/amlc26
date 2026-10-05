\# Overnight task: Amazon ML Challenge 2026 — Business Entity Resolution



You are working UNATTENDED overnight on a Windows laptop. The owner is an ML beginner who will learn from your work in the morning. Work carefully, verify everything, and document every decision in plain language.



\## Problem

First read C:\\amlc\\student\_resource\\README.md fully — it is the official problem statement (metric, output formats, rules).



\## Hard rules (breaking these = disqualification)

\- No web access, no external data/APIs/geocoding, no LLM calls anywhere in the pipeline. Use only the provided files.

\- Any pretrained model must be MIT/Apache-2.0 and <=8B params. Prefer a CPU-only pipeline with no pretrained models.

\- Never modify or delete anything under C:\\amlc\\student\_resource.

\- Treat `country` as an open set: the test set contains France, which is absent from train. No hard-coding/one-hot of {US, India}.



\## Machine constraints

\- Windows, Python 3.12 (`python`), 12 logical cores, NO GPU, 16 GB RAM but only \~5 GB free. Keep peak RAM under \~4 GB: pyarrow-backed strings / Parquet, chunked processing, `del` + `gc.collect()`.

\- Libraries: pandas, pyarrow, numpy, rapidfuzz, scikit-learn, lightgbm (pip install if missing). Nothing else without a strong reason.

\- Read TSVs with sep="\\t", quoting=csv.QUOTE\_NONE, na\_filter=False, dtype=str, encoding="utf-8". Verify row counts: train S1 2,206,821 / S2 5,034,616 / S3 5,285,603; test S1 1,732,544 / S2 4,887,273 / S3 5,082,316; train ground truth 2,206,821.

\- Write every file with encoding="utf-8". Do NOT print non-ASCII text to the console (cp1252, crashes) — write such text to files.

\- Test every long step on a small sample first. Log runtimes.



\## Folders

\- Work in C:\\amlc\\pipeline\\ (create it).

\- Final package in C:\\amlc\\submission\\ exactly as the README specifies (output/, code/business\_entity\_resolution/src, README.md, requirements.txt, Documentation\_template.md).



\## PROGRESS LOG (most important deliverable)

Maintain C:\\amlc\\pipeline\\PROGRESS.md. After every milestone append: what you did, WHY (beginner-friendly), key numbers, files produced, runtime, open problems. Keep it concise. If you stop for any reason, PROGRESS.md must let someone continue.



\## Milestones — in order; verify each before moving on

M0 Safety-net baseline: test-set matching\_results.tsv with every test S1 id and empty matches, plus candidate\_pairs.tsv likewise, in C:\\amlc\\submission\\output\_baseline\\. Run student\_resource\\utils\\validate\_submission.py on them; must PASS.



M1 Prepare: convert all TSVs to Parquet with normalized name/address columns (NFKD, strip Latin accents, lowercase, \& -> "and", punctuation -> space, Unicode digits -> ASCII, collapse spaces), address number tokens, Devanagari flags; ground truth exploded to (s1\_id, match\_id) pairs. Row-count checks. (If C:\\amlc\\work\\prepare.py exists you may review and reuse it.)



M2 Metric + EDA: F0.5 macro scorer exactly per README (singleton: empty -> 1.0, any prediction -> 0.0; empty prediction for an entity with matches -> 0.0). Unit-test with the README example (0.714). Report: singleton fraction (= predict-nothing score), matches-per-S1 distribution, S2 vs S3 share, per-country stats, whether matches always share S1's country, whether any S2/S3 id is matched to more than one S1, fraction of Devanagari names/addresses, fraction of true pairs where the names are in different scripts.



M3 Validation design: split by S1 entity (train vs validation S1 ids). Candidate generation must run against the FULL S2+S3 pools so candidate density is realistic. Document the design.



M4 Blocking v1: key-based candidate generation that scales to \~10M records within the RAM limit. On validation, measure recall ceiling (share of true pairs present in candidates), avg/max candidates per S1, runtime. Iterate; record every variant's numbers.



M5 Features + model: pairwise features (rapidfuzz name/address similarities, char n-gram TF-IDF cosine if affordable, address-number overlap/mismatch, script flags, source S2/S3, rank/gap features within each S1's candidate list). Train LightGBM on candidates from YOUR blocking (non-matches = hard negatives), training S1 entities only. Report validation AUC and feature importances.



M6 Decision layer: per S1 entity, pick matches from probabilities, tuning thresholds (empty-gate on max prob, absolute threshold, relative-to-max) DIRECTLY on validation macro F0.5. If M2 shows each S2/S3 id belongs to at most one S1, enforce it. Report validation F0.5 vs the predict-nothing floor.



M7 Test inference: full pipeline on the test set, chunked. Write output\\matching\_results.tsv and output\\candidate\_pairs.tsv (candidates = exactly what the model scored; matches must be a subset). Run the validator; must PASS.



M8 Package: code\\business\_entity\_resolution\\src (clean scripts in run order), README.md with exact reproduction steps, requirements.txt with pinned versions, and Documentation\_template.md filled in from PROGRESS.md.



If a milestone fails after reasonable attempts, write the problem in PROGRESS.md and continue with whatever still makes sense. Never leave M0 broken. Aim for correct, measured, reproducible — not fancy.

