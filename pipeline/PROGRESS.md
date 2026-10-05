# PROGRESS — Amazon ML Challenge 2026, Business Entity Resolution

Working folder: `C:\amlc\pipeline\`. Raw data (read-only): `C:\amlc\student_resource\`.
Scripts are run one at a time with `python C:/amlc/pipeline/<script>.py`.

## M0 — Safety-net baseline (DONE, PASS)

**What:** `m0_baseline.py` writes `C:\amlc\submission\output_baseline\matching_results.tsv` and
`candidate_pairs.tsv`: one row per test S1 id (1,732,544 rows) with an empty list.

**Why:** Predicting "no match" for everybody is a valid submission, and it scores exactly the share of
S1 entities that truly have no matches (each singleton scores 1.0, everything else 0.0). If everything
else breaks overnight there is still a file that validates.

**Check:** `validate_submission.py` -> `PASS` (1,732,544 rows, all empty). Runtime 1.7 s.

## M1 — Prepare (DONE)

**What:** `m1_prepare.py` (helpers in `common.py`) reads every TSV in 500k-row chunks (tab, no quoting,
no NaN conversion, all strings, UTF-8) and writes `data/{train,test}_s{1,2,3}.parquet` with:
`rid` (int64 id: source*10^10 + number, faster than strings), original columns, and
- `name_n` / `addr_n`: NFKD -> drop Latin accents (only U+0300-036F, so Indic vowel signs survive) ->
  lowercase -> `&`->`and` -> every Unicode digit to ASCII -> punctuation to space -> collapse spaces.
  Names delete `.`/apostrophes (`L.L.C.`->`llc`); addresses turn them into spaces (`No.53`->`no 53`).
  Address words are canonicalised (`street`->`st`, `road`->`rd`, `north`->`n` ...), so both spellings agree.
- `name_core`: name without legal/filler words (llc, pvt, limited, inc, sarl ...).
- `addr_nums`: all digit runs of the address, leading zeros stripped (`0477`->`477`; S2 adds fake zeros).
- `name_deva/addr_deva` (Devanagari present), `name_nonlatin/addr_nonlatin` (any non-Latin script).
Ground truth -> `train_gt_pairs.parquet` (one row per true (S1, match) pair) and `train_gt_s1.parquet`.

**Why:** all later steps need the same cleaned text; Parquet is ~5x faster to load and much smaller in RAM.

**Checks:** all row counts equal the expected ones (asserted in code); `rid` unique per file.
Train S1 81 s, S2 269 s, S3 510 s; test S1 198 s, S2 387 s, S3 334 s (runs were partly in parallel with
other jobs). All six source files finished and passed the row-count asserts (test S3 logged at 02:49:11 with
5,082,316 rows).
Ground truth: 2,206,821 S1 rows, **7,638,365 true pairs**.

**Extra (built for M4/M5):** `skeleton.py` - a hand-written, script-independent "consonant skeleton"
(e.g. `ram marketing` and `राम मार्केटिंग` both -> `RN NRKTNK`) so Latin names can be compared with
Devanagari/Tamil/Gujarati/Bengali... names (all Brahmic scripts share the same Unicode layout, one table
covers them). 5/5 hand tests agree.

## M3 — Validation design (DONE)

`m3_split.py` -> `data/folds.parquet`: every train S1 entity gets a random fold 0-9 (seed 2026).
- fold 0 = **validation** (220,210 S1), fold 1 = **model training** (221,343 S1), rest spare.
- The split is by S1 **entity** (never by pair), so no entity is seen in training and validation.
- Candidate generation for validation S1s always searches the **full** train S2+S3 pools (10.3M records),
  so each S1 faces the same amount of look-alike distractors as on the test set (test: 1.73M S1 vs 9.97M pool,
  train: 2.2M S1 vs 10.3M pool - similar density).
- Singleton share is the same in every fold (~5.6%), so the fold is representative.
- To keep runtime/RAM low, experiments use random subsets (e.g. 50k or 100k validation S1s).

## M2 — Metric (DONE) + EDA (NOT RUN YET)

- `metric.py`: macro F0.5 per README. It uses the identity F0.5 = 1.25*tp / (|pred| + 0.25*|true|) and the
  singleton rules. Self-test passes: the README example gives **0.7143**, the four singleton/empty edge
  cases are correct, and 2,000 random cases match a direct set-based implementation.
- Known from the folds already: **singleton share is about 5.6% -> predict-nothing floor about 0.056**.
  Mean matches per S1 is 3.46, so recall matters a lot even though F0.5 favours precision.
- `m2_eda.py` (full report -> `reports/eda_report.md`) was started but **stopped by Claude Code because
  the system ran low on memory** (three heavy jobs were running at once: M1 test S3, the EDA and a
  blocking experiment). It has not finished, so its questions are still open: one S1 per S2/S3 id?
  do matches always share the country? cross-script share?

## M4 — Blocking (code ready, first experiment stopped)

Files ready:
- `keys.py`: per record, hashed keys = (country, key type, token). Types: name_core words, name_core
  bigrams, full name_core, consonant-skeleton tokens (cross-script), address words, address numbers,
  "number + next word" (house number + street). Country is mixed into the hash as an opaque string, so
  France/unseen countries need no special handling. **Train keys are built**: `data/keys_train_s{1,2,3}.parquet`
  (S1 31.8M keys/138 s, S2 72.4M/383 s, S3 76.7M/398 s, about 14.4 keys per record). **Test keys: not built yet.**
- `blocking.py`: inverted index in numpy. Keys shared by more than `cap` pool records are ignored, each key
  is weighted by IDF, and the top-k pool records per S1 are kept by summed weight. It streams the pool
  key files twice, so memory is bounded by the postings of useful keys.
- `m4_block_eval.py NAME CFG_JSON N`: runs blocking for N validation S1s and appends recall@k and oracle
  F0.5@k (ceiling if a perfect model picked among the candidates) to `reports/blocking_experiments.md`.
- First experiment `v1_all_cap2000` (50k val S1, cap 2000, k 100) was **stopped by the memory guard**
  before producing numbers.

## M5-M7 — code drafted, not run
- `features.py` (smoke-tested on the 20k samples: 64 columns, about 26 s per 100k pairs). It uses rapidfuzz
  similarities, char-3gram and word TF-IDF cosines, address-number overlap, skeleton similarity, script
  flags and within-S1 context (gap to best, rank).
- `m5_train.py` (block -> features -> LightGBM, 150k train S1 / 100k val S1), `m6_decide.py`
  (grid search of gate / absolute / relative thresholds, with and without the one-owner rule), and
  `m7_infer.py` (batched test inference, writes both TSVs).

## Resume session (2026-09-25, from 11:25) — one job at a time, foreground only

**RAM guard added to `common.py`:** every script now logs `peak RAM ... MB` when it exits, and a watchdog
thread kills the script (exit code 3) if its working set passes `AMLC_MAXMB` (default 4300 MB). This
replaces the external memory guard that stopped the night run: a too-big job now stops itself cleanly.

### M2 EDA (DONE) — `reports/eda_report.md`, 34 s, peak RAM (not yet measured: reporter bug, fixed after)
Change: country is loaded as a small integer code instead of 10M Python strings (RAM), and a slow
per-group lambda was vectorised.
- Singleton share **0.0558** -> predict-nothing floor **0.0558**. Matches per S1: mean 3.46, max 11;
  2-5 matches is the bulk (78%).
- Matched records: S2 48.4%, S3 51.6%; about 92-93% of non-singleton S1s have >=1 match in each source.
- **Every S2/S3 id belongs to at most one S1** (0 of 7,638,365 matched ids repeat) -> the one-owner rule
  in M6 is legitimate.
- **True pairs always share the S1 country** (1.000000) -> blocking within country loses nothing.
- 74% of all pool records are a true match of some train S1 -> the pool is mostly "real" records, not noise.
- Scripts: S1 is always Latin. Non-Latin names: train S2 9.4%, S3 5.3%; test S2 11.2%, S3 6.3%.
  **7.2% of true pairs are cross-script** (India 18%, US 0%). Test adds France (259k S1, 26% of pool).
- String agreement of true pairs (300k sample): name_core exactly equal 51%, share >=1 name_core token 85%
  (India 75%), share >=1 address number 83%, share >=1 name token OR >=1 addr number **98%**.
  Match address empty 4.4%. Typos, word shuffles, OCR-like noise ("5ervice", "8uilnaigs") are common.

### M4 Blocking (DONE) — `reports/blocking_experiments.md`, `reports/blocking_diag.md`
**Recall ceiling** = share of true (S1, match) pairs that are among the top-K candidates (no model can find
a pair that blocking did not propose). **Oracle F0.5** = macro F0.5 if a perfect model picked the true
matches among the top-K. All on validation (fold 0) S1s against the FULL train pool (10.3M records).

| variant | #S1 | K=5 recall / avg cands | K=10 | K=20 | K=50 | K=100 | oracle F0.5 @20 | time | peak RAM |
|---|---|---|---|---|---|---|---|---|---|
| v1 keys, cap 2000 | 50k | 0.771 / 5.0 | 0.857 / 10.0 | 0.890 / 20.0 | 0.911 / 49.8 | 0.932 / 99.2 | 0.953 | 84 s | 1.7 GB |
| v1+v2 keys, cap 2000 | 10k | 0.831 / 5.0 | 0.899 / 10.0 | 0.920 / 20.0 | 0.936 / 50.0 | 0.950 / 99.8 | 0.971 | (see speed) | 2.4 GB |
| v1+v2 keys, cap 500 | 10k | 0.824 / 5.0 | 0.891 / 10.0 | 0.910 / 19.9 | 0.927 / 49.4 | 0.944 / 97.0 | 0.965 | 99 s | 2.4 GB |
| v1+v2 keys, cap 1000 | 10k | 0.827 / 5.0 | 0.896 / 10.0 | 0.916 / 20.0 | 0.932 / 49.8 | 0.947 / 99.0 | 0.969 | 135 s | 2.4 GB |
| **v1+v2 keys, cap 1000 (chosen)** | **50k** | **0.829 / 5.0** | **0.898 / 10.0** | **0.919 / 20.0** | **0.934 / 49.8** | **0.950 / 99.0** | **0.969** | 189 s | 2.4 GB |

(Almost every S1 has K candidates, so avg candidates per S1 ~ K; max = K; no S1 is left without candidates.)

**What changed and why:**
- *Diagnosis (v1):* of true pairs missed at K=20, 77% did share a usable key but were outranked, 22% only
  shared keys too common to use (e.g. "cure", "717", "huntington" each appear in thousands of records).
- *v2 keys* (`keys2.py`, types 8-12, ~15 extra keys/record, ~3 min for all train files): combinations of two
  common tokens are rare, so we add name-token x address-number, skeleton x address-number (works across
  scripts), address-number x address-word, the name without spaces (for web-style names like
  "weberssafecafecom") and address numbers from digit runs ("505b" -> 505). Each record uses at most
  5 name / 5 skeleton tokens, 4 numbers and 6 address words, chosen by smallest hash (a "bottom-k sketch",
  so shuffled addresses still pick the same tokens). Recall@20 +3 points, recall@5 +6 points.
- *cap* = keys carried by more than `cap` pool records are ignored. Lower cap = fewer (S1, pool) pairs to
  score. cap 1000 vs 2000: -0.4 pt recall@20, half the work. Chosen: **cap 1000**.
- *Speed fixes in `blocking.py`* (identical results, checked): (1) pandas hash lookup instead of
  `np.searchsorted` (5-6x faster); (2) **indirect sorts (argsort/lexsort/np.unique with inverse) were ~50x
  slower than direct sorts** on 20M+ values on this laptop (cache misses), so postings and candidate lists
  are now packed into single uint64 numbers and sorted directly; (3) hash groupby for scores; (4) 5M-pair
  chunks. Result: 50k S1 in 189 s (1.9M expanded pairs/s).
- Choice of K for the test set: postponed until feature speed is fixed (rule 5). Estimate for test blocking
  at cap 1000: ~3.4B expanded pairs, roughly 30-60 min in S1 chunks.

### Test blocking keys (DONE)
`keys.py test:1 test:2 test:3` (v1: S1 25.9M keys 92 s, S2 74.2M 301 s, S3 75.5M 339 s, peak 1.7 GB) and
`keys2.py test:1 test:2 test:3` (v2: 27.0M / 78.2M / 77.8M keys, 120 / 367 / 384 s, peak 1.7 GB).
Note: this laptop's speed varies a lot during the day (the same keys2 job ran ~4x faster on train an hour
earlier; a numpy benchmark also swung 5x). Timings below are therefore given as back-to-back comparisons.

### Feature speed fix (DONE) — `features2.py`, `records.py`, profiler `prof_feats.py`
**Profile of the old `features.py`** (100k candidate pairs): the 12 rapidfuzz scorers together took only
~1.4 s; the time went to per-record Python work redone in every chunk (sklearn HashingVectorizer
tokenising for TF-IDF ~12.7 s of 30.5 s, skeletons, building Python sets) and per-pair Python loops
(Jaccard, set differences, list indexing).
**New design:**
- `records.py SPLIT` computes everything that depends on ONE record ONCE and stores it in
  `data/rec_{split}_s{src}.parquet`: consonant skeleton, romanised core name (non-Latin only), token hash
  sets (core words, skeleton tokens, address words, address numbers) as uint32 lists, IDF weights and norms,
  first address number, lengths. Train: 12.5M records in 103 s (115-143k records/s), peak 1.2 GB.
  IDF comes from `data/tokdf_{split}.parquet` (word document frequencies over the full S2+S3 pool, 113 s once).
- `features2.RecordCache` only loads the needed rows; per pair we run rapidfuzz (C++, all cores) and
  vectorised set intersections on hashes (pair-mixed 64-bit keys + one hash-table membership test).
- TF-IDF cosines (char 3-gram name, word address) are replaced by IDF-weighted word cosines
  (`core_idf_cos`, `addr_idf_cos`); a first version had a bug (summed w instead of w^2, max 0.22) - fixed,
  now 0..1.
- Check: on the same 100k pairs, all 34 features shared with the old code have identical means (only
  `addr_word_*` differ slightly: tokens like "505b" are now excluded from address words; they are covered by
  the number features). Precomputed-record version = inline version (all stats equal).

| (100k pairs, back to back, 13:56) | old features.py | new features2.py |
|---|---|---|
| pairwise features | **16,797 pairs/s** | **189,114 pairs/s** (11x) |
| setup (tables / record cache) | 8 s (loads the whole pool: 3.3 GB) | 3 s (only needed rows) |
| peak RAM | 3.3 GB | 0.9 GB |

(An earlier run of the old code during the "slow" period measured 3,145 pairs/s; the new code measured
41,466 pairs/s in that period - the same ~13x ratio.)

**K for the test set (chosen after the speed fix): K = 30** (`block_cfg.json`: cap 1000, K 30, key types 1-12).
Recall ceiling 0.926 at K=30 vs 0.919 at 20 and 0.934 at 50. K=50 would give 5M+ training rows, beyond the
RAM budget for LightGBM with ~75 features; test cost at K=30 is ~52M pairs = ~5-20 min of pair features.

### Rule 6: Indic romanisation check (DONE) — `romanize.py`, `romanize_check.py`, `reports/romanize_check.md`
`romanize.py` is a hand-written transliterator for the 9 Brahmic scripts (one offset table, inherent-vowel
and virama rules, word-final schwa dropped), e.g. `जय फाइनेंस लिमिटेड` -> `jay phainens limited`,
`ರಾಜ್ ಇನ್ ಫೋಟೆಕ್` -> `raj in photek`. Test on train: 20,000 TRUE pairs where S1's name is Latin and the match
is in an Indian script, vs 19,998 RANDOM non-matching pairs (pool names shuffled), vs 20,000 HARD
non-matching pairs (non-Latin blocking candidates that are not matches). Similarity of S1 core name vs
romanised pool core name (0-100):

| feature | set | mean | p10 | p25 | p50 | p75 | p90 | AUC vs this set |
|---|---|---|---|---|---|---|---|---|
| roman token_set | TRUE | 55.0 | 37 | 44 | 53 | 63 | 76 | |
| roman token_set | random | 30.8 | 23 | 26 | 30 | 35 | 39 | 0.940 |
| roman token_set | hard | 31.5 | 22 | 26 | 30 | 35 | 40 | 0.926 |
| roman ratio | TRUE | 53.9 | 39 | 45 | 52 | 61 | 71 | |
| roman ratio | random | 30.6 | 22 | 26 | 30 | 35 | 39 | 0.959 |
| roman ratio | hard | 30.7 | 21 | 26 | 30 | 35 | 39 | 0.948 |
| (reference) skeleton ratio | TRUE / random / hard | 93.3 / 41.4 / 41.7 | | | | | | 0.981 / 0.975 |
| (reference) raw, not romanised | TRUE / random / hard | 7.9 / 6.0 / 7.0 | | | | | | 0.519 / 0.497 |

**Decision: keep** `roman_ratio` / `roman_tset` - the distributions barely overlap (86% of true pairs score
>= 40, vs 13% of random pairs). The existing consonant skeleton separates even better, so the model gets
both. Improvement made after the check (numbers above are WITHOUT it, i.e. conservative): romanised legal
words ("praivet limited", "pra li" ...) are removed from the romanised core (`romanize.roman_core`).

### M5 Features + model (DONE) — `m5_train.py`, `reports/m5_model.md`, `models/lgb_model.txt`
- `python m5_train.py block`: 100k validation S1 (fold 0) + 100k training S1 (fold 1), blocked against the
  full train pool with `block_cfg.json` (cap 1000, K=30) in two 100k-S1 chunks: 210 s, peak 2.0 GB.
  3.0M candidate pairs per fold, 10.7% are true matches.
- `python m5_train.py feats`: features2 on both folds: 3.0M pairs in 44 s / 42 s (~70k pairs/s end-to-end
  incl. record loading, context features and writing), peak 2.4 GB. 72 features.
- `python m5_train.py train`: LightGBM (lr 0.08, 127 leaves, early stopping on 10% of the TRAINING S1s),
  best iteration 805, 147 s, peak 2.9 GB.
  **Validation AUC 0.99974, average precision 0.99794.**
  Top features (share of gain): blocking rank 0.33, blocking score 0.22, roman_ratio 0.07 (for Latin pairs it
  equals the core-name ratio, for cross-script pairs it is the romanised comparison), addr token_set 0.07,
  nkeys 0.04, combo 0.03, number mismatches 0.02 each. Why so high: most of the 30 candidates of an S1 are
  obvious non-matches; the hard part is the few look-alikes, which the decision layer handles.

### M6 Decision layer (DONE) — `m6_decide.py`, `reports/m6_decision.md`, `models/decision.json`
Grid search on the 100k validation S1s (all of them, including the 5.6% singletons), 114 s:
| rule | validation macro F0.5 |
|---|---|
| predict nothing (floor) | 0.0556 |
| top-1 only if p >= 0.5 | 0.6822 |
| every candidate with p >= 0.5 | 0.9479 |
| tuned: gate g=0.67 on the best p, keep p >= 0.725 x best (abs t=0.01 not binding) | 0.9502 |
| **tuned + one-owner rule (chosen)** | **0.9503** |
Ceiling: blocking recall 0.926 (oracle F0.5 ~0.972 at K=30), so the model is ~2 points from its ceiling.
The one-owner rule (each S2/S3 record goes only to the S1 with the highest p; EDA: every S2/S3 id has at
most one S1) helps little on validation because only 100k of 2.2M train S1s compete there; on the test
set all 1.73M S1s compete, so its effect should be larger.

### M7 Test inference (DONE, validator PASS) — `m7_infer.py`, `m7_check.py`
18 batches of ~96k test S1s (block against the FULL test pool -> features2 -> LightGBM p -> saved as
`data/test_pred_{b}.parquet`), 200-550 s per batch (this laptop's speed varies 2-3x), peak RAM 2.3-2.7 GB;
~1.6 h in total for the 18 batches. Then `write_outputs`: candidate_pairs.tsv streamed batch by batch;
matches = one-owner rule + gate 0.67 + relative 0.725 (models/decision.json) on the rows with p >= 0.01.
**Problem hit and fixed:** the first version of `write_outputs` loaded all 51.9M candidate rows at once and the
RAM guard stopped it at 4.6 GB (no partial file was left). Rewritten to stream: 92 s, peak 1.8 GB, same
5,466,784 matches.

Output in `C:\amlc\submission\output\`:
- candidate_pairs.tsv: 51,931,423 candidates (mean 29.97 per S1, max 30 = K).
- matching_results.tsv: 5,466,784 matches (mean 3.155 per S1, max 13); 113,133 empty rows (6.5%).
- **Official validator: PASS** (also with `--check-ids`: all 9,969,589 S2/S3 ids valid). My own check
  (`reports/m7_check.md`): 0 rows where matches are not a subset of candidates; row order = test_source1.
- Per country (France was never seen in training): France 259k S1, 3.41 matches/S1, 4.8% empty;
  India 810k, 3.05, 7.4% empty; US 663k, 3.19, 6.2% empty -> France is handled like the others.
- Sanity vs validation: share of pairs with p > 0.5 on test batch 1 was 11.1% (val 10.6%).

## FINAL RESULT (2026-09-25, 16:10) — nothing has been submitted
| | value |
|---|---|
| **Validation macro F0.5** (100k held-out S1, full-pool candidates, tuned + one-owner) | **0.9503** |
| Floor (predict nothing = share of singleton S1s) | 0.0556 (0.0558 on all train) |
| Margin over the floor | +0.895 (17x) |
| Oracle at K=30 (perfect picking among candidates) | ~0.972 (recall ceiling 0.926) |
| Validator on `submission/output/` | PASS (with and without `--check-ids`) |

Caveats for the reader: (1) the decision thresholds were tuned on the same 100k validation S1s that
are reported, so 0.9503 is slightly optimistic (only 3 numbers were tuned on 100k entities; expect
~0.001-0.003); (2) on the test set the one-owner rule sees all 1.73M S1s competing, unlike validation;
(3) France is unseen, but nothing in the pipeline uses the country as a feature (it is only mixed into the
blocking keys as an opaque string), and its behaviour on the test output looks like US/India.

## What is NOT done (M8 packaging was outside this run's scope)
`submission/code/`, README.md, requirements.txt and Documentation_template.md are not written yet.
Run order for reproduction: `m1_prepare.py`, `m3_split.py`, `keys.py`, `keys2.py` (train + test),
`records.py train|test`, `m5_train.py` (block, feats, train), `m6_decide.py`, `m7_infer.py OUT 18`.
Helper/diagnostic scripts: m2_eda, m4_block_eval, m4_diag, romanize_check, prof_*, bench_guard, m7_check.
Old `features.py`, `m5_train.py` v1 approach and `memcheck.py` are superseded/utilities.
Note: a stray `C:\amlc\skeleton_test.txt` could not be deleted earlier (permission denied); safe to delete.

## DIAGNOSTIC (2026-09-25, after portal marked the upload of output/matching_results.tsv as "Failed") - nothing changed except this note

### (1) File sizes
- output/matching_results.tsv: 92,905,352 bytes = 92.91 MB (88.60 MiB)
- output/candidate_pairs.tsv: 691,679,341 bytes = 691.68 MB (659.64 MiB)
- output_baseline/matching_results.tsv: 24,063,927 bytes = 24.06 MB (22.95 MiB)
- output_baseline/candidate_pairs.tsv: 24,063,929 bytes = 24.06 MB (22.95 MiB)

### (2) First 3 lines of each matching_results.tsv, raw bytes (repr)
- output/:
    b'source1_entity_id\tmatched_entity_ids\n'
    b'S1-714132312\tS3-625880872,S3-867809779,S2-187020300\n'
    b'S1-106407869\tS3-585937637,S2-705547832,S3-613056593\n'
- output_baseline/:
    b'source1_entity_id\tmatched_entity_ids\n'
    b'S1-714132312\t\n'
    b'S1-106407869\t\n'

### (3) Header / line endings: identical
Byte scan of both matching_results.tsv files (same results for both unless noted):
- header bytes identical: source1_entity_id, one TAB, matched_entity_ids, then LF (no BOM: first 3 bytes are not EF BB BF).
- line endings: LF only, 0 CR bytes in either file (no CRLF). File ends with a single LF, no blank line at the end, 0 blank lines inside.
- 1,732,545 lines = 1 header + 1,732,544 rows, exactly 1 TAB on every line, 1,732,544 distinct S1 ids; 0 NUL bytes, 0 non-ASCII bytes, 0 quote characters, 0 spaces.
- every data row matches the pattern S1-<digits>, TAB, then empty or comma-separated S2-/S3-<digits> ids (0 exceptions in both files).
- only difference: content. output/ has 1,619,411 non-empty rows (max line 181 bytes); baseline has all rows empty (max line 36 bytes).

### (4) What the README specifies vs our file
- Leaderboard upload (README 'Submission Requirements' 1): upload matching_results.tsv, tab-separated, with the exact column names source1_entity_id and matched_entity_ids. File name matching_results.tsv, plain (not zipped). Only this file is scored; candidate_pairs.tsv is not uploaded to the leaderboard.
- Format rules (README 'Output Format'): columns separated by a single tab, ID lists comma separated with no quoting, one row per test S1 entity, empty list for singletons, no duplicate ids in a list, only S2-/S3- ids that exist in the test set. Example rows are shown as plain lines with a header line; the README does NOT mention BOM, line-ending style, encoding, a maximum file size or a row-count limit.
- The zip (<team_name>_submission.zip with output/, code/business_entity_resolution/, Documentation_template.md) is only the FINAL package, not the leaderboard upload.
- Our output/matching_results.tsv matches all of the above byte for byte as far as the README defines it: same header bytes, same separator, LF endings, 1,732,544 rows, no quoting. The official validator prints PASS for it, also with --check-ids (all ids exist in test S2/S3), and every matched id is inside the candidate list.

### Conclusion / what we cannot tell
- Nothing in the file's bytes or structure deviates from the README or from the M0 baseline file, whose format is identical. So the format is not the likely cause.
- The one measurable difference is size: 92.9 MB (5,466,784 ids) vs the 24.1 MB baseline. The README states no size limit, so an upload size / timeout limit of the portal is a hypothesis only, not confirmed.
- Not verifiable from here: the portal's own failure message (please copy its exact text), whether the portal rejects a whole-file scoring error such as an id it does not accept, or a transient upload error. Suggested next checks (not done): re-upload the same file unchanged to rule out a transient failure; upload the 24 MB baseline to see whether the portal accepts that; read the portal error text.

## DIAGNOSIS: where the validation F0.5 goes (2026-09-25 evening) - nothing in the pipeline changed, nothing submitted
**What:** new read-only script `diag_losses.py` (stages `decide`, `misses`, `owners`, `report`, run one at a time;
intermediates in the session scratchpad) -> full report `reports/diag_losses.md`. Uses the fold-0 validation set
exactly as M5/M6 left it (100k S1, `cand_trainval.parquet`, `val_pred.parquet`, `decision.json`).
**Checks:** decision reproduced exactly (0.95030). The blocking score was recomputed independently for the true
pairs and gives the pipeline's rank for 100% of 53,375 found pairs, so the rank/cause numbers below are exact.
Runtime: misses 23 min (mostly string post-processing), decide 7 s, owners 61 s, report 10 s. Peak RAM 1.3 GB.
Correction: the model has **64 features**, not 72 (`max_feature_idx=63`); the "72" above was a miscount.

**1. Blocking misses: 25,578 of 346,197 true pairs (7.4%), spread over 19,117 of the 100k S1s.**
- Cause: (a) no shared key at all: **30** (0.1%). (b) Every shared key has df > 1000, so the cap drops it: **5,450 (21%)**.
  (c) Scored but ranked below 30: **20,098 (79%)**. Of (c), 13% are at rank 31-50, 26% at 51-100, 52% at 101-1000
  and 8% above 1000, so raising K a little recovers little. The median (c) S1 retrieves ~1,175 pool records.
- By country the miss rate is India 9.0% vs US 6.3%. By source it's S2 7.4% vs S3 7.4%. **Cross-script names: 24.2% vs same-script 6.1%.**
  Cross-script pairs are 7% of true pairs but 24% of misses.
- **Reverse search** (from the missed S2/S3 record over all 2.2M train S1, same keys and scoring): the true S1 is
  rank 1 for **34%** of missed pairs, top 5 for 49% and **top 30 for 66%**. For (c) misses it's rank 1 for 42% and never unreachable;
  for (b) misses it's 44% unreachable. Found pairs as a control: rank 1 for 97.9%. Why: from the S1 side the true record competes
  with ~10M look-alikes, but from the record's side it competes with only 2.2M S1s. So a reverse pass (pool -> S1, then
  union with the forward candidates) is the biggest recall lever the data shows.
- **Close match:** 17.5% of missed records are near-duplicates (name_core + address token_set >= 90) of a TRUE
  match that we already predicted for the same S1. That opens a "sibling expansion" route: pool -> pool around predicted matches.
  Only 4% are close to a non-match candidate.
- 20 random missed pairs with raw strings are in the report. Typical ones: the pool record has an empty or short
  address, a web-style name ("heartguild.com", "Primeagro.Com"), OCR noise ("8ay"), or a name in Kannada/Devanagari script.

**2. Decision losses.** Of the 80,883 S1s whose true matches were all candidates (incl. 5,555 singletons),
10,585 score < 1 and lose **0.0170** in total: singletons we predicted something for 238 entities / **0.0024**; non-singletons
predicted empty 231 / **0.0023** (all 231 are gated: best p < 0.67); false positives in non-empty lists 2,259 / **0.0051**;
true candidates not picked in non-empty lists 8,135 / **0.0072** (this is the largest part; F0.5 still rewards recall). Of the true
candidates, 11,853 were below the threshold and 204 were removed by the one-owner rule.
**False positives (3,136 pairs):** 60% (1,875) are records that belong to **no** S1, so the one-owner rule can never remove them.
For the 1,091 FPs whose owner is an unblocked train S1 (fresh block + features + model run, out-of-sample), the owner
has the record as a candidate 95% of the time, with median p 0.96. In 56% of cases the owner's p beats the wrong S1's p.
Overall, 614 FPs would be removed by the one-owner rule once all S1s compete, as they do on the test set. Rough what-if
if only those are removed: 0.9503 -> 0.9520. That is an upper-side estimate, because full competition can also take true matches away.

**3. Country:** not a model feature. It is only hashed into the blocking keys, so France S1s only meet France records,
with key frequencies computed on the test pool. There's no one-hot and no unknown-category path. The thresholds and model are
uncalibrated for France (no French labels exist).

**Where the loss goes (macro F0.5):**
| step | loss | after |
|---|---|---|
| perfect | | 1.0000 |
| blocking (a) no shared key | -0.0000 | 1.0000 |
| blocking (b) only capped keys (df > 1000) | -0.0061 | 0.9939 |
| blocking (c) ranked below 30 | -0.0218 | **0.9721 = oracle** |
| covered S1s: singletons predicted | -0.0024 | 0.9697 |
| covered S1s: non-singletons predicted empty | -0.0023 | 0.9674 |
| covered S1s: false positives | -0.0051 | 0.9623 |
| covered S1s: true candidates not picked | -0.0072 | 0.9551 |
| S1s with blocking misses: predicted empty / FPs / true candidates not picked | -0.0014 / -0.0010 / -0.0024 | **0.9503** |

**Open (not done, no code changed):** candidate fixes suggested by these numbers, in order of size: reverse
pool->S1 blocking unioned with the forward list (covers (c) and part of (b)), sibling expansion around
predicted matches, then decision tuning for recall inside non-empty lists.

## M6b: alternative decision step "maximise expected F0.5" (2026-09-25, late) - tried, does NOT win, not used
**What:** new script `m6b_expected_f.py` (stages `tune`, `write OUT_DIR`); report `reports/m6b_expected_f.md`.
The current rule (`m6_decide.py`, `models/decision.json`, `m7_infer.py`) is untouched and remains the one in use.
The alternative has its own files and writer, so which script writes the output selects the decision step.
**Idea (for the beginner):** instead of fixed thresholds, compute for every S1 how good each possible answer
(empty list, best 1, best 2, ...) would be on average, given the model's probabilities, and pick the best answer.
This needs honest probabilities, so the model's p is first calibrated with isotonic regression.
**Design:** fold-0 validation S1s were split in half by a hash of the id (50k / 50k). Half 1 fits the calibration and c;
half 2 is only for reporting. Steps: (1) one-owner rule on raw p, as before. (2) Calibrated q = isotonic(p).
(3) Sort by q; the options are empty, top-1 ... top-n. Expected F0.5 with F = 1.25·TP/(|S|+0.25·G):
- **plug-in:** E[TP] = sum of q in S, G = sum of q + c, P(empty is right) = product of (1-q).
- **Monte Carlo:** 200 independent Bernoulli draws per candidate, G = sum y + c; the empty list scores 1 if sum y = 0.

c is the expected number of true matches that blocking never retrieved; it was grid-searched on half 1 over 0-1.5.
**Checks:** calibration is good on half 2 (sum of q 160,340 vs 160,294 true labels; every p-bin within ~0.02).
The current-rule score recomputed per entity equals m6's scorer, and the oracle on half 2 (0.9721) equals the diagnosis value.
Runtime 189 s, peak RAM 0.85 GB.

| decision (half-2 macro F0.5, 50k S1) | F0.5 |
|---|---|
| current rule, production thresholds (tuned on all of fold 0, so slightly optimistic here) | **0.95081** |
| current rule re-tuned on half 1 only (fair comparison; g 0.59, r 0.725) | 0.95034 |
| expected F0.5, plug-in (c = 1.0) | 0.94966 |
| expected F0.5, Monte Carlo 200 draws (c = 0.75) | 0.94969 |
| oracle (perfect picking among the candidates) | 0.97207 |

**Result:** both estimators lose to the current rule: -0.0011 vs production (paired std. error 0.00025) and -0.0007
vs the fairly re-tuned rule. The c curve on half 1 is flat (all values within 0.0005), so a wider c grid would not change
the conclusion. Likely reason (not verified): the Bernoulli draws treat candidates as independent, but look-alike
candidates of one S1 (near-duplicate S2/S3 records) are strongly correlated. List sizes are almost the same as the
current rule's (mean 3.13 vs 3.12, empty 6.3% vs 6.6%).
**Therefore, per the instruction, nothing was written:** no `models/decision_ef.json` / `calib_isotonic.json`, no
`submission/output_v2/`, no validator run. `output/` and `output_v1/` were not touched. Nothing submitted.
(`python m6b_expected_f.py write DIR` refuses to run unless `tune` found a win.)

## M9 / v3 candidate generation - OVERNIGHT RUN LOG (started 2026-09-25 22:40, unattended from 23:40)
Code (new files only): `blocking_v3.py` (reverse search, pool->pool near-duplicate search, per-pair forward/reverse
scores), `m9_cands.py` (driver; `python m9_cands.py STAGE`; stage list and order in its docstring). New data in
`data/v3/`, models in `models/v3/`, report `reports/m9_candidates.md`. The old pipeline files, data, models and
`submission/output*` are untouched. Long stages are resumable (finished batches/chunks are skipped on re-run).
Jobs run one at a time as background processes (the tool's foreground limit is 10 min); each logs its peak RAM, and
the RAM guard is set to 3.9 GB.

**[23:42] Checks done before the long runs:**
- `pair_scores` gives the pipeline's forward score / nkeys (nkeys 100% equal, score within 3e-5 on 89,924 known candidates).
- Reverse search with sorted-packed aggregation and 4 threads gives the same top-10 lists as a pandas-groupby reference
  (99.998% identical pairs, ranks equal on 99.994%, the rest are exact ties); 100k pool records in 36 s instead of 58-74 s.
- A first reverse run (top-10, only the 200k sample S1s) was stopped and replaced after the addendum. Its partial
  output `data/v3/rev_train/` (one chunk) is unused. The full-competition validation needs lists for all S1s,
  and the ratio variants and rev_rank need top-30.

**Plan and time budget:**
- **Train side:** rev train (top-30 for all 10.3M records, ~1.2 h) -> bucket -> crossfit -> hop (samples) -> eval -> build -> train.
- **Full-competition gate:** old pipeline and v3 over all 2.2M train S1s.
- **Test side:** rev test -> hop test -> infer -> write output_v3 -> check -> validator -> compare.
- **Can't fit:** choosing the variant by full-competition F0.5 would need one full train pass (~1.5-2 h) per variant.
  So the variant is chosen by recall ceiling among those that fit in RAM (the 03:30 fallback rule), and full
  competition is measured for the chosen v3 variant vs the old pipeline.

**[00:16] RUN STOPPED - reverse pass killed by Claude Code for low SYSTEM memory; not restarted.**
- `python m9_cands.py rev train` (top-30 reverse lists, all 2.2M train S1) ran 23:36-00:15 at ~2-3.5 min per
  500k-record chunk. The job's own peak RAM was 2.28 GB (under the 3.9 GB guard).
- At 00:15 Claude Code killed it because the machine as a whole was critically low on memory. The
  harness forbids restarting a job killed this way without the user asking, so the overnight run stopped here.
- **Done:** 13 of 21 chunks (pool records 0 - 6,500,000) in `data/v3/rev30_train/part_*.parquet`.
  Every file is complete: each chunk is written to `.tmp` and then renamed.
- **Not started:** everything after it (bucket, crossfit, hop_samples, eval, build, train, oldfull, hop, v3full,
  decide, test stages), P2, P3. No `output_v3`. Nothing submitted, nothing old touched.

**How to resume** (one job at a time; close other programs first to free RAM - the laptop has ~5 GB free in total):
1. `python m9_cands.py rev train` (skips the 13 finished chunks; ~8 left x ~2-3.5 min)
2. `python m9_cands.py bucket train`
3. smoke test of assemble: `python <scratchpad>/test_v3.py asm` - checks derived forward rank/score = block()'s
4. `python m9_cands.py crossfit`, `hop_samples`, `eval`, `build`, `train`
5. full-competition gate: `oldfull`, `hop train`, `v3full`, `decide` (writes `models/v3/decision.json` with `write_allowed`)
6. only if `write_allowed`: `rev test`, `bucket test`, `hop test`, `infer`, `write C:\amlc\submission\output_v3`,
   `check C:\amlc\submission\output_v3`, the official validator, then `compare` (P2)

Rough times at today's evening speed: rev train ~25 min left, oldfull ~1.5 h, v3full ~1.5-2 h, test side ~3 h.

**[2026-09-26 08:46] Resumed on the user's request.** Long loops now run in the foreground in slices
(`--budget SECONDS`: stop before starting a new chunk/batch after that time; re-run to continue). Free RAM at start: 7.1 GB.
- **[09:16] rev train DONE:** 21/21 chunks, 4,451 s of compute in total (2-7.5 min per 500k records; the laptop's speed
  varied 4x), peak 2.3 GB, 19.2 billion key hits.
- **[09:18] bucket train DONE:** 307,294,873 reverse-list rows (29.8 per pool record) regrouped into 46 S1 batches, plus
  each record's best reverse score; 120 s, 1.8 GB.
- **[09:23] assemble smoke test (3,000 fold-0 S1s) PASSED.** Forward score/nkeys/rank recomputed without block() equal
  block()'s (rank and nkeys 100%, score within 3e-5). rev_gap agrees with the stored list scores (max diff 0.013 =
  the 1/1024 weight rounding). Note: reverse top-5 adds ~70 pairs per S1 on this sample, because some S1s are in
  the top lists of very many records, so large reverse variants will not fit the batch RAM limit.
- **[11:50] crossfit DONE:** LightGBM on the fold-0 sample with the old features (2,995,463 pairs, same params),
  best iteration 694; predicts the fold-1 sample with out-of-fold AUC 0.99973 -> `models/v3/old_fold0_model.txt`,
  `data/v3/old_oof_fold1.parquet`. Training took 8,754 s (m5 needed 147 s) because the trainer used LightGBM
  `Dataset.subset()`, which proved ~60x slower. Fixed for all later trainings: two separate Datasets as in m5, rows
  loaded pre-ordered so both are copy-free slices. Peak 1.5 GB. Note: the CPU showed 100% load at 1.3 GHz
  (power-saving?), so all runtimes today are on the slow side.
- **[11:57] hop_samples DONE** (2nd attempt; the 1st hit the RAM guard at 4.5 GB with 1M seeds per chunk, now 150k):
  592,279 seed records (top-3 by out-of-fold p of the 200k sample S1s), 10.9M pool->pool comparisons, 1,060,389
  near-duplicate pairs with sim >= 80; 237 s, peak 2.9 GB.
- **[12:01] eval DONE** (206 s, 1.6 GB) -> `reports/m9_candidates.md` (full table of 28 variants). Fold-0 highlights:

| variant | recall ceiling | oracle F0.5 | mean / max cands per S1 | total pairs (fold 0) |
|---|---|---|---|---|
| forward K=30 (current) | 0.9261 | 0.9721 | 30.0 / 30 | 2,995,463 |
| + reverse top-3 / top-5 / top-10 | 0.9591 / 0.9626 / 0.9673 | 0.9859 / 0.9871 / 0.9889 | 34.6 / 40.6 / 59.1 (max 2,670-4,770) | 3.46M / 4.06M / 5.91M |
| + reverse ratio >= 0.9 / 0.8 / 0.7 x best | 0.9653 / 0.9675 / 0.9695 | 0.9880 / 0.9888 / 0.9896 | 34.0 / 35.0 / 37.3 (max 1,109-1,304) | 3.40M / 3.50M / 3.73M |
| + second hop sim >= 90 / 85 / 80 | 0.9311 / 0.9330 / 0.9348 | 0.9735 / 0.9741 / 0.9746 | ~30.1 (max 41-43) | 3.01M / 3.01M / 3.02M |
| **+ reverse ratio >= 0.7 + second hop >= 80 (chosen)** | **0.9729** | **0.9905** | 37.5 / 1,304 | 3.75M |

  All variants fit the RAM rules (LightGBM <= 2.6 GB est.; <= 2.85M pairs per 48k-S1 batch). The ratio variants beat
  top-k at far fewer pairs. The chosen variant has the best recall ceiling. Choosing by full-competition F0.5 instead
  would need one full train pass per variant, which is not affordable, so full competition is measured for this variant
  vs the old pipeline.
- **[12:07] build DONE** (331 s, peak 3.85 GB - close to the guard; later batches are half this size): fold-1 sample
  3,752,152 pairs (756,661 added by reverse/hop, 337,245 true); fold-0 sample 3,745,838 pairs (750,375 added,
  336,810 true = recall 0.9729). 66 features = the old 64 + rev_rank + rev_gap.
- **[12:15] train DONE** (498 s, 2.3 GB): model A (fold-1 sample) best iteration 652, 162 s; model B (fold-0 sample,
  predicts the fold-1 sample for the full competition) 805 iterations, 194 s. Fold-0 AUC 0.99967, AP 0.99703; fold-1
  out-of-fold AUC 0.99967. Gain share: rev_rank 0.58, score 0.10, rev_gap 0.047, rank 0.029.
- **[12:18] quick check, fold-0-only macro F0.5 (one-owner among the fold-0 sample, thresholds grid-tuned on fold 0,
  same method as the old 0.9503): old 0.95030, v3 0.96689 (+0.0166)** at g 0.68 / t 0.01 / r 0.725.
- **[14:11] oldfull DONE:** old pipeline (block K=30, old features, out-of-fold p: the cross-fit model for the fold-1
  sample, the production model for everyone else) over all 2,206,821 train S1s: 23 batches x ~2.87M pairs = 66.1M
  pairs, 12:18-14:10 (178-742 s per batch - the laptop's speed keeps changing), peak 2.7 GB -> `data/v3/oldfull/`.
- **[14:57] hop train DONE** (made resumable: 900k seeds per call): seeds for all 2.2M train S1s = 6,097,332 distinct
  records. The sample S1s' seeds from the full run are 100% identical to those used by eval/build. 11,169,303
  near-duplicate pairs in total; 2,724 s, peak 2.9 GB.
- **[16:45] v3full DONE:** v3 pipeline over all 2.2M train S1s (forward from oldfull + reverse ratio >= 0.7 + second hop
  >= 80, 66 features; out-of-fold p: model B for the fold-1 sample, model A for everyone else). 46 batches of ~48k S1s,
  1.63-2.39M pairs each (~80M in total), 123-202 s per batch, 14:57-16:45, peak 3.2 GB -> `data/v3/v3full/`.
- **[16:54] decide DONE** (498 s, 1.4 GB) -> `models/v3/decision.json`, table in `reports/m9_candidates.md`.
  Fold-0 macro F0.5 (100k validation S1s; one-owner rule, then gate g / absolute t / relative r grid-tuned):

| pipeline | fold-0 only competition | full competition (all 2.2M train S1s) | half 2, thresholds from half 1 (full comp.) |
|---|---|---|---|
| old | 0.95030 (g .67, r .725) | **0.95190** (g .59, r .70) | 0.95177 |
| v3 | 0.96689 (g .68, r .725) | **0.96769** (g .68, r .70) | 0.96810 |

  Expected F0.5 (v3, full competition, half 2): plug-in (c 1.5) 0.96752, Monte Carlo (c 1.5) 0.96774 vs the rule's
  0.96811, so **the rule stays**. For this comparison lists were cut to their 60 best candidates by calibrated q.
  Check: the full runs reproduce the stored fold-0 validation probabilities exactly (old and v3: all rows, max |dp| 0).
  Full competition slightly *raises* both scores: more competitors let the one-owner rule remove more wrong claims.
  **Gate: v3 0.96769 > old 0.95190 -> writing output_v3 is allowed.**
- **[17:39] rev test + bucket test DONE:** top-30 reverse lists for all 9,969,589 test pool records over the 1.73M test
  S1s: 2,527 s (~2 min per 500k records), 297,349,922 list rows regrouped into 36 batches in 108 s (1.8 GB).
- **[18:23] hop test DONE** (made resumable like train): seeds = top-3 per test S1 by the production model's p
  (`data/test_pred_*`), 4,898,893 distinct seed records, 10,462,856 near-duplicate pairs; 2,620 s, peak 3.1 GB.
- **[20:34] infer DONE:** 36 test batches (~48k S1 each). Forward candidates = the m7 ones reused (score/nkeys/rank
  recomputed, verified identical earlier), + reverse ratio >= 0.7 + second hop >= 80, 66 features, model A. 1.82-2.50M
  pairs per batch; 85-381 s per batch (speed swings); peak 3.7 GB on the first two batches, then features were built
  in two halves per batch (split at an S1 boundary, same results) -> <= 3.4 GB. One call ended silently mid-batch 16
  (no traceback, no RAM-guard message, probably stopped from outside); the rerun of that batch worked (resumable,
  nothing lost). -> `data/v3/test/pred_*.parquet`.
- **[20:38] write DONE -> `C:\amlc\submission\output_v3\`** (written to `output_v3_tmp_writing`, renamed when complete;
  200 s, 1.4 GB). Decision: one-owner over ALL test S1s, gate 0.68, relative 0.7, t 0.01 (models/v3/decision.json).
  - candidate_pairs.tsv: 72,169,455 candidates (mean 41.7, max 1,707 per S1; 952.5 MB). By source: forward only 38.4M,
    forward+reverse 13.5M (forward total 51.9M = the old candidate set), reverse only 19.8M, second hop only 0.36M,
    reverse+hop 0.10M.
  - matching_results.tsv: 5,745,731 matches (mean 3.32, max 28; old output 5,466,784); empty rows 99,154 = 5.7%
    (old 6.5%); 96.5 MB.
- **[20:40] checks:** own check - row order = test_source1, 0 rows whose matches are not a subset of candidates.
  **Official validator: PASS, also with --check-ids** (all ids exist in test S2/S3).
- **[20:45] compare (P2) DONE** (per country, v3, full competition, same decision):

| set | country | S1 | mean matches | empty | best p < 0.1 | best p 0.1-0.68 | best p >= 0.99 |
|---|---|---|---|---|---|---|---|
| fold 0 | India | 40,270 | 3.251 | 6.0% | 4.4% | 1.6% | 90.8% |
| fold 0 | US | 59,730 | 3.254 | 6.0% | 4.0% | 1.9% | 92.0% |
| test | India | 809,986 | 3.251 | 6.0% | 4.2% | 1.9% | 90.6% |
| test | US | 663,106 | 3.284 | 5.9% | 3.8% | 2.1% | 92.2% |
| test | France (unseen) | 259,452 | 3.605 | 4.4% | 2.0% | 2.1% | 93.6% |

  Test India/US look like fold 0. France gets more matches and fewer empty lists; with no French labels it can't be
  told whether that is real or model overconfidence. Full histogram in `reports/m9_candidates.md`.
- **P3 (round-2 key ideas) NOT done:** the rules made it conditional on P1+P2 finishing before 06:00; they finished at
  20:45 (the run was stopped overnight, see above). The code is not written; waiting for the user's go-ahead.

## AUDIT: why do many teams score ~0.998? (2026-09-26, ~21:50-22:10, READ-ONLY)
Nothing in the pipeline, data, models or submission folders was changed, and nothing was submitted. The audit scripts
and their raw outputs are in the Claude scratchpad (`a1_ids.py` ... `a7_empty_ex.py`, `a*_out.txt`, `a*_examples.txt`).
Runtime ~20 min; peak RAM 3.1 GB.

**Bottom line: no leak was found in ids, row order, train/test overlap or formatting.** Two data-generation fingerprints
were found; one of them explains a third of v3's missed pairs.

**1. Id / row structure (train ground truth, all 7.64M pairs).** No signal.
| S1 -> | Spearman on id number | Spearman on row position | median pct-rank gap, true vs random | exact same id number |
|---|---|---|---|---|
| S2 | +0.0001 | +0.0010 | 0.2929 vs 0.2928 (id), 0.2928 vs 0.2928 (row) | 0.00000 |
| S3 | +0.0002 | -0.0007 | 0.2931 vs 0.2927 (id), 0.2930 vs 0.2931 (row) | 0.00000 |
Ids inside each file are not sorted (Spearman(id, row) ~0). Share of pairs within 0.1% of each other: 0.00197 true vs 0.00199 random.

**2. Group adjacency.** The same S1's matched records are scattered at random.
Median gap between consecutive matched rows: S2 1,087,819 vs 1,085,884 random, S3 1,105,592 vs 1,107,972. Neighbouring
rows (gap <= 1) 0.00000; gap <= 100 is 0.00005 vs 0.00005. Id-number order gives the same (median gap 216M vs 216M).

**3. Test (v3 pairs with p > 0.99: 4.83M pairs, 1.59M S1s).** Same null result: Spearman within +/-0.001 for id and
row, median gaps equal to random, no adjacency. Test ids never reuse train ids.

**4. Train <-> test overlap.** No overlap.
- Test S1: 0 of 1,732,544 records match any train record on raw or normalized name+address. No test S1 has a twin in
  train S1, so there are no train matches to carry over.
- Test S2 / S3: raw 0.0000; normalized 0.0009 / 0.0008 against any train file. This is below the within-file duplicate
  rate (~2%), so it is generic-name collisions, not copied records.

**5. Exact equality (150k true pairs vs 150k hard negatives from blocking vs 150k random).**
| | true | hard neg | random |
|---|---|---|---|
| raw name equal | 0.049 | 0.014 | 0 |
| normalized name equal | 0.283 | 0.046 | 0 |
| raw address equal | 0.024 | 0.001 | 0 |
| normalized address equal | 0.128 | 0.004 | 0 |
| both normalized equal | 0.030 | **0.0000** | 0 |
Exact equality is rare even for true pairs (3%), so it cannot produce 0.998. When both fields are equal it is always a
true match (0 hard negatives out of 150k). Examples: `a2_examples.txt`, e.g. `Red Cafe|3455 Scenic Drive, Auburn, WA` <->
`Red-Cafe|3455 SCENIC DR, AUBURN, WA`.

**6. Formatting fingerprints.**
- **No difference between true pairs and non-pairs:** leading/trailing whitespace (0 everywhere), non-NFC Unicode
  (0 everywhere), double spaces, casing. S1 records are clean: singleton S1s look identical to S1s with matches.
- **(a) Never-matched pool records ("distractors", 26.6% of train S2/S3) are synthetic.** Their names are longer (median
  28 vs 23 chars), their address is almost never empty (0.27% vs 4.4%), and they use a special vocabulary. westgate,
  southside, eastgate, northside, midtown, lakeside and riverside each appear in ~0.7% of distractor names and in almost
  no matched names (ratio 65-76x). Also overseas (41x), infratech (31x), greater, exports.
  - A record with one of these 7 words is a distractor **99.8%** of the time (1.3% of the pool).
  - The words appear only in US records (3.6% of the US test pool; 0 in India and France).
  - Of v3's fold-0 false positives at the 0.68 gate, 64% are distractors and 36% belong to another S1. But false
    positives are few (3,626 vs 322,847 true positives). v3 already predicts only 0.16% of the US test records that
    contain these words.
  - Examples in `a4_out.txt` / `a5_out.txt`, e.g. `S2-326287525|American Inc Endocrinology`, `S2-646022591|Royal Foods Infra LLP`.
    Distractors in blocking: `Diaz Atlantic Fortress LLC` vs distractor `Emmalee Jasso Fortress Group|1236 Medlock Dr`.
- **(b) SUSPICIOUS / ACTIONABLE: an empty-address pool record is a true match 97.7% of the time** (S2 0.9776,
  S3 0.9768, against a 74% base rate). v3 handles these badly:
  - They are 4.4% of fold-0 true pairs (15,262 of 346,197) but **34% of all pairs v3 misses**.
  - Candidate recall is 0.80 vs 0.98 for the rest. Model recall at p >= 0.68 is **0.48** vs 0.95.
  - 10 missed examples in `a7_examples.txt`. Four were never candidates (p = None) and the others scored 0.04-0.59:
    `Optimal Supply Ventures` <-> `Optimal Supply Center|` p=0.04; `Silver Textiles Pvt. Ltd.` <-> `Silver Textiles Pvt Ltd|` p=0.59;
    `Fargo Education Program P.C.` <-> `FARGO  EDUCATION|` p=0.43; `Behavioral Health Associates Inc` <->
    `Behavioral Health Asochiaes Inc|` (not a candidate); `Patiala Marketing Pvt. Ltd.` <-> `Patia1a Marketing Pvt. Ltd.|` (not a candidate).

**Interpretation.**
- Nothing structural lets you read the answer off the files, so ~0.998 must come from better matching.
- The data looks synthetic: each S1 has noisy copies spread across S2/S3, every pool record belongs to at most one
  S1, plus template-generated distractors. With that structure, near-perfect recall is plausible if a team
  treats it as an assignment problem: nearly every non-distractor pool record goes to its best S1.
- v3's gap is almost all recall:
  - Fold-0 precision at the gate is 98.9% (3,626 FPs vs 322,847 TPs). Recall is 0.933, and the candidate ceiling is 0.973.
  - The biggest identifiable hole is name-only (empty-address) records. Blocking and the model need an address to
    match, so they miss them.
  - Next levers, not implemented: (1) name-only blocking and features for empty-address pool records, plus an
    "address empty" flag the model can use; (2) a pool-side assignment (every non-distractor pool record goes to its
    best S1); (3) a distractor flag from the fingerprint (length, vocabulary, address present).

## M10 / v5 record-centric assignment - PART A (measure) (2026-09-26, 22:45-23:00)
Code: `m10_v5.py` (`python m10_v5.py measure | rc | write_rc OUT`). New files only: `data/v5/`, `models/v5/`,
`reports/m10_v5.md`. Old data, models and outputs were only read. Nothing submitted.

**A1 `measure` (1 min, peak 3.2 GB):** where each owned train pool record's true owner ranks in its v3 reverse list
(top-30 S1s by reverse blocking score).
| segment | records | @1 | @2 | @3 | @5 | @10 | @30 | absent |
|---|---|---|---|---|---|---|---|---|
| all | 7,638,365 | 0.9313 | 0.9457 | 0.9522 | 0.9590 | **0.9658** | 0.9747 | 0.0253 |
| empty address | 337,018 | 0.5137 | 0.6028 | 0.6482 | 0.6942 | 0.7408 | 0.8143 | 0.1857 |
| address present | 7,301,347 | 0.9506 | 0.9615 | 0.9663 | 0.9712 | 0.9762 | 0.9821 | 0.0179 |
| cross-script | 551,240 | 0.8427 | 0.8710 | 0.8832 | 0.8966 | 0.9113 | 0.9305 | 0.0695 |
| same script | 7,087,125 | 0.9382 | 0.9515 | 0.9576 | 0.9638 | 0.9700 | 0.9782 | 0.0218 |
| India / US | 3.06M / 4.58M | 0.9250 / 0.9356 | | | 0.9544 / 0.9621 | 0.9618 / 0.9685 | 0.9719 / 0.9766 | |
| S2 / S3 | 3.69M / 3.94M | 0.9304 / 0.9322 | | | | 0.9651 / 0.9664 | 0.9741 / 0.9753 | |
- **Why it matters:** the owner is rank 1 for 93% of records, but 2.5% of owned records don't have their owner in
  their list at all.
- **Empty-address records are the weak spot:** owner at rank 1 only 51%, and absent from the top-30 for 19%.
- **Cross-script is second:** absent for 7%.

**A2 `rc` (5 min, peak 2.9 GB):** record-centric decision with the current v3 model.
- **Frame:** v3full out-of-fold p, full competition over all 2.2M train S1s.
- **Rule:** each record goes to its argmax S1 and is kept if p >= t.
- **Tuning:** t tuned on half 1 of fold 0, scored on half 2.

| decision | params | half 2 F0.5 | precision | recall | empty-addr recall |
|---|---|---|---|---|---|
| v3 rule (gate .68, rel .7) | as decision.json | **0.96811** | 0.9910 | 0.9316 | 0.474 |
| record-centric, one t | t = 0.68 | 0.96802 | 0.9906 | 0.9327 | 0.481 |
| record-centric + separate t for empty-address records | t = 0.68, t_empty = 0.78 | 0.96798 | 0.9911 | 0.9311 | 0.443 |
- **Result: not better (-0.00008) -> no output_v4b** (the stage refuses to write unless it wins).
- **Why:** v3's decision already applies the one-owner argmax; dropping the per-S1 gate / relative cut changes almost
  nothing. The losses sit in candidates (A1) and model scores for weak pairs, not in the decision rule.

**STOPPED before part B,** as instructed: recall@10 = 0.9658 < 0.97. Waiting for the go-ahead. Suggested B changes:
- **k per record:** B3's rule ("smallest k where the recall gain drops below 0.002") never triggers within the top-30.
  The steps gain +0.0144 (1->2), +0.0065 (2->3), +0.0068 (3->5), +0.0068 (5->10), +0.0057 (10->20) and +0.0032
  (20->30). So the rule would keep all 30, and the stored lists stop at 30. Stopping at k = 10 would lose 0.9 points
  vs @30.
- **Retrieval:** focus B2 (name-only keys) on empty-address and cross-script records.

## JOB 1: Kaggle export for a dense retriever (2026-09-26, 23:20-23:40) - DONE
Everything is in `C:\amlc\kaggle\`:
- **`s1_folds.tsv`** (`python m10_v5.py kaggle_folds`, 1 s): 2,206,821 rows with `entity_id`, `fold` (= folds.parquet,
  the v3 split), `part` and `v3_sample`.
  - `part`: fold-1 S1s split by a stable splitmix64 hash of the id. 1a = 110,671 (60,640 + 50,031 of v3's training
    sample); 1b = 110,672.
  - `v3_sample`: val (v3's 100k fold-0 sample) or train (v3's 100k fold-1 sample).
- **`kaggle_dense.py`:** one notebook file. Steps: eval before fine-tuning -> fine-tune on part-1a pairs -> save
  model -> eval after -> top-10 lists for TEST then TRAIN. It saves parts as it goes, stops cleanly at 11 h, and
  `SMOKE = 1` is the default.
  - Batching: each batch holds one country and never two pairs of the same S1. Conflicting pairs are pushed back
    in the queue.
  - Eval queries are stratified: 25k empty-address, 25k cross-script, 150k other. Each stratum is weighted back
    to its population share.
  - **Tested locally** (no GPU/torch here, so data prep only, first 300k rows of each TSV): reading, text
    building, flags, owners/folds, stratified queries, pairs x3 for empty addresses, batches (all single-country,
    no duplicate S1). The model/GPU parts are untested; the smoke run on Kaggle checks them.
- **`README_kaggle.md`:** which 8 files to upload, notebook settings, smoke run then full run, outputs, and the
  part-1a leakage note.
- **[2026-09-27 06:25] First Kaggle run (full size, in a draft session).**
  - Data checks matched: pool 10,320,219; owned share 0.7401; part-1a pairs 383,695, which became 418,101 training
    pairs (1,634 batches, 12.3% empty-address after x3).
  - **EVAL before fine-tuning** (base e5-small, 200k fold-0 queries, 568 s), owner recall@10 vs the v3 reverse lists
    (M10 A1):
    | segment | dense @10 | v3 @10 |
    |---|---|---|
    | all | 0.9797 | 0.9658 |
    | empty address | 0.7084 | 0.7408 |
    | cross-script | 0.9650 | 0.9113 |
    | India / US | 0.9714 / 0.9852 | 0.9618 / 0.9685 |
    @1 all 0.9571.
  - **Crashed at the first fine-tuning step:** `'str' object has no attribute 'to'`. The installed
    sentence-transformers adds non-tensor entries to the tokenizer output.
  - **Fixed:** only tensors are moved to the GPU; `preprocess` is used when it exists; the "before" evaluation is
    skipped when `eval_before.csv` already exists (same session).

## PLAN CHANGE (2026-09-27 ~18:35, user: "5 more hours") - v5a dropped, v5d = v3 + dense additions
- **v5a is dropped.** It needs ~10 more laptop-hours, and its early signal was small: stage-1 AP 0.9976 vs 0.9970,
  and the record-centric decision tied v3.
- **v5d:** keep all of v3's matches. For pool records v3 assigns to NOBODY, add (record -> dense top-1 S1) when the
  cosine and the margin to the 2nd-best S1 pass thresholds tuned on train.
  - The thresholds are tuned separately for empty-address records, optionally only for S1s that v3 already gave
    >= 1 match.
  - One owner per record still holds, because only unassigned records are added.
- **`m12_dense_merge.py open_lists` / `open_test`:** `kaggle/v3_open_train.txt` holds 3,143,398 train records
  (v3 assigns 7,176,821 of 10,320,219). `kaggle/v3_open_test.txt` holds 4,223,858 test records (output_v3 assigns
  5,745,731 of 9,969,589). The first try hit the RAM guard on the test side (Python string sets); it now uses
  numeric rids.
- **`kaggle_dense.py`, `FAST = 1`** (default): skips both evaluations, does TRAIN before TEST, and searches only the
  records in `v3_open_{split}.txt`.
- **`m12_dense_merge.py merge_eval DENSE_DIR`:**
  - Tuning uses all train S1s except part 1a (v3's train selection is out-of-fold everywhere; part-1a S1s and their
    records were in the dense model's training).
  - Thresholds are tuned on half 1 of those S1s and scored on half 2 against v3. The result goes to
    `models/v5/merge_v5d.json` and `reports/m12_dense_merge.md`.
- **`merge_write DENSE_DIR C:\amlc\submission\output_v5d`:** refuses unless the half-2 score beats v3.
- **Dry run with v3's own reverse top-2 as a stand-in for the dense lists (18:55, 388 s, peak 3.3 GB):**
  - v3 alone scores half 1 0.96729 / half 2 0.96733.
  - The stand-in adds nothing useful, as expected, and the gate refused to write.
  - Only 5.5% of the open records' top-1 guesses are correct: most open records are decoys, so the real gain
    depends on how well the dense cosine and margin separate decoys from owned records.
- **Write-path dry run (19:09, 660 s, peak 3.4 GB)** with a fake settings file, output into the scratchpad:
  - It reproduces output_v3's 5,745,731 matches exactly and adds the stand-in pairs.
  - Candidates 72,169,455 (= v3). **Official validator: PASS.**
  - `models/v5/merge_v5d.json` was reset to "better": false afterwards. The real `merge_eval` overwrites it.

**[21:20-21:30] Real run with the Kaggle output (`C:\amlc\kaggle\out\`)**
- **Kaggle:** fine-tuning loss 3.54 -> 0.005 over 1,634 batches (12 min). Train lists took 24 min, test ~45 min.
- **Coverage gap:** 550,000 records per split are missing, i.e. 597k open records, all S2. Cause: smoke-run part
  files were left in /kaggle/working, and the full run skipped those chunks (the first 200k S2 records of each
  country). The gap is the same on train and test, so tuning stays fair; those records just get no additions.
  - Fixed in merge: duplicate rows dropped, and only records v3 leaves unassigned are considered.
- **Dense top-1 on open train records** (2.41M, part 1a excluded): 11.7% correct overall.
  - Address present: 0.5-36% correct even at high cosine and large margin, i.e. decoys look like the S1 as much
    as true copies do. The dense model cannot beat v3 there.
  - Empty address with margin >= 0.1: 86-95% correct. With margin 0.05-0.1: 46-72%.
- **merge_eval, first try:** no additions, because of two tuner bugs: the cosine grid started at 0.70 (fine-tuned
  cosines are mostly 0.5-0.95), and the empty-address thresholds were only tuned when address-present additions
  had helped. Both fixed.
- **Tuned on half 1:** no address-present additions; empty-address additions at cosine >= 0.58 and margin >= 0.07.
  - **Half 2: 0.96758 vs v3 0.96733 (+0.00025).** 5,547 additions on half 2, 86.4% correct.
  - Gate passed -> `merge_write` to `C:\amlc\submission\output_v5d`.
- **[21:29] output_v5d written** (119 s, peak 3.6 GB):
  - v3's 5,745,731 matches + 17,311 empty-address additions (US 8,254, India 7,829, France 1,228) = 5,763,042
    matches. Empty rows 5.70% (v3 5.72%). Candidates 72,171,769 (v3 + the added pairs).
  - **Official validator: PASS.** Nothing was submitted by me.
  - Expected change vs v3: about +0.0003 (half 2 on train). France's thresholds come from US/India.
- **[21:33] Leaderboard:** v5d 0.956 = v3 0.956 (the +0.0003 is below display precision). The old pipeline was
  0.941.
  - Validation -> leaderboard gap is ~-0.011 for both pipelines.
  - India/US test stats look exactly like validation (6.0% empty rows, 3.25 matches/S1). France (unseen) differs:
    4.4% empty, 3.61 matches/S1, higher p.
  - If India/US score ~0.968, France must be ~0.89. Hypothesis: over-confidence on France, i.e. predictions for
    French S1s that should be empty.
- **[21:41] output_v5f** (`m12_dense_merge.py france_gate output_v5d output_v5f`, 86 s):
  - Same as v5d, except French S1s whose best v3 p (after one-owner) is < 0.982 are emptied.
  - This takes France's empty share from 4.36% to 6.00% (India+US 5.94%): 4,266 of 259,452 French rows emptied.
    India/US rows are identical to v5d.
  - **Validator: PASS.**
  - It's a blind calibration bet with no French labels: roughly +/-0.0025 overall, depending on whether those rows
    are singletons.
- **[21:59] Leaderboard: v5f 0.957** (v5d 0.956). The France over-confidence hypothesis holds; ~70% of the emptied
  rows were probably true singletons.
- **[22:05] output_v5g** (`m12_dense_merge.py france_pairs output_v5f output_v5g`, 139 s):
  - Same as v5f, except inside French rows, matches with v3 p < 0.96 are dropped. Each row keeps its best match;
    dense additions are kept.
  - France goes from 3.817 to 3.481 matches per non-empty row (India+US 3.483): 81,782 French matches dropped.
    India/US rows are unchanged.
  - **Validator: PASS.** Also a blind bet: gain if the dropped matches are mostly wrong, loss if mostly right.
- **[22:09] Leaderboard: v5g 0.959** (+0.002 over v5f). France was also padding rows with wrong matches.
- **[22:16] output_v5h** (`m12_dense_merge.py france_gt output_v5d output_v5h 0.99 0.99`):
  - France rows are emptied if their best v3 p < 0.99; otherwise they keep matches with p >= 0.99 plus their best
    match.
  - France: empty 6.47%, 3.246 matches per non-empty row, 787,694 matches (v5d 936,646; v5g ~849k).
    India/US are unchanged.
  - **Validator: PASS.** Next step in the same direction as v5g, slightly smaller.
  - Table of other settings: scratchpad `france_table.py`.
- **[~22:25] Leaderboard: v5h 0.959** (= v5g). The France adjustment has plateaued.
  - **Recommended final: output_v5g** (same score, the more moderate calibration: France's statistics = India/US).
  - Leaderboard history: baseline 0.056, first pipeline 0.941, v3 0.956, v5d 0.956, v5f 0.957, v5g 0.959,
    v5h 0.959. One submission is left today, kept in reserve.

## FINAL PACKAGE (2026-09-27 ~22:55)
- **`python m13_package.py TEAM_NAME`** -> `C:\amlc\submission\<TEAM_NAME>_submission.zip` (439 MB, 36 files, zip CRC
  check OK, 184 s). Built so far with the placeholder name TEAMNAME.
- **Contents:**
  - `output/`: output_v5g's matching_results.tsv + candidate_pairs.tsv (validator PASS);
  - `code/business_entity_resolution/src/`: the 19 pipeline files + block_cfg.json + kaggle/;
  - `models_trained/`, README.md (exact run order), requirements.txt (pinned);
  - `Documentation_template.md`, filled in.
- **Doc sources:** `pipeline/package/`. The team name and members in the document are still placeholders.
- **[~23:10] Rebuilt as `C:\amlc\submission\[object object]_submission.zip`** (439 MB, 36 files, CRC OK).
  - Team "[object object]", members Mithun Patil V N and Prahalya S.
  - Section 2.3 now has the architecture diagram.
  - Scanned for tool/AI mentions: none (the only hits are the "ai" vowel in romanize.py).
  - `TEAMNAME_submission.zip` is obsolete and can be deleted.

**Rerun instructions (for reference)** (put `dense_train.parquet` and `dense_test.parquet` in e.g. `C:\amlc\kaggle\out`):
1. `python m12_dense_merge.py merge_eval C:\amlc\kaggle\out` (~7 min)
2. only if the report says better: `python m12_dense_merge.py merge_write C:\amlc\kaggle\out C:\amlc\submission\output_v5d`
   (~11 min)
3. the validator, from `student_resource`: `python utils/validate_submission.py --matching ... --candidate ...
   --test-dir dataset/test`

## JOB 2: v5a = stacked record-level matcher on the v3 candidates (started 2026-09-26 23:35) - RUN LOG
Code: `features5.py` (v5 features) and `m11_v5a.py` (driver; `python m11_v5a.py STAGE [split] [--budget S]`).
New files only: `data/v5/`, `models/v5/`, `reports/m11_v5a.md` (every table in full).

**Stage order.** Each stage is resumable; finished batches are skipped.
1. `build1`, `train1`
2. `full1 train`, `diag3` (step 3 printout), `text`
3. `rec train`, `ctx train`, `train2`, `decide2`
4. Only if `decide2` writes `write_allowed: true` in `models/v5/decision_v5a.json`:
   `full1 test`, `rec test`, `ctx test`, `infer2`, `write2 C:\amlc\submission\output_v5a`, then the validator.

**Two changes from the plan, to save time:**
- **The test pass runs only after the fold-0 gate says v5a wins.** It costs ~3 h at tonight's laptop speed.
- **(Tried and dropped at 00:28)** building features and p1 in `full1 train` only for the pairs of records in a
  fold-0 or fold-1 sample list.
  - It truncates the lists of competing S1s, so their S1-side and cluster context would be computed over partial
    lists and bias the full-competition evaluation.
  - It would only have saved the feature time (~40 s per batch); candidate assembly (~140 s per batch) runs anyway.
  - The first run was stopped after ~2 min, before any file was written. The restart (00:29) computes ALL pairs.
- **[01:26] full1 train KILLED by Claude Code because the SYSTEM ran low on memory, not a bug in the job.**
  - It ran 00:26-01:26 at ~4.2 min per batch. The job itself was under its own 3.9 GB guard.
  - The harness forbids restarting it without the user asking, so JOB 2 is paused here.
  - **Done:** 13 of 46 batches (`data/v5/p1_train/b_000..b_012.parquet` + their feature files). Each batch is
    written to .tmp and then renamed, so every finished file is complete. A leftover `f1_train/b_013_*` feature file
    is overwritten on rerun.
  - **To resume** (close other programs first): `python m11_v5a.py full1 train`. It skips the 13 finished batches;
    ~33 x 4.2 min = ~2.3 h are left. Then continue with the stage order above.

- **[23:36] features5 check** (30k fold-0 pairs vs the stored v3 features):
  - Values are identical wherever they are not masked.
  - NaN share: address comparisons 10.4% (pool address empty), skeleton 2.2%; name/core 0% (never empty here).
  - pool_addr_empty is 10.4% of candidates but only 3.8% of positives.
- **[23:40-23:55] build1:** the RAM guard (3.9 GB) stopped it twice, cleanly. The fix is to assemble and featurise
  each 100k sample in two S1 halves, which gives identical results per S1.
  - Fold 1: 3,752,152 pairs (= v3). Assembly took 136-146 s per 50k S1s, features 96-113 s per ~2M pairs, peak 3.1 GB.
  - The laptop is in a slow phase (~3x slower than yesterday's best).
  - Fold 0: 3,745,838 pairs / 336,810 positives (= v3). The rerun took 626 s, peak 3.1 GB.
- **[00:25] train1** (1,109 s, peak 2.4 GB):
  - Model A (fold-1 sample, best iteration 677) on the fold-0 sample: AUC 0.99974 / AP 0.99762, vs the v3 model on
    the same pairs 0.99967 / 0.99703.
  - Model B (fold-0 sample, best iteration 653) on the fold-1 sample: AUC 0.99973.
  - Gain share of the new columns: pool_addr_empty 0 (the NaNs already carry it), name lengths/tokens 1.6% in total.
    rev_rank 0.62, rev_gap 0.066.
- **[00:26] full1 train started** (46 batches, resumable). Outputs: `data/v5/f1_train/b_XXX_{0,1}.parquet` (features)
  and `data/v5/p1_train/b_XXX.parquet` (s1_row, pidx, src, label, pA, pB, p1).
