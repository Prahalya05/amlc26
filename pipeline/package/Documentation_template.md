# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [object object]  
**Team Members:** Mithun Patil V N, Prahalya S  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary
We treat the task as blocking plus a pairwise classifier plus an assignment step:
- **Blocking:** hashed keys (name and address tokens, their combinations, a cross-script consonant skeleton) searched
  in both directions (S1 -> pool and pool -> S1), plus a near-duplicate "second hop".
- **Classifier:** LightGBM on ~66 string-similarity and list-context features.
- **Assignment:** each pool record goes to at most one S1; per-S1 thresholds are tuned directly on macro F0.5 under
  full competition (all 2.2M train S1s competing).
- **Two late additions:**
  - a fine-tuned multilingual-e5-small dense retriever recovers matches for pool records with an empty address;
  - a calibration step for the unseen country France removes over-confident matches.

Public leaderboard: 0.941 (first pipeline) -> 0.956 (v3) -> **0.959** (final).

---

## 2. Methodology

### 2.1 Problem Analysis
- **Size.** Train: 2,206,821 S1, 10,320,219 pool records (S2 + S3), 7,638,365 true pairs. Test: 1,732,544 S1,
  9,969,589 pool records.
- **Singletons:** 5.58% of S1s have no match. Predicting nothing scores 0.056.
- **Matches per S1:** mean 3.46 (max 11); S2 48%, S3 52%.
- **Every pool record belongs to at most one S1** (0 exceptions in train). A true match always has the S1's country.
- **26% of pool records belong to nobody.** These decoys are clearly synthetic:
  - their names are longer (median 28 vs 23 characters), the address is almost never empty, and they use a special
    vocabulary (westgate, southside, midtown, riverside, ... are 99.8% decoys);
  - they look like real entities, so they are the main source of hard negatives.
- **Noise in true pairs:**
  - Only 28% have the same normalised name and 13% the same normalised address.
  - Common noise: typos, OCR-like digit/letter swaps ("5ervice"), shuffled address parts, "##" prefixes, "null"
    tokens, casing, legal-form changes (LLC / Pvt Ltd), web-style names ("companyname.com").
  - 4.4% of matches have an empty address.
  - 7.2% of pairs are cross-script: the S1 name is Latin, the match is in an Indian script (18% of India pairs).
- **Test-only country:** France (15% of test S1s) never appears in train.
- **No leakage:** ids, row order and train/test overlap carry no signal (Spearman ~0, no adjacency, no shared records).

### 2.2 Solution Strategy
**Approach Type:** Blocking + Classifier + global one-owner assignment (hybrid with a dense retriever)  
**Core Innovation:**
- **Bidirectional blocking:** every pool record also searches for its best S1s (reverse lists). A record's rank in
  its own reverse list (`rev_rank`) is by far the strongest feature: 58% of the model's gain.
- **Full-competition validation:** thresholds are tuned while all 2.2M train S1s compete for the pool records with
  out-of-fold probabilities, exactly as on test.

### 2.3 System Architecture

```
 ┌──────────────────────────────── RAW DATA (tab-separated, read verbatim) ────────────────────────────────┐
 │  S1 (query entities)   S2 + S3 (pool, 10.3M train / 10.0M test)   train ground truth (7.64M pairs)        │
 └───────────────┬─────────────────────────────────────┬──────────────────────────────────────────────────┘
                 ▼                                     ▼
 ┌─ 1. NORMALISATION & RECORD PROFILE (once per record, Parquet) ────────────────────────────────────────┐
 │ NFKD + accent strip, lower, &→and, Unicode digits→ASCII, punctuation→space │ name_core (legal words   │
 │ removed) │ address numbers │ consonant skeleton (cross-script) │ Brahmic→Latin romanisation (9 scripts) │
 │ token-hash sets + IDF weights (name / address / numbers)                                              │
 └───────────────┬─────────────────────────────────────────────────────────────────────────────────────────┘
                 ▼
 ┌─ 2. BLOCKING KEYS: 64-bit hash(country ⊕ type ⊕ token), 12 key types, ~30 keys / record ─────────────────┐
 │ v1: word, word-bigram, full core name, skeleton token, address word, number, number+word                │
 │ v2: name×number, skeleton×number, number×address-word, name-without-spaces, digit-run numbers           │
 │     (bottom-k hash sketch → shuffled addresses select the same tokens)                                  │
 └───────────────┬─────────────────────────────────────────────────────────────────────────────────────────┘
                 ▼
 ┌─ 3. CANDIDATE GENERATION (inverted index, idf-weighted key score, keys with df > 1,000 dropped) ─────────┐
 │  ┌──────────────┐     ┌────────────────────────────┐     ┌──────────────────────────────────────┐      │
 │  │ FORWARD      │     │ REVERSE                     │     │ SECOND HOP                            │      │
 │  │ S1 → top-30  │  ∪  │ pool record → top-30 S1s;   │  ∪  │ near-duplicates (token-set ≥ 80 on    │      │
 │  │ pool records │     │ keep score ≥ 0.7 × best     │     │ name AND address) of each S1's top-3  │      │
 │  └──────────────┘     └────────────────────────────┘     └──────────────────────────────────────┘      │
 │  recall ceiling 0.926 ──────────────────────────────► 0.973   (oracle F0.5 0.991; 41.7 candidates / S1) │
 └───────────────┬─────────────────────────────────────────────────────────────────────────────────────────┘
                 ▼
 ┌─ 4. PAIRWISE FEATURES (66) ────────────────────────────────────────────────────────────────────────────┐
 │ name: ratio / token-set / token-sort / partial / Jaro-Winkler / exact │ skeleton + romanised similarity │
 │ address: fuzzy scores, word overlap, idf-cosine, number agreement     │ blocking: score, #keys, rank     │
 │ reverse: rank of this S1 in the record's own list (58% of gain), gap to the record's best              │
 │ list context: feature − best in the S1's list, rank in the list, #strong candidates                    │
 └───────────────┬─────────────────────────────────────────────────────────────────────────────────────────┘
                 ▼
 ┌─ 5. LightGBM (127 leaves, early stopping) ── cross-fitted: model A (fold-1 S1s) ⇄ model B (fold-0 S1s) ─┐
 │  p(match) for every (S1, pool record) pair; every train S1 gets an OUT-OF-FOLD probability             │
 └───────────────┬─────────────────────────────────────────────────────────────────────────────────────────┘
                 ▼
 ┌─ 6. GLOBAL ASSIGNMENT & DECISION (tuned on macro F0.5 with all 2.2M train S1s competing) ──────────────┐
 │  one-owner: each pool record → argmax_S1 p      then per S1:  best p < 0.68 → predict nothing            │
 │                                                              else keep p ≥ 0.7 × best                  │
 └───────────────┬─────────────────────────────────────────────────────────────────────────────────────────┘
                 │                ┌─ 7. DENSE RETRIEVER (GPU) ─────────────────────────────────────────────┐
                 │                │ multilingual-e5-small, contrastive fine-tuning (in-batch negatives,     │
                 │                │ single-country batches, empty-address pairs ×3) → cosine top-10 S1s    │
                 │                │ for every record the decision left unassigned                          │
                 │                └──────────────┬────────────────────────────────────────────────────────┘
                 ▼                               ▼
 ┌─ 8. MERGE: + (record → dense top-1) for empty-address records, cosine ≥ 0.58 and margin ≥ 0.07 ────────┐
 │     (thresholds tuned on held-out train S1s; 86% of additions correct)                                  │
 └───────────────┬─────────────────────────────────────────────────────────────────────────────────────────┘
                 ▼
 ┌─ 9. UNSEEN-COUNTRY CALIBRATION (France): match empty-row share and matches-per-row to India/US ─────────┐
 │     gate 0.982 on the row's best p; inside rows keep p ≥ 0.96 or the row's best                          │
 └───────────────┬─────────────────────────────────────────────────────────────────────────────────────────┘
                 ▼
   output/matching_results.tsv (5,675,442 matches)   output/candidate_pairs.tsv (72.2M pairs = model input)
```

Validation loop, used for every design decision: S1-level folds, candidate generation against the full pool, and
macro F0.5 measured under full competition (the one-owner rule over all train S1s with out-of-fold
probabilities). Final checks: half-split tuning, and the official validator on every output.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** 64-bit hashes of (country, key type, token). Using the country only as an opaque string
  keeps it an open set, so France is handled with no special code.
  - **v1 keys:** name-core words, adjacent name-word pairs, full core name, consonant-skeleton tokens (these work
    across scripts), address words, address numbers, number + following word.
  - **v2 combinations:** name token x address number, skeleton token x number, number x address word, name without
    spaces, numbers inside digit runs. They use a bottom-k hash sketch, so shuffled addresses pick the same tokens.
  - **Scoring:** idf-weighted key score; keys shared by more than 1,000 records are ignored.
- **Candidates:**
  - **forward:** the top-30 pool records per S1;
  - **reverse:** every pool record's top-30 S1s; the pair is added when its score is >= 0.7 x the record's best;
  - **second hop:** near-duplicates of each S1's 3 most likely matches (token_set_ratio >= 80 on name and address).
- **Candidate pairs generated:** 72,171,769 on test (mean 41.7 per S1, max 1,707). Train validation fold (100k
  S1s): 3.75M pairs.
- **How you ensured true matches were not lost:**
  - Recall ceiling (share of true pairs among the candidates) was measured on held-out S1s against the full 10.3M pool:
    forward-only 0.926 -> 0.973 with reverse + second hop. Oracle F0.5 is 0.991.
  - Every variant (28 in total) was measured before choosing.
  - Blocking is never filtered by country beyond the key, and every S1 gets candidates.

---

## 4. Matching Model

**Features used (66):**
- Name features:
  - rapidfuzz ratio / token_set / token_sort / partial on the normalised name;
  - ratio / token_set / Jaro-Winkler on the core name (legal words removed);
  - consonant-skeleton ratio / token_set / Jaccard (cross-script);
  - romanised-name similarities for Indian-script names (a hand-written transliterator for 9 Brahmic scripts);
  - shared core tokens, Jaccard, idf-weighted cosine, exact equality.
- Address features: rapidfuzz ratio / token_set / partial; word overlap / Jaccard / idf cosine; address numbers
  (counts, shared, Jaccard, missing on each side, first number equal / contained).
- Other:
  - blocking score, number of shared keys, forward rank;
  - reverse rank and gap to the record's best reverse score;
  - candidate source (S2/S3), script flags, lengths;
  - list context: each feature minus the best of this S1's list, rank inside the list, number of strong candidates.

**Model type:** LightGBM binary classifier: 127 leaves, learning rate 0.08, early stopping on 10% of the training S1s.
- Trained on the candidates of 100k fold-1 S1s: hard negatives from our own blocking.
- A cross-fit twin trained on 100k fold-0 S1s gives out-of-fold probabilities for the validation of all 2.2M S1s.
- Validation: AUC 0.99967, average precision 0.99703.

**Threshold selection method:** grid search of macro F0.5 on the 100k held-out S1s, under full competition.
1. **One-owner rule:** each pool record is kept only for the S1 that gives it the highest probability.
2. **Per-S1 rule:** predict nothing if the best probability < 0.68; otherwise keep candidates with p >= 0.7 x best.

Tuning on one half and scoring on the other gave the same thresholds (0.9681). Two alternatives were evaluated and
not used: an expected-F0.5 decision and a purely record-centric threshold.

**Dense retriever (v5d):**
- intfloat/multilingual-e5-small (MIT, 118M parameters), text "name | address".
- Fine-tuned for 1 epoch with MultipleNegativesRankingLoss on 418k (pool record -> owner S1) pairs from half of the
  fold-1 S1s: empty-address pairs x3, single-country batches, never two records of one S1 in a batch.
- Before fine-tuning it already finds the owner in its top-10 for 98.0% of records (v3 blocking 96.6%) and for 96.5%
  of cross-script records (v3 91.1%).
- Used only for pool records that v3 assigns to nobody: add (record -> dense top-1 S1) when cosine >= 0.58 and margin
  to the 2nd S1 >= 0.07, for empty-address records only.
- The thresholds were tuned on train half 1 and scored on half 2 (part-1a S1s excluded): 86% of additions correct,
  +0.00025 F0.5. With an address present, dense top-1 precision was too low (decoys), so none were added.
- 17,311 additions on test.

**France calibration (v5f, v5g):** France is unseen and scored far more confidently than India/US: 4.4% empty rows
vs 6.0%, 3.8 matches per non-empty row vs 3.5. Without French labels, we matched these two statistics to India/US:
1. empty a French row if its best p < 0.982 (4,266 rows);
2. inside French rows, keep matches with p >= 0.96 or the row's best (81,782 matches dropped).

Public leaderboard: 0.956 -> 0.957 -> 0.959. A stricter setting (0.99 / 0.99) gave no further gain (0.959).

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):**
  - Validation, 100k held-out train S1s under full competition: **0.9677** for v3 (half-split check 0.9681), vs
    0.9519 for the forward-only first pipeline. Predict-nothing floor: 0.056.
  - Public leaderboard: **0.959** (final v5g).
- **Precision / recall** (validation, v3): precision 0.991, recall 0.932. The loss is almost entirely missed matches.
- **Common false positives (wrong merges):**
  - decoy records that copy a real entity's name with small changes and a different or partial address;
  - look-alike companies sharing generic words and the same street number;
  - over-confident matches in the unseen country France (reduced by the calibration).
- **Common false negatives (missed matches):**
  - Pool records with an **empty address**: 4.4% of true pairs but 34% of misses. Only 80% of them are among the
    candidates, and the model finds 48%.
  - Owner not among the candidates at all: 2.5% of pool records; 19% for empty-address and 7% for cross-script records.
  - Heavy noise in both name and address, e.g. `Patiala Marketing Pvt. Ltd.` vs `Patia1a Márketing Pvt. Ltd.` with
    no address.

---

## 6. Conclusion
- **What worked:** bidirectional blocking and full-competition threshold tuning, i.e. treating the pool as records
  to assign, not only lists to rank. They raised the validation F0.5 from 0.952 to 0.968.
- **Remaining gap:** almost all recall: records with empty addresses, cross-script names, and decoys that look like
  real copies.
- **Lessons:** measure the recall ceiling first. Validate exactly as the test is scored (all S1s competing). Check
  unseen categories (France) with distribution statistics when no labels exist.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/`: `README.md` gives the exact run order and `requirements.txt` the pinned
versions. All source is in `src/`:
- data preparation: `m1_prepare.py`, `m3_split.py`, `keys.py`, `keys2.py`, `records.py`
- blocking: `blocking.py`, `blocking_v3.py`
- features: `features2.py`
- first pipeline: `m5_train.py`, `m6_decide.py`, `m7_infer.py`
- v3: `m9_cands.py`
- dense retriever notebook: `kaggle/kaggle_dense.py`
- final merge and calibration: `m12_dense_merge.py`

Entry points producing `output/matching_results.tsv` and `output/candidate_pairs.tsv`: `m9_cands.py write`, then
`m12_dense_merge.py merge_write`, `france_gate` and `france_pairs`. Trained models of the submitted run are in
`models_trained/`.

### B. Additional Results
| step | validation F0.5 (held-out S1s, full competition) | public LB |
|---|---|---|
| predict nothing | 0.056 | 0.056 |
| forward blocking K=30 + LightGBM + one-owner | 0.9519 | 0.941 |
| v3: + reverse blocking + second hop, retrained | 0.9677 | 0.956 |
| v5d: + dense additions (empty-address records) | +0.00025 on a held-out half | 0.956 |
| v5f: + France empty-row calibration | n/a (no French labels) | 0.957 |
| **v5g: + France match-count calibration** | n/a | **0.959** |

Owner recall@k of the v3 reverse lists (train, 7.64M owned records): @1 0.931, @10 0.966, @30 0.975; empty address
@1 0.514, @30 0.814; cross-script @30 0.931.
