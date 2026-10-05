# Business Entity Resolution - reproduction guide

Final submission = `output_v5g`: v3 pipeline (blocking + LightGBM + one-owner decision), plus dense-retriever additions
for empty-address records (v5d), plus a stricter calibration for the unseen country France (v5f -> v5g).
Public leaderboard: v3 0.956, v5d 0.956, v5f 0.957, **v5g 0.959**.

## Setup
- Windows 11, Python 3.12, 12 cores, no GPU, <= 4 GB RAM per job (every script has a RAM guard, `AMLC_MAXMB`).
- `pip install -r requirements.txt`
- Data layout: `<ROOT>/student_resource/dataset/{train,test}/*.tsv` (the official files, unchanged).
- Environment variables:
  - `AMLC_ROOT=<ROOT>`
  - `AMLC_WORK=<this folder>/src`, so `data/`, `models/` and `reports/` are created under `src/`.
- Run every command from `src/`. Long stages are resumable; re-running skips finished chunks or batches.
- Runtimes below are on the laptop used. Its speed varied up to 4x.

## Run order
### 1. Preparation (~40 min)
```
python m1_prepare.py            # TSV -> parquet, normalised name/address, ground-truth pairs
python m3_split.py              # S1 folds (fold 0 = validation, fold 1 = training)
python keys.py                  # blocking keys v1 (types 1-7), train + test
python keys2.py                 # blocking keys v2 (types 8-12)
python records.py train         # per-record derived data (skeleton, romanisation, token hashes, IDF)
python records.py test
```
### 2. First pipeline (production model for hop seeds, and the forward candidates of test) (~2.5 h)
```
python m5_train.py block feats train      # forward blocking K=30 for 2 x 100k S1 samples, features, LightGBM
python m6_decide.py                       # thresholds (models/decision.json)
python m7_infer.py <ROOT>/submission/output 18   # test batches -> data/test_pred_*.parquet (+ first output)
```
### 3. v3 candidates, model and decision (`m9_cands.py`, ~10 h)
```
python m9_cands.py rev train       # reverse blocking: top-30 S1s of every train pool record
python m9_cands.py bucket train
python m9_cands.py crossfit
python m9_cands.py hop_samples
python m9_cands.py eval            # chooses the candidate config -> models/v3/cand_cfg.json
python m9_cands.py build
python m9_cands.py train           # model A (fold-1 sample) + cross-fit model B
python m9_cands.py oldfull
python m9_cands.py hop train       # repeat until the log says 0 seeds left
python m9_cands.py v3full          # out-of-fold p for all 2.2M train S1s (full competition)
python m9_cands.py decide          # gate / absolute / relative thresholds -> models/v3/decision.json
python m9_cands.py rev test
python m9_cands.py bucket test
python m9_cands.py hop test        # repeat until 0 seeds left
python m9_cands.py infer
python m9_cands.py write <ROOT>/submission/output_v3
```
### 4. Dense retriever (Kaggle GPU, ~1.5 h)
```
python m10_v5.py kaggle_folds             # <ROOT>/kaggle/s1_folds.tsv
python m12_dense_merge.py open_lists      # <ROOT>/kaggle/v3_open_train.txt (records v3 leaves unassigned)
python m12_dense_merge.py open_test       # <ROOT>/kaggle/v3_open_test.txt
```
- Upload the 7 competition TSVs, `s1_folds.tsv` and the two `v3_open_*.txt` files as one Kaggle dataset.
- Run `src/kaggle/kaggle_dense.py` in a GPU notebook with `SMOKE = 0` and `FAST = 1` (details in
  `src/kaggle/README_kaggle.md`).
- Download `dense_train.parquet` and `dense_test.parquet` into `<ROOT>/kaggle/out/`.
- **Start the full run in a fresh session**, or delete `/kaggle/working/dense_*_parts` first. In the submitted run,
  leftover smoke-test parts made the notebook skip the first 200k (S2) records of each country. A clean run covers
  all open records, so it can give slightly more additions than the submitted file.

### 5. Final output
```
python m12_dense_merge.py merge_eval <ROOT>/kaggle/out
python m12_dense_merge.py merge_write <ROOT>/kaggle/out <ROOT>/submission/output_v5d
python m12_dense_merge.py france_gate <ROOT>/submission/output_v5d <ROOT>/submission/output_v5f
python m12_dense_merge.py france_pairs <ROOT>/submission/output_v5f <ROOT>/submission/output_v5g
```
- `merge_eval` tunes the dense-addition thresholds on train and only allows writing if it beats v3 on held-out S1s.
- `output_v5g/matching_results.tsv` and `candidate_pairs.tsv` are the files in `output/` of this package.
- Validate: `python <ROOT>/student_resource/utils/validate_submission.py --matching ... --candidate ... --test-dir ...`

## Source files (`src/`)
| file | role |
|---|---|
| `common.py` | paths, TSV reader, text normalisation, RAM guard |
| `m1_prepare.py`, `m3_split.py` | parquet conversion, S1 folds |
| `keys.py`, `keys2.py`, `skeleton.py` | blocking keys (name/address tokens and combinations, consonant skeleton) |
| `romanize.py`, `records.py` | Indic-script romanisation, per-record derived data |
| `blocking.py` | forward blocking (inverted index, idf-weighted key scores, top-K) |
| `blocking_v3.py` | reverse blocking, second-hop near-duplicates, pair scores |
| `features2.py` | pairwise features (rapidfuzz, token sets, numbers, list context) |
| `m4_block_eval.py` | fold sampling helper |
| `m5_train.py`, `m6_decide.py`, `m7_infer.py` | first pipeline: training, thresholds, test inference |
| `m6b_expected_f.py` | expected-F0.5 decision (evaluated, not used) + helpers |
| `m9_cands.py` | v3: candidates, retraining, full-competition validation, decision, test |
| `m10_v5.py` | diagnostics (owner recall@k), Kaggle fold file, helpers used by m12 |
| `m12_dense_merge.py` | dense additions, France calibration, final writers |
| `block_cfg.json` | blocking config (cap 1000, K 30, key types 1-12) |
| `kaggle/kaggle_dense.py`, `kaggle/README_kaggle.md` | dense retriever notebook |

`models_trained/` holds the trained LightGBM models and decision files of the submitted run, for audit. The
pipeline regenerates them.
