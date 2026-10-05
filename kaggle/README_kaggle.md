# Dense retriever on Kaggle (multilingual-e5-small) - how to run

## 1. Upload one Kaggle Dataset (e.g. named `amlc-er`) with exactly these 8 files (~2.5 GB)
| file | from | size |
|---|---|---|
| `train_source1.tsv` | `C:\amlc\student_resource\dataset\train\` | 210 MB |
| `train_source2.tsv` | same | 489 MB |
| `train_source3.tsv` | same | 504 MB |
| `train_ground_truth.tsv` | same | 127 MB |
| `test_source1.tsv` | `C:\amlc\student_resource\dataset\test\` | 175 MB |
| `test_source2.tsv` | same | 509 MB |
| `test_source3.tsv` | same | 506 MB |
| `s1_folds.tsv` | `C:\amlc\kaggle\` | 39 MB |

Folder layout inside the dataset does not matter: the script searches `DATA_DIR` recursively by file name.
Do NOT upload `kaggle\__pycache__\` (left over from a local test).

`s1_folds.tsv` columns: `entity_id`, `fold` (0-9, the split v3 uses, `pipeline/data/folds.parquet`), `part`
(fold-1 S1s split in half by a stable hash: `1a` = fine-tuning S1s, `1b` = held out; empty for other folds),
`v3_sample` (`val` = v3's 100k fold-0 validation S1s, `train` = v3's 100k fold-1 training S1s).

## 2. Notebook
1. New notebook -> *Add Input* -> your dataset.
2. Settings: **Accelerator: GPU T4 x1** (one GPU is used), **Internet: On**. The model
   `intfloat/multilingual-e5-small` (MIT licence, 118M parameters) is downloaded from Hugging Face.
3. One code cell: paste the whole of `kaggle_dense.py`. At the top, set `DATA_DIR` if needed (default
   `/kaggle/input` works for any attached dataset). `OUT_DIR` = `/kaggle/working`.
4. **Smoke run first:** keep `SMOKE = 1` (the default) and run the cell. It must finish in < 10 min
   with no error. It runs every step on 20k training pairs / 50k records and prints two EVAL tables.
   The numbers are meaningless at this size.
5. **Full run:** set `SMOKE = 0` (edit the line `SMOKE = int(os.environ.get("SMOKE", "1"))` to `SMOKE = 0`).
   Delete the smoke outputs in `/kaggle/working` first (or restart the session).
   Then use **Save Version -> Save & Run All (Commit)** so it runs in the background (up to 12 h).
   - Expected time is ~3-5 h on a T4: data loading ~5 min; 2 evaluations x ~15 min; fine-tuning ~10-20 min
     (~410k pairs, ~1,600 batches); encoding ~24M texts plus the GPU top-10 search, ~2-3 h.
   - The script stops cleanly at 11 h (`TIME_LIMIT_H`), keeping every finished part, and writes
     `STOPPED_TIME_LIMIT.txt`.

## 3. Outputs (`/kaggle/working`, download from the notebook's *Output* tab)
| file | content |
|---|---|
| `eval_before.csv`, `eval_after.csv` | owner recall@1/3/5/10 by segment (all, empty address, address present, cross-script, same script, country, S2/S3) before / after fine-tuning; 200k fold-0 queries, weighted to the population |
| `e5_small_ft/` | the fine-tuned model (sentence-transformers format) |
| `dense_test.parquet` | every TEST pool record's top-10 same-country test S1s: `record_id` (S2-/S3- id), `s1_id`, `rank` (1-10), `cosine` |
| `dense_train.parquet` | the same for TRAIN (train S1s as the index) |
| `dense_{split}_parts/` | the per-chunk parts (already merged into the file above when a split finishes; if the run stopped early, the finished parts are here) |

The log prints `EVAL before` / `EVAL after` tables (only with `FAST = 0`). Compare `@10` with the v3 reverse lists
(`python m10_v5.py measure`): v3 reaches all 0.9658, empty address 0.7408, cross-script 0.9113.

**Leakage:** records owned by part-1a S1s were used for fine-tuning. Their train lists are in-sample.
Downstream training and evaluation should use part 1b and fold 0 only.
