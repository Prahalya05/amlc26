"""M5: candidates -> features -> LightGBM matcher (train on fold 1 S1s, report on fold 0 = validation).

Usage: python m5_train.py [stage ...]   stages: block feats train  (default: all three)
Blocking config: block_cfg.json (cap 1000, top K=30 candidates per S1, key types 1-12).
Outputs (data/): cand_trainval.parquet, feats_trn.parquet, feats_val.parquet, val_pred.parquet
         (models/): lgb_model.txt, feature_list.txt   (reports/): m5_model.md
"""
import gc
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, roc_auc_score

from blocking import block_chunked
from common import DATA, WORK, log
from features2 import build_features, feature_columns
from m4_block_eval import s1_rows_for_fold

BLOCK_CFG = json.load(open(os.path.join(WORK, "block_cfg.json")))
N_VAL = 100_000     # validation S1 entities (fold 0) used for model evaluation + threshold tuning
N_TRN = 100_000     # training S1 entities (fold 1); 100k x 30 candidates = 3M rows fits the RAM budget
MODELS = os.path.join(WORK, "models")
REP = os.path.join(WORK, "reports")
os.makedirs(MODELS, exist_ok=True)
CAND = os.path.join(DATA, "cand_trainval.parquet")
FEATS = {1: os.path.join(DATA, "feats_trn.parquet"), 0: os.path.join(DATA, "feats_val.parquet")}
LGB_PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  num_threads=10, verbose=-1, max_bin=255)


def stage_block():
    t0 = time.time()
    vrows, s1_rid = s1_rows_for_fold("train", [0], N_VAL, seed=1)
    trows, _ = s1_rows_for_fold("train", [1], N_TRN, seed=2)
    rows = np.sort(np.concatenate([vrows, trows]))
    cand = block_chunked("train", rows, BLOCK_CFG, chunk=100_000)
    cand["s1_rid"] = s1_rid[cand.s1_row.values]
    pairs = pd.read_parquet(os.path.join(DATA, "train_gt_pairs.parquet"))
    pairs["label"] = np.int8(1)
    cand = cand.merge(pairs, on=["s1_rid", "m_rid"], how="left")
    cand["label"] = cand["label"].fillna(0).astype(np.int8)
    cand["fold"] = np.where(np.isin(cand.s1_row.values, vrows), 0, 1).astype(np.int8)
    cand.to_parquet(CAND, index=False)
    # every S1 entity in each role (needed for macro F0.5, incl. S1s without candidates)
    pd.DataFrame({"s1_rid": s1_rid[rows], "fold": np.where(np.isin(rows, vrows), 0, 1)}).to_parquet(
        os.path.join(DATA, "trainval_s1.parquet"), index=False)
    for f in (1, 0):
        c = cand[cand.fold == f]
        log(f"block fold {f}: {c.s1_row.nunique():,} S1, {len(c):,} pairs, positives {c.label.mean():.4f}")
    log(f"block done in {time.time() - t0:.0f}s")


def stage_feats():
    for f in (1, 0):
        cand = pd.read_parquet(CAND)
        cand = cand[cand.fold == f].reset_index(drop=True)
        log(f"features for fold {f}: {len(cand):,} pairs")
        build_features("train", cand, FEATS[f])
        del cand
        gc.collect()


def stage_train():
    t0 = time.time()
    df = pd.read_parquet(FEATS[1])
    feats = feature_columns(df)
    # early-stopping set: 10% of the TRAINING S1 entities (the validation fold stays untouched)
    es_ids = df.s1_rid.drop_duplicates().sample(frac=0.1, random_state=0)
    es = df.s1_rid.isin(es_ids).values
    y = df.label.values
    X = df[feats].to_numpy(np.float32)
    npos, n = int(y.sum()), len(y)
    del df
    gc.collect()
    dtr = lgb.Dataset(X[~es], y[~es], feature_name=feats, free_raw_data=True)
    des = lgb.Dataset(X[es], y[es], reference=dtr)
    model = lgb.train(LGB_PARAMS, dtr, num_boost_round=2000, valid_sets=[des],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)])
    del X, dtr, des
    gc.collect()
    model.save_model(os.path.join(MODELS, "lgb_model.txt"))
    with open(os.path.join(MODELS, "feature_list.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(feats))
    log(f"trained, best iteration {model.best_iteration} ({time.time() - t0:.0f}s)")
    # predict validation in batches
    out = []
    pf = pq.ParquetFile(FEATS[0])
    for i in range(pf.num_row_groups):
        v = pf.read_row_group(i, columns=feats + ["s1_rid", "m_rid", "label", "s1_row"]).to_pandas()
        vp = v[["s1_rid", "m_rid", "label", "s1_row"]].copy()
        vp["p"] = model.predict(v[feats].to_numpy(np.float32), num_threads=10).astype(np.float32)
        out.append(vp)
    vp = pd.concat(out, ignore_index=True)
    vp.to_parquet(os.path.join(DATA, "val_pred.parquet"), index=False)
    auc = roc_auc_score(vp.label, vp.p)
    ap = average_precision_score(vp.label, vp.p)
    imp = pd.Series(model.feature_importance("gain"), index=feats).sort_values(ascending=False)
    imp = (imp / imp.sum()).round(4)
    with open(os.path.join(REP, "m5_model.md"), "w", encoding="utf-8") as fh:
        fh.write(f"# M5 model\n\ntrain pairs {n:,} (positives {npos / n:.4f}), validation pairs {len(vp):,} "
                 f"(positives {vp.label.mean():.4f})\n\nbest iteration {model.best_iteration}\n\n"
                 f"**validation AUC {auc:.5f}, average precision {ap:.5f}**\n\n"
                 f"## Feature importance (share of total gain)\n\n")
        fh.write("\n".join(f"- {k}: {v}" for k, v in imp.items()))
    log(f"validation: AUC {auc:.5f} AP {ap:.5f} ({time.time() - t0:.0f}s)")
    log("top features: " + ", ".join(f"{k}={v}" for k, v in imp.head(15).items()))


if __name__ == "__main__":
    for s in sys.argv[1:] or ["block", "feats", "train"]:
        {"block": stage_block, "feats": stage_feats, "train": stage_train}[s]()
