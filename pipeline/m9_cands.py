"""M9: candidate generation v3 (forward blocking + reverse blocking + second hop), retrain, re-decide, test.

The old pipeline is untouched and stays selectable (blocking.py / block_cfg.json / m5-m7 / models/*.txt|json
/ data/*.parquet). Everything new lives in data/v3/, models/v3/, reports/m9_candidates.md.

Stages (run one at a time, in this order; long ones are resumable - finished batches are skipped on re-run):
  rev train      reverse blocking: top-30 S1s of EVERY train pool record (all 2.2M train S1 as the index)
  bucket train   regroup the reverse lists by S1 batch (+ best reverse score of every pool record)
  crossfit       old features: LightGBM trained on fold 0 (same params), predicts fold 1 (out-of-fold p)
  hop_samples    second hop for the 200k sample S1s (seeds = top-3 by out-of-fold p)
  eval           recall ceiling etc. of every candidate variant on fold-0 validation; chooses the config
  build          chosen candidates + features (64 old + rev_rank, rev_gap) for the fold-1 / fold-0 samples
  train          model A on fold 1 (predicts fold 0, folds 2-9, test), model B on fold 0 (predicts fold 1)
  oldfull        OLD pipeline over ALL 2.2M train S1: forward candidates + out-of-fold p (the S1s the
                 production model was trained on get the cross-fit model's p) -> full-competition baseline
  hop train      second hop for the remaining train S1s (seeds from oldfull), appended
  v3full         v3 pipeline over ALL train S1 (out-of-fold p) -> full-competition validation
  decide         fold-0 F0.5, old vs v3, fold-0-only vs full competition, rule + expected F0.5 -> decision
  rev test / bucket test / hop test / infer / write OUT / check OUT / compare     (test; only if v3 wins)
"""
import gc
import json
import os
import sys
import time

os.environ.setdefault("AMLC_MAXMB", "3900")  # RAM guard in common.py: stay below 4 GB
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import blocking_v3 as B
from common import DATA, ROOT, WORK, log, mem_mb, pq_path

V3 = os.path.join(DATA, "v3")
MV3 = os.path.join(WORK, "models", "v3")
REP = os.path.join(WORK, "reports")
for _d in (V3, MV3):
    os.makedirs(_d, exist_ok=True)
REV_LIST = 30           # reverse pass keeps the top-30 S1s of every pool record (variants: top-k or score ratio)
HOP_SEEDS = 3           # top-3 candidates by model p are the second-hop seeds
HOP_CAP = 20            # at most 20 second-hop candidates per S1
HOP_K_SEARCH = 20       # pool->pool search keeps the 20 best-scoring records per seed before the string check
HOP_MIN_SIM = 80        # near_duplicates keeps sim >= 80 so thresholds 80 / 85 / 90 can be evaluated
HOP_SEEDS_PER_CHUNK = 150_000  # 1M seeds at once went over 4.4 GB (postings of the tight keys)
HOP_SEEDS_PER_CALL = 900_000   # hop train: seeds per (resumable) call, ~6-8 min
NB_OLD = 23             # old-pipeline batches over the 2.2M train S1 (~96k S1 each, like m7_infer.py)
NB = {"train": 46, "test": 36}   # v3 batches (~48k S1 each: more candidates per S1 -> smaller batches)
MAX_BATCH_PAIRS = 3_000_000      # a v3 batch must stay below this many pairs (m7: 2.9M pairs -> <= 2.7 GB)
BLOCK_CFG = json.load(open(os.path.join(WORK, "block_cfg.json")))
T_START = time.time()
BUDGET = float(sys.argv[sys.argv.index("--budget") + 1]) if "--budget" in sys.argv else 1e12


def out_of_time():
    """Resumable loops stop starting new chunks / batches after --budget seconds (re-run to continue)."""
    if time.time() - T_START > BUDGET:
        log(f"time budget {BUDGET:.0f}s used - stopping here, re-run the stage to continue")
        return True
    return False


def stat_path(name):
    return os.path.join(V3, f"stats_{name}.json")


def save_stats(name, d):
    d = {**d, "peak_ram_mb": round(mem_mb()[1])}
    json.dump(d, open(stat_path(name), "w"), indent=1, default=float)
    log(f"stats {name}: {d}")


def load_stats(name):
    return json.load(open(stat_path(name))) if os.path.exists(stat_path(name)) else {}


def s1_rid_arr(split):
    return pq.read_table(pq_path(split, 1), columns=["rid"]).column("rid").to_numpy()


def pool_rid_arr(split):
    return np.concatenate([pq.read_table(pq_path(split, s), columns=["rid"]).column("rid").to_numpy() for s in (2, 3)])


def ranges(n, nb):
    return [(int(r[0]), int(r[-1]) + 1) for r in np.array_split(np.arange(n), nb)]


def target_rows():
    """S1 rows of the 100k fold-0 (validation) and 100k fold-1 (model training) samples of m5."""
    s1 = pd.read_parquet(os.path.join(DATA, "trainval_s1.parquet"))
    rid = s1_rid_arr("train")
    rows = pd.Index(rid).get_indexer(s1.s1_rid.values)
    assert (rows >= 0).all()
    return pd.DataFrame({"s1_row": rows, "s1_rid": s1.s1_rid.values, "fold": s1.fold.values})


def sample_mask(fold):
    """bool over train S1 rows: in the 100k sample of this fold."""
    T = target_rows()
    m = np.zeros(B.n_rows("train", 1), bool)
    m[T.s1_row[T.fold == fold].values] = True
    return m


def code(d):
    return d.s1_row.values.astype(np.int64) * (1 << 25) + d.pidx.values


def feats_old():
    return open(os.path.join(WORK, "models", "feature_list.txt"), encoding="utf-8").read().split("\n")


FEATS_V3_EXTRA = ["rev_rank", "rev_gap"]


def feats_v3():
    return feats_old() + FEATS_V3_EXTRA


def booster(path):
    import lightgbm as lgb
    return lgb.Booster(model_file=path)


# ------------------------------------------------------------------ reverse pass (resumable, one file per chunk)
def stage_rev(split, chunk=500_000):
    """Top-REV_LIST reverse list of EVERY pool record over ALL S1s of the split (train and test alike:
    the lists feed candidate variants, the full-competition validation and the rev_rank / rev_gap features)."""
    t0 = time.time()
    out_dir = os.path.join(V3, f"rev{REV_LIST}_{split}")
    os.makedirs(out_dir, exist_ok=True)
    k = REV_LIST
    npool = B.n_rows(split, 2) + B.n_rows(split, 3)
    st = load_stats(f"rev{REV_LIST}_{split}")
    secs, exp = st.get("seconds_sum", 0.0), st.get("expanded", 0)
    for lo in range(0, npool, chunk):
        part = os.path.join(out_dir, f"part_{lo:09d}.parquet")
        if os.path.exists(part):
            continue
        if out_of_time():
            return
        t1 = time.time()
        hi = min(lo + chunk, npool)
        row, key, kt = B.read_key_range(B.pool_files(split), lo, hi, B.TYPES)
        res = B.topk_search(row - lo, key, kt, B.s1_files(split), B.n_rows(split, 1), k)
        del row, key, kt
        exp += res.attrs["expanded"]
        pd.DataFrame({"pidx": (res.q.values + lo).astype(np.int32), "s1_row": res.idx.values.astype(np.int32),
                      "rscore": res.score.values.astype(np.float32),
                      "rrank": res["rank"].values.astype(np.int8)}).to_parquet(part + ".tmp", index=False)
        os.replace(part + ".tmp", part)
        secs += time.time() - t1
        del res
        gc.collect()
        log(f"reverse {split}: {hi:,}/{npool:,} pool records ({time.time() - t0:.0f}s this run, {secs:.0f}s total)")
        save_stats(f"rev{REV_LIST}_{split}", {"seconds_sum": round(secs, 1), "expanded": int(exp), "done_to": hi,
                                             "k": k, "pool": npool})
    log(f"reverse {split} complete")


def stage_bucket(split):
    """Regroup the reverse lists by v3 S1 batch (one file per batch) and store every pool record's best
    (rank-1) reverse score, so each batch reads only its own rows."""
    t0 = time.time()
    src_dir, out_dir = os.path.join(V3, f"rev{REV_LIST}_{split}"), os.path.join(V3, f"revb_{split}")
    os.makedirs(out_dir, exist_ok=True)
    npool, ns1 = B.n_rows(split, 2) + B.n_rows(split, 3), B.n_rows(split, 1)
    parts = sorted(f for f in os.listdir(src_dir) if f.endswith(".parquet"))
    assert len(parts) == -(-npool // 500_000), "reverse pass not complete"
    bounds = np.array([lo for lo, _ in ranges(ns1, NB[split])] + [ns1])
    best = np.zeros(npool, np.float32)
    writers, n = {}, 0
    for f in parts:
        t = pd.read_parquet(os.path.join(src_dir, f))
        r0 = t[t.rrank == 0]
        best[r0.pidx.values] = r0.rscore.values
        b = np.searchsorted(bounds, t.s1_row.values, "right") - 1
        for bi, g in t.groupby(b):
            tab = pa.Table.from_pandas(g, preserve_index=False)
            if bi not in writers:
                writers[bi] = pq.ParquetWriter(os.path.join(out_dir, f"b_{bi:03d}.parquet"), tab.schema,
                                               compression="zstd")
            writers[bi].write_table(tab)
        n += len(t)
        del t
        gc.collect()
    for w in writers.values():
        w.close()
    np.save(os.path.join(V3, f"revbest_{split}.npy"), best)
    save_stats(f"bucket_{split}", {"seconds": round(time.time() - t0, 1), "rows": n})


def rev_bucket(split, b):
    return pd.read_parquet(os.path.join(V3, f"revb_{split}", f"b_{b:03d}.parquet"))


def rev_rows(split, rows):
    """Reverse-list rows of the given S1 rows (all buckets, filtered)."""
    m = np.zeros(B.n_rows(split, 1), bool)
    m[rows] = True
    out = []
    for b in range(NB[split]):
        t = rev_bucket(split, b)
        out.append(t[m[t.s1_row.values]])
        del t
    return pd.concat(out, ignore_index=True)


def rev_best(split):
    return np.load(os.path.join(V3, f"revbest_{split}.npy"))


def rev_select(rv, best, cfg):
    """Reverse pairs to add: top-k of the record's list, or ranks 1-30 with score >= X x its rank-1 score."""
    r = cfg.get("rev")
    if not r:
        return rv.iloc[:0][["s1_row", "pidx"]]
    if r["kind"] == "topk":
        m = rv.rrank.values < r["k"]
    else:
        m = rv.rscore.values >= r["x"] * best[rv.pidx.values]
    return rv[m][["s1_row", "pidx"]]


# ------------------------------------------------------------------ LightGBM helpers
def load_matrix(path, feats, dest=None):
    """Features as one float32 matrix, filled row group by row group (no pandas copy of the whole file).
    dest[i] = output row of file row i (to load the rows already reordered)."""
    pf = pq.ParquetFile(path)
    n = pf.metadata.num_rows
    X = np.empty((n, len(feats)), np.float32)
    pos = 0
    for i in range(pf.num_row_groups):
        t = pf.read_row_group(i, columns=feats)
        m = t.num_rows
        rows = slice(pos, pos + m) if dest is None else dest[pos:pos + m]
        for j, c in enumerate(feats):
            X[rows, j] = t.column(c).to_numpy()
        pos += m
        del t
    return X


def train_lgb(path, feats, model_path):
    """Same recipe as m5_train.stage_train (params, 10% of the training S1s for early stopping, a separate
    Dataset for each). Rows are loaded in the order [training rows, early-stopping rows], so both Datasets
    are slices of one matrix (no copy). (Dataset.subset() was tried first: ~60x slower training.)"""
    import lightgbm as lgb
    from m5_train import LGB_PARAMS
    t0 = time.time()
    meta = pq.read_table(path, columns=["label", "s1_rid"])
    y, sid = meta.column("label").to_numpy(), meta.column("s1_rid").to_numpy()
    del meta
    es_ids = pd.Series(sid).drop_duplicates().sample(frac=0.1, random_state=0)
    es = np.isin(sid, es_ids.values)
    order = np.r_[np.flatnonzero(~es), np.flatnonzero(es)]
    dest = np.empty(len(order), np.int64)
    dest[order] = np.arange(len(order))
    X = load_matrix(path, feats, dest)
    y = y[order]
    ntr = int((~es).sum())
    dtr = lgb.Dataset(X[:ntr], y[:ntr], feature_name=feats, free_raw_data=True)
    des = lgb.Dataset(X[ntr:], y[ntr:], reference=dtr)
    model = lgb.train(LGB_PARAMS, dtr, num_boost_round=2000, valid_sets=[des],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(200)])
    model.save_model(model_path)
    info = dict(pairs=len(y), positives=int(y.sum()), best_iter=model.best_iteration,
                seconds=round(time.time() - t0, 1), peak_ram_mb=round(mem_mb()[1]))
    del dtr, des, X, y
    gc.collect()
    log(f"trained {model_path}: {info}")
    return model, info


def predict_file(model, path, feats, cols):
    pf = pq.ParquetFile(path)
    out = []
    for i in range(pf.num_row_groups):
        v = pf.read_row_group(i, columns=list(dict.fromkeys(feats + cols))).to_pandas()
        o = v[cols].copy()
        o["p"] = model.predict(v[feats].to_numpy(np.float32), num_threads=10).astype(np.float32)
        out.append(o)
        del v
    return pd.concat(out, ignore_index=True)


def predict_mixed(path, feats, mA, mB, maskB, cols):
    """p from model A, except rows whose S1 is in maskB (bool over S1 rows): model B (out-of-fold)."""
    pf = pq.ParquetFile(path)
    out = []
    for i in range(pf.num_row_groups):
        v = pf.read_row_group(i, columns=list(dict.fromkeys(feats + cols + ["s1_row"]))).to_pandas()
        X = v[feats].to_numpy(np.float32)
        p = np.empty(len(v), np.float32)
        b = maskB[v.s1_row.values]
        if (~b).any():
            p[~b] = mA.predict(X[~b], num_threads=10)
        if b.any():
            p[b] = mB.predict(X[b], num_threads=10)
        o = v[list(dict.fromkeys(cols + ["s1_row"]))].copy()
        o["p"] = p
        out.append(o)
        del v, X
    return pd.concat(out, ignore_index=True)


# ------------------------------------------------------------------ cross-fit (old features)
def stage_crossfit():
    from sklearn.metrics import roc_auc_score
    model, info = train_lgb(os.path.join(DATA, "feats_val.parquet"), feats_old(),
                            os.path.join(MV3, "old_fold0_model.txt"))
    pr = predict_file(model, os.path.join(DATA, "feats_trn.parquet"), feats_old(), ["s1_row", "pidx", "label"])
    pr.to_parquet(os.path.join(V3, "old_oof_fold1.parquet"), index=False)
    save_stats("crossfit", {**info, "fold1_auc": roc_auc_score(pr.label, pr.p)})


# ------------------------------------------------------------------ old pipeline over all train S1
def stage_oldfull():
    from blocking import block
    from features2 import build_features
    tstart = time.time()
    d = os.path.join(V3, "oldfull")
    os.makedirs(d, exist_ok=True)
    mA, mB = booster(os.path.join(WORK, "models", "lgb_model.txt")), booster(os.path.join(MV3, "old_fold0_model.txt"))
    maskB = sample_mask(1)  # the production model was trained on these S1s -> cross-fit model for them
    for b, (lo, hi) in enumerate(ranges(B.n_rows("train", 1), NB_OLD)):
        outp = os.path.join(d, f"pred_{b:03d}.parquet")
        if os.path.exists(outp):
            continue
        if out_of_time():
            return
        t0 = time.time()
        fwd = block("train", np.arange(lo, hi), dict(BLOCK_CFG))
        fwd.to_parquet(os.path.join(d, f"fwd_{b:03d}.parquet"), index=False)
        fp = os.path.join(d, "feats_tmp.parquet")
        build_features("train", fwd, fp)
        del fwd
        gc.collect()
        pr = predict_mixed(fp, feats_old(), mA, mB, maskB, ["pidx"])
        pr[["s1_row", "pidx", "p"]].to_parquet(outp + ".tmp", index=False)
        os.replace(outp + ".tmp", outp)
        os.remove(fp)
        log(f"oldfull batch {b + 1}/{NB_OLD}: {hi - lo:,} S1, {len(pr):,} pairs in {time.time() - t0:.0f}s "
            f"(run {time.time() - tstart:.0f}s, peak RAM {mem_mb()[1]:.0f} MB)")
        del pr
        gc.collect()
    save_stats("oldfull", {"seconds_this_run": round(time.time() - tstart, 1)})


def fwd_range(lo, hi):
    """Forward candidates (old blocking) of S1 rows [lo, hi) from the oldfull files."""
    out = []
    for b, (a, z) in enumerate(ranges(B.n_rows("train", 1), NB_OLD)):
        if z > lo and a < hi:
            t = pd.read_parquet(os.path.join(V3, "oldfull", f"fwd_{b:03d}.parquet"))
            out.append(t[(t.s1_row >= lo) & (t.s1_row < hi)])
    return pd.concat(out, ignore_index=True)


# ------------------------------------------------------------------ second hop
def top_seeds(df, n=HOP_SEEDS):
    """(s1_row, pidx, p) -> the n highest-p candidates per S1 as (s1_row, seed, p)."""
    df = df.sort_values(["s1_row", "p"], ascending=[True, False], kind="stable")
    df = df[df.groupby("s1_row").cumcount().values < n]
    return df.rename(columns={"pidx": "seed"})[["s1_row", "seed", "p"]].reset_index(drop=True)


def stage_hop_samples():
    """Second hop for the 200k sample S1s only (needed first, by eval / build): seeds = top-3 by out-of-fold
    p - fold-0 sample: the production model (trained on the fold-1 sample), data/val_pred.parquet;
    fold-1 sample: the cross-fit model trained on fold 0, data/v3/old_oof_fold1.parquet."""
    t0 = time.time()
    cand = pd.read_parquet(os.path.join(DATA, "cand_trainval.parquet"), columns=["s1_row", "pidx", "s1_rid", "m_rid", "fold"])
    vp = pd.read_parquet(os.path.join(DATA, "val_pred.parquet"), columns=["s1_rid", "m_rid", "p"])
    f0 = cand[cand.fold == 0].merge(vp, on=["s1_rid", "m_rid"])[["s1_row", "pidx", "p"]]
    assert len(f0) == len(vp)
    f1 = pd.read_parquet(os.path.join(V3, "old_oof_fold1.parquet"), columns=["s1_row", "pidx", "p"])
    seeds = top_seeds(pd.concat([f0, f1], ignore_index=True))
    del cand, vp, f0, f1
    seeds.to_parquet(os.path.join(V3, "hopseeds_train.parquet"), index=False)
    dups, st = B.near_duplicates("train", seeds.seed.values, HOP_K_SEARCH, HOP_MIN_SIM, seeds_per_chunk=HOP_SEEDS_PER_CHUNK)
    dups.to_parquet(os.path.join(V3, "hopdup_train.parquet"), index=False)
    np.save(os.path.join(V3, "hopsearched_train.npy"), np.unique(seeds.seed.values))
    save_stats("hop_train_samples", {**st, "seconds_total": round(time.time() - t0, 1), "dups": len(dups),
                                     "seed_pairs": len(seeds)})


def stage_hop(split):
    """Seeds = top-3 candidates per S1 by out-of-fold p of the OLD pipeline (train: oldfull, all 2.2M S1;
    test: the production predictions data/test_pred_*.parquet). Near-duplicate search for all seeds
    (train: only seeds not searched yet by hop_samples; their results are appended)."""
    t0 = time.time()
    parts = []
    if split == "train":
        for b in range(NB_OLD):
            parts.append(top_seeds(pd.read_parquet(os.path.join(V3, "oldfull", f"pred_{b:03d}.parquet"))))
        seeds = pd.concat(parts, ignore_index=True)
        # the sample S1s' seeds must be the ones eval / build used (same out-of-fold p)
        old = pd.read_parquet(os.path.join(V3, "hopseeds_train.parquet"))
        chk = old.merge(seeds, on=["s1_row", "seed"], how="left", suffixes=("", "_full"))
        log(f"hop seeds of the sample S1s found again in the full run: {chk.p_full.notna().mean():.4f}")
        done = np.load(os.path.join(V3, "hopsearched_train.npy"))
        todo_all = np.setdiff1d(np.unique(seeds.seed.values), done)
        prev = load_stats("hop_train")
        # resumable: at most HOP_SEEDS_PER_CALL seeds per call; dups and the searched list are saved each call
        todo = todo_all[:HOP_SEEDS_PER_CALL]
        dups_new, st = B.near_duplicates("train", todo, HOP_K_SEARCH, HOP_MIN_SIM, seeds_per_chunk=HOP_SEEDS_PER_CHUNK)
        dups = pd.concat([pd.read_parquet(os.path.join(V3, "hopdup_train.parquet")), dups_new], ignore_index=True)
        dups.to_parquet(os.path.join(V3, "hopdup_train.parquet.tmp"), index=False)
        os.replace(os.path.join(V3, "hopdup_train.parquet.tmp"), os.path.join(V3, "hopdup_train.parquet"))
        np.save(os.path.join(V3, "hopsearched_train.npy"), np.union1d(done, todo))
        left = len(todo_all) - len(todo)
        secs = prev.get("seconds_sum", 0) + time.time() - t0
        if left == 0:
            seeds.to_parquet(os.path.join(V3, "hopseeds_train.parquet"), index=False)  # all 2.2M S1s now
        save_stats("hop_train", {"seconds_sum": round(secs, 1), "dups": len(dups), "seeds_left": int(left),
                                 "seeds_all": int(seeds.seed.nunique())})
        log(f"hop train: {len(todo):,} seeds searched this call, {left:,} left")
        return
    else:
        s1rid, pidx = pd.Index(s1_rid_arr("test")), pd.Index(pool_rid_arr("test"))
        for b in range(18):
            pr = pd.read_parquet(os.path.join(DATA, f"test_pred_{b}.parquet"))
            pr["s1_row"] = s1rid.get_indexer(pr.s1_rid.values)
            pr["pidx"] = pidx.get_indexer(pr.m_rid.values)
            assert (pr.s1_row >= 0).all() and (pr.pidx >= 0).all()
            parts.append(top_seeds(pr[["s1_row", "pidx", "p"]]))
            del pr
        del s1rid, pidx
    seeds = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()
    # resumable like the train branch: HOP_SEEDS_PER_CALL seeds per call, results saved after each call
    f_done, f_dup = os.path.join(V3, "hopsearched_test.npy"), os.path.join(V3, "hopdup_test.parquet")
    done = np.load(f_done) if os.path.exists(f_done) else np.array([], np.int64)
    todo_all = np.setdiff1d(np.unique(seeds.seed.values), done)
    prev = load_stats("hop_test")
    todo = todo_all[:HOP_SEEDS_PER_CALL]
    dups_new, st = B.near_duplicates("test", todo, HOP_K_SEARCH, HOP_MIN_SIM, seeds_per_chunk=HOP_SEEDS_PER_CHUNK)
    dups = pd.concat(([pd.read_parquet(f_dup)] if len(done) else []) + [dups_new], ignore_index=True)
    dups.to_parquet(f_dup + ".tmp", index=False)
    os.replace(f_dup + ".tmp", f_dup)
    np.save(f_done, np.union1d(done, todo))
    left = len(todo_all) - len(todo)
    if left == 0:
        seeds.to_parquet(os.path.join(V3, "hopseeds_test.parquet"), index=False)
    save_stats("hop_test", {"seconds_sum": round(prev.get("seconds_sum", 0) + time.time() - t0, 1), "dups": len(dups),
                            "seeds_left": int(left), "seeds_all": int(seeds.seed.nunique()), "seed_pairs": len(seeds)})
    log(f"hop test: {len(todo):,} seeds searched this call, {left:,} left")


def hop_inputs(split, rows=None, lo=None, hi=None):
    seeds = pd.read_parquet(os.path.join(V3, f"hopseeds_{split}.parquet"))
    if rows is not None:
        m = np.zeros(B.n_rows(split, 1), bool)
        m[rows] = True
        seeds = seeds[m[seeds.s1_row.values]]
    if lo is not None:
        seeds = seeds[(seeds.s1_row >= lo) & (seeds.s1_row < hi)]
    return seeds, pd.read_parquet(os.path.join(V3, f"hopdup_{split}.parquet"))


# ------------------------------------------------------------------ evaluation of the variants (fold 0)
def union(*dfs):
    return pd.concat([d[["s1_row", "pidx"]] for d in dfs], ignore_index=True).drop_duplicates()


def metrics(c, true, rows):
    """recall ceiling, oracle F0.5, candidates per S1 for candidate set c (s1_row, pidx) over S1 rows."""
    hit = np.isin(code(true), code(c))
    h = pd.Series(hit).groupby(true.s1_row.values).sum().reindex(rows, fill_value=0).values
    nt = true.groupby("s1_row").size().reindex(rows, fill_value=0).values
    orc = np.where(nt == 0, 1.0, 1.25 * h / np.maximum(h + 0.25 * nt, 1e-9)).mean()
    per = c.groupby("s1_row").size().reindex(rows, fill_value=0)
    return dict(recall=hit.mean(), oracle=orc, mean=per.mean(), max=int(per.max()), pairs=len(c))


def true_pairs(rows):
    gt = pd.read_parquet(os.path.join(DATA, "train_gt_pairs.parquet"))
    s1rid = s1_rid_arr("train")
    gt = gt[gt.s1_rid.isin(s1rid[rows])]
    gt["s1_row"] = pd.Index(s1rid).get_indexer(gt.s1_rid.values)
    gt["pidx"] = pd.Index(pool_rid_arr("train")).get_indexer(gt.m_rid.values)
    assert (gt.pidx >= 0).all()
    return gt[["s1_row", "pidx"]].reset_index(drop=True)


REV_VARIANTS = [{"kind": "topk", "k": 3}, {"kind": "topk", "k": 5}, {"kind": "topk", "k": 10},
                {"kind": "ratio", "x": 0.9}, {"kind": "ratio", "x": 0.8}, {"kind": "ratio", "x": 0.7}]
HOP_VARIANTS = [90, 85, 80]


def rev_name(r):
    return f"reverse top-{r['k']}" if r["kind"] == "topk" else f"reverse ratio >= {r['x']} x best (ranks 1-30)"


def stage_eval():
    t0 = time.time()
    T = target_rows()
    best = rev_best("train")
    cand = pd.read_parquet(os.path.join(DATA, "cand_trainval.parquet"), columns=["s1_row", "pidx", "fold"])
    srev, sbuck, shop = load_stats(f"rev{REV_LIST}_train"), load_stats("bucket_train"), load_stats("hop_train_samples")
    S = {}
    for f in (0, 1):
        rows = np.sort(T.s1_row[T.fold == f].values)
        fwd = cand[cand.fold == f][["s1_row", "pidx"]].reset_index(drop=True)
        rv = rev_rows("train", rows)
        seeds, dups = hop_inputs("train", rows=rows)
        S[f] = dict(rows=rows, fwd=fwd, rev={i: rev_select(rv, best, {"rev": r}) for i, r in enumerate(REV_VARIANTS)},
                    hop={t: B.hop_assign(seeds, dups, fwd, t, HOP_CAP) for t in HOP_VARIANTS})
        del rv, seeds, dups
        gc.collect()
    del cand
    true = true_pairs(S[0]["rows"])
    V = [(None, None)] + [(i, None) for i in range(len(REV_VARIANTS))] + [(None, t) for t in HOP_VARIANTS] + \
        [(i, t) for i in range(len(REV_VARIANTS)) for t in HOP_VARIANTS]
    rev_min = (srev.get("seconds_sum", 0) + sbuck.get("seconds", 0)) / 60
    hop_min = shop.get("seconds_total", 0) / 60
    L = ["# M9: candidate generation v3 - variants on fold-0 validation (100k S1, full train pool)\n",
         "Recall ceiling = share of true (S1, match) pairs among the candidates; oracle F0.5 = macro F0.5 if a "
         "perfect model picked among them. **Reverse**: each of the 10.3M train pool records searches all 2.2M train "
         "S1s (same keys, idf with df over S1, cap 1000; same country via the keys) and keeps its top-30 list; a "
         "variant adds (S1, record) for the record's top-k, or for ranks 1-30 whose reverse score is >= X x the "
         "record's rank-1 score. **Second hop**: seeds = each S1's top-3 candidates by OUT-OF-FOLD p (fold 0: the "
         "fold-1 model; fold 1: the cross-fit fold-0 model); pool->pool key search (tight key types 2,6,7,8,10,11, "
         "top 20 per seed), kept if token_set_ratio on name_core AND on the address >= T; <= 20 added per S1.\n",
         f"Generation cost on the train split (all 2.2M S1 / 10.3M pool records): reverse pass + regrouping "
         f"{rev_min:.0f} min, peak RAM {max(srev.get('peak_ram_mb', 0), sbuck.get('peak_ram_mb', 0)) / 1024:.1f} GB "
         f"(one pass serves every reverse variant); second hop {hop_min:.0f} min for the {shop.get('seeds', 0):,} seed "
         f"records of the 200k sample S1s (all 2.2M S1: ~10x the seeds), peak RAM "
         f"{shop.get('peak_ram_mb', 0) / 1024:.1f} GB. Forward blocking alone: ~40 min for 2.2M S1.\n",
         "RAM fit: LightGBM on the fold-1 training pairs (estimate: 66 float32 features + bins + ids) must stay "
         "< 3.5 GB, and a 48k-S1 inference batch must stay < 3.0M pairs (the m7 batches used 2.9M pairs at <= 2.7 GB).\n",
         "| variant | recall ceiling | oracle F0.5 | mean cands/S1 | max cands/S1 | total pairs (fold 0) | "
         "extra runtime | peak RAM | training pairs (fold 1) | est. LightGBM RAM | pairs per 48k batch | fits |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    res = []
    for ri, ht in V:
        name = "forward K=30 (current)" + (f" + {rev_name(REV_VARIANTS[ri])}" if ri is not None else "") + \
            (f" + second hop (sim >= {ht})" if ht else "")
        sets = lambda f: [S[f]["fwd"]] + ([S[f]["rev"][ri]] if ri is not None else []) + \
            ([S[f]["hop"][ht]] if ht else [])  # noqa: E731
        c0 = union(*sets(0))
        m = metrics(c0, true, S[0]["rows"])
        n1 = len(union(*sets(1)))
        ram = n1 * (66 * 4 + 66 + 40) / 2**30 + 0.5
        batch = m["mean"] * 48_000
        fits = ram < 3.5 and batch < MAX_BATCH_PAIRS
        secs = (rev_min if ri is not None else 0) + (hop_min if ht else 0)
        peak = max(srev.get("peak_ram_mb", 0) if ri is not None else 0, shop.get("peak_ram_mb", 0) if ht else 0)
        res.append(dict(name=name, rev=REV_VARIANTS[ri] if ri is not None else None, hop_thr=ht, fits=fits, **m))
        L.append(f"| {name} | {m['recall']:.4f} | {m['oracle']:.4f} | {m['mean']:.1f} | {m['max']:,} | {m['pairs']:,} | "
                 f"{secs:.0f} min | {peak / 1024:.1f} GB | {n1:,} | {ram:.1f} GB | {batch / 1e6:.2f}M | "
                 f"{'yes' if fits else '**no**'} |")
        del c0
    best_v = max((r for r in res if r["fits"]), key=lambda r: (round(r["recall"], 4), -r["pairs"]))
    cfg = {k: best_v[k] for k in ("name", "rev", "hop_thr", "recall", "oracle", "mean")}
    json.dump(cfg, open(os.path.join(MV3, "cand_cfg.json"), "w"), indent=1, default=float)
    L.append(f"\n**Chosen: {best_v['name']}** - the highest recall ceiling among the variants that fit "
             f"(recall {best_v['recall']:.4f} vs {res[0]['recall']:.4f} for forward only; oracle "
             f"{best_v['oracle']:.4f} vs {res[0]['oracle']:.4f}). -> models/v3/cand_cfg.json\n")
    with open(os.path.join(REP, "m9_candidates.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    log("\n".join(L))
    save_stats("eval", {"seconds": round(time.time() - t0, 1)})


# ------------------------------------------------------------------ assemble candidates + v3 features
def assemble(split, rows, fwd, rv, best, seeds, dups, cfg, pool_rid, s1rid):
    """fwd (block() output: s1_row, pidx, score, nkeys, rank, m_rid) + the chosen reverse / second-hop pairs
    -> one frame with the columns build_features needs, plus src (bit 1 forward, 2 reverse, 4 second hop)
    and the two reverse features:
      rev_rank = rank (1-30) of this S1 in the record's reverse list, 31 if absent
      rev_gap  = record's rank-1 reverse score - this pair's reverse score
    Added pairs get their forward score / nkeys (as if the forward search had scored them) and rank = K."""
    rs = rev_select(rv, best, cfg)
    hp = B.hop_assign(seeds, dups, fwd, cfg["hop_thr"], HOP_CAP) if cfg.get("hop_thr") else rs.iloc[:0]
    derive = "score" not in fwd.columns  # forward pairs without block() columns (test: reused m7 candidates)
    if derive:
        fwd = fwd.assign(score=np.float32(0), nkeys=np.int16(0), rank=np.int16(0), m_rid=pool_rid[fwd.pidx.values])
    fwd = fwd[["s1_row", "pidx", "score", "nkeys", "rank", "m_rid"]].reset_index(drop=True)
    ex = union(rs, hp)
    ex = ex[~np.isin(code(ex), code(fwd))].reset_index(drop=True)
    c = pd.concat([fwd, pd.DataFrame({"s1_row": ex.s1_row.values.astype(np.int32),
                                      "pidx": ex.pidx.values.astype(np.int32),
                                      "score": np.float32(0), "nkeys": np.int16(0), "rank": np.int16(B.K),
                                      "m_rid": pool_rid[ex.pidx.values]})], ignore_index=True)
    nf = len(fwd)
    cc = code(c)
    c["src"] = (np.where(np.arange(len(c)) < nf, 1, 0) | np.where(np.isin(cc, code(rs)), 2, 0)
                | np.where(np.isin(cc, code(hp)), 4, 0)).astype(np.int8)
    t0 = time.time()
    fs, fk, rsc = B.pair_scores(split, rows, c)
    # forward pairs keep block()'s own score / nkeys (identical up to float rounding); added pairs get fs / fk
    if derive:
        # block()'s order inside an S1: score quantised to 1/256 descending, then pool index ascending
        c["score"], c["nkeys"] = fs, fk
        qs = np.minimum(np.round(fs[:nf].astype(np.float64) * 256), B.QMAX)
        s1r, pid = c.s1_row.values[:nf], c.pidx.values[:nf]
        o = np.lexsort((pid, -qs, s1r))
        gs = np.flatnonzero(np.r_[True, np.diff(s1r[o]) != 0])
        rk = np.empty(nf, np.int16)
        rk[o] = np.arange(nf) - np.repeat(gs, np.diff(np.r_[gs, nf]))
        c.loc[:nf - 1, "rank"] = rk
    else:
        c.loc[nf:, "score"] = fs[nf:]
        c.loc[nf:, "nkeys"] = fk[nf:]
    rr = pd.Series(rv.rrank.values.astype(np.float32) + 1, index=code(rv))
    c["rev_rank"] = rr.reindex(cc).fillna(REV_LIST + 1).values.astype(np.float32)
    c["rev_gap"] = (best[c.pidx.values] - rsc).astype(np.float32)
    c["s1_rid"] = s1rid[c.s1_row.values]
    log(f"  assembled {len(c):,} pairs ({len(ex):,} added), pair scores {time.time() - t0:.0f}s")
    return c


def stage_build():
    from features2 import build_features
    t0 = time.time()
    cfg = json.load(open(os.path.join(MV3, "cand_cfg.json")))
    pool_rid, s1rid, best = pool_rid_arr("train"), s1_rid_arr("train"), rev_best("train")
    T = target_rows()
    cand = pd.read_parquet(os.path.join(DATA, "cand_trainval.parquet"))
    gt = pd.read_parquet(os.path.join(DATA, "train_gt_pairs.parquet"))
    info = {}
    for f in (1, 0):
        rows = np.sort(T.s1_row[T.fold == f].values)
        seeds, dups = hop_inputs("train", rows=rows)
        c = assemble("train", rows, cand[cand.fold == f], rev_rows("train", rows), best, seeds, dups, cfg,
                     pool_rid, s1rid)
        del seeds, dups
        c = c.merge(gt[gt.s1_rid.isin(s1rid[rows])].assign(label=np.int8(1)), on=["s1_rid", "m_rid"], how="left")
        c["label"] = c.label.fillna(0).astype(np.int8)
        c["fold"] = np.int8(f)
        info[f"fold{f}"] = dict(pairs=len(c), positives=int(c.label.sum()), added=int((c.src & 1 == 0).sum()))
        log(f"fold {f}: {info[f'fold{f}']}")
        build_features("train", c, os.path.join(V3, f"feats_{f}.parquet"))
        del c
        gc.collect()
    save_stats("build", {"seconds": round(time.time() - t0, 1), **info})


def stage_train():
    from sklearn.metrics import average_precision_score, roc_auc_score
    t0 = time.time()
    F = feats_v3()
    mA, iA = train_lgb(os.path.join(V3, "feats_1.parquet"), F, os.path.join(MV3, "lgb_model.txt"))
    vp = predict_file(mA, os.path.join(V3, "feats_0.parquet"), F, ["s1_rid", "m_rid", "label", "s1_row", "pidx", "src"])
    vp.to_parquet(os.path.join(V3, "val_pred.parquet"), index=False)
    imp = pd.Series(mA.feature_importance("gain"), index=F).sort_values(ascending=False)
    imp = (imp / imp.sum()).round(4)
    del mA
    mB, iB = train_lgb(os.path.join(V3, "feats_0.parquet"), F, os.path.join(MV3, "lgb_fold0.txt"))
    ob = predict_file(mB, os.path.join(V3, "feats_1.parquet"), F, ["s1_row", "pidx", "label"])
    save_stats("train", {"seconds": round(time.time() - t0, 1), "model_A_fold1": iA, "model_B_fold0": iB,
                         "val_auc": roc_auc_score(vp.label, vp.p), "val_ap": average_precision_score(vp.label, vp.p),
                         "fold1_oof_auc": roc_auc_score(ob.label, ob.p),
                         "importance_top15": imp.head(15).to_dict(),
                         "importance_rev": imp[FEATS_V3_EXTRA].to_dict()})


def stage_v3full():
    """v3 pipeline over ALL train S1s: model A's p, except the fold-1 sample (A was trained on it): model B."""
    from features2 import build_features
    tstart = time.time()
    d = os.path.join(V3, "v3full")
    os.makedirs(d, exist_ok=True)
    cfg = json.load(open(os.path.join(MV3, "cand_cfg.json")))
    pool_rid, s1rid, best = pool_rid_arr("train"), s1_rid_arr("train"), rev_best("train")
    mA, mB = booster(os.path.join(MV3, "lgb_model.txt")), booster(os.path.join(MV3, "lgb_fold0.txt"))
    maskB = sample_mask(1)
    seeds_all, dups = hop_inputs("train")
    for b, (lo, hi) in enumerate(ranges(B.n_rows("train", 1), NB["train"])):
        outp = os.path.join(d, f"pred_{b:03d}.parquet")
        if os.path.exists(outp):
            continue
        if out_of_time():
            return
        t0 = time.time()
        rows = np.arange(lo, hi)
        c = assemble("train", rows, fwd_range(lo, hi), rev_bucket("train", b), best,
                     seeds_all[(seeds_all.s1_row >= lo) & (seeds_all.s1_row < hi)], dups, cfg, pool_rid, s1rid)
        fp = os.path.join(d, "feats_tmp.parquet")
        build_features("train", c, fp)
        del c
        gc.collect()
        pr = predict_mixed(fp, feats_v3(), mA, mB, maskB, ["pidx", "src"])
        pr[["s1_row", "pidx", "p", "src"]].to_parquet(outp + ".tmp", index=False)
        os.replace(outp + ".tmp", outp)
        os.remove(fp)
        log(f"v3full batch {b + 1}/{NB['train']}: {len(pr):,} pairs in {time.time() - t0:.0f}s "
            f"(run {time.time() - tstart:.0f}s, peak RAM {mem_mb()[1]:.0f} MB)")
        del pr
        gc.collect()
    save_stats("v3full", {"seconds_this_run": round(time.time() - tstart, 1)})


# ------------------------------------------------------------------ decision
def full_competition(files, keep_rows, npool):
    """One-owner rule over ALL S1s of the prediction files (each pool record stays with the S1 that gives it
    the highest p; ties -> first occurrence, as m6_decide.one_owner), then the rows of the S1s in keep_rows."""
    best = np.full(npool, -1.0, np.float32)
    owner = np.full(npool, -1, np.int64)
    for f in files:
        t = pd.read_parquet(f, columns=["s1_row", "pidx", "p"])
        g = t.groupby("pidx").p.idxmax().values
        m_, p_ = t.pidx.values[g], t.p.values[g]
        up = p_ > best[m_]
        best[m_[up]] = p_[up]
        owner[m_[up]] = t.s1_row.values[g[up]]
        del t
    out = []
    for f in files:
        t = pd.read_parquet(f)
        t = t[keep_rows[t.s1_row.values]]
        out.append(t[owner[t.pidx.values] == t.s1_row.values])
    return pd.concat(out, ignore_index=True)


def stage_decide():
    import m6b_expected_f as EF
    from m6_decide import grid, one_owner, prepare, score
    t0 = time.time()
    s1rid, pool_rid = s1_rid_arr("train"), pool_rid_arr("train")
    npool = len(pool_rid)
    T = target_rows()
    rows0 = np.sort(T.s1_row[T.fold == 0].values)
    keep0 = sample_mask(0)
    val_ids = np.sort(s1rid[rows0])
    gt = pd.read_parquet(os.path.join(DATA, "train_gt_pairs.parquet"))
    pairs = gt[gt.s1_rid.isin(val_ids)].reset_index(drop=True)
    lab = pd.Series(1, index=pd.MultiIndex.from_frame(pairs))
    del gt

    def with_labels(df):
        df = df.copy()
        df["s1_rid"] = s1rid[df.s1_row.values]
        df["m_rid"] = pool_rid[df.pidx.values]
        df["label"] = lab.reindex(pd.MultiIndex.from_arrays([df.s1_rid.values, df.m_rid.values]),
                                  fill_value=0).values.astype(np.int8)
        return df

    def sources(which):
        d = os.path.join(V3, "oldfull" if which == "old" else "v3full")
        files = sorted(os.path.join(d, f) for f in os.listdir(d) if f.startswith("pred_") and f.endswith(".parquet"))
        assert len(files) == (NB_OLD if which == "old" else NB["train"])
        own = pd.concat([pd.read_parquet(f).pipe(lambda t: t[keep0[t.s1_row.values]]) for f in files],
                        ignore_index=True)
        return files, own

    h1 = EF.split_half(val_ids)
    ids = {1: val_ids[h1], 2: val_ids[~h1]}
    pr = {h: pairs[pairs.s1_rid.isin(ids[h])] for h in (1, 2)}
    R, checks, decisions = {}, {}, {}
    for which in ("old", "v3"):
        files, own = sources(which)
        ref = pd.read_parquet(os.path.join(DATA if which == "old" else V3, "val_pred.parquet"))
        own_l = with_labels(own)
        chk = own_l.merge(ref[["s1_rid", "m_rid", "p"]], on=["s1_rid", "m_rid"], suffixes=("", "_ref"))
        checks[which] = dict(rows=len(own_l), ref_rows=len(ref), matched=len(chk),
                             max_abs_p_diff=float((chk.p - chk.p_ref).abs().max()))
        for mode in ("fold-0 only", "full competition"):
            d = one_owner(own_l) if mode == "fold-0 only" else with_labels(full_competition(files, keep0, npool))
            d = d.reset_index(drop=True)
            P = prepare(d, val_ids, pairs)
            s_all, (g, t, r) = grid(P)
            dh = {h: d[d.s1_rid.isin(ids[h])].reset_index(drop=True) for h in (1, 2)}
            P2 = prepare(dh[2], ids[2], pr[2])
            _, (g1, t1, r1) = grid(prepare(dh[1], ids[1], pr[1]))
            R[(which, mode)] = dict(tuned_on_fold0=s_all, g=g, t=t, r=r, half2_all_tuned=score(P2, g, t, r),
                                    half2_half1_tuned=score(P2, g1, t1, r1), half1_params=[g1, t1, r1])
            decisions[(which, mode)] = dict(g=g, t=t, r=r)
            if which == "v3" and mode == "full competition":
                # lists reach ~1,300 candidates; expected F0.5 only needs the best few (mean ~3.5 matches):
                # each list is cut to its 60 highest calibrated q (the dropped ones have q ~ 0)
                kmax = 60
                EF.KMAX = kmax
                cal = EF.fit_calibration(dh[1].p.values, dh[1].label.values)
                mats = {}
                for h in (1, 2):
                    dh[h]["q"] = EF.calibrate(cal, dh[h].p.values)
                    dh[h] = dh[h].sort_values(["s1_rid", "q", "p"], ascending=[True, False, False])
                    dh[h] = dh[h][dh[h].groupby("s1_rid").cumcount().values < kmax]
                    Q, M, nc, Lb = EF.to_matrix(dh[h], ids[h])
                    nt = pr[h].groupby("s1_rid").size().reindex(ids[h], fill_value=0).values.astype(np.float64)
                    mats[h] = (Q, nc, Lb, nt)
                chunk = 1000
                Q, nc, Lb, nt = mats[1]
                f_pi = [EF.true_f(k, Lb, nt).mean() for k in EF.plugin_choice(Q, nc, EF.C_GRID)]
                f_mc = [EF.true_f(k, Lb, nt).mean() for k in EF.mc_choice(Q, nc, EF.C_GRID, chunk=chunk)]
                c_pi, c_mc = EF.C_GRID[int(np.argmax(f_pi))], EF.C_GRID[int(np.argmax(f_mc))]
                Q, nc, Lb, nt = mats[2]
                R["ef"] = {f"plug-in (c={c_pi})": EF.true_f(EF.plugin_choice(Q, nc, [c_pi])[0], Lb, nt).mean(),
                           f"Monte Carlo (c={c_mc})": EF.true_f(EF.mc_choice(Q, nc, [c_mc], seed=EF.SEED + 1,
                                                                             chunk=chunk)[0], Lb, nt).mean(),
                           "kmax": kmax}
                del mats
            log(f"{which} / {mode}: {R[(which, mode)]}")
            del d, dh, P, P2
            gc.collect()
        del own, own_l, ref
    old_full = R[("old", "full competition")]["tuned_on_fold0"]
    v3_full = R[("v3", "full competition")]["tuned_on_fold0"]
    ef_best = max(v for k_, v in R["ef"].items() if k_ != "kmax")
    ef_wins = ef_best > R[("v3", "full competition")]["half2_all_tuned"]
    dec = {"kind": "rule", **decisions[("v3", "full competition")], "one_owner": True,
           "v3_full_f05": v3_full, "old_full_f05": old_full, "write_allowed": bool(v3_full > old_full),
           "ef_beats_rule_on_half2": bool(ef_wins)}
    json.dump(dec, open(os.path.join(MV3, "decision.json"), "w"), indent=1, default=float)
    L = ["\n## Decision step and fold-0 macro F0.5 (100k validation S1s)\n",
         "*fold-0 only*: the one-owner rule only among the 100k fold-0 S1s (how 0.9503 was measured). *full "
         "competition*: the one-owner rule over ALL 2.2M train S1s, each with an out-of-fold p (fold-1 sample: the "
         "model trained on fold 0; everyone else: the model trained on the fold-1 sample), as on the test set. "
         "Thresholds (gate g, absolute t, relative r) are grid-searched on the fold-0 rows; 'half 2' columns: "
         "thresholds tuned on half 1 (or on all of fold 0) and scored on half 2 only.\n",
         "| pipeline | competition | F0.5 (tuned on fold 0) | g / t / r | half 2, fold-0-tuned | half 2, half-1-tuned |",
         "|---|---|---|---|---|---|"]
    for (which, mode), v in [(k, v) for k, v in R.items() if k != "ef"]:
        L.append(f"| {which} | {mode} | **{v['tuned_on_fold0']:.5f}** | {v['g']} / {v['t']} / {v['r']} | "
                 f"{v['half2_all_tuned']:.5f} | {v['half2_half1_tuned']:.5f} |")
    L.append(f"\nExpected-F0.5 decision (m6b) on v3 / full competition, half 2: " +
             ", ".join(f"{k}: {v:.5f}" for k, v in R["ef"].items() if k != "kmax") +
             f" (rule with fold-0 thresholds on half 2: {R[('v3', 'full competition')]['half2_all_tuned']:.5f}) -> "
             f"{'expected F0.5 wins' if ef_wins else 'the rule stays'}.\n")
    L.append(f"Consistency checks (p of the full runs vs the stored validation predictions): {checks}\n")
    L.append(f"**Gate for writing output_v3: v3 full competition {v3_full:.5f} vs old full competition "
             f"{old_full:.5f} -> {'WRITE' if dec['write_allowed'] else 'DO NOT WRITE'}.**\n")
    with open(os.path.join(REP, "m9_candidates.md"), "a", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    log("\n".join(L))
    save_stats("decide", {"seconds": round(time.time() - t0, 1),
                          "results": {f"{k[0]}|{k[1]}" if isinstance(k, tuple) else k: v for k, v in R.items()},
                          "checks": checks, "decision": dec})


# ------------------------------------------------------------------ test
def stage_infer():
    from blocking import block
    from features2 import build_features
    tstart = time.time()
    dec = json.load(open(os.path.join(MV3, "decision.json")))
    assert dec["write_allowed"], "v3 did not beat the old pipeline under full competition"
    cfg = json.load(open(os.path.join(MV3, "cand_cfg.json")))
    d = os.path.join(V3, "test")
    os.makedirs(d, exist_ok=True)
    mA = booster(os.path.join(MV3, "lgb_model.txt"))
    pool_rid, s1rid, best = pool_rid_arr("test"), s1_rid_arr("test"), rev_best("test")
    seeds_all, dups = hop_inputs("test") if cfg.get("hop_thr") else (pd.DataFrame(columns=["s1_row", "seed"]), None)
    if dups is not None:  # only what hop_assign needs, compact types
        dups = pd.DataFrame({"seed": dups.seed.values.astype(np.int32), "dup": dups.dup.values.astype(np.int32),
                             "sim_name": dups.sim_name.values.astype(np.float32),
                             "sim_addr": dups.sim_addr.values.astype(np.float32)})
        seeds_all = seeds_all[["s1_row", "seed"]].astype(np.int32)
    s1ix, pix = pd.Index(s1rid), pd.Index(pool_rid)
    old_ranges = ranges(len(s1rid), 18)

    def test_fwd(lo, hi):
        """The forward candidates m7 already produced with block() (data/test_pred_*.parquet), reused
        instead of re-blocking; score / nkeys / rank are recomputed by assemble() (same weights, same order)."""
        out = []
        for ob, (a, z) in enumerate(old_ranges):
            if z > lo and a < hi:
                t = pd.read_parquet(os.path.join(DATA, f"test_pred_{ob}.parquet"), columns=["s1_rid", "m_rid"])
                r = s1ix.get_indexer(t.s1_rid.values)
                m = (r >= lo) & (r < hi)
                out.append(pd.DataFrame({"s1_row": r[m].astype(np.int32),
                                         "pidx": pix.get_indexer(t.m_rid.values[m]).astype(np.int32)}))
        f = pd.concat(out, ignore_index=True)
        assert (f.pidx >= 0).all()
        return f

    for b, (lo, hi) in enumerate(ranges(len(s1rid), NB["test"])):
        outp = os.path.join(d, f"pred_{b:03d}.parquet")
        if os.path.exists(outp):
            continue
        if out_of_time():
            return
        t0 = time.time()
        rows = np.arange(lo, hi)
        fwd = test_fwd(lo, hi)
        c = assemble("test", rows, fwd, rev_bucket("test", b), best,
                     seeds_all[(seeds_all.s1_row >= lo) & (seeds_all.s1_row < hi)], dups, cfg, pool_rid, s1rid)
        del fwd
        # features in two halves split at an S1 boundary (context features are per S1, so identical results):
        # halves the record caches - the first test batches peaked at 3.7 GB in one piece
        mid = int(np.median(c.s1_row.values))
        prs = []
        for part in (c[c.s1_row.values <= mid], c[c.s1_row.values > mid]):
            fp = os.path.join(d, "feats_tmp.parquet")
            build_features("test", part.reset_index(drop=True), fp)
            prs.append(predict_file(mA, fp, feats_v3(), ["s1_row", "pidx", "src"]))
            os.remove(fp)
            gc.collect()
        del c, part
        gc.collect()
        pr = pd.concat(prs, ignore_index=True)
        pr.to_parquet(outp + ".tmp", index=False)
        os.replace(outp + ".tmp", outp)
        log(f"test batch {b + 1}/{NB['test']}: {len(pr):,} pairs in {time.time() - t0:.0f}s "
            f"(run {time.time() - tstart:.0f}s, peak RAM {mem_mb()[1]:.0f} MB)")
        del pr
        gc.collect()
    log("all test batches done")


def test_files():
    d = os.path.join(V3, "test")
    files = sorted(os.path.join(d, f) for f in os.listdir(d) if f.startswith("pred_") and f.endswith(".parquet"))
    assert len(files) == NB["test"]
    return files


def stage_write(out_dir):
    from m7_infer import write_lists
    t0 = time.time()
    out_dir = os.path.abspath(out_dir)
    for protected in ("output", "output_v1", "output_v2", "output_baseline"):
        assert out_dir != os.path.abspath(os.path.join(ROOT, "submission", protected)), "protected folder"
    dec = json.load(open(os.path.join(MV3, "decision.json")))
    assert dec["write_allowed"] and dec["kind"] == "rule"
    assert not os.path.exists(out_dir), f"{out_dir} exists - not overwriting"
    final_dir, out_dir = out_dir, out_dir + "_tmp_writing"  # written here, renamed when complete
    os.makedirs(out_dir, exist_ok=True)
    s1rid, pool_rid = s1_rid_arr("test"), pool_rid_arr("test")
    s1_ids = pq.read_table(pq_path("test", 1), columns=["entity_id"]).column("entity_id").to_pylist()
    files = test_files()
    n_cand, srcn = 0, {}
    with open(os.path.join(out_dir, "candidate_pairs.tsv"), "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for fn, (lo, hi) in zip(files, ranges(len(s1rid), NB["test"])):
            pr = pd.read_parquet(fn, columns=["s1_row", "pidx", "src"])
            assert pr.s1_row.min() >= lo and pr.s1_row.max() < hi
            write_lists(f, s1_ids, lo, hi, pr.s1_row.values.astype(np.int64), pool_rid[pr.pidx.values])
            n_cand += len(pr)
            for k_, v in pr.src.value_counts().items():
                srcn[int(k_)] = srcn.get(int(k_), 0) + int(v)
            del pr
    # decision: one-owner over ALL test S1s (full competition), then gate / relative threshold
    sel = full_competition(files, np.ones(len(s1rid), bool), len(pool_rid))
    pmax = sel.groupby("s1_row")["p"].transform("max")
    sel = sel[(pmax >= dec["g"]) & (sel.p >= dec["t"]) & (sel.p >= dec["r"] * pmax)]
    with open(os.path.join(out_dir, "matching_results.tsv"), "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        write_lists(f, s1_ids, 0, len(s1rid), sel.s1_row.values.astype(np.int64), pool_rid[sel.pidx.values])
    os.rename(out_dir, final_dir)
    save_stats("write", {"seconds": round(time.time() - t0, 1), "candidates": n_cand, "matches": len(sel),
                         "src_counts": srcn, "out": final_dir})


def stage_check(out_dir):
    """Own checks (the official validator is run separately): row order, matches subset of candidates,
    list sizes per country. Streams both files line by line."""
    s1 = pq.read_table(pq_path("test", 1), columns=["entity_id", "country"]).to_pandas()
    bad, nm, nc = 0, np.zeros(len(s1), np.int32), np.zeros(len(s1), np.int32)
    with open(os.path.join(out_dir, "matching_results.tsv"), encoding="utf-8") as fm, \
            open(os.path.join(out_dir, "candidate_pairs.tsv"), encoding="utf-8") as fc:
        next(fm), next(fc)
        i = -1
        for i, (lm, lc) in enumerate(zip(fm, fc)):
            a, m = lm.rstrip("\n").split("\t")
            b, c = lc.rstrip("\n").split("\t")
            assert a == b == s1.entity_id.values[i], f"row order differs at {i}"
            ms, cs = (m.split(",") if m else []), (c.split(",") if c else [])
            nm[i], nc[i] = len(ms), len(cs)
            if ms and not set(ms) <= set(cs):
                bad += 1
    assert i == len(s1) - 1
    L = [f"\n## Test output check ({out_dir})\n",
         f"rows {len(s1):,}; rows whose matches are NOT a subset of candidates: {bad}; candidates {nc.sum():,} "
         f"(mean {nc.mean():.2f}, max {nc.max()}); matches {nm.sum():,} (mean {nm.mean():.3f}, max {nm.max()})\n"]
    with open(os.path.join(REP, "m9_candidates.md"), "a", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    log("\n".join(L))


def stage_compare():
    """Per country: test vs fold-0 (full competition, same decision): best-p histogram, matches per S1, % empty."""
    dec = json.load(open(os.path.join(MV3, "decision.json")))
    bins = [0, 0.1, 0.3, 0.5, dec["g"], 0.8, 0.9, 0.99, 1.0001]
    labels = [f"{a:.2f}-{b:.2f}" for a, b in zip(bins[:-1], bins[1:])]

    def summary(files, keep, country, npool):
        allrows = []
        for f in files:
            t = pd.read_parquet(f, columns=["s1_row", "p"])
            allrows.append(t[keep[t.s1_row.values]].groupby("s1_row").p.max())
        bestp = pd.concat(allrows)
        sel = full_competition(files, keep, npool)
        pmax = sel.groupby("s1_row")["p"].transform("max")
        sel = sel[(pmax >= dec["g"]) & (sel.p >= dec["t"]) & (sel.p >= dec["r"] * pmax)]
        rows = np.flatnonzero(keep)
        nmat = sel.groupby("s1_row").size().reindex(rows, fill_value=0)
        bp = bestp.reindex(rows, fill_value=0.0)
        out = {}
        for cn in np.unique(country[rows]):
            m = country[rows] == cn
            h = pd.Series(pd.cut(bp.values[m], bins, labels=labels, right=False)).value_counts(normalize=True).sort_index()
            out[cn] = dict(S1=int(m.sum()), mean_matches=float(nmat.values[m].mean()),
                           empty=float((nmat.values[m] == 0).mean()), hist=h.round(4).to_dict())
        return out

    ctr = pq.read_table(pq_path("train", 1), columns=["country"]).column("country").to_numpy(zero_copy_only=False)
    cte = pq.read_table(pq_path("test", 1), columns=["country"]).column("country").to_numpy(zero_copy_only=False)
    dtr = os.path.join(V3, "v3full")
    ftr = sorted(os.path.join(dtr, f) for f in os.listdir(dtr) if f.startswith("pred_") and f.endswith(".parquet"))
    val = summary(ftr, sample_mask(0), ctr, B.n_rows("train", 2) + B.n_rows("train", 3))
    tst = summary(test_files(), np.ones(len(cte), bool), cte, B.n_rows("test", 2) + B.n_rows("test", 3))
    L = ["\n## Test vs fold-0 per country (v3, full competition, same decision)\n",
         "| set | country | S1 | mean matches | empty | " + " | ".join(f"best p {x}" for x in labels) + " |",
         "|---|---|---|---|---|" + "---|" * len(labels)]
    for name, s in (("fold 0", val), ("test", tst)):
        for cn, v in s.items():
            L.append(f"| {name} | {cn} | {v['S1']:,} | {v['mean_matches']:.3f} | {v['empty']:.4f} | "
                     + " | ".join(f"{v['hist'].get(x, 0):.4f}" for x in labels) + " |")
    with open(os.path.join(REP, "m9_candidates.md"), "a", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    log("\n".join(L))


if __name__ == "__main__":
    st, args = sys.argv[1], sys.argv[2:]
    {"rev": lambda: stage_rev(args[0]), "bucket": lambda: stage_bucket(args[0]), "crossfit": stage_crossfit,
     "oldfull": stage_oldfull, "hop": lambda: stage_hop(args[0]), "hop_samples": stage_hop_samples,
     "eval": stage_eval, "build": stage_build,
     "train": stage_train, "v3full": stage_v3full, "decide": stage_decide, "infer": stage_infer,
     "write": lambda: stage_write(args[0]), "check": lambda: stage_check(args[0]), "compare": stage_compare}[st]()
