"""M11: v5a = stacked record-level matcher on the EXISTING v3 candidates (no new blocking).

New files only: data/v5/ (f1s_*, f1_{split}/, p1_{split}/, ...), models/v5/, reports/m11_v5a.md.

Stages (one at a time; long ones resumable with --budget SECONDS, finished batches are skipped):
  build1           v3 candidates of the two 100k samples (fold-0 validation, fold-1 training) + v5 features
  train1           stage-1 LightGBM cross-fit: A on the fold-1 sample, B on the fold-0 sample
  full1 SPLIT      all S1s of the split: v3 candidates + v5 features (kept on disk) + p1
                   (train: fold-0 sample <- A, fold-1 sample <- B, everyone else mean(A, B); test: mean(A, B))
"""
import gc
import json
import os
import sys
import time

os.environ.setdefault("AMLC_MAXMB", "3900")
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import blocking_v3 as B
import features5 as F5
import m9_cands as M9
from common import DATA, ROOT, WORK, log, mem_mb, pq_path

V5 = os.path.join(DATA, "v5")
MV5 = os.path.join(WORK, "models", "v5")
REP = os.path.join(WORK, "reports", "m11_v5a.md")
for _d in (V5, MV5):
    os.makedirs(_d, exist_ok=True)
T_START = time.time()
BUDGET = float(sys.argv[sys.argv.index("--budget") + 1]) if "--budget" in sys.argv else 1e12
FEATS1 = M9.feats_v3() + F5.NEW


def out_of_time():
    if time.time() - T_START > BUDGET:
        log(f"time budget {BUDGET:.0f}s used - stopping here, re-run the stage to continue")
        return True
    return False


def report(lines):
    with open(REP, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log("\n".join(lines))


def save_stats(name, d):
    d = {**d, "peak_ram_mb": round(mem_mb()[1])}
    json.dump(d, open(os.path.join(V5, f"stats_{name}.json"), "w"), indent=1, default=float)
    log(f"stats {name}: {d}")


def with_label(c, s1rid, rows):
    gt = pd.read_parquet(os.path.join(DATA, "train_gt_pairs.parquet"))
    c = c.merge(gt[gt.s1_rid.isin(s1rid[rows])].assign(label=np.int8(1)), on=["s1_rid", "m_rid"], how="left")
    c["label"] = c.label.fillna(0).astype(np.int8)
    return c


# ------------------------------------------------------------------ stage 1
def stage_build1():
    t0 = time.time()
    cfg = json.load(open(os.path.join(M9.MV3, "cand_cfg.json")))
    pool_rid, s1rid, best = M9.pool_rid_arr("train"), M9.s1_rid_arr("train"), M9.rev_best("train")
    T = M9.target_rows()
    cand = pd.read_parquet(os.path.join(DATA, "cand_trainval.parquet"),
                           columns=["s1_row", "pidx", "score", "nkeys", "rank", "m_rid", "fold"])
    info = {}
    for f in (1, 0):
        if os.path.exists(os.path.join(V5, f"f1s_{f}.parquet")):
            n = pq.ParquetFile(os.path.join(V5, f"f1s_{f}.parquet")).metadata.num_rows
            info[f"fold{f}"] = dict(pairs=n, note="built in an earlier run")
            continue
        rows_all = np.sort(T.s1_row[T.fold == f].values)
        # two halves of the S1 rows, each assembled + featurised separately (in one piece the assemble step
        # went over the 3.9 GB RAM guard); every step is per S1, so the result equals one piece
        halves, npairs, npos = [], 0, 0
        for h, rows in enumerate(np.array_split(rows_all, 2)):
            seeds, dups = M9.hop_inputs("train", rows=rows)
            fw = cand[(cand.fold == f) & np.isin(cand.s1_row.values, rows)].drop(columns=["fold"])
            c = M9.assemble("train", rows, fw, M9.rev_rows("train", rows), best, seeds, dups, cfg, pool_rid, s1rid)
            del seeds, dups, fw
            gc.collect()
            c = with_label(c, s1rid, rows)
            c["fold"] = np.int8(f)
            npairs, npos = npairs + len(c), npos + int(c.label.sum())
            fp = os.path.join(V5, f"f1s_{f}_h{h}.parquet")
            F5.build_features("train", c, fp)
            halves.append(fp)
            del c
            gc.collect()
        info[f"fold{f}"] = dict(pairs=npairs, positives=npos)
        w = None
        out = os.path.join(V5, f"f1s_{f}.parquet")
        for fp in halves:
            with open(fp, "rb") as fh:  # explicit handle: Windows cannot delete a file pyarrow still holds
                pf = pq.ParquetFile(fh)
                for i in range(pf.num_row_groups):
                    t = pf.read_row_group(i)
                    if w is None:
                        w = pq.ParquetWriter(out + ".tmp", t.schema, compression="zstd")
                    w.write_table(t)
                del pf, t
        w.close()
        os.replace(out + ".tmp", out)
        for fp in halves:
            os.remove(fp)
    save_stats("build1", {"seconds": round(time.time() - t0, 1), **info})


def stage_train1():
    from sklearn.metrics import average_precision_score, roc_auc_score
    t0 = time.time()
    mA, iA = M9.train_lgb(os.path.join(V5, "f1s_1.parquet"), FEATS1, os.path.join(MV5, "s1_A.txt"))
    va = M9.predict_file(mA, os.path.join(V5, "f1s_0.parquet"), FEATS1, ["s1_rid", "m_rid", "label", "s1_row", "pidx"])
    imp = pd.Series(mA.feature_importance("gain"), index=FEATS1)
    imp = (imp / imp.sum()).sort_values(ascending=False).round(4)
    del mA
    mB, iB = M9.train_lgb(os.path.join(V5, "f1s_0.parquet"), FEATS1, os.path.join(MV5, "s1_B.txt"))
    vb = M9.predict_file(mB, os.path.join(V5, "f1s_1.parquet"), FEATS1, ["label"])
    v3 = pd.read_parquet(os.path.join(M9.V3, "val_pred.parquet"), columns=["s1_rid", "m_rid", "label", "p"])
    st = dict(seconds=round(time.time() - t0, 1), A=iA, B=iB,
              auc_A_on_fold0=roc_auc_score(va.label, va.p), ap_A_on_fold0=average_precision_score(va.label, va.p),
              auc_B_on_fold1=roc_auc_score(vb.label, vb.p),
              v3_auc_fold0=roc_auc_score(v3.label, v3.p), v3_ap_fold0=average_precision_score(v3.label, v3.p),
              importance_top15=imp.head(15).to_dict(), importance_new=imp[F5.NEW].to_dict())
    save_stats("train1", st)
    report(["\n## Stage 1 (v3 LightGBM retrained with the v5 empty-field features, cross-fit)\n",
            f"- model A (fold-1 sample, {iA['pairs']:,} pairs, best iter {iA['best_iter']}) -> fold-0 sample: "
            f"AUC {st['auc_A_on_fold0']:.5f}, AP {st['ap_A_on_fold0']:.5f} (v3 model on the same pairs: AUC "
            f"{st['v3_auc_fold0']:.5f}, AP {st['v3_ap_fold0']:.5f})",
            f"- model B (fold-0 sample, best iter {iB['best_iter']}) -> fold-1 sample: AUC {st['auc_B_on_fold1']:.5f}",
            f"- new features' gain share: {st['importance_new']}", f"- top-15 gain: {st['importance_top15']}",
            f"- {st['seconds']:.0f}s\n"])


def predict_p1(fp, mA, mB, maskA, maskB, cols):
    """p1 per row of feature file fp: A for maskA S1 rows, B for maskB rows, mean(A, B) otherwise."""
    pf = pq.ParquetFile(fp)
    out = []
    for i in range(pf.num_row_groups):
        v = pf.read_row_group(i, columns=list(dict.fromkeys(FEATS1 + cols))).to_pandas()
        X = v[FEATS1].to_numpy(np.float32)
        pa_ = mA.predict(X, num_threads=10).astype(np.float32)
        pb_ = mB.predict(X, num_threads=10).astype(np.float32)
        s = v.s1_row.values
        p1 = np.where(maskA[s], pa_, np.where(maskB[s], pb_, (pa_ + pb_) / 2)).astype(np.float32)
        o = v[cols].copy()
        o["pA"], o["pB"], o["p1"] = pa_, pb_, p1
        out.append(o)
        del v, X
    return pd.concat(out, ignore_index=True)


def stage_full1(split):
    tstart = time.time()
    cfg = json.load(open(os.path.join(M9.MV3, "cand_cfg.json")))
    fdir, pdir = os.path.join(V5, f"f1_{split}"), os.path.join(V5, f"p1_{split}")
    os.makedirs(fdir, exist_ok=True)
    os.makedirs(pdir, exist_ok=True)
    pool_rid, s1rid, best = M9.pool_rid_arr(split), M9.s1_rid_arr(split), M9.rev_best(split)
    mA, mB = M9.booster(os.path.join(MV5, "s1_A.txt")), M9.booster(os.path.join(MV5, "s1_B.txt"))
    n1 = len(s1rid)
    if split == "train":
        maskA, maskB = M9.sample_mask(0), M9.sample_mask(1)
        # ALL pairs get features + p1: a filter to the records of the sample lists was tried first and dropped,
        # because it truncates the lists of competing S1s (their S1-side / cluster context would be wrong)
    else:
        maskA = maskB = np.zeros(n1, bool)
    seeds_all, dups = M9.hop_inputs(split)
    dups = pd.DataFrame({"seed": dups.seed.values.astype(np.int32), "dup": dups.dup.values.astype(np.int32),
                         "sim_name": dups.sim_name.values.astype(np.float32),
                         "sim_addr": dups.sim_addr.values.astype(np.float32)})
    seeds_all = seeds_all[["s1_row", "seed"]].astype(np.int32)
    if split == "test":
        s1ix, pix = pd.Index(s1rid), pd.Index(pool_rid)
        old_ranges = M9.ranges(n1, 18)

    def fwd(lo, hi):
        if split == "train":
            return M9.fwd_range(lo, hi)
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

    nb = M9.NB[split]
    for b, (lo, hi) in enumerate(M9.ranges(n1, nb)):
        outp = os.path.join(pdir, f"b_{b:03d}.parquet")
        if os.path.exists(outp):
            continue
        if out_of_time():
            return
        t0 = time.time()
        rows = np.arange(lo, hi)
        c = M9.assemble(split, rows, fwd(lo, hi), M9.rev_bucket(split, b), best,
                        seeds_all[(seeds_all.s1_row >= lo) & (seeds_all.s1_row < hi)], dups, cfg, pool_rid, s1rid)
        c = with_label(c, s1rid, rows) if split == "train" else c.assign(label=np.int8(-1))
        # two halves split at an S1 boundary: bounds the record caches (v3 test batches peaked at 3.7 GB in one piece)
        mid = int(np.median(c.s1_row.values))
        prs = []
        for h, part in enumerate((c[c.s1_row.values <= mid], c[c.s1_row.values > mid])):
            fp = os.path.join(fdir, f"b_{b:03d}_{h}.parquet")
            F5.build_features(split, part.reset_index(drop=True), fp)
            prs.append(predict_p1(fp, mA, mB, maskA, maskB, ["s1_row", "pidx", "src", "label"]))
            gc.collect()
        del c, part
        pr = pd.concat(prs, ignore_index=True)
        pr.to_parquet(outp + ".tmp", index=False)
        os.replace(outp + ".tmp", outp)
        log(f"full1 {split} batch {b + 1}/{nb}: {len(pr):,} pairs in {time.time() - t0:.0f}s "
            f"(run {time.time() - tstart:.0f}s, peak RAM {mem_mb()[1]:.0f} MB)")
        del pr, prs
        gc.collect()
    done = len([f for f in os.listdir(pdir) if f.endswith(".parquet")])
    if done == nb:
        save_stats(f"full1_{split}", {"batches": nb, "seconds_last_run": round(time.time() - tstart, 1)})


def p1_files(split):
    d = os.path.join(V5, f"p1_{split}")
    files = sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".parquet") and f.startswith("b_"))
    assert len(files) == M9.NB[split], f"p1_{split} incomplete: {len(files)}/{M9.NB[split]}"
    return files


def argmax_owner(files, col, npool):
    """highest-`col` S1 row of every pool record over all rows of the files (ties: first file / row)."""
    best = np.full(npool, -1.0, np.float32)
    owner = np.full(npool, -1, np.int64)
    for f in files:
        t = pd.read_parquet(f, columns=["s1_row", "pidx", col])
        g = t.groupby("pidx")[col].idxmax().values
        m_, p_ = t.pidx.values[g], t[col].values[g]
        up = p_ > best[m_]
        best[m_[up]] = p_[up]
        owner[m_[up]] = t.s1_row.values[g[up]]
        del t
    return best, owner


def segments(split, pidx):
    """A1 segments of pool records pidx (names -> bool masks)."""
    import m10_v5 as M10
    a = M10.pool_attrs(split).iloc[pidx].reset_index(drop=True)
    seg = {"all": np.ones(len(a), bool), "empty address": a.empty_addr.values, "address present": ~a.empty_addr.values,
           "cross-script": a.nonlatin.values, "same script": ~a.nonlatin.values}
    for c in a.country.cat.categories:
        if (a.country == c).any():
            seg[f"country {c}"] = (a.country == c).values
    seg["S2"], seg["S3"] = a.src.values == 2, a.src.values == 3
    return seg


def stage_diag3():
    """Step 3: can stage 1 already pick the owner? (fold-0 sample, full competition over all train S1s)"""
    import m10_v5 as M10
    t0 = time.time()
    npool = B.n_rows("train", 2) + B.n_rows("train", 3)
    best1, own1 = argmax_owner(p1_files("train"), "p1", npool)
    own = M10.owner_arrays()
    keep0 = M9.sample_mask(0)
    rec = np.flatnonzero((own >= 0) & keep0[np.maximum(own, 0)])        # records owned by fold-0 sample S1s
    # is the owner among the record's candidates? (a label-1 row exists)
    present = np.zeros(npool, bool)
    for f in p1_files("train"):
        t = pd.read_parquet(f, columns=["pidx", "label"])
        present[t.pidx.values[t.label.values == 1]] = True
    pres = rec[present[rec]]
    hit = own1[pres] == own[pres]
    seg = segments("train", pres)
    L = ["\n## Step 3. Stage 1 as a record-level picker (fold-0 sample, full competition, out-of-fold p1)\n",
         f"Records owned by fold-0 sample S1s: {len(rec):,}; owner among the record's candidates: {len(pres):,} "
         f"({len(pres) / len(rec):.4f}).\n",
         "| segment | records (owner present) | argmax p1 = owner | argmax = owner and p1 >= 0.5 |", "|---|---|---|---|"]
    for k, m in seg.items():
        L.append(f"| {k} | {m.sum():,} | {hit[m].mean():.4f} | {(hit[m] & (best1[pres][m] >= 0.5)).mean():.4f} |")
    # v3's fold-0 misses with the owner present
    dec = json.load(open(os.path.join(M9.MV3, "decision.json")))
    v3files = sorted(os.path.join(M9.V3, "v3full", f) for f in os.listdir(os.path.join(M9.V3, "v3full"))
                     if f.startswith("pred_") and f.endswith(".parquet"))
    sel = M9.full_competition(v3files, keep0, npool)
    pmax = sel.groupby("s1_row").p.transform("max")
    sel = sel[(pmax >= dec["g"]) & (sel.p >= dec["t"]) & (sel.p >= dec["r"] * pmax)]
    got = np.zeros(npool, bool)
    got[sel.pidx.values[own[sel.pidx.values] == sel.s1_row.values]] = True
    v3present = np.zeros(npool, bool)
    for f in v3files:
        t = pd.read_parquet(f, columns=["s1_row", "pidx"])
        m = own[t.pidx.values] == t.s1_row.values
        v3present[t.pidx.values[m]] = True
    miss = rec[v3present[rec] & ~got[rec]]
    am = own1[miss] == own[miss]
    pm = best1[miss]
    L += [f"\nv3's fold-0 misses with the owner among v3's candidates: {len(miss):,} (of {v3present[rec].sum():,} "
          f"present). Owner IS argmax p1: {am.mean():.4f}. Of those, p1 of the owner: "
          f"< 0.1 {np.mean(pm[am] < 0.1):.3f}, 0.1-0.5 {np.mean((pm[am] >= 0.1) & (pm[am] < 0.5)):.3f}, "
          f"0.5-0.68 {np.mean((pm[am] >= 0.5) & (pm[am] < 0.68)):.3f}, >= 0.68 {np.mean(pm[am] >= 0.68):.3f} "
          f"(>= 0.68 = would pass v3's gate as a record-level pick).",
          "Misses by segment (share where the owner = argmax p1): " + ", ".join(
              f"{k} {am[v].mean():.3f} (n={v.sum():,})" for k, v in segments("train", miss).items()),
          f"\n({time.time() - t0:.0f}s)\n"]
    report(L)


# ------------------------------------------------------------------ step 4: record text score
TXT_HASH_BITS = 20


def _splitmix(x):
    z = x.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    with np.errstate(over="ignore"):
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def _vec():
    from sklearn.feature_extraction.text import HashingVectorizer
    return HashingVectorizer(analyzer="char_wb", ngram_range=(2, 4), n_features=2 ** TXT_HASH_BITS,
                             alternate_sign=False, norm="l2", dtype=np.float32)


def _vectorize(texts):
    return _vec().transform(texts)


def _pool_text_batches(split, bs=100_000):
    for s in (2, 3):
        for b in pq.ParquetFile(pq_path(split, s)).iter_batches(bs, columns=["rid", "business_name", "business_address"]):
            d = b.to_pandas()
            yield d.rid.values, (d.business_name + " | " + d.business_address).values


def stage_text(limit=None):
    """Record text score: P(record is owned by some S1) from char 2-4-grams of 'name | address'.
    2-fold by record-id hash on train (out-of-fold), test = mean of both models."""
    from concurrent.futures import ProcessPoolExecutor
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import roc_auc_score
    import m10_v5 as M10
    t0 = time.time()
    owned = np.unique(pd.read_parquet(os.path.join(DATA, "train_gt_pairs.parquet"), columns=["m_rid"]).m_rid.values)
    models = [SGDClassifier(loss="log_loss", alpha=2e-7, random_state=s) for s in (0, 1)]
    ex = ProcessPoolExecutor(max_workers=4)

    def vec_batches(split):
        it = _pool_text_batches(split)
        n = 0
        while True:
            chunk = []
            for _ in range(4):
                try:
                    chunk.append(next(it))
                except StopIteration:
                    break
            if not chunk:
                return
            for (rid, _), X in zip(chunk, ex.map(_vectorize, [c[1] for c in chunk])):
                yield rid, X
                n += len(rid)
            if limit and n >= limit:
                return

    EPOCHS = 2
    for ep in range(EPOCHS):
        for rid, X in vec_batches("train"):
            y = np.isin(rid, owned).astype(np.int8)
            f = (_splitmix(rid) >> np.uint64(7)) & np.uint64(1)
            for k in (0, 1):
                m = f == k
                models[k].partial_fit(X[m], y[m], classes=[0, 1])
        log(f"text: epoch {ep + 1} done ({time.time() - t0:.0f}s)")
    out = {}
    for split in ("train", "test"):
        rids, sc = [], []
        for rid, X in vec_batches(split):
            p0, p1_ = (m.predict_proba(X)[:, 1].astype(np.float32) for m in models)
            if split == "train":
                f = ((_splitmix(rid) >> np.uint64(7)) & np.uint64(1)).astype(bool)
                s = np.where(f, p0, p1_)          # record in fold 1 -> model trained on fold 0, and vice versa
            else:
                s = (p0 + p1_) / 2
            rids.append(rid)
            sc.append(s)
        out[split] = (np.concatenate(rids), np.concatenate(sc))
        log(f"text: scored {split} ({time.time() - t0:.0f}s)")
    ex.shutdown()
    L = ["\n## Step 4. Record text score (P(record is owned) from char 2-4-grams, SGD log-loss, 2-fold by record hash)\n"]
    stats = {}
    for split in ("train", "test"):
        rid, s = out[split]
        a = M10.pool_attrs(split)
        if not limit:
            assert np.array_equal(rid, M9.pool_rid_arr(split))
        ctry = a.country.values[:len(rid)].astype(str)
        for c in np.unique(ctry):
            m = ctry == c
            stats[(split, c)] = dict(mean=float(s[m].mean()), p10=float(np.quantile(s[m], 0.1)),
                                     p50=float(np.quantile(s[m], 0.5)), p90=float(np.quantile(s[m], 0.9)))
            if split == "train":
                y = np.isin(rid[m], owned)
                stats[(split, c)]["auc"] = float(roc_auc_score(y, s[m]))
                stats[(split, c)]["owned_share"] = float(y.mean())
    L += ["| split | country | AUC (out-of-fold) | owned share | mean score | p10 | p50 | p90 |", "|---|---|---|---|---|---|---|---|"]
    for (sp, c), v in stats.items():
        L.append(f"| {sp} | {c} | {v.get('auc', float('nan')):.4f} | {v.get('owned_share', float('nan')):.4f} | "
                 f"{v['mean']:.4f} | {v['p10']:.4f} | {v['p50']:.4f} | {v['p90']:.4f} |")
    te = {c: v["mean"] for (sp, c), v in stats.items() if sp == "test"}
    lo_, hi_ = min(te.get("US", np.nan), te.get("India", np.nan)), max(te.get("US", np.nan), te.get("India", np.nan))
    fr = te.get("France")
    france_nan = fr is not None and not (lo_ <= fr <= hi_)
    L.append(f"\nFrance test mean {fr} vs US-India test range [{lo_:.4f}, {hi_:.4f}] -> "
             f"{'OUTSIDE: France score set to NaN' if france_nan else 'inside: kept'}.\n")
    if not limit:
        for split in ("train", "test"):
            rid, s = out[split]
            s = s.astype(np.float32)
            if split == "test" and france_nan:
                s[M10.pool_attrs("test").country.values.astype(str) == "France"] = np.nan
            np.save(os.path.join(V5, f"txt_{split}.npy"), s)
    L.append(f"({time.time() - t0:.0f}s{', LIMITED TEST RUN' if limit else ''})\n")
    report(L)


# ------------------------------------------------------------------ step 5: context features
TOPR = 10  # per record: its 10 highest p1 values are kept (rank / best / 2nd best / pruning up to k = 10)


def stage_rec(split):
    """Per pool record over ALL its candidate pairs (full competition): top-10 p1 values (desc, -1 = none),
    # pairs with p1 > 0.1, list length. One direct sort of packed (pidx, inverted p1 bits)."""
    t0 = time.time()
    npool = B.n_rows(split, 2) + B.n_rows(split, 3)
    parts = [pd.read_parquet(f, columns=["pidx", "p1"]) for f in p1_files(split)]
    pidx = np.concatenate([p.pidx.values for p in parts]).astype(np.uint64)
    p1 = np.concatenate([p.p1.values for p in parts]).astype(np.float32)
    del parts
    n01 = np.bincount(pidx[p1 > 0.1].astype(np.int64), minlength=npool).astype(np.int16)
    ln = np.bincount(pidx.astype(np.int64), minlength=npool).astype(np.int16)
    packed = (pidx << np.uint64(32)) | (np.uint64(0xFFFFFFFF) - p1.view(np.uint32).astype(np.uint64))
    del pidx, p1
    packed.sort()
    pid = (packed >> np.uint64(32)).astype(np.int64)
    p = (np.uint32(0xFFFFFFFF) - (packed & np.uint64(0xFFFFFFFF)).astype(np.uint32)).view(np.float32)
    del packed
    gs = np.flatnonzero(np.r_[True, pid[1:] != pid[:-1]])
    rank = np.arange(len(pid)) - np.repeat(gs, np.diff(np.r_[gs, len(pid)]))
    m = rank < TOPR
    top = np.full((npool, TOPR), -1.0, np.float32)
    top[pid[m], rank[m]] = p[m]
    np.savez(os.path.join(V5, f"rec_{split}.npz"), top=top, n01=n01, ln=ln)
    save_stats(f"rec_{split}", {"seconds": round(time.time() - t0, 1), "pairs": int(len(pid)),
                                "records_with_candidates": int((ln > 0).sum()), "mean_list": float(ln[ln > 0].mean())})


def load_rec(split):
    z = np.load(os.path.join(V5, f"rec_{split}.npz"))
    return z["top"], z["n01"], z["ln"]


def rec_rank(top, pidx, p1):
    """1 + number of the record's top-10 p1 values above p1 (11 = below the record's 10th best)."""
    return 1 + (top[pidx] > p1[:, None]).sum(1)


FEATS2_EXTRA = ["p1", "r_rank", "r_diff", "r_best", "r_2nd", "r_n01", "r_len", "s_rank", "s_n05",
                "clu_name", "clu_addr", "clu_n", "txt"]
CLU_MIN, CLU_MAX = 0.9, 20


def context(d, rec, txt, strings):
    """d: rows of one batch (whole S1 lists: s1_row, pidx, p1, keep). Adds the stage-2 context columns
    for the rows with keep=True and returns only those."""
    from rapidfuzz import fuzz, process
    top, n01, ln = rec
    d = d.copy()
    g = d.groupby("s1_row", sort=False).p1
    d["s_rank"] = g.rank(ascending=False, method="min").astype(np.float32)
    d["s_n05"] = (d.p1 > 0.5).groupby(d.s1_row.values, sort=False).transform("sum").astype(np.float32)
    # confident copies of each S1: its rows with p1 > 0.9, best 20
    cp = d[d.p1 > CLU_MIN].sort_values(["s1_row", "p1"], ascending=[True, False])
    cp = cp[cp.groupby("s1_row").cumcount().values < CLU_MAX][["s1_row", "pidx"]].rename(columns={"pidx": "cpidx"})
    d = d[d.keep.values].reset_index(drop=True)
    pid, p1 = d.pidx.values, d.p1.values
    t = top[pid]
    d["r_rank"] = rec_rank(top, pid, p1).astype(np.float32)
    d["r_best"] = np.where(t[:, 0] >= 0, t[:, 0], np.nan).astype(np.float32)
    d["r_2nd"] = np.where(t[:, 1] >= 0, t[:, 1], np.nan).astype(np.float32)
    other = np.where(p1 < t[:, 0], t[:, 0], t[:, 1])
    d["r_diff"] = np.where(other >= 0, p1 - other, np.nan).astype(np.float32)
    d["r_n01"], d["r_len"] = n01[pid].astype(np.float32), ln[pid].astype(np.float32)
    d["txt"] = txt[pid]
    # cluster: max name_core similarity to the S1's other confident copies, address similarity to that copy
    j = pd.DataFrame({"row": np.arange(len(d)), "s1_row": d.s1_row.values, "pidx": pid}).merge(cp, on="s1_row")
    j = j[j.pidx.values != j.cpidx.values]
    d["clu_n"] = np.bincount(j.row.values, minlength=len(d)).astype(np.float32)
    d["clu_name"], d["clu_addr"] = np.float32(np.nan), np.float32(np.nan)
    if len(j):
        W = dict(workers=-1, dtype=np.float32)
        sn = process.cpdist(strings.get("core", j.pidx.values), strings.get("core", j.cpidx.values),
                            scorer=fuzz.token_set_ratio, **W)
        j = j.assign(sn=sn).sort_values(["row", "sn"], ascending=[True, False]).drop_duplicates("row")
        sa = process.cpdist(strings.get("addr", j.pidx.values), strings.get("addr", j.cpidx.values),
                            scorer=fuzz.token_set_ratio, **W)
        cn, ca = d.clu_name.values.copy(), d.clu_addr.values.copy()
        cn[j.row.values], ca[j.row.values] = j.sn.values, sa
        d["clu_name"], d["clu_addr"] = cn, ca
    return d


def batch_frames(split, b):
    """stage-1 features (both halves) + p1 of batch b, aligned row by row."""
    fdir, pdir = os.path.join(V5, f"f1_{split}"), os.path.join(V5, f"p1_{split}")
    f = pd.concat([pd.read_parquet(os.path.join(fdir, f"b_{b:03d}_{h}.parquet")) for h in (0, 1)], ignore_index=True)
    p = pd.read_parquet(os.path.join(pdir, f"b_{b:03d}.parquet"))
    assert len(f) == len(p) and (f.s1_row.values == p.s1_row.values).all() and (f.pidx.values == p.pidx.values).all()
    f["p1"] = p.p1.values
    return f


def stage_ctx(split, k=None):
    """Stage-2 rows with all features. train: the pairs of every record that appears in a fold-0 sample list
    (full-competition evaluation) + the fold-1 sample pairs (training) -> f2s_0 / f2s_1 / f2c_train.
    test: the pairs with record rank r_rank <= k -> f2_test/b_*.parquet."""
    import pyarrow as pa
    t0 = time.time()
    rec = load_rec(split)
    txt = np.load(os.path.join(V5, f"txt_{split}.npy"))
    strings = B.PoolStrings(split)
    out_dir = os.path.join(V5, f"f2_{split}")
    os.makedirs(out_dir, exist_ok=True)
    if split == "train":
        keep0, keep1 = M9.sample_mask(0), M9.sample_mask(1)
        R0 = np.zeros(len(rec[2]), bool)
        for f in p1_files("train"):
            t = pd.read_parquet(f, columns=["s1_row", "pidx"])
            R0[t.pidx.values[keep0[t.s1_row.values]]] = True
        log(f"records in fold-0 sample lists: {R0.sum():,}")
    n_rows = 0
    for b in range(M9.NB[split]):
        outp = os.path.join(out_dir, f"b_{b:03d}.parquet")
        if os.path.exists(outp):
            continue
        if out_of_time():
            return
        t1 = time.time()
        f = batch_frames(split, b)
        if split == "train":
            f["keep"] = R0[f.pidx.values] | keep1[f.s1_row.values]
        elif k is None:
            f["keep"] = True
        else:
            f["keep"] = rec_rank(rec[0], f.pidx.values, f.p1.values) <= k
        cols = ["s1_row", "pidx", "p1", "keep"]
        c = context(f[cols], rec, txt, strings)
        f = f[f.keep.values].reset_index(drop=True)
        assert (f.s1_row.values == c.s1_row.values).all() and (f.pidx.values == c.pidx.values).all()
        for col in FEATS2_EXTRA:
            f[col] = c[col].values
        f = f.drop(columns=["keep"])
        f.to_parquet(outp + ".tmp", index=False)
        os.replace(outp + ".tmp", outp)
        n_rows += len(f)
        log(f"ctx {split} batch {b + 1}/{M9.NB[split]}: {len(f):,} rows in {time.time() - t1:.0f}s "
            f"(run {time.time() - t0:.0f}s, peak RAM {mem_mb()[1]:.0f} MB)")
        del f, c
        gc.collect()
    save_stats(f"ctx_{split}", {"seconds_last_run": round(time.time() - t0, 1), "rows_last_run": n_rows, "k": k})


# ------------------------------------------------------------------ step 6: stage 2
FEATS2 = FEATS1 + FEATS2_EXTRA
P2COLS = ["s1_row", "pidx", "label", "p1", "r_rank", "pool_addr_empty"]


def f2_files(split):
    d = os.path.join(V5, f"f2_{split}")
    files = sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".parquet") and f.startswith("b_"))
    assert len(files) == M9.NB[split], f"f2_{split} incomplete"
    return files


def gather(mask, out):
    """rows of f2_train whose S1 row is in mask -> one file (for LightGBM training)."""
    import pyarrow as pa
    w = None
    for f in f2_files("train"):
        t = pd.read_parquet(f)
        t = t[mask[t.s1_row.values]]
        tab = pa.Table.from_pandas(t, preserve_index=False)
        if w is None:
            w = pq.ParquetWriter(out, tab.schema, compression="zstd")
        w.write_table(tab, row_group_size=500_000)
        del t, tab
    w.close()


def stage_train2():
    from sklearn.metrics import average_precision_score, roc_auc_score
    t0 = time.time()
    keep0, keep1 = M9.sample_mask(0), M9.sample_mask(1)
    s0, s1 = os.path.join(V5, "f2s_0.parquet"), os.path.join(V5, "f2s_1.parquet")
    gather(keep1, s1)
    gather(keep0, s0)
    mA, iA = M9.train_lgb(s1, FEATS2, os.path.join(MV5, "s2_A.txt"))
    imp = pd.Series(mA.feature_importance("gain"), index=FEATS2)
    imp = (imp / imp.sum()).sort_values(ascending=False).round(4)
    mB, iB = M9.train_lgb(s0, FEATS2, os.path.join(MV5, "s2_B.txt"))
    parts = []
    for f in f2_files("train"):
        v = pd.read_parquet(f)
        X = v[FEATS2].to_numpy(np.float32)
        b = keep1[v.s1_row.values]
        p = np.empty(len(v), np.float32)
        p[~b] = mA.predict(X[~b], num_threads=10)
        if b.any():
            p[b] = mB.predict(X[b], num_threads=10)
        o = v[P2COLS].copy()
        o["p2"] = p
        o["fold"] = np.where(keep0[v.s1_row.values], 0, np.where(b, 1, -1)).astype(np.int8)
        parts.append(o)
        del v, X
    d = pd.concat(parts, ignore_index=True)
    d.to_parquet(os.path.join(V5, "p2_train.parquet"), index=False)
    z = d[d.fold == 0]
    st = dict(seconds=round(time.time() - t0, 1), A=iA, B=iB, rows=len(d),
              auc_p2_fold0=roc_auc_score(z.label, z.p2), ap_p2_fold0=average_precision_score(z.label, z.p2),
              auc_p1_fold0=roc_auc_score(z.label, z.p1), ap_p1_fold0=average_precision_score(z.label, z.p1),
              importance_top20=imp.head(20).to_dict(), importance_extra=imp[FEATS2_EXTRA].to_dict())
    save_stats("train2", st)
    report(["\n## Step 6. Stage 2 (features = stage-1 v5 features + p1 + context)\n",
            f"- model A2 trained on the fold-1 sample ({iA['pairs']:,} pairs, best iter {iA['best_iter']}, "
            f"{iA['seconds']:.0f}s); B2 on the fold-0 sample (only to give the fold-1 sample S1s out-of-fold p2 "
            f"as competitors), best iter {iB['best_iter']}",
            f"- fold-0 sample pairs: AUC p2 {st['auc_p2_fold0']:.5f} / AP {st['ap_p2_fold0']:.5f} vs p1 "
            f"{st['auc_p1_fold0']:.5f} / {st['ap_p1_fold0']:.5f}",
            f"- gain share of the new features: {st['importance_extra']}",
            f"- top-20 gain: {st['importance_top20']}", f"- rows scored: {len(d):,}; {st['seconds']:.0f}s\n"])


# ------------------------------------------------------------------ step 7-8: decision + pruning
def f05_parts(sel_idx, sel_lab, n, ntrue):
    npred = np.bincount(sel_idx, minlength=n).astype(np.float64)
    tp = np.bincount(sel_idx, weights=sel_lab, minlength=n)
    f = np.where(ntrue == 0, (npred == 0).astype(float),
                 np.divide(1.25 * tp, npred + 0.25 * ntrue, out=np.zeros_like(tp), where=(npred + 0.25 * ntrue) > 0))
    return f, npred, tp


def owner_frame(d, pcol, keep0, s1rid, pool_rid):
    """one-owner argmax of pcol over all rows of d, then the rows of fold-0 sample S1s (M10.Frame input)."""
    g = d.groupby("pidx")[pcol].idxmax().values
    o = d.loc[g]
    o = o[keep0[o.s1_row.values]]
    return pd.DataFrame({"s1_rid": s1rid[o.s1_row.values], "pidx": o.pidx.values, "p": o[pcol].values,
                         "label": o.label.values, "s1_row": o.s1_row.values})


def seg_report(name, F, sel, S1seg):
    f, npred, tp = f05_parts(F.idx[sel], F.lab[sel], F.n, F.ntrue)
    r = {"F0.5": f.mean(), "precision": tp.sum() / max(npred.sum(), 1), "recall": tp.sum() / F.ntrue.sum(),
         "singleton acc": float((npred[F.ntrue == 0] == 0).mean())}
    for k, m in S1seg.items():
        r[k] = f[m].mean()
    return name, r


def stage_decide2():
    import itertools
    import m10_v5 as M10
    import m6b_expected_f as EF
    t0 = time.time()
    s1rid, pool_rid = M9.s1_rid_arr("train"), M9.pool_rid_arr("train")
    keep0 = M9.sample_mask(0)
    empty = M10.pool_attrs("train").empty_addr.values
    nonlatin = M10.pool_attrs("train").nonlatin.values
    T = M9.target_rows()
    val_ids = np.sort(s1rid[T.s1_row[T.fold == 0].values])
    h1 = EF.split_half(val_ids)
    gt = pd.read_parquet(os.path.join(DATA, "train_gt_pairs.parquet"))
    pairs = gt[gt.s1_rid.isin(val_ids)].reset_index(drop=True)
    del gt
    pairs["pidx"] = pd.Index(pool_rid).get_indexer(pairs.m_rid.values)
    ctry = pq.read_table(pq_path("train", 1), columns=["country"]).column("country").to_numpy(zero_copy_only=False)
    ids = {"all": val_ids, "h1": val_ids[h1], "h2": val_ids[~h1]}

    def s1_segments(idv):
        rows = pd.Index(s1rid).get_indexer(idv)
        pe = pairs.assign(e=empty[pairs.pidx.values], x=nonlatin[pairs.pidx.values]).groupby("s1_rid")[["e", "x"]].any()
        pe = pe.reindex(idv, fill_value=False)
        seg = {f"country {c}": ctry[rows] == c for c in np.unique(ctry[rows])}
        seg["S1s with an empty-address match"] = pe.e.values
        seg["S1s with a cross-script match"] = pe.x.values
        return seg

    def frames(o):
        out = {}
        for k, idv in ids.items():
            dd = o[o.s1_rid.isin(idv)].reset_index(drop=True)
            nt = pairs[pairs.s1_rid.isin(idv)].groupby("s1_rid").size().reindex(idv, fill_value=0).values.astype(float)
            out[k] = M10.Frame(dd, idv, nt, empty)
        return out

    d = pd.read_parquet(os.path.join(V5, "p2_train.parquet"))
    res, table = {}, []
    SEG2 = s1_segments(ids["h2"])
    # v3 on the same half (full competition, decision.json)
    dec3 = json.load(open(os.path.join(M9.MV3, "decision.json")))
    v3files = sorted(os.path.join(M9.V3, "v3full", f) for f in os.listdir(os.path.join(M9.V3, "v3full"))
                     if f.startswith("pred_") and f.endswith(".parquet"))
    Fv3 = M10.fold0_frames(v3files, empty)
    s, sel = Fv3["h2"].rule(dec3["g"], dec3["t"], dec3["r"])
    table.append(seg_report(f"v3 (rule g {dec3['g']} r {dec3['r']})", Fv3["h2"], sel, SEG2))
    del Fv3
    gc.collect()
    # stage 1 alone, record-centric on p1, and stage 2 with pruning
    for pcol, ks in (("p1", [None]), ("p2", [1, 2, 3, 5, 10, None])):
        for k in ks:
            dk = d if k is None else d[d.r_rank.values <= k]
            Fr = frames(owner_frame(dk, pcol, keep0, s1rid, pool_rid))
            s1_, (t, te) = M10.tune_rc(Fr["h1"], split_t=True)
            s2, sel2 = Fr["h2"].rc(t, te)
            sa, (ta, tea) = M10.tune_rc(Fr["all"], split_t=True)
            cands = dk[dk.fold == 0].groupby("s1_row").size().reindex(np.flatnonzero(keep0), fill_value=0)
            name = f"{'stage 1 (p1)' if pcol == 'p1' else 'v5a stage 2 (p2)'} top-{k if k else 'all'} per record"
            res[(pcol, k)] = dict(h2=s2, t=t, te=te, all_tuned=sa, t_all=ta, te_all=tea,
                                  cands_mean=float(cands.mean()), cands_median=float(cands.median()),
                                  cands_p95=float(cands.quantile(0.95)))
            table.append(seg_report(f"{name} (t {t}, t_empty {te})", Fr["h2"], sel2, SEG2))
            del Fr
            gc.collect()
    f_none = res[("p2", None)]["all_tuned"]
    k_best = next(k for k in [1, 2, 3, 5, 10, None] if res[("p2", k)]["all_tuned"] >= f_none - 0.0005)
    ch = res[("p2", k_best)]
    v3h2 = table[0][1]["F0.5"]
    better = ch["h2"] > v3h2
    cols = list(table[0][1].keys())
    L = ["\n## Step 7-8. Decision and candidate pruning (fold-0 sample, full competition)\n",
         "Record-centric: each pool record goes to its argmax S1 (p2; p1 for the stage-1 row) and is kept if p >= t "
         "(t_empty for empty-address records). t / t_empty tuned on half 1, all numbers below on half 2 "
         f"({(~h1).sum():,} S1s). Segment columns = macro F0.5 over those S1s.\n",
         "| decision | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for name, r in table:
        L.append(f"| {name} | " + " | ".join(f"{v:.4f}" for v in r.values()) + " |")
    L += ["\nPruning (stage 2 scores only each record's top-k pairs by p1; t / t_empty re-tuned per k on all of "
          "fold 0) and candidates per fold-0 S1:\n",
          "| k | fold-0 F0.5 (tuned on fold 0) | half 2 (half-1-tuned) | cands/S1 mean | median | p95 |",
          "|---|---|---|---|---|---|"]
    for k in [1, 2, 3, 5, 10, None]:
        r = res[("p2", k)]
        L.append(f"| {k or 'all (= v3 candidates)'} | {r['all_tuned']:.5f} | {r['h2']:.5f} | {r['cands_mean']:.1f} | "
                 f"{r['cands_median']:.0f} | {r['cands_p95']:.0f} |")
    L.append(f"\n**Chosen k = {k_best or 'all'}** (smallest k within 0.0005 of no pruning on fold 0). "
             f"Gate: v5a half 2 {ch['h2']:.5f} vs v3 half 2 {v3h2:.5f} ({ch['h2'] - v3h2:+.5f}) -> "
             f"{'WRITE output_v5a' if better else 'DO NOT WRITE'}.\n")
    report(L)
    json.dump({"k": k_best, "t": ch["t_all"], "t_empty": ch["te_all"], "h2": ch["h2"], "v3_h2": v3h2,
               "write_allowed": bool(better), "results": {f"{a}|{b}": v for (a, b), v in res.items()}},
              open(os.path.join(MV5, "decision_v5a.json"), "w"), indent=1, default=float)
    save_stats("decide2", {"seconds": round(time.time() - t0, 1)})


# ------------------------------------------------------------------ step 9: test
def stage_infer2():
    t0 = time.time()
    dec = json.load(open(os.path.join(MV5, "decision_v5a.json")))
    mA = M9.booster(os.path.join(MV5, "s2_A.txt"))
    parts = []
    for f in f2_files("test"):
        v = pd.read_parquet(f, columns=list(dict.fromkeys(FEATS2 + ["s1_row", "pidx"])))
        o = v[["s1_row", "pidx", "pool_addr_empty"]].copy()
        o["p2"] = mA.predict(v[FEATS2].to_numpy(np.float32), num_threads=10).astype(np.float32)
        parts.append(o)
        del v
    d = pd.concat(parts, ignore_index=True)
    d.to_parquet(os.path.join(V5, "p2_test.parquet"), index=False)
    save_stats("infer2", {"seconds": round(time.time() - t0, 1), "pairs": len(d), "k": dec["k"]})


def stage_write2(out_dir):
    from m7_infer import write_lists
    t0 = time.time()
    out_dir = os.path.abspath(out_dir)
    assert os.path.basename(out_dir) == "output_v5a"
    dec = json.load(open(os.path.join(MV5, "decision_v5a.json")))
    assert dec["write_allowed"], "v5a did not beat v3 on half 2"
    assert not os.path.exists(out_dir), f"{out_dir} exists - not overwriting"
    tmp = out_dir + "_tmp_writing"
    os.makedirs(tmp, exist_ok=True)
    s1rid, pool_rid = M9.s1_rid_arr("test"), M9.pool_rid_arr("test")
    s1_ids = pq.read_table(pq_path("test", 1), columns=["entity_id"]).column("entity_id").to_pylist()
    d = pd.read_parquet(os.path.join(V5, "p2_test.parquet"))
    with open(os.path.join(tmp, "candidate_pairs.tsv"), "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        write_lists(f, s1_ids, 0, len(s1rid), d.s1_row.values.astype(np.int64), pool_rid[d.pidx.values])
    g = d.groupby("pidx").p2.idxmax().values
    o = d.loc[g]
    o = o[np.where(o.pool_addr_empty.values > 0, o.p2.values >= dec["t_empty"], o.p2.values >= dec["t"])]
    with open(os.path.join(tmp, "matching_results.tsv"), "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        write_lists(f, s1_ids, 0, len(s1rid), o.s1_row.values.astype(np.int64), pool_rid[o.pidx.values])
    os.rename(tmp, out_dir)
    nc = d.groupby("s1_row").size().reindex(np.arange(len(s1rid)), fill_value=0)
    nm = o.groupby("s1_row").size().reindex(np.arange(len(s1rid)), fill_value=0)
    ctry = pq.read_table(pq_path("test", 1), columns=["country"]).column("country").to_numpy(zero_copy_only=False)
    per = {c: dict(mean_matches=float(nm.values[ctry == c].mean()), empty=float((nm.values[ctry == c] == 0).mean()))
           for c in np.unique(ctry)}
    info = dict(seconds=round(time.time() - t0, 1), candidates=len(d), cand_mean=float(nc.mean()),
                cand_median=float(nc.median()), cand_p95=float(nc.quantile(0.95)), cand_max=int(nc.max()),
                matches=len(o), match_mean=float(nm.mean()), empty_rows=float((nm == 0).mean()), per_country=per)
    save_stats("write2", info)
    report([f"\n## Step 9. Test output -> {out_dir}\n", f"{info}\n"])


if __name__ == "__main__":
    st, args = sys.argv[1], [a for a in sys.argv[2:] if not a.startswith("--")]
    {"build1": stage_build1, "train1": stage_train1, "full1": lambda: stage_full1(args[0]),
     "diag3": stage_diag3, "text": lambda: stage_text(int(args[0]) if args else None),
     "rec": lambda: stage_rec(args[0]),
     "ctx": lambda: stage_ctx(args[0], json.load(open(os.path.join(MV5, "decision_v5a.json")))["k"]
                              if args[0] == "test" else None),
     "train2": stage_train2, "decide2": stage_decide2, "infer2": stage_infer2,
     "write2": lambda: stage_write2(args[0])}[st]()
