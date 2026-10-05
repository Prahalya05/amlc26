"""DIAGNOSIS ONLY (changes no pipeline code / data / outputs): where does validation macro F0.5 go?

Validation = the 100k fold-0 S1s of m5_train.py, with the EXISTING candidates (cand_trainval.parquet,
K=30, cap 1000) and model probabilities (val_pred.parquet), decision = models/decision.json.

Usage: python diag_losses.py STAGE      (run one at a time, in this order)
  misses  : blocking misses. Cause (a) no shared key / (b) only keys with df > cap / (c) retrieved
            but rank > K; forward rank of the true pool record, reverse rank of the true S1
            (pool record -> all 2.2M train S1, same keys, same idf/cap scoring, df counted over S1),
            close match (name_core + address) of the missed record to a candidate of the same S1.
  decide  : decision losses on the validation S1s, false-positive list with the true owner.
  owners  : for false positives whose true owner is not a validation S1: block + features + model
            for those owners (same config as the pipeline) -> did the owner have it, with which p?
  report  : features / country section and the final loss table -> reports/diag_losses.md
Intermediate files: %AMLC_DIAG% (default: session scratchpad\\diag).
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
from rapidfuzz import fuzz, process

from blocking import DEFAULT_CFG, QMAX, KeyLookup, block_chunked, key_files, load_s1_keys, pool_info
from common import DATA, WORK, log, pq_path

SCR = os.environ.get("AMLC_DIAG", r"C:\Users\Prahalya\AppData\Local\Temp\claude\C--amlc"
                     r"\50a3ac7e-0498-4cfb-b107-2f12544cd93a\scratchpad\diag")
os.makedirs(SCR, exist_ok=True)
REP = os.path.join(WORK, "reports")
MODELS = os.path.join(WORK, "models")
BLOCK_CFG = json.load(open(os.path.join(WORK, "block_cfg.json")))
CFG = {**DEFAULT_CFG, **BLOCK_CFG}
TYPES = np.array(CFG["types"], np.int8)
K, CAP = CFG["k"], CFG["cap"]
TW = np.zeros(16)
for _k, _v in CFG["type_w"].items():
    TW[_k] = _v
CLOSE = 90  # "close match": token_set_ratio >= 90 on name_core AND on addr_n


def sp(name):
    return os.path.join(SCR, name)


def md_table(df, floatfmt="{:.4f}"):
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join([df.index.name or ""] + cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for i, r in df.iterrows():
        cells = [f"{int(v):,}" if isinstance(v, (float, np.floating)) and float(v).is_integer() and abs(v) >= 2
                 else floatfmt.format(v) if isinstance(v, (float, np.floating)) else f"{v:,}" if isinstance(
            v, (int, np.integer)) else str(v) for v in r.values]
        lines.append("| " + " | ".join([str(i)] + cells) + " |")
    return "\n".join(lines)


# ------------------------------------------------------------------ generic "rank of the target" search
def scan(files, U, gather_mask):
    """df of every key of U over the index files; also every key row of the records in gather_mask."""
    look = KeyLookup(U)
    df = np.zeros(len(U), np.int64)
    got = []
    for path, off in files:
        pf = pq.ParquetFile(path)
        for i in range(pf.num_row_groups):
            t = pf.read_row_group(i)
            row = t.column("row").to_numpy().astype(np.int64) + off
            key = t.column("key").to_numpy()
            pos = look(key)
            hit = pos >= 0
            df += np.bincount(pos[hit], minlength=len(U))
            g = gather_mask[row]
            if g.any():
                got.append(pd.DataFrame({"row": row[g], "key": key[g], "kt": t.column("kt").to_numpy()[g]}))
            del t, row, key, pos, hit, g
    return df, pd.concat(got, ignore_index=True)


def postings(files, U, allowed, df):
    """Same as blocking.block pass B: postings of the allowed keys, grouped by key position."""
    ua = np.flatnonzero(allowed)
    look = KeyLookup(U[ua])
    packed = []
    for path, off in files:
        pf = pq.ParquetFile(path)
        for i in range(pf.num_row_groups):
            t = pf.read_row_group(i)
            row = t.column("row").to_numpy().astype(np.int64) + off
            pos = look(t.column("key").to_numpy())
            hit = pos >= 0
            packed.append((ua[pos[hit]].astype(np.uint64) << np.uint64(32)) | row[hit].astype(np.uint64))
            del t, row, pos, hit
    packed = np.concatenate(packed)
    packed.sort()
    post = (packed & np.uint64(0xFFFFFFFF)).astype(np.int32)
    del packed
    offs = np.zeros(len(U) + 1, np.int64)
    offs[1:] = np.cumsum(np.where(allowed, df, 0))
    return post, offs


def rank_targets(q_id, q_key, q_kt, pairs, files, n_index, gather_mask, max_pairs_chunk=4_000_000):
    """Blocking-identical search for queries q (local ids 0..nq-1, their keys q_key/q_kt) against an
    index (key files `files`, n_index records): key weight = log(n_index/df) * type weight if
    1 <= df <= cap, score = sum of shared key weights, order = (quantised score desc, index row asc).
    For every row of `pairs` (q, t) returns: score, rank (0-based, -1 if the target shares no allowed
    key = never retrieved), n_ret (#index records retrieved for q), and the shared-key analysis
    n_any (shared keys of any df), n_allowed (shared keys with df <= cap), min_df, kt_bits."""
    t0 = time.time()
    U, inv = np.unique(q_key, return_inverse=True)
    ktU = np.zeros(len(U), np.int8)
    ktU[inv] = q_kt
    df, got = scan(files, U, gather_mask)
    allowed = (df >= 1) & (df <= CAP)
    w = np.where(allowed, np.log(n_index / np.maximum(df, 1)) * TW[ktU], 0.0).astype(np.float32)
    log(f"  keys {len(q_key):,} (distinct {len(U):,}), allowed {allowed.sum():,} ({time.time() - t0:.0f}s)")
    post, offs = postings(files, U, allowed, df)
    log(f"  postings {len(post):,} ({time.time() - t0:.0f}s)")

    P = pairs[["q", "t"]].reset_index(drop=True)
    n = len(P)
    out_rank = np.full(n, -1, np.int64)
    out_score = np.zeros(n, np.float32)
    out_nret = np.zeros(n, np.int64)
    keep = allowed[inv]
    packed = np.sort((q_id[keep].astype(np.uint64) << np.uint64(32)) | inv[keep].astype(np.uint64))
    qq = (packed >> np.uint64(32)).astype(np.int64)
    pp = (packed & np.uint64(0xFFFFFFFF)).astype(np.int64)
    del packed
    lens = offs[pp + 1] - offs[pp]
    first = np.flatnonzero(np.r_[True, np.diff(qq) != 0]) if len(qq) else np.array([], np.int64)
    urow = qq[first]
    per_row = np.add.reduceat(lens, first) if len(first) else np.array([], np.int64)
    bounds, acc, start = [], 0, 0
    for i, v in enumerate(per_row):
        if acc + v > max_pairs_chunk and i > start:
            bounds.append((start, i))
            start, acc = i, 0
        acc += v
    bounds.append((start, len(urow)))
    first = np.append(first, len(qq))
    Pq, Pt = P.q.values, P.t.values
    for a, b in bounds:
        if a >= b:
            continue
        lo, hi = first[a], first[b]
        r, p, L = qq[lo:hi], pp[lo:hi], lens[lo:hi]
        tot = int(L.sum())
        if tot == 0:
            continue
        rr = np.repeat(r, L)
        starts = np.repeat(offs[p] - np.cumsum(L) + L, L)
        cand = post[starts + np.arange(tot)].astype(np.int64)
        ww = np.repeat(w[p], L)
        code = rr * n_index + cand
        del rr, starts, cand
        g = pd.DataFrame({"code": code, "w": ww}).groupby("code", sort=False).w.agg(["sum", "size"])
        del code, ww
        uc = g.index.to_numpy()
        score = g["sum"].to_numpy().astype(np.float32)
        del g
        cr = (uc // n_index).astype(np.uint64)
        cp = (uc % n_index).astype(np.uint64)
        qs = np.minimum(np.round(score.astype(np.float64) * 256), QMAX).astype(np.uint64)
        packed = np.sort((cr << np.uint64(42)) | ((np.uint64(QMAX) - qs) << np.uint64(24)) | cp)
        cr = (packed >> np.uint64(42)).astype(np.int64)
        cp = (packed & np.uint64((1 << 24) - 1)).astype(np.int64)
        del packed, qs
        gstart = np.flatnonzero(np.r_[True, np.diff(cr) != 0])
        gsize = np.diff(np.r_[gstart, len(cr)])
        rank = np.arange(len(cr)) - np.repeat(gstart, gsize)
        sel = np.flatnonzero((Pq >= urow[a]) & (Pq <= urow[b - 1]))
        tcode = Pq[sel] * n_index + Pt[sel]
        ix = pd.Index(cr * n_index + cp).get_indexer(tcode)
        ok = ix >= 0
        out_rank[sel[ok]] = rank[ix[ok]]
        out_score[sel[ok]] = score[pd.Index(uc).get_indexer(tcode[ok])]
        nret = pd.Series(gsize, index=cr[gstart])
        out_nret[sel] = nret.reindex(Pq[sel], fill_value=0).values
        del uc, score, cr, cp, rank, ix
        gc.collect()
    del post, offs
    log(f"  scoring done ({time.time() - t0:.0f}s)")

    # shared-key analysis on the (q, t) pairs
    look = KeyLookup(U)
    tk = pd.DataFrame({"t": got.row.values, "pos": look(got.key.values)})
    tk = tk[tk.pos >= 0]
    qk = pd.DataFrame({"q": q_id, "pos": inv})
    m = P.reset_index().rename(columns={"index": "pi"}).merge(qk, on="q").merge(tk, on=["t", "pos"])
    m["allowed"] = allowed[m.pos.values]
    m["df"] = df[m.pos.values]
    m["bit"] = np.left_shift(1, ktU[m.pos.values].astype(np.int64))
    agg = m.groupby("pi").agg(n_any=("pos", "size"), n_allowed=("allowed", "sum"), min_df=("df", "min"))
    bits = m.drop_duplicates(["pi", "bit"]).groupby("pi").bit.sum()
    res = pd.DataFrame({"score": out_score, "rank": out_rank, "n_ret": out_nret})
    res["n_any"] = agg.n_any.reindex(res.index, fill_value=0).values
    res["n_allowed"] = agg.n_allowed.reindex(res.index, fill_value=0).values
    res["min_df"] = agg.min_df.reindex(res.index, fill_value=0).values
    res["kt_bits"] = bits.reindex(res.index, fill_value=0).values
    return res, got


def load_rows(split, src, rows, cols):
    """Selected rows (row numbers inside the source parquet) with the given columns, indexed by row."""
    rows = np.unique(rows)
    pf = pq.ParquetFile(pq_path(split, src))
    mask = np.zeros(pf.metadata.num_rows, bool)
    mask[rows] = True
    out, pos = [], 0
    for b in pf.iter_batches(500_000, columns=cols):
        m = mask[pos:pos + b.num_rows]
        if m.any():
            t = pa.Table.from_batches([b]).filter(pa.array(m)).to_pandas()
            t.index = np.flatnonzero(m) + pos
            out.append(t)
        pos += b.num_rows
    return pd.concat(out)


def load_pool_rows(pidx, cols, n2):
    pidx = np.unique(pidx)
    a = load_rows("train", 2, pidx[pidx < n2], cols)
    b = load_rows("train", 3, pidx[pidx >= n2] - n2, cols)
    b.index = b.index + n2
    return pd.concat([a, b])


# ------------------------------------------------------------------ decision (identical to m6 / m7)
def decided():
    from m6_decide import one_owner, prepare, score
    vp = pd.read_parquet(os.path.join(DATA, "val_pred.parquet"))
    s1 = pd.read_parquet(os.path.join(DATA, "trainval_s1.parquet"))
    val_ids = s1.s1_rid[s1.fold == 0].values
    gt = pd.read_parquet(os.path.join(DATA, "train_gt_pairs.parquet"))
    pairs = gt[gt.s1_rid.isin(val_ids)].reset_index(drop=True)
    dec = json.load(open(os.path.join(MODELS, "decision.json")))
    d = (one_owner(vp) if dec["one_owner"] else vp).reset_index(drop=True)
    P = prepare(d, val_ids, pairs)
    f = score(P, dec["g"], dec["t"], dec["r"])
    sel = (P["pmax"] >= dec["g"]) & (P["p"] >= dec["t"]) & (P["p"] >= dec["r"] * P["pmax"])
    d["pmax_after_owner"] = P["pmax"]
    d["pred"] = sel
    return vp, d, val_ids, pairs, gt, dec, f


# ------------------------------------------------------------------ stage: misses
def stage_misses():
    t0 = time.time()
    s1rid = pq.read_table(pq_path("train", 1), columns=["rid"]).column("rid").to_numpy()
    pool_rid, n2 = pool_info("train")
    npool, ns1 = len(pool_rid), len(s1rid)
    vp, d, val_ids, pairs, gt, dec, f = decided()
    log(f"decision reproduced: macro F0.5 = {f:.5f}")
    del gt
    hit = vp.loc[vp.label == 1, ["s1_rid", "m_rid"]].assign(found=True)
    tp = pairs.merge(hit, on=["s1_rid", "m_rid"], how="left")
    tp["found"] = tp.found.fillna(False).astype(bool)
    tp["s1_row"] = pd.Index(s1rid).get_indexer(tp.s1_rid.values)
    tp["pidx"] = pd.Index(pool_rid).get_indexer(tp.m_rid.values)
    assert (tp.s1_row >= 0).all() and (tp.pidx >= 0).all()
    if SMOKE:
        tp = tp[tp.s1_rid.isin(val_ids[:2000])].reset_index(drop=True)
    cand = pd.read_parquet(os.path.join(DATA, "cand_trainval.parquet"),
                           columns=["s1_rid", "m_rid", "pidx", "rank", "label", "fold"])
    cand = cand[cand.fold == 0].drop(columns="fold")
    tp = tp.merge(cand[["s1_rid", "m_rid", "rank"]].rename(columns={"rank": "cand_rank"}),
                  on=["s1_rid", "m_rid"], how="left")
    assert (tp.found == tp.cand_rank.notna()).all()
    log(f"val true pairs {len(tp):,}, missed {(~tp.found).sum():,} ({time.time() - t0:.0f}s)")

    # ---- forward: every true pair of the S1s that have >= 1 miss
    qrows = np.unique(tp.s1_row[~tp.found].values)
    k_row, k_key, k_kt = load_s1_keys("train", qrows, TYPES)
    q_id = np.searchsorted(qrows, k_row)
    fw = tp[tp.s1_row.isin(qrows)].copy()
    fw["q"] = np.searchsorted(qrows, fw.s1_row.values)
    fw["t"] = fw.pidx.values
    control = tp[tp.found].sample(500 if SMOKE else 20_000, random_state=0)
    gather = np.zeros(npool, bool)
    gather[fw.pidx.values] = True
    gather[control.pidx.values] = True
    pool_files = [(p, off) for s, off in ((2, 0), (3, n2)) for p in key_files("train", s, TYPES)]
    log(f"forward search: {len(qrows):,} S1 queries, {len(fw):,} true pairs")
    res, got_pool = rank_targets(q_id, k_key, k_kt, fw, pool_files, npool, gather)
    for c in res.columns:
        fw[f"f_{c}"] = res[c].values
    del res, k_row, k_key, k_kt, q_id
    gc.collect()
    chk = fw[fw.found]
    agree = (chk.f_rank == chk.cand_rank).mean()
    log(f"check: recomputed forward rank == pipeline rank for {agree:.4%} of {len(chk):,} found pairs")

    # ---- reverse: missed pool records + 20k found controls -> all 2.2M train S1
    rv = pd.concat([fw[~fw.found][["s1_row", "pidx"]].assign(control=False),
                    control[["s1_row", "pidx"]].assign(control=True)], ignore_index=True)
    qp = np.unique(rv.pidx.values)
    rv["q"] = np.searchsorted(qp, rv.pidx.values)
    rv["t"] = rv.s1_row.values
    gk = got_pool[np.isin(got_pool.row.values, qp)]
    gather_s1 = np.zeros(ns1, bool)
    gather_s1[rv.s1_row.values] = True
    s1_files = [(p, 0) for p in key_files("train", 1, TYPES)]
    log(f"reverse search: {len(qp):,} pool queries against {ns1:,} S1")
    res, _ = rank_targets(np.searchsorted(qp, gk.row.values), gk.key.values, gk.kt.values, rv, s1_files, ns1,
                          gather_s1)
    for c in res.columns:
        rv[f"r_{c}"] = res[c].values
    del res, got_pool, gk
    gc.collect()

    # ---- attributes: country, source, script
    s1a = load_rows("train", 1, tp.s1_row.values, ["country", "name_nonlatin", "name_core", "addr_n", "addr_nums"])
    miss = fw[~fw.found].copy()
    miss = miss.merge(rv[~rv.control][["s1_row", "pidx"] + [c for c in rv.columns if c.startswith("r_")]],
                      on=["s1_row", "pidx"], how="left")
    ctl = control.merge(rv[rv.control][["s1_row", "pidx"] + [c for c in rv.columns if c.startswith("r_")]],
                        on=["s1_row", "pidx"], how="left")
    miss["cause"] = np.where(miss.f_n_any == 0, "a_no_shared_key",
                             np.where(miss.f_n_allowed == 0, "b_only_capped_keys", "c_ranked_below_K"))
    # candidates of the S1s with misses (for the close-match test)
    cand_m = cand[cand.s1_rid.isin(miss.s1_rid.unique())].merge(
        d[["s1_rid", "m_rid", "pred"]], on=["s1_rid", "m_rid"], how="left")
    cand_m["pred"] = cand_m.pred.fillna(False).astype(bool)
    need = np.unique(np.r_[tp.pidx.values, cand_m.pidx.values])
    pa_ = load_pool_rows(need, ["name_nonlatin", "name_core", "addr_n", "addr_nums"], n2)
    for df_ in (tp, miss, ctl):
        df_["country"] = s1a.country.reindex(df_.s1_row.values).values
        df_["src"] = "S" + (df_.m_rid.values // 10**10).astype(str)
        df_["cross_script"] = pa_.name_nonlatin.reindex(df_.pidx.values).values != \
            s1a.name_nonlatin.reindex(df_.s1_row.values).values

    # ---- close match: missed record vs every candidate of the same S1
    cm = miss[["s1_rid", "pidx"]].reset_index().rename(columns={"index": "mi"}).merge(
        cand_m[["s1_rid", "pidx", "label", "pred"]].rename(columns={"pidx": "cpidx"}), on="s1_rid")
    W = dict(workers=-1, dtype=np.float32)
    s = lambda col, ix: pa_[col].reindex(ix).fillna("").to_numpy(dtype=object)  # noqa: E731
    nm = process.cpdist(s("name_core", cm.pidx.values), s("name_core", cm.cpidx.values), scorer=fuzz.token_set_ratio, **W)
    ad = process.cpdist(s("addr_n", cm.pidx.values), s("addr_n", cm.cpidx.values), scorer=fuzz.token_set_ratio, **W)
    cm["close"] = (nm >= CLOSE) & (ad >= CLOSE)
    cm["exact"] = (s("name_core", cm.pidx.values) == s("name_core", cm.cpidx.values)) & \
        (s("addr_n", cm.pidx.values) == s("addr_n", cm.cpidx.values)) & (s("name_core", cm.pidx.values) != "")
    cm["close_true"] = cm.close & (cm.label == 1)
    cm["close_true_pred"] = cm.close_true & cm.pred
    cm["close_false"] = cm.close & (cm.label == 0)
    ag = cm.groupby("mi")[["close", "exact", "close_true", "close_true_pred", "close_false"]].any()
    for c in ag.columns:
        miss[f"cm_{c}"] = ag[c].reindex(miss.index, fill_value=False).values
    # string-level overlap for cause (a): does the pair share anything at all?
    tokset = lambda a: [set(x.split()) for x in a]  # noqa: E731
    s1c, s1n = tokset(s1a.name_core.reindex(miss.s1_row).fillna("")), tokset(s1a.addr_nums.reindex(miss.s1_row).fillna(""))
    s1w = tokset(s1a.addr_n.reindex(miss.s1_row).fillna(""))
    pc_, pn, pw = (tokset(pa_[c].reindex(miss.pidx).fillna("")) for c in ("name_core", "addr_nums", "addr_n"))
    miss["str_name_tok"] = [len(a & b) > 0 for a, b in zip(s1c, pc_)]
    miss["str_addr_num"] = [len(a & b) > 0 for a, b in zip(s1n, pn)]
    miss["str_addr_word"] = [len({x for x in a & b if len(x) >= 3 and not x.isdigit()}) > 0 for a, b in zip(s1w, pw)]

    keep_cols = ["s1_rid", "m_rid", "s1_row", "pidx", "country", "src", "cross_script", "cause"] + \
        [c for c in miss.columns if c.startswith(("f_", "r_", "cm_", "str_"))]
    miss[keep_cols].to_parquet(sp("miss_pairs.parquet"), index=False)
    ctl[["s1_row", "pidx", "country", "src", "cross_script", "cand_rank"] +
        [c for c in ctl.columns if c.startswith("r_")]].to_parquet(sp("control_pairs.parquet"), index=False)
    tp[["s1_rid", "m_rid", "s1_row", "pidx", "found", "country", "src", "cross_script"]].to_parquet(
        sp("true_pairs.parquet"), index=False)
    # 20 random missed pairs with the raw strings (UTF-8 file only)
    smp = miss.sample(20, random_state=7)
    raw1 = load_rows("train", 1, smp.s1_row.values, ["business_name", "business_address"])
    raw2 = load_pool_rows(smp.pidx.values, ["business_name", "business_address"], n2)
    smp = smp.assign(s1_name=raw1.business_name.reindex(smp.s1_row).values,
                     s1_addr=raw1.business_address.reindex(smp.s1_row).values,
                     m_name=raw2.business_name.reindex(smp.pidx).values,
                     m_addr=raw2.business_address.reindex(smp.pidx).values)
    smp.to_parquet(sp("miss_sample.parquet"), index=False)
    json.dump(dict(check_rank_agree=float(agree), n_found_checked=int(len(chk)), f05=float(f)),
              open(sp("misses_meta.json"), "w"))
    log(f"misses stage done ({time.time() - t0:.0f}s)")


# ------------------------------------------------------------------ stage: decide
def shapley_fp_fn(tp, npred, fp, fn_c, nt):
    """Split F_oracle - F (oracle = perfect picking among candidates) between the false positives and the
    true candidates that were not predicted, with a 2-player Shapley split (symmetric, sums exactly)."""
    def F(tp_, np_):
        return np.divide(1.25 * tp_, np_ + 0.25 * nt, out=np.zeros_like(tp_, dtype=float), where=(np_ + 0.25 * nt) > 0)
    v0 = F(tp, npred)
    v_fp = F(tp, npred - fp)                   # false positives removed
    v_fn = F(tp + fn_c, npred + fn_c)           # missed true candidates added
    v_all = F(tp + fn_c, tp + fn_c)             # = oracle
    return 0.5 * ((v_fp - v0) + (v_all - v_fn)), 0.5 * ((v_fn - v0) + (v_all - v_fp))


def stage_decide():
    t0 = time.time()
    vp, d, val_ids, pairs, gt, dec, f = decided()
    log(f"decision reproduced: macro F0.5 = {f:.5f}")
    ids = pd.Index(val_ids)
    E = pd.DataFrame(index=ids)
    E["nt"] = pairs.groupby("s1_rid").size().reindex(ids, fill_value=0)
    E["h"] = vp.groupby("s1_rid").label.sum().reindex(ids, fill_value=0)
    pr = d[d.pred]
    E["npred"] = pr.groupby("s1_rid").size().reindex(ids, fill_value=0)
    E["tp"] = pr.groupby("s1_rid").label.sum().reindex(ids, fill_value=0)
    E["pmax_raw"] = vp.groupby("s1_rid").p.max().reindex(ids, fill_value=0)
    E["pmax_owner"] = d.groupby("s1_rid").p.max().reindex(ids, fill_value=0)
    # true candidates that were not predicted, by reason
    tc = vp[vp.label == 1].merge(d[["s1_rid", "m_rid", "pred"]], on=["s1_rid", "m_rid"], how="left")
    tc["why"] = np.where(tc.pred.isna(), "removed_by_one_owner",
                         np.where(tc.pred.astype(bool), "predicted", "below_threshold"))
    nt, h, npred, tpv = (E[c].values.astype(float) for c in ("nt", "h", "npred", "tp"))
    fp, fn_c = npred - tpv, h - tpv
    E["F"] = np.where(nt == 0, (npred == 0).astype(float),
                      np.divide(1.25 * tpv, npred + 0.25 * nt, out=np.zeros_like(tpv), where=(npred + 0.25 * nt) > 0))
    E["F_oracle"] = np.where(nt == 0, 1.0, 1.25 * h / (h + 0.25 * np.maximum(nt, 1e-9)))
    assert abs(E.F.mean() - f) < 1e-9
    E["covered"] = E.h == E.nt
    l_fp, l_fn = shapley_fp_fn(tpv, npred, fp, fn_c, nt)
    single, empty = nt == 0, (nt > 0) & (npred == 0)
    E["cat"] = np.select([single & (npred > 0), single, empty, (E.F.values < E.F_oracle.values - 1e-12)],
                         ["singleton_predicted", "singleton_ok", "nonsingleton_empty", "nonempty_imperfect"], "ok")
    E["loss_single"] = np.where(single & (npred > 0), 1.0, 0.0)
    E["loss_empty"] = np.where(empty, E.F_oracle.values, 0.0)
    E["loss_fp"] = np.where(E.cat == "nonempty_imperfect", l_fp, 0.0)
    E["loss_fn"] = np.where(E.cat == "nonempty_imperfect", l_fn, 0.0)
    E["loss_block"] = 1.0 - E.F_oracle
    E["fp"], E["fn_c"] = fp, fn_c
    E["gate_fail"] = empty & (E.pmax_owner.values < dec["g"])
    tot = E.loss_block + E.loss_single + E.loss_empty + E.loss_fp + E.loss_fn
    assert np.allclose(1 - tot, E.F.values)
    E.to_parquet(sp("entities.parquet"))
    tc.groupby("why").size().to_frame("n").to_csv(sp("true_cand_status.csv"))
    # false positives with their true owner
    fpr = pr[pr.label == 0][["s1_rid", "m_rid", "p", "pmax_after_owner"]].copy()
    owner = gt.set_index("m_rid").s1_rid
    fpr["owner"] = owner.reindex(fpr.m_rid.values).fillna(-1).astype(np.int64).values  # -1 = no owner
    fpr["owner_in_val"] = np.isin(fpr.owner.values, val_ids)
    vo = vp.rename(columns={"s1_rid": "owner", "p": "p_owner_val"})[["owner", "m_rid", "p_owner_val"]]
    fpr = fpr.merge(vo, on=["owner", "m_rid"], how="left")
    fpr["wrong_nt"] = E.nt.reindex(fpr.s1_rid.values).values
    fpr.to_parquet(sp("fp.parquet"), index=False)
    log(f"entities: {len(E):,}; false positives {len(fpr):,} (owner known {(fpr.owner >= 0).sum():,}, "
        f"owner in val {fpr.owner_in_val.sum():,}) ({time.time() - t0:.0f}s)")


# ------------------------------------------------------------------ stage: owners
def stage_owners():
    import lightgbm as lgb
    from features2 import build_features
    t0 = time.time()
    fpr = pd.read_parquet(sp("fp.parquet"))
    s1rid = pq.read_table(pq_path("train", 1), columns=["rid"]).column("rid").to_numpy()
    own = np.unique(fpr.owner[(fpr.owner >= 0) & ~fpr.owner_in_val].values)
    rows = np.sort(pd.Index(s1rid).get_indexer(own))
    assert (rows >= 0).all()
    log(f"owners to block: {len(rows):,}")
    cand = block_chunked("train", rows, BLOCK_CFG, chunk=100_000)
    cand["s1_rid"] = s1rid[cand.s1_row.values]
    out = sp("owner_feats.parquet")
    build_features("train", cand, out)
    del cand
    gc.collect()
    feats = open(os.path.join(MODELS, "feature_list.txt"), encoding="utf-8").read().split("\n")
    model = lgb.Booster(model_file=os.path.join(MODELS, "lgb_model.txt"))
    res = []
    pf = pq.ParquetFile(out)
    for i in range(pf.num_row_groups):
        v = pf.read_row_group(i, columns=feats + ["s1_rid", "m_rid"]).to_pandas()
        r = v[["s1_rid", "m_rid", "rank"]].copy()
        r["p"] = model.predict(v[feats].to_numpy(np.float32), num_threads=10).astype(np.float32)
        res.append(r)
    res = pd.concat(res, ignore_index=True)
    res.to_parquet(sp("owner_pred.parquet"), index=False)
    log(f"owners stage done: {len(res):,} owner candidates ({time.time() - t0:.0f}s)")


# ------------------------------------------------------------------ stage: report
def pct(x):
    return f"{100 * x:.1f}%"


def stage_report():
    import lightgbm as lgb
    L = ["# Diagnosis: where the validation F0.5 goes (fold 0, 100k S1, K=30, cap 1000)\n",
         "Diagnosis only: no pipeline code, data or output was changed. Script: `diag_losses.py`.\n"]
    meta = json.load(open(sp("misses_meta.json")))
    E = pd.read_parquet(sp("entities.parquet"))
    N = len(E)
    miss = pd.read_parquet(sp("miss_pairs.parquet"))
    ctl = pd.read_parquet(sp("control_pairs.parquet"))
    tp = pd.read_parquet(sp("true_pairs.parquet"))

    # ---------------- 1. blocking misses
    L.append("## 1. Blocking misses (true pairs that are not among the 30 candidates)\n")
    L.append(f"Validation true pairs {len(tp):,}; missed {len(miss):,} ({pct(len(miss) / len(tp))}); "
             f"S1 entities with >= 1 miss: {miss.s1_rid.nunique():,} of {N:,}.\n")
    L.append(f"Check: re-running the blocking score for the true pairs reproduces the pipeline's rank for "
             f"{pct(meta['check_rank_agree'])} of {meta['n_found_checked']:,} found pairs.\n")
    cz = miss.cause.value_counts().sort_index()
    L.append("Causes: (a) the pair shares no blocking key at all; (b) it shares keys, but every shared key is "
             f"carried by > {CAP} pool records, so the cap removes them; (c) it shares an allowed key (so it is "
             "scored) but >= 30 pool records score higher.\n")
    L.append(md_table(pd.DataFrame({"missed pairs": cz, "share": (cz / len(miss)).round(4)}).rename_axis("cause")))
    for col, name in (("country", "country"), ("src", "source"), ("cross_script", "cross-script (name)")):
        tab = pd.crosstab(miss[col], miss.cause)
        tab["total missed"] = tab.sum(axis=1)
        tab["true pairs"] = tp.groupby(col).size()
        tab["miss rate"] = (tab["total missed"] / tab["true pairs"]).round(4)
        L.append(f"\nBy {name}:\n")
        L.append(md_table(tab.rename_axis(name)))
    # cause details
    b = miss[miss.cause == "b_only_capped_keys"]
    c = miss[miss.cause == "c_ranked_below_K"]
    a = miss[miss.cause == "a_no_shared_key"]
    L.append("\n**(a) no shared key** - string-level overlap anyway (the key rules did not produce a common key): "
             f"share a name_core word {pct(a.str_name_tok.mean())}, an address number {pct(a.str_addr_num.mean())}, "
             f"an address word (>= 3 letters) {pct(a.str_addr_word.mean())}; none of the three "
             f"{pct((~(a.str_name_tok | a.str_addr_num | a.str_addr_word)).mean())}; cross-script "
             f"{pct(a.cross_script.mean())}.\n")
    if len(b):
        bins = pd.cut(b.f_min_df, [CAP, 2000, 5000, 20000, 10**9], labels=["1001-2000", "2001-5000", "5001-20000", ">20000"])
        L.append(f"**(b) only capped keys** - smallest df among the shared keys: "
                 + ", ".join(f"{k}: {v:,}" for k, v in bins.value_counts().sort_index().items())
                 + f"; median number of shared (capped) keys {b.f_n_any.median():.0f}.\n")
    if len(c):
        rk = pd.cut(c.f_rank + 1, [30, 50, 100, 1000, 10**9], labels=["31-50", "51-100", "101-1000", ">1000"])
        L.append("**(c) ranked below K** - forward rank of the true record (1 = best): "
                 + ", ".join(f"{k}: {v:,} ({pct(v / len(c))})" for k, v in rk.value_counts().sort_index().items())
                 + f"; recomputed rank <= 30 (tie/rounding disagreements): {(c.f_rank < K).sum():,}; "
                 f"median records retrieved for these S1s {c.f_n_ret.median():,.0f}.\n")
    # reverse rank
    def rbins(df_):
        r = df_.r_rank
        return pd.Series({"rank 1": (r == 0).mean(), "2-5": r.between(1, 4).mean(), "6-30": r.between(5, 29).mean(),
                          "31-100": r.between(30, 99).mean(), ">100": (r >= 100).mean(),
                          "not retrieved": (r < 0).mean(), "n": len(df_)})
    rv = pd.DataFrame({k: rbins(g) for k, g in miss.groupby("cause")})
    rv["all missed"] = rbins(miss)
    rv["control: 20k found pairs"] = rbins(ctl)
    rv = rv.T
    rv["n"] = rv["n"].astype(int)
    L.append("\n**Reverse search** (query = the missed S2/S3 record, index = all 2,206,821 train S1, same key "
             "types, weights log(N/df) x type weight with df counted over S1, same cap 1000). Where does the TRUE "
             "S1 rank? (shares of pairs)\n")
    L.append(md_table(rv.rename_axis("pairs")))
    L.append(f"\nMissed pairs whose true S1 is in the reverse top 30: {(miss.r_rank.between(0, 29)).sum():,} "
             f"({pct(miss.r_rank.between(0, 29).mean())}); top 5: {pct(miss.r_rank.between(0, 4).mean())}.\n")
    # close match
    L.append(f"\n**Close match to another candidate of the same S1** (token_set_ratio >= {CLOSE} on name_core AND "
             "on the normalised address; only same-script comparisons can pass):\n")
    cmt = pd.DataFrame({
        "close to any candidate": miss.groupby("cause").cm_close.mean(),
        "exactly equal (name_core + address)": miss.groupby("cause").cm_exact.mean(),
        "close to a TRUE-match candidate": miss.groupby("cause").cm_close_true.mean(),
        "... that we also predicted": miss.groupby("cause").cm_close_true_pred.mean(),
        "close to a non-match candidate": miss.groupby("cause").cm_close_false.mean()})
    cmt.loc["all missed"] = [miss.cm_close.mean(), miss.cm_exact.mean(), miss.cm_close_true.mean(),
                             miss.cm_close_true_pred.mean(), miss.cm_close_false.mean()]
    L.append(md_table(cmt.rename_axis("cause")))
    smp = pd.read_parquet(sp("miss_sample.parquet"))
    L.append("\n**20 random missed pairs** (raw strings; fwd = forward rank of the true record, rev = reverse "
             "rank of the true S1, '-' = not retrieved; close = close to a candidate of the same S1)\n")
    L.append("| # | country | src | cause | fwd | rev | close | S1 name / address | missed S2/S3 name / address |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    esc = lambda x: str(x).replace("|", "/")  # noqa: E731
    for i, r in enumerate(smp.itertuples(), 1):
        L.append(f"| {i} | {r.country} | {r.src} | {r.cause[0]} | {r.f_rank + 1 if r.f_rank >= 0 else '-'} | "
                 f"{r.r_rank + 1 if r.r_rank >= 0 else '-'} | {'y' if r.cm_close else 'n'} | "
                 f"{esc(r.s1_name)} / {esc(r.s1_addr)} | {esc(r.m_name)} / {esc(r.m_addr)} |")

    # ---------------- 2. decision losses
    L.append("\n## 2. Decision losses\n")
    st = pd.read_csv(sp("true_cand_status.csv"), index_col=0).n
    L.append(f"Decision reproduced: macro F0.5 = {E.F.mean():.5f} (g={0.67}, r={0.725}, one-owner). True pairs "
             f"that were candidates: {st.sum():,} -> predicted {st.get('predicted', 0):,}, below threshold "
             f"{st.get('below_threshold', 0):,}, removed by the one-owner rule {st.get('removed_by_one_owner', 0):,}.\n")
    cov = E[E.covered]
    lost = cov[cov.F < 1]
    L.append(f"Entities whose true matches were ALL candidates: {len(cov):,} (incl. {int((cov.nt == 0).sum()):,} "
             f"singletons). Of these, {len(lost):,} score < 1; they lose {(1 - lost.F).sum():,.1f} entity-points = "
             f"{(1 - lost.F).sum() / N:.4f} macro F0.5.\n")
    rows = [("singletons we predicted something for", cov.loss_single, cov.cat == "singleton_predicted"),
            ("non-singletons predicted empty", cov.loss_empty, cov.cat == "nonsingleton_empty"),
            ("false positives in non-empty lists", cov.loss_fp, (cov.cat == "nonempty_imperfect") & (cov.fp > 0)),
            ("misses (true candidates not picked) in non-empty lists", cov.loss_fn,
             (cov.cat == "nonempty_imperfect") & (cov.fn_c > 0))]
    t2 = pd.DataFrame([(n, int(m.sum()), s.sum(), s.sum() / N) for n, s, m in rows],
                      columns=["part", "entities", "entity-points lost", "macro F0.5 lost"]).set_index("part")
    t2.loc["total"] = [len(lost), t2["entity-points lost"].sum(), t2["macro F0.5 lost"].sum()]
    t2["entities"] = t2.entities.astype(int)
    L.append(md_table(t2))
    ne = cov[cov.cat == "nonsingleton_empty"]
    L.append(f"\nNon-singletons predicted empty: {len(ne):,}; best p (after one-owner) below the gate 0.67: "
             f"{int(ne.gate_fail.sum()):,}; all candidates taken by the one-owner rule: {int((ne.pmax_owner == 0).sum()):,}. "
             f"Singletons predicted non-empty: {int((cov.cat == 'singleton_predicted').sum()):,} "
             f"(mean {cov[cov.cat == 'singleton_predicted'].npred.mean():.2f} predicted ids). "
             f"FP/FN split inside non-empty lists = 2-player Shapley split of (1 - F).\n")
    # false positives
    fpr = pd.read_parquet(sp("fp.parquet"))
    op = pd.read_parquet(sp("owner_pred.parquet")).rename(columns={"s1_rid": "owner", "p": "p_owner_new",
                                                                  "rank": "owner_rank"})
    fpr = fpr.merge(op, on=["owner", "m_rid"], how="left")
    folds = pd.read_parquet(os.path.join(DATA, "folds.parquet"))[["s1_rid", "fold"]].rename(columns={"s1_rid": "owner"})
    fpr = fpr.merge(folds, on="owner", how="left")
    fpr["p_owner"] = fpr.p_owner_val.fillna(fpr.p_owner_new)
    fpr["owner_has"] = fpr.p_owner.notna()
    grp = np.select([fpr.owner < 0, fpr.owner_in_val, fpr.fold == 1],
                    ["record belongs to no S1", "owner is a validation S1 (competed in one-owner)",
                     "owner is a fold-1 TRAINING S1 (p in-sample)"], "owner is another train S1 (fresh run)")
    fpr["grp"] = grp

    def fsum(g):
        has = g[g.owner_has]
        return pd.Series({"FPs": len(g), "owner has it as candidate": has.shape[0] / max(len(g), 1),
                          "median p_owner": has.p_owner.median() if len(has) else np.nan,
                          "p_owner >= 0.5": (has.p_owner >= 0.5).sum() / max(len(g), 1),
                          "p_owner > p_wrong": (has.p_owner > has.p).sum() / max(len(g), 1),
                          "median p_wrong": g.p.median()})
    ft = fpr.groupby("grp").apply(fsum)
    ft.loc["all FPs"] = fsum(fpr)
    ft["FPs"] = ft["FPs"].astype(int)
    L.append(f"\n**False positives** ({len(fpr):,} predicted pairs that are wrong). 'owner' = the S1 the record truly "
             "belongs to; shares are of all FPs in the row; p_wrong = our p for the wrong S1.\n")
    L.append(md_table(ft.rename_axis("true owner of the wrong record")))
    wrong_single = (fpr.wrong_nt == 0).mean()
    L.append(f"\nFPs sitting in a singleton's list: {pct(wrong_single)}. Out-of-sample FPs (owner not in fold 1) "
             "that the owner would claim with a higher p (the one-owner rule removes them once all S1s compete, "
             "as on the test set): "
             f"{int(((fpr.fold != 1) & fpr.owner_has & (fpr.p_owner > fpr.p)).sum()):,}.\n")
    # rough effect of removing those
    rm = fpr[(fpr.fold != 1) & fpr.owner_has & (fpr.p_owner > fpr.p) & ~fpr.owner_in_val]
    E2 = E.copy()
    dn = rm.groupby("s1_rid").size().reindex(E2.index, fill_value=0).values
    npred2 = E2.npred.values - dn
    nt, tpv = E2.nt.values.astype(float), E2.tp.values.astype(float)
    F2 = np.where(nt == 0, (npred2 == 0).astype(float),
                  np.divide(1.25 * tpv, npred2 + 0.25 * nt, out=np.zeros_like(tpv), where=(npred2 + 0.25 * nt) > 0))
    L.append(f"Rough what-if (only removing those FPs, ignoring that full competition can also take TRUE matches "
             f"away): macro F0.5 {E.F.mean():.4f} -> {F2.mean():.4f}.\n")

    # ---------------- 3. features
    model = lgb.Booster(model_file=os.path.join(MODELS, "lgb_model.txt"))
    fn = model.feature_name()
    L.append(f"\n## 3. Model features\n\nThe saved model uses **{len(fn)} features** (models/lgb_model.txt, "
             f"max_feature_idx={len(fn) - 1}; the '72 features' in PROGRESS.md was a miscount). Country is a model "
             f"feature: **{'yes' if any('country' in x for x in fn) else 'no'}**.\n")
    L.append(", ".join(f"{i}. `{x}`" for i, x in enumerate(fn, 1)) + "\n")
    L.append("**How the unseen country France is handled:** country appears in exactly one place, the blocking keys "
             "(keys.py / keys2.py): the country string is hashed and XOR-ed into every key, so a France S1 can only "
             "share keys with France pool records. Key df / IDF weights and the cap are computed per key over the test "
             "pool, which contains the France records, so France keys get their own frequencies. No feature, "
             "threshold or model split looks at the country value, so there is no one-hot and no 'unknown category' "
             "path. What is NOT covered: the model and the thresholds were fitted on US + India only, so their "
             "calibration on French names/addresses is unmeasured (no French labels exist). The normalisation has "
             "partial French support: accents stripped, legal words sarl/sas/sci/eurl/snc removed, avenue/boulevard/"
             "saint abbreviated; 'rue', 'chemin', 'allee', 'cedex' etc. are left as plain words.\n")

    # ---------------- final table
    oracle = E.F_oracle.mean()
    # blocking loss by cause: each entity's oracle loss split in proportion to its missed pairs per cause
    mc = pd.crosstab(miss.s1_rid, miss.cause)
    share = mc.div(mc.sum(axis=1), axis=0)
    lb = E.loss_block.reindex(share.index).values[:, None] * share.values
    by_cause = pd.Series(lb.sum(axis=0) / N, index=share.columns)
    unc = E[~E.covered]
    T = [("Perfect", "", 1.0),
         ("blocking (a) no shared key", by_cause.get("a_no_shared_key", 0), None),
         ("blocking (b) only keys with df > 1000 (cap)", by_cause.get("b_only_capped_keys", 0), None),
         ("blocking (c) scored but rank > 30", by_cause.get("c_ranked_below_K", 0), None),
         ("= oracle at K=30 (perfect picking among the candidates)", "", oracle),
         ("decision, fully covered: singletons predicted non-empty", cov.loss_single.sum() / N, None),
         ("decision, fully covered: non-singletons predicted empty", cov.loss_empty.sum() / N, None),
         ("decision, fully covered: false positives in lists", cov.loss_fp.sum() / N, None),
         ("decision, fully covered: true candidates not picked", cov.loss_fn.sum() / N, None),
         ("decision, entities with blocking misses: predicted empty", unc.loss_empty.sum() / N, None),
         ("decision, entities with blocking misses: false positives", unc.loss_fp.sum() / N, None),
         ("decision, entities with blocking misses: true candidates not picked", unc.loss_fn.sum() / N, None),
         ("= validation macro F0.5", "", E.F.mean())]
    L.append("\n## Where the loss goes (1.000 -> oracle -> actual), in macro F0.5 points\n")
    L.append("| step | loss | F0.5 after |\n|---|---|---|")
    run = 1.0
    for name, loss, val in T:
        if val is not None:
            L.append(f"| **{name}** | | **{val:.4f}** |")
            run = val
        else:
            run -= loss
            L.append(f"| {name} | -{loss:.4f} | {run:.4f} |")
    with open(os.path.join(REP, "diag_losses.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    log("report written: reports/diag_losses.md")


SMOKE = False
if __name__ == "__main__":
    if sys.argv[2:] == ["smoke"]:  # small test run into a separate folder
        SMOKE, SCR = True, os.path.join(SCR, "smoke")
        os.makedirs(SCR, exist_ok=True)
    {"misses": stage_misses, "decide": stage_decide, "owners": stage_owners, "report": stage_report}[sys.argv[1]]()
