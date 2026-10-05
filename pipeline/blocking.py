"""Candidate generation (blocking) with an inverted index on hashed keys.

For a set of S1 records and the FULL S2+S3 pool of the same split:
  1. Collect the distinct keys of the chosen S1 records (U).
  2. Stream the pool key files once, count how many pool records carry each key of U ("df").
     Keys carried by more than `cap` pool records are too common to be useful (e.g. "road",
     "mumbai") and are ignored; each remaining key gets weight idf = log(N_pool / df) * type weight.
  3. Stream the pool again and keep the postings (key -> pool records) of the useful keys.
  4. For chunks of S1 records, look up all their keys, add up the weights of shared keys per
     (S1, pool record) pair, and keep the `k` best-scoring pool records per S1.
Everything is numpy on integer arrays, so memory stays bounded (see log lines for sizes).
"""
import gc
import os
import time

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

from common import DATA, log, pq_path

DEFAULT_CFG = dict(types=(1, 2, 3, 4, 5, 6, 7), cap=2000, k=50,
                   type_w={1: 1.0, 2: 1.0, 3: 0.5, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0,
                           8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0},
                   max_pairs_chunk=5_000_000)


def pool_info(split):
    """rid of every pool record in pool order (S2 rows first, then S3 rows)."""
    rids = [pq.read_table(pq_path(split, s), columns=["rid"]).column("rid").to_numpy() for s in (2, 3)]
    return np.concatenate(rids), len(rids[0])


FILE_TYPES = {False: np.arange(1, 8), True: np.arange(8, 13)}  # key types in keys_* / keys2_* files


def key_files(split, src, types):
    """v1 keys (types 1-7, keys.py) and v2 keys (types 8-12, keys2.py); only files that are needed."""
    f = []
    if any(t <= 7 for t in types):
        f.append(os.path.join(DATA, f"keys_{split}_s{src}.parquet"))
    if any(t >= 8 for t in types):
        f.append(os.path.join(DATA, f"keys2_{split}_s{src}.parquet"))
    return f


def _iter_pool_keys(split, types, n2):
    jobs = [(path, off) for s, off in ((2, 0), (3, n2)) for path in key_files(split, s, types)]
    for path, off in jobs:
        pf = pq.ParquetFile(path)
        for i in range(pf.num_row_groups):
            t = pf.read_row_group(i)
            kt = t.column("kt").to_numpy()
            row, key = t.column("row").to_numpy(), t.column("key").to_numpy()
            if not np.isin(FILE_TYPES[path.split("keys")[-1][:1] == "2"], types).all():
                m = np.isin(kt, types)
                row, key, kt = row[m], key[m], kt[m]
            yield row.astype(np.int32) + off, key, kt
            del t


def load_s1_keys(split, s1_rows, types):
    """Keys of the selected S1 rows (row numbers in the S1 parquet), read row group by row group
    so only the selected rows' keys are ever held in memory."""
    sel = np.zeros(int(np.max(s1_rows)) + 1, bool)
    sel[s1_rows] = True
    rows, keys, kts = [], [], []
    for path in key_files(split, 1, types):
        pf = pq.ParquetFile(path)
        for i in range(pf.num_row_groups):
            t = pf.read_row_group(i)
            row, kt = t.column("row").to_numpy(), t.column("kt").to_numpy()
            m = (row < len(sel)) & np.isin(kt, types)
            m[m] = sel[row[m]]
            rows.append(row[m])
            keys.append(t.column("key").to_numpy()[m])
            kts.append(kt[m])
    return np.concatenate(rows), np.concatenate(keys), np.concatenate(kts)


def block_chunked(split, s1_rows, cfg=None, chunk=100_000):
    """block() over S1 chunks (each chunk re-scans the pool; results are identical to one call,
    because key weights depend only on the pool). Bounds RAM for large S1 sets."""
    s1_rows = np.sort(np.asarray(s1_rows))
    parts = []
    for lo in range(0, len(s1_rows), chunk):
        log(f"  blocking S1 chunk {lo:,}-{min(lo + chunk, len(s1_rows)):,} of {len(s1_rows):,}")
        parts.append(block(split, s1_rows[lo:lo + chunk], cfg))
        gc.collect()
    return pd.concat(parts, ignore_index=True)


class KeyLookup:
    """Position of each queried key in the key array U, or -1 (pandas hash table; 5-6x faster
    than np.searchsorted here. A bit-filter in front of it was tried and made it slower.)"""

    def __init__(self, U):
        self.index = pd.Index(U)

    def __call__(self, keys):
        return self.index.get_indexer(keys)


QMAX = (1 << 18) - 1


def block(split, s1_rows, cfg=None, return_stats=False):
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    types = np.array(cfg["types"], np.int8)
    t0 = time.time()
    s1_row, s1_key, s1_kt = load_s1_keys(split, s1_rows, types)
    U = np.unique(s1_key)
    pool_rid, n2 = pool_info(split)
    npool = len(pool_rid)
    assert npool < 1 << 24 and max(s1_rows) < 1 << 22  # bit budget of the packed sort keys
    # pass A: document frequency of every key of U in the pool
    df = np.zeros(len(U), np.int64)
    ktU = np.zeros(len(U), np.int8)
    look = KeyLookup(U)
    for prow, pkey, pkt in _iter_pool_keys(split, types, n2):
        pos = look(pkey)
        hit = pos >= 0
        df += np.bincount(pos[hit], minlength=len(U))
        ktU[pos[hit]] = pkt[hit]
    allowed = (df >= 1) & (df <= cfg["cap"])
    tw = np.zeros(16)
    for k, v in cfg["type_w"].items():
        tw[k] = v
    w = np.where(allowed, np.log(npool / np.maximum(df, 1)) * tw[ktU], 0.0).astype(np.float32)
    n_post = int(df[allowed].sum())
    log(f"  S1 keys {len(s1_key):,} (distinct {len(U):,}); allowed {allowed.sum():,}; "
        f"postings {n_post:,} ({time.time() - t0:.0f}s)")
    # pass B: postings of allowed keys, grouped by key position in U
    # Group postings by key: pack (key position, pool idx) into one uint64 and sort it DIRECTLY.
    # (Indirect sorts - argsort/lexsort/np.unique(return_inverse) - are ~50x slower on this laptop
    # for tens of millions of values because every comparison is a cache miss.)
    packed = []
    ua = np.flatnonzero(allowed)  # look up allowed keys only
    look = KeyLookup(U[ua])
    for prow, pkey, _ in _iter_pool_keys(split, types, n2):
        pos = look(pkey)
        hit = pos >= 0
        packed.append((ua[pos[hit]].astype(np.uint64) << np.uint64(32)) | prow[hit].astype(np.uint64))
    log(f"  pass B scanned ({time.time() - t0:.0f}s)")
    packed = np.concatenate(packed)
    packed.sort()
    post_pidx = (packed & np.uint64(0xFFFFFFFF)).astype(np.int32)
    del packed
    gc.collect()
    offs = np.zeros(len(U) + 1, np.int64)
    offs[1:] = np.cumsum(np.where(allowed, df, 0))
    log(f"  index built ({time.time() - t0:.0f}s)")

    # S1 side: keep allowed keys only
    spos = np.searchsorted(U, s1_key)
    keep = allowed[spos]
    s1_row, spos = s1_row[keep], spos[keep]
    packed = np.sort((s1_row.astype(np.uint64) << np.uint64(32)) | spos.astype(np.uint64))
    s1_row = (packed >> np.uint64(32)).astype(np.int64)
    spos = (packed & np.uint64(0xFFFFFFFF)).astype(np.int64)
    del packed
    lens = (offs[spos + 1] - offs[spos])
    # chunk boundaries over S1 rows so that each chunk expands to <= max_pairs_chunk pairs
    first = np.flatnonzero(np.r_[True, np.diff(s1_row) != 0]) if len(s1_row) else np.array([], np.int64)
    urow = s1_row[first]
    per_row = np.add.reduceat(lens, first) if len(first) else np.array([], np.int64)
    bounds, acc, start = [], 0, 0
    for i, v in enumerate(per_row):
        if acc + v > cfg["max_pairs_chunk"] and i > start:
            bounds.append((start, i))
            start, acc = i, 0
        acc += v
    bounds.append((start, len(urow)))
    first = np.append(first, len(s1_row))
    K = cfg["k"]
    res = []
    n_expanded = 0
    for a, b in bounds:
        lo, hi = first[a], first[b]
        r, p, L = s1_row[lo:hi], spos[lo:hi], lens[lo:hi]
        tot = int(L.sum())
        if tot == 0:
            continue
        n_expanded += tot
        rr = np.repeat(r, L)
        starts = np.repeat(offs[p] - np.cumsum(L) + L, L)
        cand = post_pidx[starts + np.arange(tot)]
        ww = np.repeat(w[p], L)
        code = rr.astype(np.int64) * npool + cand
        del rr, starts, cand
        # sum of key weights and number of shared keys per (S1, pool record): hash groupby, no sort
        g = pd.DataFrame({"code": code, "w": ww}).groupby("code", sort=False).w.agg(["sum", "size"])
        del code, ww
        uc = g.index.to_numpy()
        score = g["sum"].to_numpy().astype(np.float32)
        nk = g["size"].to_numpy().astype(np.int16)
        del g
        cr = (uc // npool).astype(np.uint64)
        cp = (uc % npool).astype(np.uint64)
        # order by (S1 row, score desc) with one direct sort of a packed uint64:
        # 22 bits S1 row | 18 bits inverted score (quantised to 1/256) | 24 bits pool idx
        q = np.minimum(np.round(score.astype(np.float64) * 256), QMAX).astype(np.uint64)
        packed = np.sort((cr << np.uint64(42)) | ((np.uint64(QMAX) - q) << np.uint64(24)) | cp)
        cr = (packed >> np.uint64(42)).astype(np.int64)
        cp = (packed & np.uint64((1 << 24) - 1)).astype(np.int64)
        del packed, q
        gstart = np.flatnonzero(np.r_[True, np.diff(cr) != 0])
        rank = np.arange(len(cr)) - np.repeat(gstart, np.diff(np.r_[gstart, len(cr)]))
        m = rank < K
        cr, cp, rank = cr[m], cp[m], rank[m]
        back = pd.Index(uc).get_indexer(cr * npool + cp)  # exact score / nkeys of the kept pairs
        res.append(pd.DataFrame({"s1_row": cr.astype(np.int32), "pidx": cp.astype(np.int32),
                                 "score": score[back], "nkeys": nk[back], "rank": rank.astype(np.int16)}))
        del uc, cr, cp, score, nk, rank, back
        gc.collect()
    out = pd.concat(res, ignore_index=True) if res else pd.DataFrame(
        columns=["s1_row", "pidx", "score", "nkeys", "rank"])
    out["m_rid"] = pool_rid[out.pidx.values]
    stats = dict(n_s1=len(s1_rows), s1_keys=len(s1_key), postings=n_post, expanded=n_expanded,
                 seconds=round(time.time() - t0, 1))
    log(f"  blocking done: {len(out):,} candidates, expanded {n_expanded:,} pairs, "
        f"{time.time() - t0:.0f}s")
    del post_pidx
    gc.collect()
    return (out, stats) if return_stats else out
