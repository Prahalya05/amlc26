"""Pairwise features v5 = features2 + honest empty fields.

Every feature that compares a field is NaN (not 0) when the field is empty on either side
(LightGBM then learns a separate branch for "not comparable" instead of reading it as "very different"):
  name_n    : name_ratio, name_tset, name_tsort, name_partial
  name_core : core_*, roman_* (romanised core vs core)
  skeleton  : skel_ratio, skel_tset, skel_jacc
  address   : addr_ratio, addr_tset, addr_partial, addr_word_*, addr_idf_cos and the address-number comparisons
              num_shared, num_jacc, num_miss1, num_miss2, num_first_eq, num_first_in2
New per-pair columns: pool_addr_empty, s1_addr_empty, len_name1/2 (chars of name_n), ntok_name1/2 (tokens).
Context features (add_context) are computed after the masking, so combo / *_gap / *_rk are NaN where their
base feature is NaN (pandas max / rank skip NaN).
"""
import gc
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import features2 as F2
from common import log

NAME = ["name_ratio", "name_tset", "name_tsort", "name_partial"]
CORE = ["core_ratio", "core_tset", "core_jw", "core_eq", "roman_ratio", "roman_tset", "core_shared", "core_jacc",
        "core_idf_cos"]
SKEL = ["skel_ratio", "skel_tset", "skel_jacc"]
ADDR = ["addr_ratio", "addr_tset", "addr_partial", "addr_word_shared", "addr_word_jacc", "addr_idf_cos",
        "num_shared", "num_jacc", "num_miss1", "num_miss2", "num_first_eq", "num_first_in2"]
NEW = ["pool_addr_empty", "s1_addr_empty", "len_name1", "len_name2", "ntok_name1", "ntok_name2"]


def _name_stats(cache):
    """chars and tokens of name_n per cache record (computed once per cache)."""
    if not hasattr(cache, "_nlen"):
        cache._nlen = pc.utf8_length(cache.name).to_numpy(zero_copy_only=False).astype(np.float32)
        ntok = pc.list_value_length(pc.split_pattern(cache.name, " ")).to_numpy(zero_copy_only=False).astype(np.float32)
        cache._ntok = np.where(cache._nlen > 0, ntok, 0).astype(np.float32)
        cache._skel_empty = np.diff(cache.skel_set[0]) == 0
    return cache._nlen, cache._ntok, cache._skel_empty


def pair_features(A, B, ia, ib):
    f = F2.pair_features(A, B, ia, ib)
    nl1, nt1, se1 = _name_stats(A)
    nl2, nt2, se2 = _name_stats(B)
    empty = {"name": (nl1[ia] == 0) | (nl2[ib] == 0),
             "core": (A.len_core[ia] == 0) | (B.len_core[ib] == 0),
             "skel": se1[ia] | se2[ib],
             "addr": (A.len_addr[ia] == 0) | (B.len_addr[ib] == 0)}
    for cols, key in ((NAME, "name"), (CORE, "core"), (SKEL, "skel"), (ADDR, "addr")):
        m = empty[key]
        if m.any():
            for c in cols:
                v = f[c].values.astype(np.float32, copy=True)
                v[m] = np.nan
                f[c] = v
    f["pool_addr_empty"] = (B.len_addr[ib] == 0).astype(np.float32)
    f["s1_addr_empty"] = (A.len_addr[ia] == 0).astype(np.float32)
    f["len_name1"], f["len_name2"] = nl1[ia], nl2[ib]
    f["ntok_name1"], f["ntok_name2"] = nt1[ia], nt2[ib]
    return f


def build_features(split, cand, out_path, chunk=500_000, extra_cols=None):
    """features2.build_features with the v5 pair features. cand: s1_row, pidx, score, nkeys, rank, m_rid (+ any
    extra columns, kept). Chunks never split an S1's list (the context features are per S1)."""
    t0 = time.time()
    A = F2.RecordCache(split, [1], cand.s1_row.values)
    B = F2.RecordCache(split, [2, 3], cand.pidx.values)
    cand = cand.sort_values(["s1_row", "rank"], kind="stable").reset_index(drop=True)
    starts = np.r_[0, np.flatnonzero(np.diff(cand.s1_row.values)) + 1]
    cuts = [0]
    for s in starts:
        if s - cuts[-1] >= chunk:
            cuts.append(s)
    cuts.append(len(cand))
    writer = None
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        c = cand.iloc[lo:hi].reset_index(drop=True)
        feats = pair_features(A, B, A.idx(c.s1_row.values), B.idx(c.pidx.values))
        df = pd.concat([c, feats], axis=1)
        df["msrc3"] = (df.m_rid.values // 10**10 == 3).astype(np.float32)
        for col in ("score", "nkeys", "rank"):
            df[col] = df[col].astype(np.float32)
        df = F2.add_context(df)
        tab = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, tab.schema, compression="zstd")
        writer.write_table(tab)
        del c, feats, df, tab
        gc.collect()
    if writer is not None:
        writer.close()
    dt = time.time() - t0
    log(f"  v5 features for {len(cand):,} pairs in {dt:.0f}s -> {out_path}")
    del A, B
    gc.collect()
