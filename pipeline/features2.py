"""Pairwise features, v2 (fast). Same role as features.py, rewritten after profiling.

Principle: everything that depends on ONE record (normalised strings, consonant skeleton,
Indic romanisation, token sets, number sets, IDF weights and norms) is computed ONCE per record
by records.py and stored in data/rec_*.parquet; `RecordCache` only loads the rows needed. Per pair we only run rapidfuzz string scores (C++, multi-threaded) and
vectorised set intersections on integer hashes (no Python loop per pair).

Feature groups (country is never a feature, so France is handled like US/India):
  blocking : score, nkeys, rank (+ gap/rank inside the S1's candidate list)
  name     : rapidfuzz ratio / token_set / token_sort / partial on the name, ratio / token_set /
             Jaro-Winkler on the core name, skeleton ratio / token_set / Jaccard (cross-script),
             romanised ratio / token_set (cross-script pairs only, see romanize.py),
             core-token shared count / Jaccard / IDF-weighted cosine, exact core equality
  address  : rapidfuzz ratio / token_set / partial; word shared / Jaccard / IDF-weighted cosine;
             address numbers: counts, shared, Jaccard, missing on each side, first number equal / contained
  other    : source S3 flag, script flags, lengths
  context  : for key features f: f - max(f in this S1's list) and rank of f in the list
"""
import gc
import os
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from common import DATA, log, pq_path
from keys import hash_strings, tokens

COLS = ["name_n", "name_core", "addr_n", "addr_nums", "name_nonlatin"]
MIX = np.uint64(0x9E3779B97F4A7C15)
USE_ROMAN = True  # kept after the rule-6 check (romanize_check.py, reports/romanize_check.md)


# ------------------------------------------------------------------ token document frequencies
def token_df(split):
    """Document frequency of name-core words and address words over the FULL S2+S3 pool of a split
    (for IDF weights). Cached in data/tokdf_{split}.parquet."""
    path = os.path.join(DATA, f"tokdf_{split}.parquet")
    if os.path.exists(path):
        t = pd.read_parquet(path)
        return {k: g.set_index("h")["df"] for k, g in t.groupby("kind")}, int(t.attrs_n.iloc[0])
    t0 = time.time()
    acc = {"core": [], "addr": []}
    n = 0
    for s in (2, 3):
        for b in pq.ParquetFile(pq_path(split, s)).iter_batches(1_000_000, columns=["name_core", "addr_n"]):
            n += b.num_rows
            for kind, col, rx in (("core", "name_core", None), ("addr", "addr_n", r"^[^0-9]+$")):
                par, h = _tok_hashes(b.column(col), rx)
                u = pd.DataFrame({"p": par, "h": h}).drop_duplicates().h.value_counts()
                acc[kind].append(u)
    out = []
    for kind, parts in acc.items():
        v = pd.concat(parts).groupby(level=0).sum()
        out.append(pd.DataFrame({"kind": kind, "h": v.index.values.astype(np.uint64), "df": v.values}))
    t = pd.concat(out, ignore_index=True)
    t["attrs_n"] = n
    t.to_parquet(path, index=False)
    log(f"  token df for {split}: {len(t):,} tokens over {n:,} pool records ({time.time() - t0:.0f}s)")
    return {k: g.set_index("h")["df"] for k, g in t.groupby("kind")}, n


def _tok_hashes(strs, keep_regex=None, minlen=1):
    if isinstance(strs, pa.ChunkedArray):
        strs = strs.combine_chunks()
    flat, par = tokens(strs)
    keep = pc.greater_equal(pc.utf8_length(flat), minlen)
    if keep_regex:
        keep = pc.and_(keep, pc.match_substring_regex(flat, keep_regex))
    keep = keep.to_numpy(zero_copy_only=False)
    flat = flat.filter(pa.array(keep))
    h = hash_strings(flat) if len(flat) else np.array([], np.uint64)
    return par[keep], h


class RecordCache:
    """Per-record data for the records `rows` (positions in the concatenation of the `srcs` files).
    Nothing is computed here: strings come from data/{split}_s{src}.parquet and everything derived
    (skeleton, romanisation, hash sets, IDF weights) from data/rec_{split}_s{src}.parquet, which
    records.py computed once per record."""

    def __init__(self, split, srcs, rows):
        from records import rec_path
        t0 = time.time()
        rows = np.unique(rows)
        sizes = [pq.ParquetFile(pq_path(split, s)).metadata.num_rows for s in srcs]
        self.local = np.full(sum(sizes), -1, np.int32)  # global row -> cache index
        self.local[rows] = np.arange(len(rows), dtype=np.int32)
        parts, off = [], 0
        for s, size in zip(srcs, sizes):
            mask = np.zeros(size, bool)
            mask[rows[(rows >= off) & (rows < off + size)] - off] = True
            base = pq.ParquetFile(pq_path(split, s)).iter_batches(500_000, columns=COLS)
            rec = pq.ParquetFile(rec_path(split, s)).iter_batches(500_000)
            pos = 0
            for b1, b2 in zip(base, rec):
                assert b1.num_rows == b2.num_rows
                m = mask[pos: pos + b1.num_rows]
                pos += b1.num_rows
                if m.any():
                    fm = pa.array(m)
                    tb = pa.Table.from_batches([b1.filter(fm)])
                    rt = pa.Table.from_batches([b2.filter(fm)])
                    for c in rt.column_names:
                        tb = tb.append_column(c, rt.column(c))
                    parts.append(tb)
            assert pos == size
            off += size
        t = pa.concat_tables(parts).combine_chunks()
        del parts
        n = self.n = t.num_rows
        assert n == len(rows)
        col = lambda c: t.column(c).chunk(0) if t.column(c).num_chunks else t.column(c).combine_chunks()  # noqa: E731
        self.name, self.core, self.addr = col("name_n"), col("name_core"), col("addr_n")
        self.skel, self.roman = col("skel"), col("roman")
        self.nonlatin = t.column("name_nonlatin").to_numpy().astype(bool)

        def csr(c):
            a = col(c)
            return a.offsets.to_numpy().astype(np.int64), a.values.to_numpy().astype(np.uint64)

        self.core_set, self.skel_set, self.addr_set, self.num_set = (csr(c) for c in ("core_h", "skel_h", "addr_h", "num_h"))
        self.core_set_w = col("core_w").values.to_numpy()
        self.addr_set_w = col("addr_w").values.to_numpy()
        self.core_set_norm = t.column("core_norm").to_numpy()
        self.addr_set_norm = t.column("addr_norm").to_numpy()
        self.num_first = t.column("num_first").to_numpy().astype(np.uint64)
        self.len_core = t.column("len_core").to_numpy()
        self.len_addr = t.column("len_addr").to_numpy()
        del t
        gc.collect()
        log(f"  record cache {srcs}: {n:,} records ({time.time() - t0:.0f}s)")

    def idx(self, global_rows):
        li = self.local[global_rows]
        assert (li >= 0).all()
        return li

    def strings(self, attr, li):
        return getattr(self, attr).take(pa.array(li)).to_numpy(zero_copy_only=False)


def _gather(csr, li, w=None):
    """Expand CSR rows li -> (pair index, value[, weight]) for all tokens of all pairs."""
    offs, vals = csr
    L = offs[li + 1] - offs[li]
    tot = int(L.sum())
    pidx = np.repeat(np.arange(len(li)), L)
    start = np.repeat(offs[li] - np.cumsum(L) + L, L) + np.arange(tot)
    return pidx, vals[start], (w[start] if w is not None else None), L


def overlap(csrA, liA, csrB, liB, wA=None):
    """Per pair: number of shared tokens (and sum of wA^2 over shared tokens), plus set sizes."""
    pa_, va, wa, La = _gather(csrA, liA, wA)
    pb_, vb, _, Lb = _gather(csrB, liB)
    with np.errstate(over="ignore"):
        ka = va ^ (pa_.astype(np.uint64) * MIX)
        kb = vb ^ (pb_.astype(np.uint64) * MIX)
    hit = pd.Series(ka).isin(kb).values  # hash-table membership (no sort)
    P = len(liA)
    shared = np.bincount(pa_[hit], minlength=P).astype(np.float32)
    # sum of squared weights of shared tokens = numerator of the IDF-weighted cosine
    wsum = (np.bincount(pa_[hit], weights=wa[hit].astype(np.float64) ** 2, minlength=P).astype(np.float32)
            if wA is not None else None)
    return shared, wsum, La.astype(np.float32), Lb.astype(np.float32)


def contains_first(A, liA, B, liB):
    """Is the first address number of A among B's numbers?"""
    pb_, vb, _, _ = _gather(B.num_set, liB)
    f = A.num_first[liA]
    with np.errstate(over="ignore"):
        kb = vb ^ (pb_.astype(np.uint64) * MIX)
        ka = f ^ (np.arange(len(liA), dtype=np.uint64) * MIX)
    return (pd.Series(ka).isin(kb).values & (f != 0)).astype(np.float32)


def _jacc(sh, la, lb):
    u = la + lb - sh
    return np.where(u > 0, sh / np.maximum(u, 1), 0).astype(np.float32)


def pair_features(A, B, ia, ib):
    """A: S1 RecordCache, B: pool RecordCache, ia/ib: cache indices per pair."""
    W = dict(workers=-1, dtype=np.float32)
    f = {}
    n1, n2 = A.strings("name", ia), B.strings("name", ib)
    f["name_ratio"] = process.cpdist(n1, n2, scorer=fuzz.ratio, **W)
    f["name_tset"] = process.cpdist(n1, n2, scorer=fuzz.token_set_ratio, **W)
    f["name_tsort"] = process.cpdist(n1, n2, scorer=fuzz.token_sort_ratio, **W)
    f["name_partial"] = process.cpdist(n1, n2, scorer=fuzz.partial_ratio, **W)
    del n1, n2
    c1, c2 = A.strings("core", ia), B.strings("core", ib)
    f["core_ratio"] = process.cpdist(c1, c2, scorer=fuzz.ratio, **W)
    f["core_tset"] = process.cpdist(c1, c2, scorer=fuzz.token_set_ratio, **W)
    f["core_jw"] = process.cpdist(c1, c2, scorer=JaroWinkler.normalized_similarity, **W)
    f["core_eq"] = ((c1 == c2) & (A.len_core[ia] > 0)).astype(np.float32)
    nl1, nl2 = A.nonlatin[ia], B.nonlatin[ib]
    cross = nl1 != nl2
    if USE_ROMAN:
        # romanised core name vs the Latin core name, only for cross-script pairs; other pairs copy
        # the plain core scores so the feature means "best same-alphabet comparison"
        f["roman_ratio"] = f["core_ratio"].copy()
        f["roman_tset"] = f["core_tset"].copy()
        x = np.flatnonzero(cross & nl2)
        if len(x):
            r2 = B.strings("roman", ib[x])
            f["roman_ratio"][x] = process.cpdist(c1[x], r2, scorer=fuzz.ratio, **W)
            f["roman_tset"][x] = process.cpdist(c1[x], r2, scorer=fuzz.token_set_ratio, **W)
    del c1, c2
    k1, k2 = A.strings("skel", ia), B.strings("skel", ib)
    f["skel_ratio"] = process.cpdist(k1, k2, scorer=fuzz.ratio, **W)
    f["skel_tset"] = process.cpdist(k1, k2, scorer=fuzz.token_set_ratio, **W)
    del k1, k2
    a1, a2 = A.strings("addr", ia), B.strings("addr", ib)
    f["addr_ratio"] = process.cpdist(a1, a2, scorer=fuzz.ratio, **W)
    f["addr_tset"] = process.cpdist(a1, a2, scorer=fuzz.token_set_ratio, **W)
    f["addr_partial"] = process.cpdist(a1, a2, scorer=fuzz.partial_ratio, **W)
    del a1, a2
    # set features on hashes
    sh, ws, la, lb = overlap(A.core_set, ia, B.core_set, ib, A.core_set_w)
    f["core_shared"], f["core_jacc"] = sh, _jacc(sh, la, lb)
    f["core_idf_cos"] = np.where(A.core_set_norm[ia] * B.core_set_norm[ib] > 0,
                                 ws / np.maximum(A.core_set_norm[ia] * B.core_set_norm[ib], 1e-9), 0).astype(np.float32)
    f["ntok_core1"], f["ntok_core2"] = la, lb
    sh, _, la, lb = overlap(A.skel_set, ia, B.skel_set, ib)
    f["skel_jacc"] = _jacc(sh, la, lb)
    sh, ws, la, lb = overlap(A.addr_set, ia, B.addr_set, ib, A.addr_set_w)
    f["addr_word_shared"], f["addr_word_jacc"] = sh, _jacc(sh, la, lb)
    f["addr_idf_cos"] = np.where(A.addr_set_norm[ia] * B.addr_set_norm[ib] > 0,
                                 ws / np.maximum(A.addr_set_norm[ia] * B.addr_set_norm[ib], 1e-9), 0).astype(np.float32)
    sh, _, la, lb = overlap(A.num_set, ia, B.num_set, ib)
    f["num_n1"], f["num_n2"], f["num_shared"] = la, lb, sh
    f["num_jacc"] = _jacc(sh, la, lb)
    f["num_miss1"], f["num_miss2"] = la - sh, lb - sh
    f["num_first_eq"] = ((A.num_first[ia] == B.num_first[ib]) & (A.num_first[ia] != 0)).astype(np.float32)
    f["num_first_in2"] = contains_first(A, ia, B, ib)
    f["len_core1"], f["len_core2"] = A.len_core[ia], B.len_core[ib]
    f["len_addr1"], f["len_addr2"] = A.len_addr[ia], B.len_addr[ib]
    f["nonlatin1"], f["nonlatin2"] = nl1.astype(np.float32), nl2.astype(np.float32)
    f["cross_script"] = cross.astype(np.float32)
    return pd.DataFrame(f)


CONTEXT = ["score", "name_tset", "core_jw", "skel_ratio", "addr_tset", "num_jacc", "core_idf_cos",
           "addr_idf_cos", "combo"]


def add_context(df):
    """Relative-to-the-best features inside each S1's candidate list (df sorted by s1_row)."""
    df["combo"] = (df.core_idf_cos + df.addr_idf_cos + df.skel_ratio / 100).astype(np.float32)
    g = df.groupby("s1_row", sort=False)
    df["n_cands"] = g["pidx"].transform("size").astype(np.float32)
    for c in CONTEXT:
        df[f"{c}_gap"] = (df[c] - g[c].transform("max")).astype(np.float32)
        df[f"{c}_rk"] = g[c].rank(ascending=False, method="min").astype(np.float32)
    # how many candidates of this S1 look like strong matches
    df["n_strong"] = (df["combo"] > 1.2).astype(np.float32).groupby(df["s1_row"].values, sort=False).transform("sum")
    return df


def build_features(split, cand, out_path, chunk=500_000):
    """cand: DataFrame with s1_row, pidx, score, nkeys, rank, m_rid (whole S1 lists).
    Writes cand + features to out_path, chunk by chunk (chunks never split an S1's list)."""
    t0 = time.time()
    A = RecordCache(split, [1], cand.s1_row.values)
    B = RecordCache(split, [2, 3], cand.pidx.values)
    log(f"  record caches ready ({time.time() - t0:.0f}s)")
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
        df = add_context(df)
        tab = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, tab.schema, compression="zstd")
        writer.write_table(tab)
        log(f"    features {hi:,}/{len(cand):,} pairs ({time.time() - t0:.0f}s)")
        del c, feats, df, tab
        gc.collect()
    if writer is not None:
        writer.close()
    dt = time.time() - t0
    log(f"  features for {len(cand):,} pairs in {dt:.0f}s ({len(cand) / max(dt, 1e-9):,.0f} pairs/s incl. setup)"
        f" -> {out_path}")
    del A, B
    gc.collect()


def feature_columns(df):
    skip = {"s1_row", "pidx", "m_rid", "s1_rid", "label", "fold"}
    return [c for c in df.columns if c not in skip]
