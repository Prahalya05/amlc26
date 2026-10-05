"""Candidate generation v3: the old forward blocking (blocking.py, UNCHANGED) plus two additions.

  reverse   : for every S2/S3 record, the same keys and the same scoring (idf = log(N/df) x type weight,
              keys with df > cap ignored; df counted over the S1 file because S1 is the index here) find
              its top-k S1 records; (S1, record) is added to that S1's candidates. Keys contain the
              country, so only same-country S1s are found.
  second hop: for chosen "seed" pool records (an S1's top-3 candidates by model p), find near-duplicate
              pool records with a pool->pool key search (tight key types only), keep those with strong
              name_core AND address similarity; they are added as candidates of the seed's S1.
  pair_scores: the forward blocking score / number of shared keys for any extra (S1, record) pair,
              so added pairs get the same 'score' / 'nkeys' features as forward candidates.
All searches are chunked numpy like blocking.block (same packed direct sorts).
"""
import gc
import json
import os
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from blocking import DEFAULT_CFG, QMAX, KeyLookup, key_files, load_s1_keys
from common import WORK, log, pq_path

CFG = {**DEFAULT_CFG, **json.load(open(os.path.join(WORK, "block_cfg.json")))}
TYPES = np.array(CFG["types"], np.int8)
CAP, K = CFG["cap"], CFG["k"]
TW = np.zeros(16)
for _k, _v in CFG["type_w"].items():
    TW[_k] = _v
# near-duplicate search: only keys that combine tokens (rare, precise): name bigram, number+word,
# full core name, name x number, number x address word, name without spaces
HOP_TYPES = np.array([2, 6, 7, 8, 10, 11], np.int8)


def n_rows(split, src):
    return pq.ParquetFile(pq_path(split, src)).metadata.num_rows


def pool_files(split, types=TYPES):
    n2 = n_rows(split, 2)
    return [(p, off) for s, off in ((2, 0), (3, n2)) for p in key_files(split, s, types)]


def s1_files(split, types=TYPES):
    return [(p, 0) for p in key_files(split, 1, types)]


def read_key_range(files, lo, hi, types, mask=None):
    """(row, key, kt) of all key rows with global row in [lo, hi), kt in types (and mask[row] if given).
    Only row groups whose 'row' statistics overlap the range are read."""
    rows, keys, kts = [], [], []
    for path, off in files:
        pf = pq.ParquetFile(path)
        ci = pf.schema_arrow.get_field_index("row")
        for i in range(pf.num_row_groups):
            st = pf.metadata.row_group(i).column(ci).statistics
            if st is not None and st.has_min_max and (st.max + off < lo or st.min + off >= hi):
                continue
            t = pf.read_row_group(i)
            row = t.column("row").to_numpy().astype(np.int64) + off
            kt = t.column("kt").to_numpy()
            m = (row >= lo) & (row < hi) & np.isin(kt, types)
            if mask is not None:
                m[m] = mask[row[m]]
            if m.any():
                rows.append(row[m])
                keys.append(t.column("key").to_numpy()[m])
                kts.append(kt[m])
            del t
    if not rows:
        return np.array([], np.int64), np.array([], np.uint64), np.array([], np.int8)
    return np.concatenate(rows), np.concatenate(keys), np.concatenate(kts)


def _iter_index(files, kt_filter, cols):
    for path, off in files:
        pf = pq.ParquetFile(path)
        for i in range(pf.num_row_groups):
            t = pf.read_row_group(i, columns=cols)
            kt = t.column("kt").to_numpy()
            m = np.isin(kt, kt_filter) if kt_filter is not None else None
            yield t, off, m
            del t


def scan_df(files, U, kt_filter=None, gather_mask=None):
    """df of every key of U over the index files (+ all key rows of the records in gather_mask)."""
    look = KeyLookup(U)
    df = np.zeros(len(U), np.int64)
    got = []
    cols = ["row", "key", "kt"] if gather_mask is not None else ["key", "kt"]
    for t, off, m in _iter_index(files, kt_filter, cols):
        key = t.column("key").to_numpy()
        pos = look(key if m is None else key[m])
        df += np.bincount(pos[pos >= 0], minlength=len(U))
        if gather_mask is not None:
            row = t.column("row").to_numpy().astype(np.int64) + off
            g = gather_mask[row]
            if g.any():
                got.append(pd.DataFrame({"row": row[g], "key": key[g]}))
    return df, (pd.concat(got, ignore_index=True) if got else None)


def postings(files, U, allowed, df, kt_filter=None):
    ua = np.flatnonzero(allowed)
    look = KeyLookup(U[ua])
    packed = []
    for t, off, m in _iter_index(files, kt_filter, ["row", "key", "kt"]):
        row = t.column("row").to_numpy().astype(np.int64) + off
        key = t.column("key").to_numpy()
        if m is not None:
            row, key = row[m], key[m]
        pos = look(key)
        hit = pos >= 0
        packed.append((ua[pos[hit]].astype(np.uint64) << np.uint64(32)) | row[hit].astype(np.uint64))
    packed = np.concatenate(packed)
    packed.sort()
    post = (packed & np.uint64(0xFFFFFFFF)).astype(np.int32)
    del packed
    offs = np.zeros(len(U) + 1, np.int64)
    offs[1:] = np.cumsum(np.where(allowed, df, 0))
    assert offs[-1] == len(post)
    return post, offs


WBITS, WSCALE = 15, 1024.0  # weight <= log(10.3M) ~ 16.2 per key -> < 2^15 units of 1/1024
WMASK = (1 << WBITS) - 1


def topk_search(q, key, kt, files, n_index, k, self_idx=None, kt_filter=None, max_pairs_chunk=5_000_000,
                fast_agg=True, threads=4):
    """Top-k index records per query. q: local query id (0..nq-1) of every query key row.
    self_idx[q] = index id of the query itself (excluded), for pool->pool searches.
    Returns DataFrame q, idx, score, nkeys, rank (0-based)."""
    assert len(q) == 0 or q.max() < (1 << 22)
    assert (1 << 22) * n_index < (1 << (64 - WBITS))
    codes, U = pd.factorize(key)
    U = np.asarray(U, np.uint64)
    ktU = np.zeros(len(U), np.int8)
    ktU[codes] = kt
    df, _ = scan_df(files, U, kt_filter)
    allowed = (df >= 1) & (df <= CAP)
    w = np.where(allowed, np.log(n_index / np.maximum(df, 1)) * TW[ktU], 0.0).astype(np.float32)
    wq_all = np.round(w.astype(np.float64) * WSCALE).astype(np.uint64)
    assert wq_all.max(initial=0) <= WMASK
    post, offs = postings(files, U, allowed, df, kt_filter)
    keep = allowed[codes]
    packed = np.sort((q[keep].astype(np.uint64) << np.uint64(32)) | codes[keep].astype(np.uint64))
    qq = (packed >> np.uint64(32)).astype(np.int64)
    pp = (packed & np.uint64(0xFFFFFFFF)).astype(np.int64)
    del packed, codes, keep
    lens = offs[pp + 1] - offs[pp]
    first = np.flatnonzero(np.r_[True, np.diff(qq) != 0]) if len(qq) else np.array([], np.int64)
    per_row = np.add.reduceat(lens, first) if len(first) else np.array([], np.int64)
    bounds, acc, start = [], 0, 0
    for i, v in enumerate(per_row):
        if acc + v > max_pairs_chunk and i > start:
            bounds.append((start, i))
            start, acc = i, 0
        acc += v
    bounds.append((start, len(first)))
    first = np.append(first, len(qq))

    def work(ab):
        a, b = ab
        lo, hi = first[a], first[b]
        r, p, L = qq[lo:hi], pp[lo:hi], lens[lo:hi]
        tot = int(L.sum())
        if tot == 0:
            return None, 0
        rr = np.repeat(r, L)
        starts = np.repeat(offs[p] - np.cumsum(L) + L, L)
        cand = post[starts + np.arange(tot)].astype(np.int64)
        if fast_agg:
            # one DIRECT sort of packed (query, index record, weight in 1/1024 units), then reduceat:
            # much faster than a hash groupby here; the weight quantisation (< 0.001 per key) only
            # matters for exact ties
            wq = np.repeat(wq_all[p], L)
            packed = np.sort((((rr - r[0]).astype(np.uint64) * np.uint64(n_index) + cand.astype(np.uint64))
                              << np.uint64(WBITS)) | wq)
            del rr, starts, cand, wq
            code = packed >> np.uint64(WBITS)
            gs = np.flatnonzero(np.r_[True, code[1:] != code[:-1]])
            score = (np.add.reduceat((packed & np.uint64(WMASK)).astype(np.int64), gs) / WSCALE).astype(np.float32)
            nk = np.diff(np.r_[gs, len(packed)]).astype(np.int16)
            uc = code[gs].astype(np.int64) + r[0] * n_index
            del packed, code, gs
        else:
            ww = np.repeat(w[p], L)
            code = rr * n_index + cand
            del rr, starts, cand
            g = pd.DataFrame({"code": code, "w": ww}).groupby("code", sort=True).w.agg(["sum", "size"])
            del code, ww
            uc = g.index.to_numpy()
            score = g["sum"].to_numpy().astype(np.float32)
            nk = g["size"].to_numpy().astype(np.int16)
            del g
        cr, cp = uc // n_index, uc % n_index
        if self_idx is not None:
            m = cp != self_idx[cr]
            uc, score, nk, cr, cp = uc[m], score[m], nk[m], cr[m], cp[m]
        # order by (query, score desc, index record asc): the low bits hold the pair's POSITION j in the
        # (query, record)-sorted arrays - within a query j grows with the record, so ties break exactly as
        # before, and score / nkeys / record come back by j without a hash lookup
        assert len(uc) < (1 << 24)
        qs = np.minimum(np.round(score.astype(np.float64) * 256), QMAX).astype(np.uint64)
        packed = np.sort((cr.astype(np.uint64) << np.uint64(42)) | ((np.uint64(QMAX) - qs) << np.uint64(24))
                         | np.arange(len(uc), dtype=np.uint64))
        sr = (packed >> np.uint64(42)).astype(np.int64)
        j = (packed & np.uint64((1 << 24) - 1)).astype(np.int64)
        del packed, qs
        gstart = np.flatnonzero(np.r_[True, np.diff(sr) != 0])
        rank = np.arange(len(sr)) - np.repeat(gstart, np.diff(np.r_[gstart, len(sr)]))
        m = rank < k
        sr, j, rank = sr[m], j[m], rank[m]
        return (sr.astype(np.int32), cp[j].astype(np.int32), score[j], nk[j], rank.astype(np.int16)), tot

    # sub-chunks in parallel threads: the heavy numpy steps (sort, repeat, gathers) release the GIL
    from concurrent.futures import ThreadPoolExecutor
    res, n_exp = [], 0
    with ThreadPoolExecutor(max_workers=threads) as ex:
        for out_, tot in ex.map(work, [ab for ab in bounds if ab[0] < ab[1]]):
            n_exp += tot
            if out_ is not None:
                res.append(pd.DataFrame(dict(zip(["q", "idx", "score", "nkeys", "rank"], out_))))
    gc.collect()
    del post, offs
    out = pd.concat(res, ignore_index=True) if res else pd.DataFrame(
        {"q": [], "idx": [], "score": [], "nkeys": [], "rank": []})
    out.attrs["expanded"] = n_exp
    return out


# ------------------------------------------------------------------ reverse blocking
def reverse_pass(split, k, out_path, keep_s1=None, chunk=500_000, limit=None):
    """For every pool record: top-k S1 records (same keys/scoring, S1 as the index). Writes
    (pidx, s1_row, rscore, rrank) for rank < k, optionally only for S1 rows with keep_s1[row]."""
    t0 = time.time()
    pf_files, sf_files = pool_files(split), s1_files(split)
    npool = n_rows(split, 2) + n_rows(split, 3)
    ns1 = n_rows(split, 1)
    end = min(npool, limit) if limit else npool
    writer, n_out, n_exp = None, 0, 0
    for lo in range(0, end, chunk):
        hi = min(lo + chunk, end)
        row, key, kt = read_key_range(pf_files, lo, hi, TYPES)
        res = topk_search(row - lo, key, kt, sf_files, ns1, k)
        n_exp += res.attrs["expanded"]
        del row, key, kt
        if keep_s1 is not None:
            res = res[keep_s1[res.idx.values]]
        tab = pa.table({"pidx": (res.q.values + lo).astype(np.int32), "s1_row": res.idx.values.astype(np.int32),
                        "rscore": res.score.values.astype(np.float32), "rrank": res["rank"].values.astype(np.int8)})
        if writer is None:
            writer = pq.ParquetWriter(out_path, tab.schema, compression="zstd")
        writer.write_table(tab)
        n_out += len(res)
        del res, tab
        gc.collect()
        log(f"  reverse {split}: pool {hi:,}/{end:,}, kept pairs {n_out:,}, expanded {n_exp:,} ({time.time() - t0:.0f}s)")
    writer.close()
    return dict(seconds=round(time.time() - t0, 1), pairs=n_out, expanded=n_exp, pool=end)


# ------------------------------------------------------------------ second hop (pool -> pool near duplicates)
class PoolStrings:
    """name_core / addr_n of every pool record as Arrow arrays (take() per chunk)."""

    def __init__(self, split):
        cols = {}
        for c in ("name_core", "addr_n"):
            parts = [pq.read_table(pq_path(split, s), columns=[c]).column(c).combine_chunks() for s in (2, 3)]
            cols[c] = pa.concat_arrays(parts)
        self.core, self.addr = cols["name_core"], cols["addr_n"]

    def get(self, attr, idx):
        return getattr(self, attr).take(pa.array(idx)).to_numpy(zero_copy_only=False)


def near_duplicates(split, seeds, k_search=20, min_sim=80, seeds_per_chunk=500_000, strings=None):
    """Pool->pool search for the seed records (global pool indices). Returns (seed, dup, sim_name,
    sim_addr, hscore) for dups among the seed's top-k_search by key score (tight key types) whose
    token_set_ratio on name_core AND on addr_n are both >= min_sim."""
    from rapidfuzz import fuzz, process
    t0 = time.time()
    seeds = np.unique(np.asarray(seeds, np.int64))
    npool = n_rows(split, 2) + n_rows(split, 3)
    files = pool_files(split, HOP_TYPES)
    mask = np.zeros(npool, bool)
    mask[seeds] = True
    S = strings or PoolStrings(split)
    out, n_cmp = [], 0
    for c0 in range(0, len(seeds), seeds_per_chunk):
        sd = seeds[c0:c0 + seeds_per_chunk]
        row, key, kt = read_key_range(files, int(sd[0]), int(sd[-1]) + 1, HOP_TYPES, mask)
        q = np.searchsorted(sd, row)
        res = topk_search(q, key, kt, files, npool, k_search, self_idx=sd, kt_filter=HOP_TYPES)
        del row, key, kt, q
        a, b = sd[res.q.values], res.idx.values.astype(np.int64)
        n_cmp += len(a)
        W = dict(workers=-1, dtype=np.float32)
        sn = process.cpdist(S.get("core", a), S.get("core", b), scorer=fuzz.token_set_ratio, **W)
        m = sn >= min_sim
        sa = process.cpdist(S.get("addr", a[m]), S.get("addr", b[m]), scorer=fuzz.token_set_ratio, **W)
        mm = sa >= min_sim
        out.append(pd.DataFrame({"seed": a[m][mm], "dup": b[m][mm], "sim_name": sn[m][mm], "sim_addr": sa[mm],
                                 "hscore": res.score.values[m][mm]}))
        del res, a, b, sn, sa
        gc.collect()
        log(f"  near-dup {split}: seeds {min(c0 + seeds_per_chunk, len(seeds)):,}/{len(seeds):,}, compared {n_cmp:,}, "
            f"kept {sum(len(x) for x in out):,} ({time.time() - t0:.0f}s)")
    return pd.concat(out, ignore_index=True), dict(seconds=round(time.time() - t0, 1), seeds=len(seeds), compared=n_cmp)


def hop_assign(s1_seeds, dups, existing, thr, cap=20):
    """s1_seeds: (s1_row, seed); dups: near_duplicates output; existing: (s1_row, pidx) already candidates.
    Returns the added (s1_row, pidx) pairs: dups of the S1's seeds with both similarities >= thr, not
    already candidates, best (sim_name + sim_addr) first, at most `cap` per S1."""
    d = dups[(dups.sim_name >= thr) & (dups.sim_addr >= thr)]
    m = s1_seeds.merge(d, on="seed")[["s1_row", "dup", "sim_name", "sim_addr"]].rename(columns={"dup": "pidx"})
    m["sim"] = m.sim_name + m.sim_addr
    m = m.sort_values(["s1_row", "sim"], ascending=[True, False]).drop_duplicates(["s1_row", "pidx"])
    ex = pd.MultiIndex.from_arrays([existing.s1_row.values, existing.pidx.values])
    m = m[~pd.MultiIndex.from_arrays([m.s1_row.values, m.pidx.values]).isin(ex)]
    m = m[m.groupby("s1_row").cumcount().values < cap]
    return m[["s1_row", "pidx"]].reset_index(drop=True)


# ------------------------------------------------------------------ forward score of extra pairs
def pair_scores(split, s1_rows, pairs, chunk=2_000_000):
    """For (s1_row, pidx) pairs whose S1 rows are s1_rows:
      fscore, fnk : forward blocking score (sum of weights of shared keys with 1 <= POOL df <= cap) and
                    number of such keys - same weights as blocking.block, so an added pair gets the score
                    it would have had in the forward search;
      rscore      : reverse score (weights log(N_S1/df) with df over the S1 file, cap on that df) - the score
                    the reverse search gives the pair.
    Shared keys are always S1 keys, so only the S1 keys (U) are looked up: one scan of the pool key files
    (pool df + the keys in U of the pairs' pool records) and one of the S1 key files (S1 df)."""
    row, key, kt = load_s1_keys(split, np.asarray(s1_rows), TYPES)
    codes, U = pd.factorize(key)
    U = np.asarray(U, np.uint64)
    ktU = np.zeros(len(U), np.int8)
    ktU[codes] = kt
    npool, ns1 = n_rows(split, 2) + n_rows(split, 3), n_rows(split, 1)
    gm = np.zeros(npool, bool)
    gm[pairs.pidx.values] = True
    look = KeyLookup(U)
    dfp = np.zeros(len(U), np.int64)
    gp = []
    for t, off, _ in _iter_index(pool_files(split), None, ["row", "key", "kt"]):
        pos = look(t.column("key").to_numpy())
        hit = pos >= 0
        dfp += np.bincount(pos[hit], minlength=len(U))
        r = t.column("row").to_numpy().astype(np.int64) + off
        g = hit & gm[r]
        gp.append((r[g].astype(np.uint64) << np.uint64(32)) | pos[g].astype(np.uint64))
    gp = np.sort(np.concatenate(gp))
    gpid = (gp >> np.uint64(32)).astype(np.int64)
    gpos = (gp & np.uint64(0xFFFFFFFF)).astype(np.int64)
    del gp
    dfs, _ = scan_df(s1_files(split), U)
    okf, oks = (dfp >= 1) & (dfp <= CAP), (dfs >= 1) & (dfs <= CAP)
    wf = np.where(okf, np.log(npool / np.maximum(dfp, 1)) * TW[ktU], 0.0).astype(np.float32)
    ws = np.where(oks, np.log(ns1 / np.maximum(dfs, 1)) * TW[ktU], 0.0).astype(np.float32)
    nU = np.int64(len(U))
    s1code = pd.Index(np.unique(row.astype(np.int64) * nU + codes))
    del row, key, kt, codes
    P_s1, P_pid = pairs.s1_row.values.astype(np.int64), pairs.pidx.values.astype(np.int64)
    n = len(P_s1)
    fscore, rscore = np.zeros(n, np.float32), np.zeros(n, np.float32)
    fnk = np.zeros(n, np.int16)
    for a in range(0, n, chunk):
        s1c, pid = P_s1[a:a + chunk], P_pid[a:a + chunk]
        st = np.searchsorted(gpid, pid, "left")
        L = np.searchsorted(gpid, pid, "right") - st
        tot = int(L.sum())
        pi = np.repeat(np.arange(len(pid)), L)
        kpos = gpos[np.repeat(st - np.cumsum(L) + L, L) + np.arange(tot)]
        hit = s1code.get_indexer(s1c[pi] * nU + kpos) >= 0
        pi, kpos = pi[hit], kpos[hit]
        f = okf[kpos]
        fscore[a:a + chunk] = np.bincount(pi[f], weights=wf[kpos[f]], minlength=len(pid))
        fnk[a:a + chunk] = np.bincount(pi[f], minlength=len(pid))
        rscore[a:a + chunk] = np.bincount(pi, weights=ws[kpos], minlength=len(pid))
        del st, L, pi, kpos, hit, f
    return fscore, fnk, rscore
