"""Blocking keys for every record, saved once per file so blocking experiments are cheap.

A "key" is a 64-bit hash of (country, key type, token). Two records can only become candidates
if they share at least one key, and the country is mixed into every key, so records are only
compared within the same country label (the label is used as an opaque string -> works for
France or any unseen country).

Key types (int8 code):
  1 N  : word of name_core (name without legal words)          e.g. "orelees"
  2 NB : two adjacent words of name_core                       e.g. "orelees barbershop"
  3 K  : consonant skeleton token of the name (cross-script)   e.g. "RLS"
  4 A  : address word (>= 3 chars, not a number)               e.g. "westchester"
  5 D  : address number (leading zeros removed)                e.g. "1795"
  6 DB : address number + the following word                   e.g. "1795 westchester"
  7 NF : full name_core string                                 e.g. "orelees barbershop"

Output: data/keys_{split}_s{src}.parquet with columns row (int32, row number in the source
parquet), key (uint64), kt (int8 key type).  Usage: python keys.py [split:src ...]
"""
import gc
import os
import sys
import time
from functools import lru_cache

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from common import DATA, log, pq_path
from skeleton import GENERIC_SKEL, skel_token

KT = {"N": 1, "NB": 2, "K": 3, "A": 4, "D": 5, "DB": 6, "NF": 7}


def hash_strings(arr: pa.Array) -> np.ndarray:
    """Deterministic uint64 hash of each string (hash the distinct values only)."""
    d = pc.dictionary_encode(arr)
    if isinstance(d, pa.ChunkedArray):
        d = d.combine_chunks()
    uniq = d.dictionary.to_numpy(zero_copy_only=False)
    hu = pd.util.hash_array(uniq.astype(object), categorize=False)
    return hu[d.indices.to_numpy(zero_copy_only=False)]


def tokens(strings: pa.Array):
    lst = pc.split_pattern(strings, " ")
    flat = pc.list_flatten(lst)
    par = pc.list_parent_indices(lst).to_numpy(zero_copy_only=False).astype(np.int32)
    return flat, par


def bigrams(flat, par, first_mask=None):
    """Adjacent token pairs inside the same record; optionally the first token must satisfy a mask."""
    n = len(flat)
    if n < 2:
        return pa.array([], pa.string()), np.array([], np.int32)
    same = par[:-1] == par[1:]
    if first_mask is not None:
        same &= first_mask[:-1]
    idx = np.flatnonzero(same)
    a = pc.take(flat, pa.array(idx))
    b = pc.take(flat, pa.array(idx + 1))
    return pc.binary_join_element_wise(a, b, " "), par[idx]


@lru_cache(maxsize=2_000_000)
def _skel(tok):
    return skel_token(tok)


def skel_strings(name_n: pa.Array) -> pa.Array:
    out = []
    for s in name_n.to_pylist():
        sk = [x for x in (_skel(t) for t in (s or "").split()) if len(x) >= 2 and x not in GENERIC_SKEL]
        out.append(" ".join(sk))
    return pa.array(out, pa.string())


def make_keys(t: pa.Table, row0: int):
    """Return (row int32, key uint64, kt int8) for a chunk of records."""
    ch = hash_strings(t.column("country").combine_chunks())
    rows, keys, kts = [], [], []

    def add(strs, par, kt, minlen=2):
        if len(strs) == 0:
            return
        keep = pc.greater_equal(pc.utf8_length(strs), minlen).to_numpy(zero_copy_only=False)
        strs = strs.filter(pa.array(keep))
        par = par[keep]
        h = hash_strings(strs) ^ np.uint64((kt * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF)
        with np.errstate(over="ignore"):
            h = h ^ (ch[par] * np.uint64(0xC2B2AE3D27D4EB4F))
        rows.append(par)
        keys.append(h)
        kts.append(np.full(len(par), kt, np.int8))

    core = t.column("name_core").combine_chunks()
    flat, par = tokens(core)
    add(flat, par, KT["N"])
    bg, bp = bigrams(flat, par)
    add(bg, bp, KT["NB"])
    add(core, np.arange(len(core), dtype=np.int32), KT["NF"], minlen=3)
    sk = skel_strings(t.column("name_n").combine_chunks())
    flat_k, par_k = tokens(sk)
    add(flat_k, par_k, KT["K"])
    addr = t.column("addr_n").combine_chunks()
    flat, par = tokens(addr)
    isnum = pc.match_substring_regex(flat, r"^[0-9]+$").to_numpy(zero_copy_only=False)
    words = flat.filter(pa.array(~isnum))
    add(words, par[~isnum], KT["A"], minlen=3)
    nums = pc.replace_substring_regex(flat.filter(pa.array(isnum)), r"^0+([0-9])", r"\1")
    add(nums, par[isnum], KT["D"], minlen=1)
    flat_z = pc.if_else(pa.array(isnum), pc.replace_substring_regex(flat, r"^0+([0-9])", r"\1"), flat)
    bg, bp = bigrams(flat_z, par, first_mask=isnum)
    add(bg, bp, KT["DB"])

    row = np.concatenate(rows).astype(np.int64)
    key = np.concatenate(keys)
    kt = np.concatenate(kts)
    # one (row, key) only once
    df = pd.DataFrame({"row": row, "key": key, "kt": kt}).drop_duplicates(["row", "key"])
    df["row"] = (df["row"].values + row0).astype(np.int32)
    return df


def build(split, src, chunk=500_000):
    t0 = time.time()
    pf = pq.ParquetFile(pq_path(split, src))
    out = os.path.join(DATA, f"keys_{split}_s{src}.parquet")
    writer, row0, nkeys = None, 0, 0
    cols = ["country", "name_core", "name_n", "addr_n"]
    for batch in pf.iter_batches(batch_size=chunk, columns=cols):
        t = pa.Table.from_batches([batch])
        df = make_keys(t, row0)
        tab = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out, tab.schema, compression="zstd")
        writer.write_table(tab, row_group_size=2_000_000)
        row0 += len(t)
        nkeys += len(df)
        del t, df, tab
        gc.collect()
    writer.close()
    log(f"keys {split} S{src}: {row0:,} records, {nkeys:,} keys ({nkeys / row0:.1f}/record) "
        f"in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    jobs = sys.argv[1:] or ["train:1", "train:2", "train:3", "test:1", "test:2", "test:3"]
    for j in jobs:
        sp, s = j.split(":")
        build(sp, int(s))
