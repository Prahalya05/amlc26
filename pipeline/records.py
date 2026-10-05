"""Per-record derived data, computed ONCE per record and saved: data/rec_{split}_s{src}.parquet
(same row order as data/{split}_s{src}.parquet). Feature building only loads these rows.

Columns:
  skel      consonant skeleton of name_n (skeleton.py; generic legal words removed)
  roman     romanised core name for non-Latin names (romanize.roman_core), "" for Latin names
  core_h    distinct name_core word hashes (uint32)   core_w  their IDF weights (float32)
  skel_h    distinct skeleton token hashes (>= 2 chars)
  addr_h    distinct address word hashes (no digits)  addr_w  their IDF weights
  num_h     distinct address number hashes (digit runs, zeros stripped)
  num_first hash of the first address number (0 = none)
  core_norm / addr_norm   sqrt(sum of squared IDF weights)  (for IDF-weighted cosine)
  len_core / len_addr     string lengths
IDF = log((N+1)/(df+1)), df = number of S2+S3 records of the split containing the word (features2.token_df).
Usage: python records.py SPLIT [src ...]
"""
import gc
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from common import DATA, log, pq_path
from keys import hash_strings, skel_strings
from romanize import roman_core

COLS = ["name_n", "name_core", "addr_n", "addr_nums", "name_nonlatin"]


def rec_path(split, src):
    return os.path.join(DATA, f"rec_{split}_s{src}.parquet")


def _sets(strs, keep_regex=None, minlen=1):
    """Distinct token hashes per record -> (offsets int32 [n+1], uint64 hashes)."""
    from features2 import _tok_hashes
    n = len(strs)
    par, h = _tok_hashes(strs, keep_regex, minlen)
    d = pd.DataFrame({"p": par, "h": h}).drop_duplicates()
    offs = np.zeros(n + 1, np.int32)
    offs[1:] = np.cumsum(np.bincount(d.p.values, minlength=n))
    return offs, d.h.values


def _list(offs, vals, typ):
    return pa.ListArray.from_arrays(pa.array(offs, pa.int32()), pa.array(vals, typ))


def _weights(offs, h, df_series, N):
    d = df_series.reindex(h).fillna(0).values
    w = np.log((N + 1) / (d + 1)).astype(np.float32)
    sq = np.zeros(len(offs) - 1, np.float64)
    nz = np.diff(offs) > 0
    if nz.any():
        sq[nz] = np.add.reduceat(w.astype(np.float64) ** 2, offs[:-1][nz])
    return w, np.sqrt(sq).astype(np.float32)


def make(b: pa.RecordBatch, dfd, N):
    n = b.num_rows
    name, core, addr = b.column("name_n"), b.column("name_core"), b.column("addr_n")
    nonlatin = b.column("name_nonlatin").to_numpy(zero_copy_only=False).astype(bool)
    skel = skel_strings(name)
    rom = np.full(n, "", dtype=object)
    nl = np.flatnonzero(nonlatin)
    if len(nl):
        rom[nl] = [roman_core(x) for x in core.take(pa.array(nl)).to_pylist()]
    out = {"skel": skel, "roman": pa.array(rom, pa.string())}
    co, ch = _sets(core)
    cw, cn = _weights(co, ch, dfd["core"], N)
    so, sh = _sets(skel, minlen=2)
    ao, ah = _sets(addr, keep_regex=r"^[^0-9]+$")
    aw, an = _weights(ao, ah, dfd["addr"], N)
    nums = b.column("addr_nums")
    no, nh = _sets(nums)
    u32 = lambda x: (x & np.uint64(0xFFFFFFFF)).astype(np.uint32)  # noqa: E731
    out["core_h"] = _list(co, u32(ch), pa.uint32())
    out["core_w"] = _list(co, cw, pa.float32())
    out["skel_h"] = _list(so, u32(sh), pa.uint32())
    out["addr_h"] = _list(ao, u32(ah), pa.uint32())
    out["addr_w"] = _list(ao, aw, pa.float32())
    out["num_h"] = _list(no, u32(nh), pa.uint32())
    # first number = first element of the (ordered, deduplicated) number list
    has = np.diff(no) > 0
    nf = np.zeros(n, np.uint32)
    nf[has] = u32(nh)[no[:-1][has]]
    nf[has & (nf == 0)] = 1  # 0 is reserved for "no number"
    out["num_first"] = pa.array(nf, pa.uint32())
    out["core_norm"] = pa.array(cn)
    out["addr_norm"] = pa.array(an)
    out["len_core"] = pc.cast(pc.utf8_length(core), pa.float32())
    out["len_addr"] = pc.cast(pc.utf8_length(addr), pa.float32())
    return pa.table(out)


def build(split, src, chunk=500_000):
    from features2 import token_df
    t0 = time.time()
    dfd, N = token_df(split)
    writer, n = None, 0
    for b in pq.ParquetFile(pq_path(split, src)).iter_batches(chunk, columns=COLS):
        t = make(b, dfd, N)
        if writer is None:
            writer = pq.ParquetWriter(rec_path(split, src), t.schema, compression="zstd")
        writer.write_table(t)
        n += b.num_rows
        del t
        gc.collect()
    writer.close()
    log(f"records {split} S{src}: {n:,} in {time.time() - t0:.0f}s ({n / (time.time() - t0):,.0f} records/s)")


if __name__ == "__main__":
    split = sys.argv[1]
    for s in (sys.argv[2:] or ["1", "2", "3"]):
        build(split, int(s))
