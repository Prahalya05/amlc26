"""Extra blocking keys (v2), written to data/keys2_{split}_s{src}.parquet next to the v1 keys.

Why: the M4 diagnostics (reports/blocking_diag.md) showed that most missed true pairs DO share
keys, but only common ones ("cure", "717", "huntington" are each carried by thousands of pool
records), so they are either ignored (df > cap) or outscored. A COMBINATION of two common tokens
is usually rare: "cure + 717" is carried by a handful of records. So we add:

  8  NxD : name_core token  x address number          e.g. "cure|717"
  9  KxD : name skeleton token x address number       (works across scripts: "JN|39")
  10 DxA : address number x address word (>= 4 chars)  e.g. "717|huntington"
  11 NC  : name_core with spaces removed, trailing "com" dropped  ("weberssafecafecom" -> "weberssafecafe")
  12 D2  : address number from digit runs ("505b" -> "505"; v1 type 5 needs a pure-number token)

To bound the number of keys per record, each record uses at most 5 name tokens, 5 skeleton tokens,
4 numbers and 6 address words. They are picked by smallest hash ("bottom-k sketch"), NOT by position:
addresses are often shuffled between sources, and two records that share a token make the same
choice for it no matter where it stands.

Usage: python keys2.py [split:src ...]   (same row numbering as keys.py)
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
from keys import hash_strings, skel_strings, tokens

KT2 = {"NxD": 8, "KxD": 9, "DxA": 10, "NC": 11, "D2": 12}
LIMITS = {"name": 5, "skel": 5, "num": 4, "word": 6}
M1 = np.uint64(0x9E3779B97F4A7C15)
M2 = np.uint64(0xC2B2AE3D27D4EB4F)
M3 = np.uint64(0x165667B19E3779F9)


def tok_frame(strs: pa.Array, keep_regex=None, minlen=2):
    """(par, h) of distinct tokens per record; optional filter regex on the token."""
    flat, par = tokens(strs)
    keep = pc.greater_equal(pc.utf8_length(flat), minlen)
    if keep_regex:
        keep = pc.and_(keep, pc.match_substring_regex(flat, keep_regex))
    keep = keep.to_numpy(zero_copy_only=False)
    flat = flat.filter(pa.array(keep))
    df = pd.DataFrame({"par": par[keep], "h": hash_strings(flat) if len(flat) else np.array([], np.uint64)})
    return df.drop_duplicates()


def bottom(df, k):
    """Keep the k tokens with the smallest hash per record."""
    df = df.sort_values(["par", "h"])
    r = df.groupby("par").cumcount().values
    return df[r < k]


def cross(a, b, kt):
    m = a.merge(b, on="par", suffixes=("_a", "_b"))
    with np.errstate(over="ignore"):
        h = (m.h_a.values * M1) ^ (m.h_b.values * M3) ^ np.uint64((kt * 0x632BE59BD9B4E019) & 0xFFFFFFFFFFFFFFFF)
    return m.par.values.astype(np.int64), h


def make_keys2(t: pa.Table, row0: int):
    ch = hash_strings(t.column("country").combine_chunks())
    n = t.num_rows
    name = bottom(tok_frame(t.column("name_core").combine_chunks()), LIMITS["name"])
    skel = bottom(tok_frame(skel_strings(t.column("name_n").combine_chunks())), LIMITS["skel"])
    nums_all = tok_frame(t.column("addr_nums").combine_chunks(), minlen=1)
    nums = bottom(nums_all, LIMITS["num"])
    words = bottom(tok_frame(t.column("addr_n").combine_chunks(), keep_regex=r"^[^0-9]+$", minlen=4),
                   LIMITS["word"])
    parts = [cross(name, nums, KT2["NxD"]) + (KT2["NxD"],),
             cross(skel, nums, KT2["KxD"]) + (KT2["KxD"],),
             cross(nums, words, KT2["DxA"]) + (KT2["DxA"],)]
    with np.errstate(over="ignore"):
        parts.append((nums_all.par.values.astype(np.int64),
                      nums_all.h.values * M3 ^ np.uint64(KT2["D2"] * 0x9E3779B97F4A7C15 & 0xFFFFFFFFFFFFFFFF),
                      KT2["D2"]))
    nc = pc.replace_substring(t.column("name_core").combine_chunks(), " ", "")
    nc = pc.replace_substring_regex(nc, "com$", "")
    ok = pc.greater_equal(pc.utf8_length(nc), 5).to_numpy(zero_copy_only=False)
    with np.errstate(over="ignore"):
        parts.append((np.flatnonzero(ok).astype(np.int64),
                      hash_strings(nc.filter(pa.array(ok))) * M1 ^ np.uint64(KT2["NC"] * 0x632BE59BD9B4E019 & 0xFFFFFFFFFFFFFFFF),
                      KT2["NC"]))
    rows = np.concatenate([p[0] for p in parts])
    with np.errstate(over="ignore"):
        keys = np.concatenate([p[1] ^ (ch[p[0]] * M2) for p in parts])
    kts = np.concatenate([np.full(len(p[0]), p[2], np.int8) for p in parts])
    df = pd.DataFrame({"row": rows, "key": keys, "kt": kts}).drop_duplicates(["row", "key"])
    df["row"] = (df["row"].values + row0).astype(np.int32)
    assert df.row.max() < row0 + n
    return df


def build(split, src, chunk=500_000, limit=None):
    t0 = time.time()
    pf = pq.ParquetFile(pq_path(split, src))
    out = os.path.join(DATA, f"keys2_{split}_s{src}.parquet" if not limit else f"keys2_sample_{split}_s{src}.parquet")
    writer, row0, nkeys, cnt = None, 0, 0, {}
    cols = ["country", "name_core", "name_n", "addr_n", "addr_nums"]
    for batch in pf.iter_batches(batch_size=chunk if not limit else limit, columns=cols):
        t = pa.Table.from_batches([batch])
        df = make_keys2(t, row0)
        for k, v in df.kt.value_counts().items():
            cnt[k] = cnt.get(k, 0) + v
        tab = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out, tab.schema, compression="zstd")
        writer.write_table(tab, row_group_size=2_000_000)
        row0 += len(t)
        nkeys += len(df)
        del t, df, tab
        gc.collect()
        if limit:
            break
    writer.close()
    log(f"keys2 {split} S{src}: {row0:,} records, {nkeys:,} keys ({nkeys / row0:.1f}/record), "
        f"per type/record { {int(k): round(v / row0, 2) for k, v in sorted(cnt.items())} } in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    args = sys.argv[1:]
    limit = None
    if args and args[0].startswith("--sample="):
        limit = int(args.pop(0).split("=")[1])
    for j in args or ["train:1", "train:2", "train:3", "test:1", "test:2", "test:3"]:
        sp, s = j.split(":")
        build(sp, int(s), limit=limit)
