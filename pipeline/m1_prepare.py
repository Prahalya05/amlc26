"""M1: raw TSV -> Parquet with normalised columns.

Usage: python m1_prepare.py [--sample N]   (sample = only first N rows of each file, for testing)

Output columns per source file:
  rid (int64 unique id), entity_id, business_name, business_address, country,
  name_n (normalised name), name_core (name without legal words), addr_n (normalised address,
  street-type abbreviations canonicalised), addr_nums (digit runs, space separated),
  name_deva / addr_deva (contains Devanagari), name_nonlatin / addr_nonlatin (any non-Latin script)
Ground truth -> train_gt_pairs.parquet (s1_rid, m_rid) and train_gt_s1.parquet (s1_rid, n_match).
"""
import argparse
import gc
import time

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from common import (EXPECTED_ROWS, addr_numbers, canon_addr, id_to_rid, log, name_core, normalize,
                    pq_path, raw_path, read_tsv_chunks, script_flags, DATA)
import os


def process_chunk(df):
    t = pa.Table.from_pandas(df, preserve_index=False)
    ids = t.column("entity_id").combine_chunks()
    name = t.column("business_name").combine_chunks()
    addr = t.column("business_address").combine_chunks()
    name_n = normalize(name, "name")
    addr_n = canon_addr(normalize(addr, "addr"))
    nd, nn = script_flags(name)
    ad, an = script_flags(addr)
    return pa.table({
        "rid": id_to_rid(ids), "entity_id": ids, "business_name": name, "business_address": addr,
        "country": t.column("country").combine_chunks(),
        "name_n": name_n, "name_core": name_core(name_n), "addr_n": addr_n,
        "addr_nums": addr_numbers(addr_n),
        "name_deva": nd, "name_nonlatin": nn, "addr_deva": ad, "addr_nonlatin": an,
    })


def convert_source(split, src, sample):
    t0 = time.time()
    out = pq_path(split, src) if not sample else os.path.join(DATA, f"sample_{split}_s{src}.parquet")
    writer, n = None, 0
    for df in read_tsv_chunks(raw_path(split, src)):
        assert list(df.columns) == ["entity_id", "business_name", "business_address", "country"]
        if sample:
            df = df.iloc[: sample - n]
        tab = process_chunk(df)
        if writer is None:
            writer = pq.ParquetWriter(out, tab.schema, compression="zstd")
        writer.write_table(tab, row_group_size=250_000)
        n += len(df)
        del df, tab
        gc.collect()
        if sample and n >= sample:
            break
    writer.close()
    if not sample:
        assert n == EXPECTED_ROWS[(split, src)], (split, src, n)
        rid = pq.read_table(out, columns=["rid"]).column("rid")
        assert pc.count_distinct(rid).as_py() == n, "rid not unique"
        assert pc.all(pc.equal(pc.divide(rid, 10**10), src)).as_py()
    log(f"{split} S{src}: {n:,} rows -> {out} ({time.time() - t0:.0f}s)")


def convert_gt(sample):
    t0 = time.time()
    s1_parts, pair_s1, pair_m = [], [], []
    n = 0
    for df in read_tsv_chunks(raw_path("train", "gt")):
        assert list(df.columns) == ["source1_entity_id", "matched_entity_ids"]
        if sample:
            df = df.iloc[: sample - n]
        s1 = id_to_rid(pa.array(df["source1_entity_id"].values, pa.string()))
        lists = pc.split_pattern(pa.array(df["matched_entity_ids"].values, pa.string()), ",")
        flat = pc.list_flatten(lists)
        parent = pc.list_parent_indices(lists)
        keep = pc.greater(pc.utf8_length(flat), 0)
        flat, parent = flat.filter(keep), parent.filter(keep)
        s1_parts.append(pa.table({"s1_rid": s1, "n_match": pc.cast(
            pc.subtract(pc.list_value_length(lists), pc.cast(pc.equal(
                pc.list_element(lists, 0), ""), pa.int32())), pa.int32())}))
        pair_s1.append(pc.take(s1, parent))
        pair_m.append(id_to_rid(flat))
        n += len(df)
        if sample and n >= sample:
            break
    s1t = pa.concat_tables(s1_parts)
    pairs = pa.table({"s1_rid": pa.chunked_array(pair_s1), "m_rid": pa.chunked_array(pair_m)})
    if not sample:
        assert n == EXPECTED_ROWS[("train", "gt")], n
        assert pc.count_distinct(s1t.column("s1_rid")).as_py() == n
        assert pc.sum(s1t.column("n_match")).as_py() == len(pairs)
    pre = "sample_" if sample else ""
    pq.write_table(s1t, os.path.join(DATA, f"{pre}train_gt_s1.parquet"))
    pq.write_table(pairs, os.path.join(DATA, f"{pre}train_gt_pairs.parquet"))
    log(f"ground truth: {n:,} S1 rows, {len(pairs):,} true pairs ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--only", default="")  # e.g. "train:2"
    args = ap.parse_args()
    jobs = [("train", 1), ("train", 2), ("train", 3), ("test", 1), ("test", 2), ("test", 3)]
    if args.only:
        sp, s = args.only.split(":")
        jobs = [(sp, int(s))] if s != "gt" else []
    if not args.only or args.only.endswith("gt"):
        convert_gt(args.sample)
    for split, src in jobs:
        convert_source(split, src, args.sample)
