"""M7: full test inference.

Steps (each S1 batch independently, so memory stays bounded):
  block (test S1 batch vs FULL test S2+S3 pool) -> features -> LightGBM probability
then, over all batches: one-owner rule + thresholds from models/decision.json, and write
  <out>/matching_results.tsv  (final matches, subset of the candidates)
  <out>/candidate_pairs.tsv   (exactly the pairs the model scored)
Usage: python m7_infer.py OUT_DIR N_BATCHES [TIME_BUDGET_S]
  Batches already predicted (data/test_pred_{b}.parquet) are skipped, and no new batch is started
  after TIME_BUDGET_S seconds, so the script can simply be re-run until it reports "all batches done";
  the TSVs are written only then.
"""
import gc
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from blocking import block
from common import DATA, ROOT, WORK, log, pq_path, rid_to_id
from features2 import build_features
from m6_decide import one_owner

MODELS = os.path.join(WORK, "models")
BLOCK_CFG = json.load(open(os.path.join(WORK, "block_cfg.json"))) if os.path.exists(
    os.path.join(WORK, "block_cfg.json")) else {}
if "type_w" in BLOCK_CFG:
    BLOCK_CFG["type_w"] = {int(k): v for k, v in BLOCK_CFG["type_w"].items()}


def run_batches(n_batches, budget=1e9):
    tstart = time.time()
    s1_rid = pq.read_table(pq_path("test", 1), columns=["rid"]).column("rid").to_numpy()
    model = lgb.Booster(model_file=os.path.join(MODELS, "lgb_model.txt"))
    feats = open(os.path.join(MODELS, "feature_list.txt"), encoding="utf-8").read().split("\n")
    for b, rows in enumerate(np.array_split(np.arange(len(s1_rid)), n_batches)):
        outp = os.path.join(DATA, f"test_pred_{b}.parquet")
        if os.path.exists(outp):
            continue
        if time.time() - tstart > budget:
            log("time budget used up; re-run to continue")
            return False
        t0 = time.time()
        cand = block("test", rows, BLOCK_CFG)
        fp = os.path.join(DATA, f"test_feats_tmp.parquet")
        build_features("test", cand, fp)
        del cand
        gc.collect()
        parts = []
        with pq.ParquetFile(fp) as pf:  # closed afterwards so the temp file can be deleted
            for i in range(pf.num_row_groups):
                df = pf.read_row_group(i, columns=feats + ["s1_row", "m_rid"]).to_pandas()
                p = model.predict(df[feats].to_numpy(np.float32), num_threads=10)
                parts.append(pd.DataFrame({"s1_rid": s1_rid[df.s1_row.values], "m_rid": df.m_rid.values,
                                           "p": p.astype(np.float32)}))
                del df
        pd.concat(parts, ignore_index=True).to_parquet(outp, index=False)
        os.remove(fp)
        log(f"batch {b + 1}/{n_batches}: {len(rows):,} S1, done in {time.time() - t0:.0f}s")
        gc.collect()
    log("all batches done")
    return True


def write_lists(f, s1_ids, lo, hi, pos, m):
    """Write one line per S1 position lo..hi-1: 'S1-id<TAB>id,id,...' (empty when it has no rows).
    pos = S1 position of each row (absolute), m = pool rid of each row. Rows are grouped by S1 with
    one direct sort of packed (position, row) values."""
    packed = np.sort(((pos - lo).astype(np.uint64) << np.uint64(32)) | np.arange(len(pos), dtype=np.uint64))
    order = (packed & np.uint64(0xFFFFFFFF)).astype(np.int64)
    pp, mm = (packed >> np.uint64(32)).astype(np.int64), m[order]
    del packed, order
    assert not pd.Series(pp * 10**11 + mm).duplicated().any(), "duplicate id in a list"
    step = 100_000
    for c in range(0, hi - lo, step):
        x, y = np.searchsorted(pp, [c, c + step])
        g = pd.Series(rid_to_id(mm[x:y])).groupby(pp[x:y]).agg(",".join)
        g = g.reindex(np.arange(c, min(c + step, hi - lo)), fill_value="")
        f.writelines(f"{s1_ids[lo + i]}\t{v}\n" for i, v in zip(g.index, g.values))


def write_outputs(out_dir, n_batches):
    """RAM-bounded: candidate_pairs.tsv is streamed batch by batch (each batch is a contiguous range of
    S1 positions); only rows with p >= t are kept globally for the one-owner rule and thresholds."""
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    s1_rid = pq.read_table(pq_path("test", 1), columns=["rid"]).column("rid").to_numpy()
    s1_ids = pq.read_table(pq_path("test", 1), columns=["entity_id"]).column("entity_id").to_pylist()
    dec = json.load(open(os.path.join(MODELS, "decision.json")))
    # Rows with p < t can never be selected, and (because the gate g >= t) dropping them first changes
    # neither the one-owner winner of any selectable row nor any gate decision -> same result, less RAM.
    assert dec["g"] >= dec["t"]
    ranges = [(int(r[0]), int(r[-1]) + 1) for r in np.array_split(np.arange(len(s1_rid)), n_batches)]
    keep, n_cand, n_s1_seen = [], 0, 0
    with open(os.path.join(out_dir, "candidate_pairs.tsv"), "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for b, (lo, hi) in enumerate(ranges):
            pred = pd.read_parquet(os.path.join(DATA, f"test_pred_{b}.parquet"))
            pos = pd.Index(s1_rid[lo:hi]).get_indexer(pred.s1_rid.values)
            assert (pos >= 0).all()
            write_lists(f, s1_ids, lo, hi, pos + lo, pred.m_rid.values)
            n_cand += len(pred)
            n_s1_seen += pred.s1_rid.nunique()
            keep.append(pred[pred.p >= dec["t"]])
            del pred, pos
            gc.collect()
    log(f"candidate_pairs.tsv written: {n_cand:,} candidates ({time.time() - t0:.0f}s)")
    sel = pd.concat(keep, ignore_index=True)
    del keep
    sel = one_owner(sel) if dec["one_owner"] else sel
    pmax = sel.groupby("s1_rid")["p"].transform("max")
    sel = sel[(pmax >= dec["g"]) & (sel.p >= dec["r"] * pmax)]
    log(f"candidates {n_cand:,}, matches {len(sel):,}; S1 with >=1 match "
        f"{sel.s1_rid.nunique() / len(s1_rid):.4f}")
    pos = pd.Index(s1_rid).get_indexer(sel.s1_rid.values)
    assert (pos >= 0).all()
    with open(os.path.join(out_dir, "matching_results.tsv"), "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        write_lists(f, s1_ids, 0, len(s1_rid), pos, sel.m_rid.values)
    log(f"outputs written to {out_dir} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "submission", "output")
    nb = int(sys.argv[2]) if len(sys.argv) > 2 else 18
    budget = float(sys.argv[3]) if len(sys.argv) > 3 else 1e9
    if run_batches(nb, budget):
        write_outputs(out, nb)
