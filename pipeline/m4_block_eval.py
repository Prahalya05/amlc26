"""M4: run blocking for (a sample of) the VALIDATION S1 entities against the full train pool and
measure the recall ceiling.

Usage: python m4_block_eval.py NAME '{"cap":2000,"k":50,"types":[1,2,3,4,5,6,7]}' [n_s1_sample]
Appends a line to reports/blocking_experiments.md.

Metrics
  recall@k        share of true (S1, match) pairs that are among the first k candidates
  oracle F0.5@k   macro F0.5 if a perfect model picked exactly the true matches among the
                  first k candidates (singletons -> empty prediction). This is the ceiling
                  on what any matcher can score after this blocking.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from blocking import block
from common import DATA, WORK, log, pq_path

REP = os.path.join(WORK, "reports")
os.makedirs(REP, exist_ok=True)


def s1_rows_for_fold(split, folds, n_sample=0, seed=0):
    rid = pq.read_table(pq_path(split, 1), columns=["rid"]).column("rid").to_numpy()
    f = pd.read_parquet(os.path.join(DATA, "folds.parquet"))
    sel = f.s1_rid[f.fold.isin(folds)].values
    rows = np.flatnonzero(np.isin(rid, sel))
    if n_sample and n_sample < len(rows):
        rows = np.sort(np.random.default_rng(seed).choice(rows, n_sample, replace=False))
    return rows, rid


def evaluate(cand, rows, s1_rid, ks=(5, 10, 20, 30, 50, 100)):
    pairs = pd.read_parquet(os.path.join(DATA, "train_gt_pairs.parquet"))
    target = s1_rid[rows]
    pairs = pairs[pairs.s1_rid.isin(target)]
    cand = cand.assign(s1_rid=s1_rid[cand.s1_row.values])
    hit = cand.merge(pairs, on=["s1_rid", "m_rid"])
    ntrue = pairs.groupby("s1_rid").size().reindex(target, fill_value=0).values.astype(float)
    res = {}
    for k in ks:
        if k > cand["rank"].max() + 1:
            continue
        h = hit[hit["rank"] < k].groupby("s1_rid").size().reindex(target, fill_value=0).values
        f = np.where(ntrue == 0, 1.0, 1.25 * h / np.maximum(h + 0.25 * ntrue, 1e-9))
        res[k] = dict(recall=h.sum() / ntrue.sum(), oracle=f.mean(),
                      avg_cands=(cand["rank"] < k).sum() / len(target))
    per_s1 = cand.groupby("s1_row").size()
    res["max_cands"] = int(per_s1.max())
    res["s1_without_cands"] = float(1 - len(per_s1) / len(target))
    return res


if __name__ == "__main__":
    name = sys.argv[1]
    cfg = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    if "type_w" in cfg:
        cfg["type_w"] = {int(k): v for k, v in cfg["type_w"].items()}
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 50_000
    rows, s1_rid = s1_rows_for_fold("train", [0], n)
    log(f"experiment {name}: {len(rows):,} validation S1, cfg={cfg}")
    cand, st = block("train", rows, cfg, return_stats=True)
    res = evaluate(cand, rows, s1_rid)
    line = [f"| {name} | {json.dumps(cfg)} | {len(rows):,} | {st['postings']:,} | {st['seconds']}s |"]
    for k in (5, 10, 20, 30, 50, 100):
        if k in res:
            r = res[k]
            line.append(f" {r['recall']:.4f} / {r['oracle']:.4f} / {r['avg_cands']:.1f} |")
        else:
            line.append(" - |")
    line.append(f" {res['max_cands']} | {res['s1_without_cands']:.4f} |")
    path = os.path.join(REP, "blocking_experiments.md")
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("Cells: recall / oracle F0.5 / avg candidates per S1, at top-k.\n\n"
                     "| name | cfg | #S1 | postings | time | k=5 | k=10 | k=20 | k=30 | k=50 | k=100 | max cands | S1 w/o cands |\n"
                     "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("".join(line) + "\n")
    log("".join(line))
    if len(sys.argv) > 4:
        cand.to_parquet(os.path.join(DATA, f"cand_{name}.parquet"), index=False)
