"""Post-hoc sanity checks of submission/output (my own, besides the official validator):
matches are a subset of candidates; per-country empty share and average list sizes."""
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from common import DATA, ROOT, log, pq_path

out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "submission", "output")
s1 = pq.read_table(pq_path("test", 1), columns=["entity_id", "country"]).to_pandas()


def read(name):
    d = pd.read_csv(os.path.join(out, name), sep="\t", dtype=str, na_filter=False, quoting=3)
    assert (d.iloc[:, 0].values == s1.entity_id.values).all(), "row order differs from test_source1"
    return d.iloc[:, 1].values


mt, cd = read("matching_results.tsv"), read("candidate_pairs.tsv")
bad = 0
for m, c in zip(mt, cd):
    if m and not set(m.split(",")) <= set(c.split(",")):
        bad += 1
nm = np.array([len(x.split(",")) if x else 0 for x in mt])
nc = np.array([len(x.split(",")) if x else 0 for x in cd])
lines = [f"# M7 output checks ({out})\n", f"S1 rows {len(s1):,}; rows whose matches are NOT a subset of candidates: {bad}",
         f"total candidates {nc.sum():,} (mean {nc.mean():.2f}, max {nc.max()}); total matches {nm.sum():,} "
         f"(mean {nm.mean():.3f}, max {nm.max()})", "",
         "| country | S1 | empty match share | mean matches | mean candidates |", "|---|---|---|---|---|"]
for c, g in s1.groupby("country"):
    i = g.index.values
    lines.append(f"| {c} | {len(g):,} | {(nm[i] == 0).mean():.4f} | {nm[i].mean():.3f} | {nc[i].mean():.2f} |")
# country of matched ids is not stored in the output; check via the pool files
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports", "m7_check.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
log("\n".join(lines))
