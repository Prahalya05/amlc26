"""M4 diagnostics: why are true pairs missing from the top-k candidates?
Uses data/cand_diag10k.parquet (saved by m4_block_eval.py ... save). Writes reports/blocking_diag.md."""
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from blocking import pool_info
from common import DATA, WORK, log, pq_path
from m4_block_eval import s1_rows_for_fold

rows, s1_rid = s1_rows_for_fold("train", [0], 10_000)
cand = pd.read_parquet(os.path.join(DATA, "cand_diag10k.parquet"))
cand["s1_rid"] = s1_rid[cand.s1_row.values]
pairs = pd.read_parquet(os.path.join(DATA, "train_gt_pairs.parquet"))
pairs = pairs[pairs.s1_rid.isin(s1_rid[rows])]
pairs = pairs.merge(cand[["s1_rid", "m_rid", "rank", "score"]], on=["s1_rid", "m_rid"], how="left")
pool_rid, n2 = pool_info("train")
o = np.argsort(pool_rid)  # rids are not sorted in file order
pairs["pidx"] = o[np.searchsorted(pool_rid, pairs.m_rid.values, sorter=o)]
assert (pool_rid[pairs.pidx.values] == pairs.m_rid.values).all()
o1 = np.argsort(s1_rid)
pairs["s1_row"] = o1[np.searchsorted(s1_rid, pairs.s1_rid.values, sorter=o1)]
assert (s1_rid[pairs.s1_row.values] == pairs.s1_rid.values).all()

# keys of the S1 rows and of the matched pool rows
k1 = pq.read_table(os.path.join(DATA, "keys_train_s1.parquet")).to_pandas()
k1 = k1[k1.row.isin(pairs.s1_row.unique())]
kp = []
want = np.unique(pairs.pidx.values)
for s, off in ((2, 0), (3, n2)):
    pf = pq.ParquetFile(os.path.join(DATA, f"keys_train_s{s}.parquet"))
    for i in range(pf.num_row_groups):
        t = pf.read_row_group(i).to_pandas()
        t["row"] = t.row.astype(np.int64) + off
        kp.append(t[np.isin(t.row.values, want)])
        del t
kp = pd.concat(kp)
# document frequency of the involved keys over the whole pool
need = np.unique(np.concatenate([k1.key.values, kp.key.values]))
df = np.zeros(len(need), np.int64)
for s in (2, 3):
    pf = pq.ParquetFile(os.path.join(DATA, f"keys_train_s{s}.parquet"))
    for i in range(pf.num_row_groups):
        key = pf.read_row_group(i, columns=["key"]).column("key").to_numpy()
        pos = np.searchsorted(need, key)
        pos[pos >= len(need)] = 0
        hit = need[pos] == key
        df += np.bincount(pos[hit], minlength=len(need))
dfs = pd.Series(df, index=need)

shared = pairs[["s1_row", "pidx", "rank"]].merge(k1.rename(columns={"row": "s1_row"}), on="s1_row").merge(
    kp.rename(columns={"row": "pidx"}), on=["pidx", "key"], suffixes=("", "_p"))
shared["df"] = dfs.reindex(shared.key.values).values
g = shared.groupby(["s1_row", "pidx"])
summ = pd.DataFrame({"n_shared": g.size(), "min_df": g.df.min(),
                     "n_allowed": g.df.apply(lambda x: int((x <= 2000).sum()))})
pairs = pairs.merge(summ.reset_index(), on=["s1_row", "pidx"], how="left").fillna(
    {"n_shared": 0, "n_allowed": 0})
pairs["miss20"] = ~(pairs["rank"] < 20)
L = ["# Blocking diagnostics (10k validation S1, cap 2000)\n",
     f"true pairs {len(pairs):,}; missed at k=20: {pairs.miss20.mean():.4f}; not in top-100: "
     f"{pairs['rank'].isna().mean():.4f}\n",
     "Among pairs missed at k=20:",
     f"- share no key at all: {(pairs[pairs.miss20].n_shared == 0).mean():.4f}",
     f"- share keys, but all above cap (df>2000): "
     f"{((pairs[pairs.miss20].n_shared > 0) & (pairs[pairs.miss20].n_allowed == 0)).mean():.4f}",
     f"- have >=1 allowed shared key (ranked too low): {(pairs[pairs.miss20].n_allowed > 0).mean():.4f}",
     "\nShared key types among missed pairs (share of missed pairs having that type shared):"]
ms = shared.merge(pairs.loc[pairs.miss20, ["s1_row", "pidx"]], on=["s1_row", "pidx"])
for kt, grp in ms.groupby("kt"):
    L.append(f"- type {kt}: {grp[['s1_row', 'pidx']].drop_duplicates().shape[0] / pairs.miss20.sum():.4f} "
             f"(median df {grp.df.median():.0f})")
L.append("\nShared key types among FOUND pairs (rank<20):")
fs = shared.merge(pairs.loc[~pairs.miss20, ["s1_row", "pidx"]], on=["s1_row", "pidx"])
for kt, grp in fs.groupby("kt"):
    L.append(f"- type {kt}: {grp[['s1_row', 'pidx']].drop_duplicates().shape[0] / (~pairs.miss20).sum():.4f} "
             f"(median df {grp.df.median():.0f})")
# examples of misses
samp = pairs[pairs.miss20].sample(40, random_state=0)


def fetch(src, rids):
    import pyarrow as pa
    import pyarrow.compute as pc
    w = pa.array(np.unique(rids))
    parts = [b.filter(pc.is_in(b.column("rid"), w)) for b in
             pq.ParquetFile(pq_path("train", src)).iter_batches(500_000, columns=["rid", "name_n", "addr_n"])]
    return pa.Table.from_batches(parts).to_pandas().set_index("rid")


t1 = fetch(1, samp.s1_rid.values)
tp = {s: fetch(s, samp.m_rid.values) for s in (2, 3)}
L.append("\n## 40 missed pairs (rank, #shared keys, #allowed)\n")
for r in samp.itertuples():
    a = t1.loc[r.s1_rid]
    b = tp[r.m_rid // 10**10].loc[r.m_rid]
    L.append(f"- rank {r.rank} sh {r.n_shared:.0f}/{r.n_allowed:.0f}: `{a.name_n}` / `{a.addr_n}` <-> "
             f"`{b.name_n}` / `{b.addr_n}`")
with open(os.path.join(WORK, "reports", "blocking_diag.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(L) + "\n")
log("diag written")
