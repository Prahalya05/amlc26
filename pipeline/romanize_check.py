"""Rule-6 check: does Indic romanisation separate true cross-script pairs from non-matches?

Sample true train pairs where the S1 name is Latin and the matched S2/S3 name is non-Latin, and
compare name similarity (S1 core name vs romanised pool core name) for
  (a) those TRUE pairs,
  (b) RANDOM non-matching pairs: the same S1 names with the pool names shuffled across S1s,
  (c) HARD non-matching pairs: non-Latin blocking candidates of validation S1s that are not matches.
For reference the same is computed for the consonant skeleton (skeleton.py) and for the raw,
un-romanised strings. Separation is measured by AUC (0.5 = useless, 1.0 = perfect).
Writes reports/romanize_check.md.
"""
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from rapidfuzz import fuzz, process
from sklearn.metrics import roc_auc_score

from common import DATA, WORK, log, pq_path
from keys import skel_strings
from romanize import romanize

N = 20_000
rng = np.random.default_rng(0)


def fetch(src, rids, cols=("rid", "name_core", "name_n", "name_nonlatin")):
    w = pa.array(np.unique(rids))
    parts = [b.filter(pc.is_in(b.column("rid"), w))
             for b in pq.ParquetFile(pq_path("train", src)).iter_batches(500_000, columns=list(cols))]
    return pa.Table.from_batches(parts).to_pandas().set_index("rid")


pairs = pd.read_parquet(os.path.join(DATA, "train_gt_pairs.parquet"))
nl_rids = np.concatenate([
    (lambda t: t.column("rid").to_numpy()[t.column("name_nonlatin").to_numpy(zero_copy_only=False)])(
        pq.read_table(pq_path("train", s), columns=["rid", "name_nonlatin"])) for s in (2, 3)])
true = pairs[pairs.m_rid.isin(nl_rids)].sample(N, random_state=0)
nl = pd.concat([fetch(s, true.m_rid.values) for s in (2, 3)])
s1 = fetch(1, true.s1_rid.values)
a = s1.loc[true.s1_rid.values]
b = nl.loc[true.m_rid.values]
# random non-matches: shuffle the pool side (drop the rare accidental true pair)
perm = rng.permutation(N)
rand_ok = true.s1_rid.values != true.s1_rid.values[perm]
# hard non-matches: non-Latin blocking candidates that are not true matches
cand = pd.read_parquet(os.path.join(DATA, "cand_diag10k.parquet"))
from m4_block_eval import s1_rows_for_fold  # noqa: E402

_, s1_rid = s1_rows_for_fold("train", [0], 0)
cand["s1_rid"] = s1_rid[cand.s1_row.values]
cand = cand.merge(pairs.assign(y=1), on=["s1_rid", "m_rid"], how="left")
hard = cand[cand.y.isna()]
hp = pd.concat([fetch(s, hard.m_rid.values) for s in (2, 3)])
hard = hard[hard.m_rid.isin(hp.index[hp.name_nonlatin])]
hard = hard.sample(min(N, len(hard)), random_state=0)
ha = fetch(1, hard.s1_rid.values).loc[hard.s1_rid.values]
hb = hp.loc[hard.m_rid.values]
log(f"true cross-script pairs {N:,}, random negatives {rand_ok.sum():,}, hard negatives {len(hard):,}")


def sims(c1, c2, n1, n2):
    r2 = [romanize(x) for x in c2]
    k1 = skel_strings(pa.array(n1, pa.string())).to_pylist()
    k2 = skel_strings(pa.array(n2, pa.string())).to_pylist()
    W = dict(workers=-1, dtype=np.float32)
    return {
        "roman_tset": process.cpdist(c1, r2, scorer=fuzz.token_set_ratio, **W),
        "roman_ratio": process.cpdist(c1, r2, scorer=fuzz.ratio, **W),
        "skel_ratio": process.cpdist(k1, k2, scorer=fuzz.ratio, **W),
        "raw_ratio": process.cpdist(c1, c2, scorer=fuzz.ratio, **W),
    }


S_true = sims(list(a.name_core), list(b.name_core), list(a.name_n), list(b.name_n))
S_rand = sims(list(a.name_core[rand_ok]), list(b.name_core.values[perm][rand_ok]),
              list(a.name_n[rand_ok]), list(b.name_n.values[perm][rand_ok]))
S_hard = sims(list(ha.name_core), list(hb.name_core), list(ha.name_n), list(hb.name_n))

qs = [0.1, 0.25, 0.5, 0.75, 0.9]
L = ["# Rule 6: Indic romanisation check (train)\n",
     f"Pairs: {N:,} true cross-script pairs (S1 Latin, match non-Latin), {rand_ok.sum():,} random "
     f"non-matching pairs (pool names shuffled), {len(hard):,} hard non-matching pairs (non-Latin "
     f"blocking candidates of validation S1s).\n",
     "Similarity = rapidfuzz score 0-100 between the S1 core name and the transformed pool core name.\n",
     "| feature | set | mean | p10 | p25 | p50 | p75 | p90 |", "|---|---|---|---|---|---|---|---|"]
for k in S_true:
    for nm, S in (("TRUE", S_true), ("random", S_rand), ("hard", S_hard)):
        v = S[k]
        L.append(f"| {k} | {nm} | {v.mean():.1f} | " + " | ".join(f"{np.quantile(v, q):.0f}" for q in qs) + " |")
L += ["", "| feature | AUC true vs random | AUC true vs hard |", "|---|---|---|"]
auc = {}
for k in S_true:
    ar = roc_auc_score(np.r_[np.ones(N), np.zeros(rand_ok.sum())], np.r_[S_true[k], S_rand[k]])
    ah = roc_auc_score(np.r_[np.ones(N), np.zeros(len(hard))], np.r_[S_true[k], S_hard[k]])
    auc[k] = (ar, ah)
    L.append(f"| {k} | {ar:.4f} | {ah:.4f} |")
# histogram of the main feature (10-point bins) for the human reader
L += ["", "Histogram of roman_tset (share of pairs per 10-point bin):", "",
      "| bin | TRUE | random | hard |", "|---|---|---|---|"]
bins = np.arange(0, 101, 10)
h = {nm: np.histogram(S["roman_tset"], bins=bins)[0] / len(S["roman_tset"])
     for nm, S in (("t", S_true), ("r", S_rand), ("h", S_hard))}
for i in range(len(bins) - 1):
    L.append(f"| {bins[i]}-{bins[i + 1]} | {h['t'][i]:.3f} | {h['r'][i]:.3f} | {h['h'][i]:.3f} |")
L += ["", "## 30 true pairs: S1 core name | pool core name -> romanised (score)", ""]
r2 = [romanize(x) for x in b.name_core[:30]]
for x, y, z, s in zip(a.name_core[:30], b.name_core[:30], r2, S_true["roman_tset"][:30]):
    L.append(f"- `{x}` | `{y}` -> `{z}` ({s:.0f})")
with open(os.path.join(WORK, "reports", "romanize_check.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(L) + "\n")
log("AUC (true vs random, true vs hard): " + ", ".join(f"{k} {v[0]:.3f}/{v[1]:.3f}" for k, v in auc.items()))
