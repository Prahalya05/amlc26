"""M10: v5 = record-centric assignment. Everything new lives in data/v5/, models/v5/, reports/m10_v5.md.
Old paths (v1/v3 data, models, outputs) are only read.

Part A (measure, from existing v3 data - no new blocking pass):
  measure        owner recall@k in the v3 reverse lists (top-30 S1s of every train pool record), by segment
  rc             record-centric decision with the CURRENT v3 model: each pool record goes to its highest-p S1
                 (over all 2.2M train S1s, out-of-fold p from v3full) and is kept if p >= t
                 (optionally a separate t for empty-address records); t tuned on half 1 of fold 0,
                 scored on half 2 - vs v3's gate/relative rule on the same frame
  write_rc OUT   test output with the record-centric decision (candidates = the v3 candidate file)
"""
import gc
import json
import os
import shutil
import sys
import time

os.environ.setdefault("AMLC_MAXMB", "3900")
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import blocking_v3 as B
import m9_cands as M9
from common import DATA, ROOT, WORK, log, mem_mb, pq_path

V5 = os.path.join(DATA, "v5")
MV5 = os.path.join(WORK, "models", "v5")
REP = os.path.join(WORK, "reports", "m10_v5.md")
for _d in (V5, MV5):
    os.makedirs(_d, exist_ok=True)


def report(lines):
    with open(REP, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log("\n".join(lines))


def save_stats(name, d):
    d = {**d, "peak_ram_mb": round(mem_mb()[1])}
    json.dump(d, open(os.path.join(V5, f"stats_{name}.json"), "w"), indent=1, default=float)


def pool_attrs(split):
    """Per pool record (pool order: S2 rows then S3 rows): empty normalised address, non-Latin name,
    country, source (2/3)."""
    parts = [pq.read_table(pq_path(split, s), columns=["addr_n", "name_nonlatin", "country"]).to_pandas()
             for s in (2, 3)]
    n2 = len(parts[0])
    d = pd.concat(parts, ignore_index=True)
    return pd.DataFrame({"empty_addr": (d.addr_n.str.len() == 0).values, "nonlatin": d.name_nonlatin.values.astype(bool),
                         "country": d.country.astype("category").values,
                         "src": np.where(np.arange(len(d)) < n2, 2, 3).astype(np.int8)})


def owner_arrays(split="train"):
    """owner S1 row of every pool record (-1 = none) from the ground truth."""
    gt = pd.read_parquet(os.path.join(DATA, "train_gt_pairs.parquet"))
    pidx = pd.Index(M9.pool_rid_arr(split)).get_indexer(gt.m_rid.values)
    srow = pd.Index(M9.s1_rid_arr(split)).get_indexer(gt.s1_rid.values)
    assert (pidx >= 0).all() and (srow >= 0).all()
    own = np.full(B.n_rows(split, 2) + B.n_rows(split, 3), -1, np.int64)
    own[pidx] = srow
    return own


KS = [1, 2, 3, 5, 10, 20, 30]


def recall_table(rank, seg):
    """rank: owner's rank 1..30 (99 = absent) per owned record; seg: DataFrame of segment columns."""
    rows = []

    def one(name, m):
        r = rank[m]
        rows.append([name, f"{m.sum():,}"] + [f"{(r <= k).mean():.4f}" for k in KS] + [f"{(r > 30).mean():.4f}"])
    one("all", np.ones(len(rank), bool))
    one("empty address", seg.empty_addr.values)
    one("address present", ~seg.empty_addr.values)
    one("cross-script (pool name non-Latin)", seg.nonlatin.values)
    one("same script", ~seg.nonlatin.values)
    for c in seg.country.cat.categories:
        one(f"country {c}", (seg.country == c).values)
    one("S2", seg.src.values == 2)
    one("S3", seg.src.values == 3)
    return ["| segment | records | " + " | ".join(f"@{k}" for k in KS) + " | absent from top-30 |",
            "|---|---|" + "---|" * (len(KS) + 1)] + ["| " + " | ".join(r) + " |" for r in rows]


# ------------------------------------------------------------------ A1
def stage_measure():
    t0 = time.time()
    own = owner_arrays()
    owned = np.flatnonzero(own >= 0)
    rank = np.full(len(own), 99, np.int16)
    d = os.path.join(M9.V3, "rev30_train")
    for f in sorted(os.listdir(d)):
        t = pd.read_parquet(os.path.join(d, f))
        m = own[t.pidx.values] == t.s1_row.values
        rank[t.pidx.values[m]] = t.rrank.values[m].astype(np.int16) + 1
        del t
    seg = pool_attrs("train").iloc[owned].reset_index(drop=True)
    r = rank[owned]
    np.save(os.path.join(V5, "owner_rank_train.npy"), rank)
    L = ["# M10 / v5: record-centric assignment\n",
         "## A1. Owner recall@k in the v3 reverse lists (train, every pool record that has a true owner)\n",
         "Each of the 10.3M train pool records has a reverse list = its top-30 train S1s by reverse blocking score "
         "(v3 `rev30_train`). recall@k = share of owned records whose owner is ranked <= k. "
         "'Empty address' = normalised address is empty.\n"] + recall_table(r, seg)
    report(L)
    save_stats("measure", {"seconds": round(time.time() - t0, 1), "owned": len(owned),
                           "recall_at": {k: float((r <= k).mean()) for k in KS}})


# ------------------------------------------------------------------ A2
def f05_per_s1(idx, lab, n, ntrue):
    npred = np.bincount(idx, minlength=n).astype(np.float64)
    tp = np.bincount(idx, weights=lab, minlength=n)
    return np.where(ntrue == 0, (npred == 0).astype(float),
                    np.divide(1.25 * tp, npred + 0.25 * ntrue, out=np.zeros_like(tp), where=(npred + 0.25 * ntrue) > 0))


class Frame:
    """Owner rows (after the one-owner argmax over all S1s) of a set of S1s, with labels, for fast scoring."""

    def __init__(self, d, ids, ntrue, empty):
        self.idx = pd.Index(ids).get_indexer(d.s1_rid.values)
        assert (self.idx >= 0).all()
        self.p, self.lab = d.p.values, d.label.values.astype(np.float64)
        self.pmax = d.groupby("s1_rid").p.transform("max").values
        self.empty = empty[d.pidx.values]
        self.n, self.ntrue = len(ids), ntrue

    def rc(self, t, te=None):
        te = t if te is None else te
        sel = np.where(self.empty, self.p >= te, self.p >= t)
        return f05_per_s1(self.idx[sel], self.lab[sel], self.n, self.ntrue).mean(), sel

    def rule(self, g, t, r):
        sel = (self.pmax >= g) & (self.p >= t) & (self.p >= r * self.pmax)
        return f05_per_s1(self.idx[sel], self.lab[sel], self.n, self.ntrue).mean(), sel

    def pr(self, sel):
        """micro precision / recall of a selection"""
        tp = self.lab[sel].sum()
        return tp / max(sel.sum(), 1), tp / self.ntrue.sum()


T_GRID = np.round(np.r_[np.arange(0.01, 0.1, 0.01), np.arange(0.1, 0.96, 0.02), [0.97, 0.98, 0.99]], 3)


def tune_rc(F, split_t=False):
    best = (-1, None)
    for t in T_GRID:
        if split_t:
            for te in T_GRID:
                s = F.rc(t, te)[0]
                if s > best[0]:
                    best = (s, (float(t), float(te)))
        else:
            s = F.rc(t)[0]
            if s > best[0]:
                best = (s, (float(t), None))
    return best


def fold0_frames(files, split_empty):
    """Full competition over all train S1s -> owner rows of the fold-0 S1s, split into halves."""
    import m6b_expected_f as EF
    s1rid, pool_rid = M9.s1_rid_arr("train"), M9.pool_rid_arr("train")
    T = M9.target_rows()
    rows0 = np.sort(T.s1_row[T.fold == 0].values)
    val_ids = np.sort(s1rid[rows0])
    gt = pd.read_parquet(os.path.join(DATA, "train_gt_pairs.parquet"))
    pairs = gt[gt.s1_rid.isin(val_ids)].reset_index(drop=True)
    del gt
    d = M9.full_competition(files, M9.sample_mask(0), len(pool_rid))
    d["s1_rid"] = s1rid[d.s1_row.values]
    d["m_rid"] = pool_rid[d.pidx.values]
    lab = pd.Series(1, index=pd.MultiIndex.from_frame(pairs))
    d["label"] = lab.reindex(pd.MultiIndex.from_arrays([d.s1_rid.values, d.m_rid.values]), fill_value=0).values
    h1 = EF.split_half(val_ids)
    out = {}
    for name, ids in (("all", val_ids), ("h1", val_ids[h1]), ("h2", val_ids[~h1])):
        dd = d[d.s1_rid.isin(ids)].reset_index(drop=True)
        nt = pairs[pairs.s1_rid.isin(ids)].groupby("s1_rid").size().reindex(ids, fill_value=0).values.astype(float)
        out[name] = Frame(dd, ids, nt, split_empty)
    return out


def stage_rc():
    t0 = time.time()
    dec = json.load(open(os.path.join(M9.MV3, "decision.json")))
    files = sorted(os.path.join(M9.V3, "v3full", f) for f in os.listdir(os.path.join(M9.V3, "v3full"))
                   if f.startswith("pred_") and f.endswith(".parquet"))
    assert len(files) == M9.NB["train"]
    empty = pool_attrs("train").empty_addr.values
    Fr = fold0_frames(files, empty)
    res = {}

    def row(name, F2, sel2, s_h2, params, s_all=None):
        p, r = F2.pr(sel2)
        e = F2.empty & (F2.lab > 0)
        res[name] = dict(half2=s_h2, all=s_all, params=params, precision=p, recall=r)
        return (f"| {name} | {params} | {'' if s_all is None else f'{s_all:.5f}'} | **{s_h2:.5f}** | {p:.4f} | {r:.4f} | "
                f"{sel2[e].sum() / max(Fr['h2_true_empty'], 1):.4f} |")

    L = ["\n## A2. Record-centric decision with the current v3 model (fold 0, full competition)\n",
         "Frame: v3full out-of-fold p for all 2.2M train S1s; every pool record is kept only under its highest-p S1 "
         "(one-owner argmax - identical in all rows). *v3 rule*: then per S1 gate g on the best p, absolute t, "
         "relative r (models/v3/decision.json). *record-centric*: then keep the record if its p >= t (no per-S1 "
         "gate / relative cut); *+ t_empty*: a separate threshold for empty-address pool records. "
         "Thresholds tuned on half 1 of fold 0, scored on half 2 (half split as in m9 decide).\n"]
    # recall of empty-address true pairs on half 2: count them from the ground truth
    import m6b_expected_f as EF
    s1rid, pool_rid = M9.s1_rid_arr("train"), M9.pool_rid_arr("train")
    T = M9.target_rows()
    val_ids = np.sort(s1rid[T.s1_row[T.fold == 0].values])
    h2_ids = val_ids[~EF.split_half(val_ids)]
    gt = pd.read_parquet(os.path.join(DATA, "train_gt_pairs.parquet"))
    g2 = gt[gt.s1_rid.isin(h2_ids)]
    Fr["h2_true_empty"] = int(empty[pd.Index(pool_rid).get_indexer(g2.m_rid.values)].sum())
    del gt, g2
    L += ["| decision | params (tuned on half 1) | fold 0 (tuned on fold 0) | half 2 (half-1-tuned) | precision h2 | "
          "recall h2 | empty-address recall h2 |", "|---|---|---|---|---|---|---|"]
    F1, F2, FA = Fr["h1"], Fr["h2"], Fr["all"]
    # v3 rule, thresholds from decision.json (tuned on all of fold 0) and re-tuned on half 1
    import itertools
    s_all, _ = FA.rule(dec["g"], dec["t"], dec["r"])
    s2, sel2 = F2.rule(dec["g"], dec["t"], dec["r"])
    L.append(row("v3 rule (decision.json, tuned on all fold 0)", F2, sel2, s2, (dec["g"], dec["t"], dec["r"]), s_all))
    best = (-1, None)
    for g, r in itertools.product(np.round(np.arange(0.3, 0.9, 0.02), 2), np.round(np.arange(0.4, 0.9, 0.025), 3)):
        s = F1.rule(g, dec["t"], r)[0]
        if s > best[0]:
            best = (s, (float(g), dec["t"], float(r)))
    s2, sel2 = F2.rule(*best[1])
    L.append(row("v3 rule (re-tuned on half 1)", F2, sel2, s2, best[1]))
    for split_t in (False, True):
        s1_, (t, te) = tune_rc(F1, split_t)
        s2, sel2 = F2.rc(t, te)
        sa = tune_rc(FA, split_t)
        name = "record-centric" + (" + t_empty" if split_t else "")
        L.append(row(name, F2, sel2, s2, (t, te), sa[0]))
        res[name]["fold0_tuned_params"] = sa[1]
    # hybrid: record-centric threshold + the v3 gate (for information)
    best = (-1, None)
    for g in np.round(np.arange(0.0, 0.9, 0.05), 2):
        for t in T_GRID:
            sel = (F1.pmax >= g) & (F1.p >= t)
            s = f05_per_s1(F1.idx[sel], F1.lab[sel], F1.n, F1.ntrue).mean()
            if s > best[0]:
                best = (s, (float(g), float(t)))
    g, t = best[1]
    sel2 = (F2.pmax >= g) & (F2.p >= t)
    s2 = f05_per_s1(F2.idx[sel2], F2.lab[sel2], F2.n, F2.ntrue).mean()
    L.append(row("record-centric + per-S1 gate g (hybrid, info)", F2, sel2, s2, (g, t)))
    v3h2 = res["v3 rule (decision.json, tuned on all fold 0)"]["half2"]
    rc_name = max(("record-centric", "record-centric + t_empty"), key=lambda k: res[k]["half2"])
    better = res[rc_name]["half2"] > v3h2
    L.append(f"\nBest record-centric variant: **{rc_name}**, half 2 {res[rc_name]['half2']:.5f} vs v3 rule "
             f"{v3h2:.5f} ({res[rc_name]['half2'] - v3h2:+.5f}) -> "
             f"{'BETTER: write output_v4b' if better else 'not better: no output_v4b'}.\n")
    report(L)
    json.dump({"better": bool(better), "variant": rc_name, "params_fold0": res[rc_name]["fold0_tuned_params"],
               "results": res}, open(os.path.join(MV5, "rc_decision.json"), "w"), indent=1, default=float)
    save_stats("rc", {"seconds": round(time.time() - t0, 1)})


def stage_write_rc(out_dir):
    """Test output with the record-centric decision: one-owner argmax over all test S1s, then p >= t
    (t_empty for empty-address records). candidate_pairs.tsv = copy of the v3 candidate file (same lists)."""
    from m7_infer import write_lists
    t0 = time.time()
    out_dir = os.path.abspath(out_dir)
    dec = json.load(open(os.path.join(MV5, "rc_decision.json")))
    assert dec["better"], "record-centric did not beat v3 on half 2"
    assert not os.path.exists(out_dir), f"{out_dir} exists - not overwriting"
    t, te = dec["params_fold0"]
    te = t if te is None else te
    tmp = out_dir + "_tmp_writing"
    os.makedirs(tmp, exist_ok=True)
    s1rid, pool_rid = M9.s1_rid_arr("test"), M9.pool_rid_arr("test")
    s1_ids = pq.read_table(pq_path("test", 1), columns=["entity_id"]).column("entity_id").to_pylist()
    empty = pool_attrs("test").empty_addr.values
    sel = M9.full_competition(M9.test_files(), np.ones(len(s1rid), bool), len(pool_rid))
    sel = sel[np.where(empty[sel.pidx.values], sel.p.values >= te, sel.p.values >= t)]
    with open(os.path.join(tmp, "matching_results.tsv"), "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        write_lists(f, s1_ids, 0, len(s1rid), sel.s1_row.values.astype(np.int64), pool_rid[sel.pidx.values])
    shutil.copyfile(os.path.join(ROOT, "submission", "output_v3", "candidate_pairs.tsv"),
                    os.path.join(tmp, "candidate_pairs.tsv"))
    os.rename(tmp, out_dir)
    nm = sel.groupby("s1_row").size().reindex(np.arange(len(s1rid)), fill_value=0)
    info = dict(seconds=round(time.time() - t0, 1), matches=len(sel), mean=float(nm.mean()),
                empty_rows=float((nm == 0).mean()), t=t, t_empty=te, out=out_dir)
    save_stats("write_rc", info)
    report([f"\n**write_rc -> {out_dir}:** {info}\n"])


KAGGLE = os.path.join(ROOT, "kaggle")


def stage_kaggle_folds():
    """kaggle/s1_folds.tsv: entity_id, fold (data/folds.parquet, the split v3 uses), part (fold-1 S1s split in
    half by a stable splitmix64 hash of the id: 1a / 1b; '' otherwise), v3_sample (val = v3's 100k fold-0
    validation sample, train = v3's 100k fold-1 training sample)."""
    import m6b_expected_f as EF
    os.makedirs(KAGGLE, exist_ok=True)
    f = pd.read_parquet(os.path.join(DATA, "folds.parquet"))
    ids = pq.read_table(pq_path("train", 1), columns=["rid", "entity_id"]).to_pandas()
    d = ids.merge(f, left_on="rid", right_on="s1_rid", how="left")
    assert d.fold.notna().all() and len(d) == len(ids)
    d["part"] = ""
    m1 = (d.fold == 1).values
    d.loc[m1, "part"] = np.where(EF.split_half(d.rid.values[m1]), "1a", "1b")
    t = pd.read_parquet(os.path.join(DATA, "trainval_s1.parquet"))
    role = pd.Series(np.where(t.fold.values == 0, "val", "train"), index=t.s1_rid.values)
    d["v3_sample"] = role.reindex(d.rid.values).fillna("").values
    d[["entity_id", "fold", "part", "v3_sample"]].to_csv(os.path.join(KAGGLE, "s1_folds.tsv"), sep="\t", index=False,
                                                         lineterminator="\n")
    ct = pd.crosstab(d.fold, d.part.replace("", "-") + "/" + d.v3_sample.replace("", "-"))
    report(["\n## JOB 1a: kaggle/s1_folds.tsv\n", f"{len(d):,} S1 rows; fold x (part / v3_sample):\n",
            "```", ct.to_string(), "```"])


if __name__ == "__main__":
    st, args = sys.argv[1], sys.argv[2:]
    {"measure": stage_measure, "rc": stage_rc, "write_rc": lambda: stage_write_rc(args[0]),
     "kaggle_folds": stage_kaggle_folds}[st]()
