"""M12: v5d = v3 + dense-retriever additions for the pool records v3 left unassigned.

Stages:
  open_lists   ids of the train / test pool records v3 assigns to nobody (train: v3full out-of-fold p, one-owner
               over all 2.2M train S1s + decision.json; test: submission/output_v3/matching_results.tsv)
               -> C:/amlc/kaggle/v3_open_train.txt, v3_open_test.txt (upload to Kaggle for kaggle_dense_fast.py)
"""
import json
import os
import sys
import time

os.environ.setdefault("AMLC_MAXMB", "3900")
import numpy as np
import pandas as pd

import m9_cands as M9
from common import ROOT, log, rid_to_id

KAGGLE = os.path.join(ROOT, "kaggle")


def stage_open_test():
    """test: exactly what output_v3 submitted (numeric rids, no Python string sets - those hit the RAM guard)"""
    t0 = time.time()
    mr = pd.read_csv(os.path.join(ROOT, "submission", "output_v3", "matching_results.tsv"), sep="\t", dtype=str,
                     na_filter=False, usecols=[1]).iloc[:, 0]
    m = mr[mr != ""].str.split(",").explode()
    got = (m.str[1].astype(np.int64) * 10**10 + m.str[3:].astype(np.int64)).values
    del mr, m
    pool_rid = M9.pool_rid_arr("test")
    open_rid = pool_rid[~np.isin(pool_rid, got)]
    with open(os.path.join(KAGGLE, "v3_open_test.txt"), "w", encoding="utf-8", newline="\n") as f:
        for a in range(0, len(open_rid), 1_000_000):
            f.write("\n".join(rid_to_id(open_rid[a:a + 1_000_000])) + "\n")
    log(f"test: {len(pool_rid):,} pool records, v3 assigns {len(np.unique(got)):,}, open {len(open_rid):,} "
        f"({time.time() - t0:.0f}s)")


def stage_open_lists():
    t0 = time.time()
    dec = json.load(open(os.path.join(M9.MV3, "decision.json")))
    # train
    pool_rid = M9.pool_rid_arr("train")
    files = sorted(os.path.join(M9.V3, "v3full", f) for f in os.listdir(os.path.join(M9.V3, "v3full"))
                   if f.startswith("pred_") and f.endswith(".parquet"))
    sel = M9.full_competition(files, np.ones(len(M9.s1_rid_arr("train")), bool), len(pool_rid))
    pmax = sel.groupby("s1_row").p.transform("max")
    sel = sel[(pmax >= dec["g"]) & (sel.p >= dec["t"]) & (sel.p >= dec["r"] * pmax)]
    assigned = np.zeros(len(pool_rid), bool)
    assigned[sel.pidx.values] = True
    open_tr = rid_to_id(pool_rid[~assigned])
    with open(os.path.join(KAGGLE, "v3_open_train.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(open_tr) + "\n")
    log(f"train: {len(pool_rid):,} pool records, v3 assigns {assigned.sum():,}, open {len(open_tr):,}")
    # test: exactly what output_v3 submitted
    mr = pd.read_csv(os.path.join(ROOT, "submission", "output_v3", "matching_results.tsv"), sep="\t", dtype=str,
                     na_filter=False)
    got = set(x for v in mr.iloc[:, 1].values if v for x in v.split(","))
    te_ids = rid_to_id(M9.pool_rid_arr("test"))
    open_te = [x for x in te_ids if x not in got]
    with open(os.path.join(KAGGLE, "v3_open_test.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(open_te) + "\n")
    log(f"test: {len(te_ids):,} pool records, v3 assigns {len(got):,}, open {len(open_te):,} ({time.time() - t0:.0f}s)")


# ------------------------------------------------------------------ merge
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from common import DATA, WORK, pq_path

V5 = os.path.join(DATA, "v5")
MV5 = os.path.join(WORK, "models", "v5")
REP = os.path.join(WORK, "reports", "m12_dense_merge.md")
C_GRID = np.round(np.arange(0.50, 0.991, 0.02), 3)   # fine-tuned cosines sit mostly in 0.5-0.95
M_GRID = [0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.09, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3]


def report(lines):
    with open(REP, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log("\n".join(lines))


def _rid(arr):
    """'S2-123' -> 2*10^10 + 123 (arrow, no Python strings)"""
    src = pc.cast(pc.utf8_slice_codeunits(arr, 1, 2), pa.int64())
    num = pc.cast(pc.utf8_slice_codeunits(arr, 3), pa.int64())
    return pc.add(pc.multiply(src, 10**10), num).to_numpy()


def dense_top2(path):
    """per record: top-1 S1 rid, cosine 1, cosine 2 (NaN if no 2nd)."""
    pf = pq.ParquetFile(path)
    R, S, K, C = [], [], [], []
    for i in range(pf.num_row_groups):
        t = pf.read_row_group(i)
        t = t.filter(pc.less_equal(t.column("rank"), 2))
        R.append(_rid(t.column("record_id")))
        S.append(_rid(t.column("s1_id")))
        K.append(t.column("rank").to_numpy())
        C.append(t.column("cosine").to_numpy())
    r, s, k, c = (np.concatenate(x) for x in (R, S, K, C))
    # leftover smoke-run parts in the Kaggle output repeat some records: keep the first (record, rank) row
    u = ~pd.DataFrame({"r": r, "k": k}).duplicated().values
    r, s, k, c = r[u], s[u], k[u], c[u]
    one = k == 1
    d = pd.DataFrame({"rid": r[one], "s1": s[one], "c1": c[one]})
    c2 = pd.Series(c[k == 2], index=r[k == 2])
    d["c2"] = c2.reindex(d.rid.values).values
    d["margin"] = (d.c1 - d.c2.fillna(-1)).astype(np.float32)
    return d


def part1a_rows():
    import m6b_expected_f as EF
    f = pd.read_parquet(os.path.join(DATA, "folds.parquet"))
    rid = f.s1_rid.values[f.fold.values == 1]
    s1rid = M9.s1_rid_arr("train")
    m = np.zeros(len(s1rid), bool)
    m[pd.Index(s1rid).get_indexer(rid[EF.split_half(rid)])] = True
    return m


def v3_selection(split):
    dec = json.load(open(os.path.join(M9.MV3, "decision.json")))
    if split == "train":
        files = sorted(os.path.join(M9.V3, "v3full", f) for f in os.listdir(os.path.join(M9.V3, "v3full"))
                       if f.startswith("pred_") and f.endswith(".parquet"))
    else:
        files = M9.test_files()
    sel = M9.full_competition(files, np.ones(len(M9.s1_rid_arr(split)), bool),
                              len(M9.pool_rid_arr(split)))
    pmax = sel.groupby("s1_row").p.transform("max")
    return sel[(pmax >= dec["g"]) & (sel.p >= dec["t"]) & (sel.p >= dec["r"] * pmax)][["s1_row", "pidx"]], files


def additions(split, dense_dir):
    """candidate additions (s1_row, pidx, c1, margin, empty) from the dense top-1 of every open record."""
    import m10_v5 as M10
    d = dense_top2(os.path.join(dense_dir, f"dense_{split}.parquet"))
    pidx = pd.Index(M9.pool_rid_arr(split)).get_indexer(d.rid.values)
    s1row = pd.Index(M9.s1_rid_arr(split)).get_indexer(d.s1.values)
    assert (pidx >= 0).all() and (s1row >= 0).all()
    empty = M10.pool_attrs(split).empty_addr.values
    return pd.DataFrame({"s1_row": s1row, "pidx": pidx, "c1": d.c1.values, "margin": d.margin.values,
                         "eaddr": empty[pidx]})


def stage_merge_eval(dense_dir):
    """Tune on train S1s (all folds; part-1a S1s and their records excluded: the dense model trained on them;
    v3's train selection is out-of-fold everywhere), tune on half 1, score on half 2."""
    import m10_v5 as M10
    import m6b_expected_f as EF
    t0 = time.time()
    open(REP, "w", encoding="utf-8").close()  # each evaluation starts a fresh report
    s1rid = M9.s1_rid_arr("train")
    n1 = len(s1rid)
    own = M10.owner_arrays()
    p1a = part1a_rows()
    sel, _ = v3_selection("train")
    base_n = np.bincount(sel.s1_row.values, minlength=n1).astype(np.float64)
    base_tp = np.bincount(sel.s1_row.values, weights=(own[sel.pidx.values] == sel.s1_row.values).astype(float),
                          minlength=n1)
    assigned = np.zeros(len(own), bool)
    assigned[sel.pidx.values] = True
    del sel
    import gc
    gc.collect()
    add = additions("train", dense_dir)
    add = add[~assigned[add.pidx.values]]  # only records v3 leaves unassigned (smoke leftovers may include others)
    add = add[~p1a[add.s1_row.values] & ~((own[add.pidx.values] >= 0) & p1a[np.maximum(own[add.pidx.values], 0)])]
    add["label"] = (own[add.pidx.values] == add.s1_row.values).astype(np.float64)
    ntrue = np.bincount(own[own >= 0], minlength=n1).astype(np.float64)
    h1 = np.zeros(n1, bool)
    h1[EF.split_half(s1rid)] = True  # split_half returns a mask over the ids passed
    keep = ~p1a
    halves = {"h1": keep & h1, "h2": keep & ~h1}
    has_v3 = base_n > 0

    def F(mask_add, S):
        a = add[mask_add]
        n = base_n + np.bincount(a.s1_row.values, minlength=n1)
        tp = base_tp + np.bincount(a.s1_row.values, weights=a.label.values, minlength=n1)
        f = np.where(ntrue == 0, (n == 0).astype(float),
                     np.divide(1.25 * tp, n + 0.25 * ntrue, out=np.zeros_like(tp), where=(n + 0.25 * ntrue) > 0))
        return f[S].mean(), f

    c1, mg, em = add.c1.values, add.margin.values, add.eaddr.values
    tgt = has_v3[add.s1_row.values]

    def mask(p):
        (cn, mn, ce, me, only_v3) = p
        m = np.where(em, (c1 >= ce) & (mg >= me), (c1 >= cn) & (mg >= mn))
        return m & tgt if only_v3 else m

    S1 = halves["h1"]
    base_h1 = F(np.zeros(len(add), bool), S1)[0]
    best = (base_h1, None)
    for only_v3 in (False, True):
        for cn in C_GRID:
            for mn in M_GRID:
                p = (cn, mn, 9.0, 9.0, only_v3)  # empty-address additions off while tuning the rest
                s = F(mask(p), S1)[0]
                if s > best[0]:
                    best = (s, p)
    # empty-address thresholds: tuned on top of the best address-present setting, or alone if none helped
    cn, mn, _, _, ov = best[1] if best[1] is not None else (9.0, 9.0, None, None, None)
    for only_v3 in ((ov,) if ov is not None else (False, True)):
        for ce in C_GRID:
            for me in M_GRID:
                p = (cn, mn, ce, me, only_v3)
                s = F(mask(p), S1)[0]
                if s > best[0]:
                    best = (s, p)
    S2 = halves["h2"]
    v3_h2 = F(np.zeros(len(add), bool), S2)[0]
    L = ["# M12: v5d = v3 + dense top-1 additions for the records v3 leaves unassigned\n",
         f"Train S1s excluding part 1a ({keep.sum():,}); open train records with a dense list: {len(add):,} after "
         f"removing part-1a S1s / records. Additions correct: {add.label.mean():.4f} overall.\n",
         f"v3 alone: half 1 {base_h1:.5f}, half 2 {v3_h2:.5f}\n"]
    res = {"v3_h2": v3_h2, "params": best[1]}
    if best[1] is None:
        L.append("No threshold improves half 1 -> no additions, no output_v5d.")
        res["better"] = False
    else:
        m = mask(best[1])
        s2, f2 = F(m, S2)
        a = add[m & S2[add.s1_row.values]]
        res.update(h2=s2, better=bool(s2 > v3_h2), added_h2=len(a), precision_h2=float(a.label.mean()))
        L.append(f"Tuned on half 1: {dict(zip(['c1', 'margin', 'c1_empty', 'margin_empty', 'only_S1s_with_v3_matches'], best[1]))}")
        L.append(f"**Half 2: v5d {s2:.5f} vs v3 {v3_h2:.5f} ({s2 - v3_h2:+.5f})**; additions on half 2: {len(a):,}, "
                 f"correct {a.label.mean():.4f}, of them empty-address {a.eaddr.mean():.3f}.")
        # sensitivity table on half 2 for the non-empty threshold
        L.append("\n| c1 >= | margin >= | half 2 F0.5 |\n|---|---|---|")
        for cn in (0.8, 0.85, 0.9, 0.93, 0.95, 0.97):
            p = (cn, best[1][1], best[1][2], best[1][3], best[1][4])
            L.append(f"| {cn} | {best[1][1]} | {F(mask(p), S2)[0]:.5f} |")
    L.append(f"\n({time.time() - t0:.0f}s)\n")
    report(L)
    json.dump(res, open(os.path.join(MV5, "merge_v5d.json"), "w"), indent=1, default=float)


def stage_merge_write(dense_dir, out_dir):
    from m7_infer import write_lists
    t0 = time.time()
    res = json.load(open(os.path.join(MV5, "merge_v5d.json")))
    assert res["better"], "v5d did not beat v3 on half 2"
    out_dir = os.path.abspath(out_dir)
    assert os.path.basename(out_dir) == "output_v5d" and not os.path.exists(out_dir)
    tmp = out_dir + "_tmp_writing"
    os.makedirs(tmp, exist_ok=True)
    cn, mn, ce, me, only_v3 = res["params"]
    s1rid, pool_rid = M9.s1_rid_arr("test"), M9.pool_rid_arr("test")
    s1_ids = pq.read_table(pq_path("test", 1), columns=["entity_id"]).column("entity_id").to_pylist()
    sel, files = v3_selection("test")
    add = additions("test", dense_dir)
    add = add[~np.isin(add.pidx.values, sel.pidx.values)]           # open records only (safety)
    has_v3 = np.bincount(sel.s1_row.values, minlength=len(s1rid)) > 0
    m = np.where(add.eaddr.values, (add.c1.values >= ce) & (add.margin.values >= me),
                 (add.c1.values >= cn) & (add.margin.values >= mn))
    if only_v3:
        m &= has_v3[add.s1_row.values]
    add = add[m]
    final = pd.concat([sel, add[["s1_row", "pidx"]]], ignore_index=True)
    with open(os.path.join(tmp, "matching_results.tsv"), "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        write_lists(f, s1_ids, 0, len(s1rid), final.s1_row.values.astype(np.int64), pool_rid[final.pidx.values])
    n_cand = 0
    with open(os.path.join(tmp, "candidate_pairs.tsv"), "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for fn, (lo, hi) in zip(files, M9.ranges(len(s1rid), M9.NB["test"])):
            pr = pd.read_parquet(fn, columns=["s1_row", "pidx"])
            a = add[(add.s1_row >= lo) & (add.s1_row < hi)][["s1_row", "pidx"]]
            pr = pd.concat([pr, a], ignore_index=True).drop_duplicates()
            write_lists(f, s1_ids, lo, hi, pr.s1_row.values.astype(np.int64), pool_rid[pr.pidx.values])
            n_cand += len(pr)
    os.rename(tmp, out_dir)
    nm = final.groupby("s1_row").size().reindex(np.arange(len(s1rid)), fill_value=0)
    ctry = pq.read_table(pq_path("test", 1), columns=["country"]).column("country").to_numpy(zero_copy_only=False)
    ca = pd.Series(ctry[add.s1_row.values]).value_counts().to_dict()
    report([f"\n## Test output -> {out_dir}\n", f"v3 matches {len(sel):,} + additions {len(add):,} "
            f"(empty-address {add.eaddr.mean():.3f}; by country {ca}) = {len(final):,} matches, mean {nm.mean():.3f} "
            f"per S1, empty rows {(nm == 0).mean():.4f}; candidates {n_cand:,}. ({time.time() - t0:.0f}s)\n"])


def stage_france_gate(src_dir, out_dir):
    """output_v5f = src (output_v5d) with a stricter empty gate for FRANCE S1s only: France (unseen in train) is
    predicted empty far less often than India / US (4.4% vs 6.0%), which suggests over-confidence. The gate on the
    S1's best v3 p (after the one-owner rule) is raised for France until France's empty-row share equals the
    India + US empty share of the same file. India / US rows are copied unchanged."""
    import shutil
    t0 = time.time()
    out_dir = os.path.abspath(out_dir)
    assert os.path.basename(out_dir) == "output_v5f" and not os.path.exists(out_dir)
    s1 = pq.read_table(pq_path("test", 1), columns=["entity_id", "country"]).to_pandas()
    mr = pd.read_csv(os.path.join(src_dir, "matching_results.tsv"), sep="\t", dtype=str, na_filter=False)
    assert (mr.iloc[:, 0].values == s1.entity_id.values).all()
    empty = (mr.iloc[:, 1] == "").values
    fr = (s1.country == "France").values
    target = empty[~fr].mean()
    own = M9.full_competition(M9.test_files(), np.ones(len(s1), bool), len(M9.pool_rid_arr("test")))
    pmax = own.groupby("s1_row").p.max().reindex(np.arange(len(s1)), fill_value=0.0).values
    best_g, rows = None, None
    for g in np.round(np.arange(0.68, 0.999, 0.002), 3):
        e = empty | (fr & (pmax < g))
        if e[fr].mean() >= target:
            best_g, rows = float(g), fr & (pmax < g) & ~empty
            break
    assert best_g is not None
    mr.iloc[rows, 1] = ""
    tmp = out_dir + "_tmp_writing"
    os.makedirs(tmp, exist_ok=True)
    mr.to_csv(os.path.join(tmp, "matching_results.tsv"), sep="\t", index=False, lineterminator="\n")
    shutil.copyfile(os.path.join(src_dir, "candidate_pairs.tsv"), os.path.join(tmp, "candidate_pairs.tsv"))
    os.rename(tmp, out_dir)
    n_rm = mr.iloc[:, 1].str.len()  # noqa: F841 (kept simple)
    report([f"\n## output_v5f: stricter empty gate for France only -> {out_dir}\n",
            f"India+US empty share {target:.4f}; France empty share {empty[fr].mean():.4f} -> "
            f"{(empty | rows)[fr].mean():.4f} with gate g_France = {best_g} (others keep 0.68). "
            f"France S1 rows emptied: {int(rows.sum()):,} of {int(fr.sum()):,}. ({time.time() - t0:.0f}s)\n"])


def stage_france_pairs(src_dir, out_dir):
    """output_v5g = src (output_v5f) with fewer matches per FRANCE row: France non-empty rows hold ~3.8 matches vs
    ~3.5 for India / US. Inside France rows, matches with v3 p < t_F are dropped (each row keeps its best match;
    dense additions have no v3 p and are kept); t_F is the smallest value that brings France's mean matches per
    non-empty row down to the India + US mean of the same file. India / US rows are copied unchanged."""
    import shutil
    t0 = time.time()
    out_dir = os.path.abspath(out_dir)
    assert os.path.basename(out_dir) == "output_v5g" and not os.path.exists(out_dir)
    s1 = pq.read_table(pq_path("test", 1), columns=["entity_id", "country"]).to_pandas()
    mr = pd.read_csv(os.path.join(src_dir, "matching_results.tsv"), sep="\t", dtype=str, na_filter=False)
    assert (mr.iloc[:, 0].values == s1.entity_id.values).all()
    fr = (s1.country == "France").values
    lst = mr.iloc[:, 1]
    n = np.where(lst.values == "", 0, lst.str.count(",").values + 1)
    target = n[~fr & (n > 0)].mean()
    pool_rid = M9.pool_rid_arr("test")
    own = M9.full_competition(M9.test_files(), fr, len(pool_rid))
    p_of = np.full(len(pool_rid), np.nan, np.float32)
    p_of[own.pidx.values] = own.p.values
    rows = np.flatnonzero(fr & (n > 0))
    ex = lst.iloc[rows].str.split(",").explode()
    d = pd.DataFrame({"row": ex.index.values, "id": ex.values})
    rid = (d.id.str[1].astype(np.int64) * 10**10 + d.id.str[3:].astype(np.int64)).values
    d["p"] = p_of[pd.Index(pool_rid).get_indexer(rid)]
    d["top"] = d.groupby("row").p.transform("max")
    before = len(d) / len(rows)
    best_t = None
    for t in np.round(np.arange(0.0, 0.999, 0.01), 3):
        keep = d.p.isna() | (d.p >= t) | (d.p >= d.top)
        if keep.sum() / len(rows) <= target:
            best_t = float(t)
            break
    assert best_t is not None
    d = d[d.p.isna() | (d.p >= best_t) | (d.p >= d.top)]
    new = d.groupby("row").id.agg(",".join)
    mr.iloc[new.index.values, 1] = new.values
    tmp = out_dir + "_tmp_writing"
    os.makedirs(tmp, exist_ok=True)
    mr.to_csv(os.path.join(tmp, "matching_results.tsv"), sep="\t", index=False, lineterminator="\n")
    shutil.copyfile(os.path.join(src_dir, "candidate_pairs.tsv"), os.path.join(tmp, "candidate_pairs.tsv"))
    os.rename(tmp, out_dir)
    report([f"\n## output_v5g: fewer matches per France row -> {out_dir}\n",
            f"India+US mean matches per non-empty row {target:.3f}; France {before:.3f} -> {len(d) / len(rows):.3f} "
            f"with t_France = {best_t} (each row keeps its best). France matches dropped: "
            f"{int(round((before - len(d) / len(rows)) * len(rows))):,}. ({time.time() - t0:.0f}s)\n"])


def stage_france_gt(src_dir, out_dir, g, t):
    """France rows of src (output_v5d = v3 rule + dense additions): empty the row if its best v3 p < g; else keep
    matches with v3 p >= t or = the row's best (dense additions, no v3 p, are kept). India / US unchanged."""
    import shutil
    t0 = time.time()
    g, t = float(g), float(t)
    out_dir = os.path.abspath(out_dir)
    assert not os.path.exists(out_dir)
    s1 = pq.read_table(pq_path("test", 1), columns=["entity_id", "country"]).to_pandas()
    mr = pd.read_csv(os.path.join(src_dir, "matching_results.tsv"), sep="\t", dtype=str, na_filter=False)
    assert (mr.iloc[:, 0].values == s1.entity_id.values).all()
    fr = (s1.country == "France").values
    lst = mr.iloc[:, 1]
    n = np.where(lst.values == "", 0, lst.str.count(",").values + 1)
    pool_rid = M9.pool_rid_arr("test")
    own = M9.full_competition(M9.test_files(), fr, len(pool_rid))
    p_of = np.full(len(pool_rid), np.nan, np.float32)
    p_of[own.pidx.values] = own.p.values
    rows = np.flatnonzero(fr & (n > 0))
    ex = lst.iloc[rows].str.split(",").explode()
    d = pd.DataFrame({"row": ex.index.values, "id": ex.values})
    rid = (d.id.str[1].astype(np.int64) * 10**10 + d.id.str[3:].astype(np.int64)).values
    d["p"] = p_of[pd.Index(pool_rid).get_indexer(rid)]
    d["top"] = d.groupby("row").p.transform("max")
    live = d[d.top.fillna(1.0) >= g]
    keep = live[live.p.isna() | (live.p >= t) | (live.p >= live.top)]
    new = keep.groupby("row").id.agg(",".join).reindex(rows, fill_value="")
    mr.iloc[rows, 1] = new.values
    tmp = out_dir + "_tmp_writing"
    os.makedirs(tmp, exist_ok=True)
    mr.to_csv(os.path.join(tmp, "matching_results.tsv"), sep="\t", index=False, lineterminator="\n")
    shutil.copyfile(os.path.join(src_dir, "candidate_pairs.tsv"), os.path.join(tmp, "candidate_pairs.tsv"))
    os.rename(tmp, out_dir)
    ne = (new.values != "").sum()
    report([f"\n## {os.path.basename(out_dir)}: France gate {g} / pair cut {t} from {os.path.basename(src_dir)}\n",
            f"France: empty {1 - ne / fr.sum():.4f}, matches per non-empty row {len(keep) / max(ne, 1):.3f}, "
            f"matches {len(keep):,} (was {len(d):,}). ({time.time() - t0:.0f}s)\n"])


if __name__ == "__main__":
    a = sys.argv[2:]
    {"open_lists": stage_open_lists, "open_test": stage_open_test,
     "merge_eval": lambda: stage_merge_eval(a[0]), "merge_write": lambda: stage_merge_write(a[0], a[1]),
     "france_gate": lambda: stage_france_gate(a[0], a[1]),
     "france_pairs": lambda: stage_france_pairs(a[0], a[1]),
     "france_gt": lambda: stage_france_gt(*a)}[sys.argv[1]]()
