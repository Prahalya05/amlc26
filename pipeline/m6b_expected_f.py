"""M6b: ALTERNATIVE decision step - pick the list that maximises the EXPECTED per-entity F0.5.
The current rule (m6_decide.py + models/decision.json, used by m7_infer.py) is untouched; this script
has its own files (models/decision_ef.json, models/calib_isotonic.json) and its own writer, so the
decision step is selected by which script writes the output.

Method, per S1 entity:
  1. one-owner rule on the raw model p (exactly as m6/m7: each pool record stays only with the S1
     that gives it the highest p);
  2. calibrated q = isotonic(p) (fitted on validation labels; applied as linear interpolation);
  3. candidates sorted by q (raw p breaks ties); options = empty list, top-1, ..., top-n;
  4. each option scored by expected F0.5 with F0.5 = 1.25*TP / (|S| + 0.25*G), empty list = 1 if G=0:
       plug-in     : E[TP] = sum of q in S, G = sum of q over all candidates + c, P(empty ok) = prod(1-q)
       Monte Carlo : 200 draws y_i ~ Bernoulli(q_i) (independent); G = sum y + c; empty scores 1 if sum y = 0
     c = expected number of true matches that blocking never retrieved (tuned on validation half 1);
  5. the option with the highest expected F0.5 is predicted (ties -> the shorter list).

Usage:
  python m6b_expected_f.py tune           validation: half 1 (by id) fits calibration + c, half 2 reports
  python m6b_expected_f.py write OUT_DIR  test output with the chosen method (only if tune says it wins)
"""
import gc
import json
import os
import shutil
import sys
import time

os.environ.setdefault("AMLC_MAXMB", "3900")  # RAM guard in common.py: stay below 4 GB
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.isotonic import IsotonicRegression

from common import DATA, ROOT, WORK, log, pq_path
from m6_decide import grid, one_owner, prepare, score

MODELS = os.path.join(WORK, "models")
REP = os.path.join(WORK, "reports")
KMAX = 30          # candidates per S1 (block_cfg K)
N_MC = 200
C_GRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.75, 1.0, 1.5]
SEED = 2026


# ------------------------------------------------------------------ calibration
def fit_calibration(p, y):
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True)
    iso.fit(p.astype(np.float64), y.astype(np.float64))
    return dict(x=iso.X_thresholds_.tolist(), y=iso.y_thresholds_.tolist())


def calibrate(cal, p):
    # IsotonicRegression.predict = linear interpolation between thresholds, clipped at the ends
    return np.interp(p.astype(np.float64), cal["x"], cal["y"]).astype(np.float32)


# ------------------------------------------------------------------ per-S1 matrices
def to_matrix(df, s1_ids):
    """df: rows (s1_rid, m_rid, p, q) after one-owner. Returns Q (n x KMAX, sorted desc by q, zero padded),
    M (m_rid, -1 padded), ncand, and the S1 order (= s1_ids)."""
    n = len(s1_ids)
    li = pd.Index(s1_ids).get_indexer(df.s1_rid.values)
    assert (li >= 0).all()
    order = np.lexsort((-df.p.values, -df.q.values, li))
    li, q, m = li[order], df.q.values[order], df.m_rid.values[order]
    gstart = np.flatnonzero(np.r_[True, np.diff(li) != 0])
    j = np.arange(len(li)) - np.repeat(gstart, np.diff(np.r_[gstart, len(li)]))
    assert j.max() < KMAX
    Q = np.zeros((n, KMAX), np.float32)
    M = np.full((n, KMAX), -1, np.int64)
    Q[li, j] = q
    M[li, j] = m
    ncand = np.bincount(li, minlength=n)
    if "label" not in df.columns:
        return Q, M, ncand
    L = np.zeros((n, KMAX), np.int16)
    L[li, j] = df.label.values[order]
    return Q, M, ncand, L


def choose(ef, empty, ncand):
    """ef: (n, KMAX) expected F of top-k (k=1..KMAX), empty: (n,). Returns chosen k (0 = empty list)."""
    valid = np.arange(1, KMAX + 1)[None, :] <= ncand[:, None]
    allv = np.concatenate([empty[:, None], np.where(valid, ef, -1.0)], axis=1)
    return np.argmax(allv, axis=1)  # first maximum -> shorter list on ties


def plugin_choice(Q, ncand, cs):
    """Chosen k for each c in cs (plug-in estimate). Returns (len(cs), n) int array."""
    etp = np.cumsum(Q, axis=1, dtype=np.float64)
    tot = etp[:, -1]
    empty = np.prod(1.0 - Q.astype(np.float64), axis=1)
    k = np.arange(1, KMAX + 1)[None, :]
    return np.stack([choose(1.25 * etp / (k + 0.25 * (tot + c)[:, None]), empty, ncand) for c in cs])


def mc_choice(Q, ncand, cs, seed=SEED, chunk=2000):
    """Chosen k for each c in cs (Monte Carlo, N_MC independent Bernoulli draws per candidate)."""
    rng = np.random.default_rng(seed)
    out = np.zeros((len(cs), len(Q)), np.int64)
    k = np.arange(1, KMAX + 1, dtype=np.float32)[None, None, :]
    for lo in range(0, len(Q), chunk):
        q = Q[lo:lo + chunk]
        y = rng.random((len(q), N_MC, KMAX), dtype=np.float32) < q[:, None, :]
        tpk = np.cumsum(y, axis=2, dtype=np.int16)
        s = tpk[:, :, -1].astype(np.float32)
        del y
        empty = (s == 0).mean(axis=1)
        for ci, c in enumerate(cs):
            ef = (1.25 * tpk / (k + 0.25 * (s + c)[:, :, None])).mean(axis=1)
            out[ci, lo:lo + chunk] = choose(ef, empty, ncand[lo:lo + chunk])
        del tpk, s
    return out


def true_f(kchosen, L, nt):
    """Macro F0.5 of predicting the top-k (k per S1) given true labels L (n x KMAX, sorted like Q)."""
    tp = np.take_along_axis(np.cumsum(L, axis=1), np.maximum(kchosen - 1, 0)[:, None], axis=1)[:, 0]
    tp = np.where(kchosen == 0, 0, tp).astype(np.float64)
    f = np.where(nt == 0, (kchosen == 0).astype(float),
                 np.divide(1.25 * tp, kchosen + 0.25 * nt, out=np.zeros_like(tp), where=(kchosen + 0.25 * nt) > 0))
    return f


# ------------------------------------------------------------------ tune on validation
def split_half(ids):
    """Deterministic half split by S1 id (splitmix64 hash of the id, lower half of the hashes = half 1)."""
    z = ids.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    with np.errstate(over="ignore"):
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    h1 = np.zeros(len(ids), bool)
    h1[np.argsort(z, kind="stable")[: len(ids) // 2]] = True
    return h1


def stage_tune():
    t0 = time.time()
    vp = pd.read_parquet(os.path.join(DATA, "val_pred.parquet"))
    s1 = pd.read_parquet(os.path.join(DATA, "trainval_s1.parquet"))
    val_ids = np.sort(s1.s1_rid[s1.fold == 0].values)
    gt = pd.read_parquet(os.path.join(DATA, "train_gt_pairs.parquet"))
    pairs = gt[gt.s1_rid.isin(val_ids)].reset_index(drop=True)
    del gt
    dec = json.load(open(os.path.join(MODELS, "decision.json")))
    h1 = split_half(val_ids)
    ids = {1: val_ids[h1], 2: val_ids[~h1]}
    d = one_owner(vp).reset_index(drop=True)  # one-owner over ALL validation S1s, as in m6
    lines = ["# M6b: expected-F0.5 decision (alternative to the m6 thresholds)\n",
             f"Validation fold 0: {len(val_ids):,} S1, split by hashed id into half 1 ({len(ids[1]):,}, fits "
             f"calibration and c) and half 2 ({len(ids[2]):,}, reported). One-owner rule applied first on raw p "
             f"(over all fold-0 S1s, as in m6). Candidates after one-owner: {len(d):,} "
             f"(removed {len(vp) - len(d):,}).\n"]

    # current rule on half 2 (thresholds as in production, tuned on all of fold 0) + re-tuned on half 1
    res = {}
    P2 = prepare(d[d.s1_rid.isin(ids[2])], ids[2], pairs[pairs.s1_rid.isin(ids[2])])
    res["current rule (decision.json, as in production)"] = score(P2, dec["g"], dec["t"], dec["r"])
    P1 = prepare(d[d.s1_rid.isin(ids[1])], ids[1], pairs[pairs.s1_rid.isin(ids[1])])
    s_h1, (g1, t1, r1) = grid(P1)
    res[f"current rule re-tuned on half 1 (g={g1}, t={t1}, r={r1})"] = score(P2, g1, t1, r1)
    del P1, P2
    log(f"current rule on half 2: {res}")

    # calibration on half 1
    dh = {h: d[d.s1_rid.isin(ids[h])].reset_index(drop=True) for h in (1, 2)}
    cal = fit_calibration(dh[1].p.values, dh[1].label.values)
    for h in (1, 2):
        dh[h]["q"] = calibrate(cal, dh[h].p.values)
    lines.append("## Calibration (isotonic, half 1)\n")
    lines.append(f"{len(cal['x']):,} thresholds. Reliability on half 2 (mean label vs mean p / calibrated q "
                 "per raw-p bin):\n")
    lines.append("| raw p bin | pairs | mean label | mean raw p | mean q |\n|---|---|---|---|---|")
    b = pd.cut(dh[2].p, [0, 0.01, 0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 0.99, 1.0], include_lowest=True)
    for iv, g in dh[2].groupby(b, observed=True):
        lines.append(f"| {iv} | {len(g):,} | {g.label.mean():.4f} | {g.p.mean():.4f} | {g.q.mean():.4f} |")
    lines.append(f"\nSum over half 2: labels {dh[2].label.sum():,}, raw p {dh[2].p.sum():,.0f}, q {dh[2].q.sum():,.0f}.\n")

    # matrices, true labels
    mats = {}
    for h in (1, 2):
        Q, M, nc, L = to_matrix(dh[h], ids[h])
        nt = pairs[pairs.s1_rid.isin(ids[h])].groupby("s1_rid").size().reindex(ids[h], fill_value=0).values
        mats[h] = (Q, M, nc, L, nt.astype(np.float64))
        assert L.sum() == dh[h].label.sum() and nc.sum() == len(dh[h])

    # tune c on half 1 (both estimators), report half 2
    lines.append("## c = expected number of never-retrieved true matches (tuned on half 1)\n")
    lines.append("| c | plug-in half 1 | Monte Carlo half 1 |\n|---|---|---|")
    Q, M, nc, L, nt = mats[1]
    t1_ = time.time()
    f_pi = [true_f(k, L, nt).mean() for k in plugin_choice(Q, nc, C_GRID)]
    f_mc = [true_f(k, L, nt).mean() for k in mc_choice(Q, nc, C_GRID)]
    log(f"half-1 c grid done ({time.time() - t1_:.0f}s)")
    for c, a, b_ in zip(C_GRID, f_pi, f_mc):
        lines.append(f"| {c} | {a:.5f} | {b_:.5f} |")
    c_pi, c_mc = C_GRID[int(np.argmax(f_pi))], C_GRID[int(np.argmax(f_mc))]
    Q, M, nc, L, nt = mats[2]
    k_pi = plugin_choice(Q, nc, [c_pi])[0]
    k_mc = mc_choice(Q, nc, [c_mc], seed=SEED + 1)[0]
    f_alt = {"plug-in": true_f(k_pi, L, nt), "mc": true_f(k_mc, L, nt)}
    res[f"expected F0.5, plug-in (c={c_pi})"] = f_alt["plug-in"].mean()
    res[f"expected F0.5, Monte Carlo 200 draws (c={c_mc})"] = f_alt["mc"].mean()
    # oracle on the same half for reference
    h_ = L.sum(axis=1)
    res["(reference) oracle: perfect picking among the candidates"] = np.where(
        nt == 0, 1.0, 1.25 * h_ / (h_ + 0.25 * np.maximum(nt, 1e-9))).mean()

    lines.append("\n## Half 2 macro F0.5 (never used for fitting anything except the production thresholds)\n")
    lines.append("| decision | half-2 macro F0.5 |\n|---|---|")
    for k_, v in res.items():
        lines.append(f"| {k_} | {v:.5f} |")
    cur = res["current rule (decision.json, as in production)"]
    best_name = max(f_alt, key=lambda m_: f_alt[m_].mean())
    best_val = f_alt[best_name].mean()
    wins = bool(best_val > cur)
    # list-size behaviour on half 2
    for nm_, kk in (("plug-in", k_pi), ("Monte Carlo", k_mc)):
        lines.append(f"\n{nm_}: empty lists {np.mean(kk == 0):.4f}, mean list size {kk.mean():.3f}")
    Pc = prepare(dh[2], ids[2], pairs[pairs.s1_rid.isin(ids[2])])
    selc = (Pc["pmax"] >= dec["g"]) & (Pc["p"] >= dec["t"]) & (Pc["p"] >= dec["r"] * Pc["pmax"])
    nsel = np.bincount(Pc["idx"][selc], minlength=Pc["n"])
    lines.append(f"\ncurrent rule: empty lists {np.mean(nsel == 0):.4f}, mean list size {nsel.mean():.3f}")
    # per-entity comparison, paired (same S1 order: ids[2])
    tpc = np.bincount(Pc["idx"][selc], weights=Pc["label"][selc], minlength=Pc["n"])
    f_cur = np.where(Pc["ntrue"] == 0, (nsel == 0).astype(float),
                     1.25 * tpc / np.maximum(nsel + 0.25 * Pc["ntrue"], 1e-9))
    diff = f_alt[best_name] - f_cur
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    lines.append(f"\nPaired difference best - current on half 2: {diff.mean():+.5f} (std. error {se:.5f}); "
                 f"entities better {int((diff > 1e-12).sum()):,}, worse {int((diff < -1e-12).sum()):,}.")
    lines.append(f"\n**Best alternative: {best_name} ({best_val:.5f}) vs current rule ({cur:.5f}) -> "
                 f"{'BEATS the current rule' if wins else 'does NOT beat the current rule'}.**\n")
    choice = dict(method=best_name, c=c_pi if best_name == "plug-in" else c_mc, wins_on_half2=wins,
                  half2_best=float(best_val), half2_current=float(cur), n_mc=N_MC, seed=SEED)
    if wins:
        # refit the calibration on ALL of fold 0 (c stays as tuned on half 1)
        cal_all = fit_calibration(d.p.values, d.label.values)
        with open(os.path.join(MODELS, "calib_isotonic.json"), "w", encoding="utf-8") as f:
            json.dump(cal_all, f)
        with open(os.path.join(MODELS, "decision_ef.json"), "w", encoding="utf-8") as f:
            json.dump(choice, f, indent=1)
        lines.append(f"Calibration refitted on all fold 0 ({len(cal_all['x']):,} thresholds) -> "
                     f"models/calib_isotonic.json; decision -> models/decision_ef.json: {choice}\n")
    assert abs(f_cur.mean() - cur) < 1e-9
    with open(os.path.join(REP, "m6b_expected_f.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log("\n".join(x for x in lines if x.startswith("|") or x.startswith("**")))
    log(f"tune done ({time.time() - t0:.0f}s)")


# ------------------------------------------------------------------ write test output
def stage_write(out_dir):
    from m7_infer import write_lists
    t0 = time.time()
    dec = json.load(open(os.path.join(MODELS, "decision_ef.json")))
    assert dec["wins_on_half2"], "alternative did not win on validation; not writing"
    cal = json.load(open(os.path.join(MODELS, "calib_isotonic.json")))
    out_dir = os.path.abspath(out_dir)
    for protected in ("output", "output_v1", "output_baseline"):
        assert out_dir != os.path.abspath(os.path.join(ROOT, "submission", protected)), "protected folder"
    os.makedirs(out_dir, exist_ok=True)
    s1_rid = pq.read_table(pq_path("test", 1), columns=["rid"]).column("rid").to_numpy()
    s1_ids = pq.read_table(pq_path("test", 1), columns=["entity_id"]).column("entity_id").to_pylist()
    nb = 18
    ranges = [(int(r[0]), int(r[-1]) + 1) for r in np.array_split(np.arange(len(s1_rid)), nb)]
    pool = np.concatenate([pq.read_table(pq_path("test", s), columns=["rid"]).column("rid").to_numpy() for s in (2, 3)])
    pidx = pd.Index(pool)
    # pass 1: one-owner winner of every pool record over ALL candidate rows (same semantics as
    # m6_decide.one_owner on the concatenation of the batches: highest p, first occurrence on ties)
    best = np.full(len(pool), -1.0, np.float32)
    owner = np.full(len(pool), -1, np.int64)
    n_rows = 0
    for b in range(nb):
        pred = pd.read_parquet(os.path.join(DATA, f"test_pred_{b}.parquet"))
        n_rows += len(pred)
        mi = pidx.get_indexer(pred.m_rid.values)
        assert (mi >= 0).all()
        g = pd.DataFrame({"mi": mi, "p": pred.p.values}).groupby("mi").p.idxmax()
        rows = g.values
        m_, p_ = mi[rows], pred.p.values[rows]
        upd = p_ > best[m_]
        best[m_[upd]] = p_[upd]
        owner[m_[upd]] = pred.s1_rid.values[rows[upd]]
        del pred, mi, g
        gc.collect()
    log(f"pass 1 (one-owner winners) over {n_rows:,} rows ({time.time() - t0:.0f}s)")
    # pass 2: per batch, keep owner rows, calibrate, choose
    sel_s1, sel_m, n_after, kall = [], [], 0, []
    for b, (lo, hi) in enumerate(ranges):
        pred = pd.read_parquet(os.path.join(DATA, f"test_pred_{b}.parquet"))
        mi = pidx.get_indexer(pred.m_rid.values)
        pred = pred[owner[mi] == pred.s1_rid.values].reset_index(drop=True)
        n_after += len(pred)
        pred["q"] = calibrate(cal, pred.p.values)
        ids_b = s1_rid[lo:hi]
        Q, M, nc = to_matrix(pred, ids_b)
        if dec["method"] == "plug-in":
            k = plugin_choice(Q, nc, [dec["c"]])[0]
        else:
            k = mc_choice(Q, nc, [dec["c"]], seed=dec["seed"] + 100 + b)[0]
        take = np.arange(KMAX)[None, :] < k[:, None]
        r_, c_ = np.nonzero(take)
        sel_s1.append(r_ + lo)
        sel_m.append(M[r_, c_])
        kall.append(k)
        del pred, Q, M, take
        gc.collect()
        log(f"  batch {b + 1}/{nb}: matches {int(k.sum()):,} ({time.time() - t0:.0f}s)")
    pos, m = np.concatenate(sel_s1), np.concatenate(sel_m)
    kall = np.concatenate(kall)
    assert (m >= 0).all()
    with open(os.path.join(out_dir, "matching_results.tsv"), "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        write_lists(f, s1_ids, 0, len(s1_rid), pos, m)
    src = os.path.join(ROOT, "submission", "output", "candidate_pairs.tsv")
    shutil.copyfile(src, os.path.join(out_dir, "candidate_pairs.tsv"))  # candidates unchanged
    country = pq.read_table(pq_path("test", 1), columns=["country"]).column("country").to_pandas()
    lines = [f"\n## Test output ({out_dir}), method {dec['method']} c={dec['c']}\n",
             f"candidate rows {n_rows:,}; after one-owner {n_after:,}; matches {len(m):,} "
             f"(mean {kall.mean():.3f} per S1, max {kall.max()}); empty rows {np.mean(kall == 0):.4f}. "
             f"candidate_pairs.tsv copied byte for byte from submission/output. ({time.time() - t0:.0f}s)\n",
             "| country | S1 | empty share | mean matches |", "|---|---|---|---|"]
    for c, g in pd.Series(kall).groupby(country.values):
        lines.append(f"| {c} | {len(g):,} | {(g == 0).mean():.4f} | {g.mean():.3f} |")
    with open(os.path.join(REP, "m6b_expected_f.md"), "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log("\n".join(lines))


if __name__ == "__main__":
    if sys.argv[1] == "tune":
        stage_tune()
    elif sys.argv[1] == "write":
        stage_write(sys.argv[2])
