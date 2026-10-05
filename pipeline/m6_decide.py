"""M6: decision layer, tuned directly on validation macro F0.5.

For each S1 entity with candidate probabilities p (from the LightGBM matcher):
  1. (optional) one-owner rule: a pool record may belong to only ONE S1 -> it is kept only for
     the S1 that gives it the highest probability (EDA showed each S2/S3 id has at most one S1).
  2. empty gate: if the best probability < g, predict nothing (the entity looks like a singleton).
  3. otherwise keep every candidate with p >= t (absolute) AND p >= r * best (relative to best).
Grid search over (g, t, r); the score is the macro F0.5 over ALL validation S1 entities, including
those for which blocking found no candidate (they get an empty prediction).
Writes reports/m6_decision.md and models/decision.json.
"""
import itertools
import json
import os
import time

import numpy as np
import pandas as pd

from common import DATA, WORK, log

REP = os.path.join(WORK, "reports")
MODELS = os.path.join(WORK, "models")


def one_owner(df):
    """Keep each m_rid only under the S1 with the highest p."""
    best = df.groupby("m_rid")["p"].transform("max")
    return df[df.p >= best].drop_duplicates("m_rid")


def prepare(pred, s1_ids, true_pairs):
    s1_ids = np.asarray(s1_ids)
    ntrue = true_pairs.groupby("s1_rid").size().reindex(s1_ids, fill_value=0)
    code = pd.Index(s1_ids)
    idx = code.get_indexer(pred.s1_rid.values)
    assert (idx >= 0).all()
    pmax = pred.groupby("s1_rid")["p"].transform("max").values
    return dict(idx=idx, p=pred.p.values, pmax=pmax, label=pred.label.values.astype(np.int64),
                ntrue=ntrue.values.astype(np.float64), n=len(s1_ids))


def score(P, g, t, r):
    sel = (P["pmax"] >= g) & (P["p"] >= t) & (P["p"] >= r * P["pmax"])
    npred = np.bincount(P["idx"][sel], minlength=P["n"]).astype(np.float64)
    tp = np.bincount(P["idx"][sel], weights=P["label"][sel], minlength=P["n"])
    nt = P["ntrue"]
    f = np.where(nt == 0, (npred == 0).astype(float),
                 np.divide(1.25 * tp, npred + 0.25 * nt, out=np.zeros_like(tp), where=(npred + 0.25 * nt) > 0))
    return f.mean()


def grid(P):
    G = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    T = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7]
    R = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    best = (-1, None)
    for g, t, r in itertools.product(G, T, R):
        s = score(P, g, t, r)
        if s > best[0]:
            best = (s, (g, t, r))
    # local refinement around the best point
    g0, t0, r0 = best[1]
    for g, t, r in itertools.product(np.clip(g0 + np.arange(-0.05, 0.051, 0.01), 0, 1),
                                     np.clip(t0 + np.arange(-0.05, 0.051, 0.01), 0.01, 1),
                                     np.clip(r0 + np.arange(-0.1, 0.101, 0.025), 0, 1)):
        s = score(P, g, t, r)
        if s > best[0]:
            best = (s, (round(float(g), 3), round(float(t), 3), round(float(r), 3)))
    return best


if __name__ == "__main__":
    t0 = time.time()
    vp = pd.read_parquet(os.path.join(DATA, "val_pred.parquet"))
    s1 = pd.read_parquet(os.path.join(DATA, "trainval_s1.parquet"))
    val_ids = s1.s1_rid[s1.fold == 0].values
    pairs = pd.read_parquet(os.path.join(DATA, "train_gt_pairs.parquet"))
    pairs = pairs[pairs.s1_rid.isin(val_ids)]
    floor = (pairs.groupby("s1_rid").size().reindex(val_ids, fill_value=0) == 0).mean()
    lines = [f"# M6 decision layer\n\nvalidation S1 entities: {len(val_ids):,}; "
             f"predict-nothing floor F0.5 = {floor:.4f}\n"]
    rec = vp.label.sum() / len(pairs)
    lines.append(f"recall ceiling of the scored candidates: {rec:.4f}\n")
    results = {}
    for name, df in [("plain", vp), ("one_owner", one_owner(vp))]:
        P = prepare(df, val_ids, pairs)
        s, (g, t, r) = grid(P)
        results[name] = dict(f05=s, g=g, t=t, r=r)
        lines.append(f"- **{name}**: best macro F0.5 = {s:.5f} at gate g={g}, abs t={t}, rel r={r}")
        # simple reference points
        lines.append(f"  - top-1 only if p>=0.5: {score(P, 0.5, 0.5, 1.0):.5f}; "
                     f"all p>=0.5: {score(P, 0.0, 0.5, 0.0):.5f}")
    best_name = max(results, key=lambda k: results[k]["f05"])
    dec = {**results[best_name], "one_owner": best_name == "one_owner"}
    lines.append(f"\nchosen: {best_name} -> {dec}\n")
    with open(os.path.join(REP, "m6_decision.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(MODELS, "decision.json"), "w", encoding="utf-8") as f:
        json.dump(dec, f, indent=1)
    log("\n".join(lines))
    log(f"done in {time.time() - t0:.0f}s")
