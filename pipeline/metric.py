"""Macro F0.5 exactly as in the README.

Per S1 entity with truth set T and prediction set P:
  T empty  -> 1.0 if P empty else 0.0
  T non-empty, P empty -> 0.0
  otherwise F0.5 = 1.25*P*R / (0.25*P + R), which simplifies to 1.25*tp / (|P| + 0.25*|T|).
The final score is the plain mean over ALL S1 entities of the evaluation set.
"""
import numpy as np
import pandas as pd


def per_entity_f05(s1_ids, pred_pairs, true_pairs):
    """s1_ids: array of every S1 rid in the evaluation set.
    pred_pairs / true_pairs: DataFrames with columns s1_rid, m_rid (int64), no duplicate rows.
    Returns a DataFrame indexed by s1_rid with n_pred, n_true, tp, f05."""
    s1_ids = pd.Index(np.asarray(s1_ids, dtype=np.int64), name="s1_rid")
    pred = pred_pairs[["s1_rid", "m_rid"]].drop_duplicates()
    true = true_pairs[["s1_rid", "m_rid"]].drop_duplicates()
    pred = pred[pred.s1_rid.isin(s1_ids)]
    true = true[true.s1_rid.isin(s1_ids)]
    out = pd.DataFrame(index=s1_ids)
    out["n_pred"] = pred.groupby("s1_rid").size().reindex(s1_ids, fill_value=0).values
    out["n_true"] = true.groupby("s1_rid").size().reindex(s1_ids, fill_value=0).values
    tp = pred.merge(true, on=["s1_rid", "m_rid"]).groupby("s1_rid").size()
    out["tp"] = tp.reindex(s1_ids, fill_value=0).values
    npred, ntrue, tpv = (out[c].values.astype(np.float64) for c in ("n_pred", "n_true", "tp"))
    f = np.where(ntrue == 0, (npred == 0).astype(np.float64),
                 np.divide(1.25 * tpv, npred + 0.25 * ntrue, out=np.zeros_like(tpv),
                           where=(npred + 0.25 * ntrue) > 0))
    out["f05"] = f
    return out


def macro_f05(s1_ids, pred_pairs, true_pairs):
    return float(per_entity_f05(s1_ids, pred_pairs, true_pairs)["f05"].mean())


def _f05_sets(P, T):
    """Direct set-based reference implementation (slow, for testing)."""
    P, T = set(P), set(T)
    if not T:
        return 1.0 if not P else 0.0
    if not P:
        return 0.0
    tp = len(P & T)
    if tp == 0:
        return 0.0
    p, r = tp / len(P), tp / len(T)
    return 1.25 * p * r / (0.25 * p + r)


def _self_test():
    # README example: predict [47, 193, 812], truth [47, 812] -> 0.714
    ex = per_entity_f05([1], pd.DataFrame({"s1_rid": [1, 1, 1], "m_rid": [47, 193, 812]}),
                        pd.DataFrame({"s1_rid": [1, 1], "m_rid": [47, 812]}))
    assert abs(ex.f05.iloc[0] - 0.7142857) < 1e-6, ex
    assert round(ex.f05.iloc[0], 3) == 0.714
    # singleton rules
    s = per_entity_f05([1, 2, 3, 4],
                       pd.DataFrame({"s1_rid": [2, 4], "m_rid": [9, 5]}),
                       pd.DataFrame({"s1_rid": [3, 4], "m_rid": [7, 8]}))
    # 1: T empty P empty -> 1 ; 2: T empty, P non-empty -> 0 ; 3: T non-empty P empty -> 0 ; 4: tp=0 -> 0
    assert list(s.f05) == [1.0, 0.0, 0.0, 0.0], s
    # random comparison with the set implementation
    rng = np.random.default_rng(0)
    pr, tr, ref = [], [], []
    for e in range(2000):
        P = set(rng.integers(0, 8, rng.integers(0, 5)).tolist())
        T = set(rng.integers(0, 8, rng.integers(0, 5)).tolist())
        pr += [(e, m) for m in P]
        tr += [(e, m) for m in T]
        ref.append(_f05_sets(P, T))
    got = per_entity_f05(np.arange(2000), pd.DataFrame(pr, columns=["s1_rid", "m_rid"]),
                         pd.DataFrame(tr, columns=["s1_rid", "m_rid"]))
    assert np.allclose(got.f05.values, ref)
    print("metric self-test OK: README example =", round(ex.f05.iloc[0], 4))


if __name__ == "__main__":
    _self_test()
