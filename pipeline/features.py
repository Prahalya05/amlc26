"""Pairwise features for (S1 record, candidate S2/S3 record) pairs.

Groups of features (all country-agnostic: country is never a feature, so France is handled the
same way as US/India):
  blocking : score, nkeys, rank from blocking + relative versions within the S1's candidate list
  name     : rapidfuzz ratio / token_set / token_sort / partial / Jaro-Winkler on the normalised
             name and on the "core" name (legal words removed); token Jaccard; exact equality;
             consonant-skeleton similarity (works across scripts); char 3-gram TF-IDF cosine
  address  : rapidfuzz ratio / token_set / partial; word Jaccard; word TF-IDF cosine;
             address-number overlap / mismatches / first number equal
  other    : source (S2 or S3), script flags (Latin vs non-Latin, cross-script), lengths
  context  : for each feature f in a key set, f - max(f over this S1's candidates) and the rank
             of f inside the S1's list ("is this the best-looking candidate?")
"""
import gc
import os
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize as l2norm

from common import DATA, log, pq_path
from keys import skel_strings

COLS = ["rid", "name_n", "name_core", "addr_n", "addr_nums", "name_nonlatin"]


def load_side(split, srcs):
    tabs = [pq.read_table(pq_path(split, s), columns=COLS) for s in srcs]
    t = pa.concat_tables(tabs).combine_chunks()
    return t


class Tfidf:
    """Hashed TF-IDF; IDF fitted on (a sample of) the unlabeled pool of the split being processed."""

    def __init__(self, analyzer, ngram, n_features=2**20):
        self.hv = HashingVectorizer(analyzer=analyzer, ngram_range=ngram, n_features=n_features,
                                    alternate_sign=False, norm=None, lowercase=False,
                                    token_pattern=r"(?u)\b\w+\b")
        self.tf = TfidfTransformer(sublinear_tf=True)

    def fit(self, texts):
        self.tf.fit(self.hv.transform(texts))
        return self

    def vec(self, texts):
        return l2norm(self.tf.transform(self.hv.transform(texts)))


def rowcos(A, B):
    return np.asarray(A.multiply(B).sum(axis=1)).ravel().astype(np.float32)


def fit_tfidf(pool, n=500_000, seed=0):
    idx = np.random.default_rng(seed).choice(pool.num_rows, min(n, pool.num_rows), replace=False)
    sub = pool.take(pa.array(idx))
    tn = Tfidf("char_wb", (3, 3), n_features=2**18).fit(sub.column("name_n").to_pylist())
    ta = Tfidf("word", (1, 1)).fit(sub.column("addr_n").to_pylist())
    return tn, ta


def _sets(strs):
    return [set(s.split()) if s else set() for s in strs]


def _jacc(A, B):
    return np.array([len(a & b) / len(a | b) if (a or b) else 0.0 for a, b in zip(A, B)], np.float32)


class Side:
    """Per-record derived values, computed once per UNIQUE record of a chunk and then indexed."""

    def __init__(self, t, idx, tfn, tfa):
        u, inv = np.unique(idx, return_inverse=True)
        s = t.take(pa.array(u))
        self.inv = inv
        self.n = s.column("name_n").to_pylist()
        self.c = s.column("name_core").to_pylist()
        self.a = s.column("addr_n").to_pylist()
        self.d = s.column("addr_nums").to_pylist()
        self.k = skel_strings(s.column("name_n")).to_pylist()
        self.cs = _sets(self.c)
        self.ks = _sets(self.k)
        self.aw = [{w for w in x if not w.isdigit()} for x in _sets(self.a)]
        self.ds = _sets(self.d)
        self.nl = s.column("name_nonlatin").to_numpy(zero_copy_only=False)
        self.vn = tfn.vec(self.n)
        self.va = tfa.vec(self.a)

    def get(self, attr):
        v = getattr(self, attr)
        return [v[i] for i in self.inv]


def pair_features(s1t, poolt, s1_idx, p_idx, tfn, tfa):
    """s1_idx / p_idx: row positions into s1t / poolt. Returns a DataFrame of raw pair features."""
    A = Side(s1t, s1_idx, tfn, tfa)
    B = Side(poolt, p_idx, tfn, tfa)
    n1, n2 = A.get("n"), B.get("n")
    c1, c2 = A.get("c"), B.get("c")
    a1, a2 = A.get("a"), B.get("a")
    d1, d2 = A.get("d"), B.get("d")
    k1, k2 = A.get("k"), B.get("k")
    W = dict(workers=-1, dtype=np.float32)
    f = {}
    f["name_ratio"] = process.cpdist(n1, n2, scorer=fuzz.ratio, **W)
    f["name_tset"] = process.cpdist(n1, n2, scorer=fuzz.token_set_ratio, **W)
    f["name_tsort"] = process.cpdist(n1, n2, scorer=fuzz.token_sort_ratio, **W)
    f["name_partial"] = process.cpdist(n1, n2, scorer=fuzz.partial_ratio, **W)
    f["core_ratio"] = process.cpdist(c1, c2, scorer=fuzz.ratio, **W)
    f["core_tset"] = process.cpdist(c1, c2, scorer=fuzz.token_set_ratio, **W)
    f["core_jw"] = process.cpdist(c1, c2, scorer=JaroWinkler.normalized_similarity, **W)
    f["core_eq"] = np.array([x == y and x != "" for x, y in zip(c1, c2)], np.float32)
    cs1, cs2 = A.get("cs"), B.get("cs")
    f["core_jacc"] = _jacc(cs1, cs2)
    f["skel_ratio"] = process.cpdist(k1, k2, scorer=fuzz.ratio, **W)
    f["skel_tset"] = process.cpdist(k1, k2, scorer=fuzz.token_set_ratio, **W)
    f["skel_jacc"] = _jacc(A.get("ks"), B.get("ks"))
    f["addr_ratio"] = process.cpdist(a1, a2, scorer=fuzz.ratio, **W)
    f["addr_tset"] = process.cpdist(a1, a2, scorer=fuzz.token_set_ratio, **W)
    f["addr_partial"] = process.cpdist(a1, a2, scorer=fuzz.partial_ratio, **W)
    as1, as2 = A.get("aw"), B.get("aw")
    f["addr_word_jacc"] = _jacc(as1, as2)
    f["addr_word_shared"] = np.array([len(x & y) for x, y in zip(as1, as2)], np.float32)
    ds1, ds2 = A.get("ds"), B.get("ds")
    f["num_n1"] = np.array([len(x) for x in ds1], np.float32)
    f["num_n2"] = np.array([len(x) for x in ds2], np.float32)
    f["num_shared"] = np.array([len(x & y) for x, y in zip(ds1, ds2)], np.float32)
    f["num_jacc"] = _jacc(ds1, ds2)
    f["num_miss1"] = np.array([len(x - y) for x, y in zip(ds1, ds2)], np.float32)
    f["num_miss2"] = np.array([len(y - x) for x, y in zip(ds1, ds2)], np.float32)
    f["num_first_eq"] = np.array([(x.split()[:1] == y.split()[:1]) and x != "" for x, y in zip(d1, d2)],
                                 np.float32)
    f["num_first_in2"] = np.array([(x.split()[0] in y) if x else False for x, y in zip(d1, ds2)],
                                  np.float32)
    f["len_core1"] = np.array([len(x) for x in c1], np.float32)
    f["len_core2"] = np.array([len(x) for x in c2], np.float32)
    f["len_addr1"] = np.array([len(x) for x in a1], np.float32)
    f["len_addr2"] = np.array([len(x) for x in a2], np.float32)
    f["ntok_core1"] = np.array([len(x) for x in cs1], np.float32)
    f["ntok_core2"] = np.array([len(x) for x in cs2], np.float32)
    nl1 = A.nl[A.inv]
    nl2 = B.nl[B.inv]
    f["nonlatin1"] = nl1.astype(np.float32)
    f["nonlatin2"] = nl2.astype(np.float32)
    f["cross_script"] = (nl1 != nl2).astype(np.float32)
    f["tfidf_name"] = rowcos(A.vn[A.inv], B.vn[B.inv])
    f["tfidf_addr"] = rowcos(A.va[A.inv], B.va[B.inv])
    return pd.DataFrame(f)


CONTEXT = ["score", "name_tset", "core_jw", "skel_ratio", "addr_tset", "num_jacc", "tfidf_name",
           "tfidf_addr", "combo"]


def add_context(df):
    """Relative-to-the-best features inside each S1's candidate list."""
    df["combo"] = (df.tfidf_name + df.tfidf_addr + df.skel_ratio / 100).astype(np.float32)
    g = df.groupby("s1_row", sort=False)
    df["n_cands"] = g["pidx"].transform("size").astype(np.float32)
    for c in CONTEXT:
        mx = g[c].transform("max")
        df[f"{c}_gap"] = (df[c] - mx).astype(np.float32)
        df[f"{c}_rk"] = g[c].rank(ascending=False, method="min").astype(np.float32)
    # how many candidates of this S1 look like strong matches (dup clusters in S2/S3)
    df["_strong"] = (df["combo"] > 1.2).astype(np.float32)
    df["n_strong"] = df.groupby("s1_row", sort=False)["_strong"].transform("sum").astype(np.float32)
    df.drop(columns="_strong", inplace=True)
    return df


def build_features(split, cand, out_path, chunk=500_000):
    """cand: DataFrame with s1_row, pidx, score, nkeys, rank, m_rid.
    Writes cand + features to out_path (parquet), processing whole S1 lists chunk by chunk
    so that the context features (which look at one S1's full list) stay exact."""
    t0 = time.time()
    s1t = load_side(split, [1])
    poolt = load_side(split, [2, 3])
    tfn, tfa = fit_tfidf(poolt)
    log(f"  feature tables loaded + tfidf fitted ({time.time() - t0:.0f}s)")
    cand = cand.sort_values(["s1_row", "rank"]).reset_index(drop=True)
    # chunk boundaries at S1 changes
    starts = np.r_[0, np.flatnonzero(np.diff(cand.s1_row.values)) + 1]
    cuts = [0]
    for s in starts:
        if s - cuts[-1] >= chunk:
            cuts.append(s)
    cuts.append(len(cand))
    writer = None
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        c = cand.iloc[lo:hi].reset_index(drop=True)
        feats = pair_features(s1t, poolt, c.s1_row.values, c.pidx.values, tfn, tfa)
        df = pd.concat([c, feats], axis=1)
        df["msrc3"] = (df.m_rid.values // 10**10 == 3).astype(np.float32)
        df["score"] = df["score"].astype(np.float32)
        df["nkeys"] = df["nkeys"].astype(np.float32)
        df["rank"] = df["rank"].astype(np.float32)
        df = add_context(df)
        tab = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, tab.schema, compression="zstd")
        writer.write_table(tab)
        log(f"    features {hi:,}/{len(cand):,} pairs ({time.time() - t0:.0f}s)")
        del c, feats, df, tab
        gc.collect()
    if writer is not None:
        writer.close()
    del s1t, poolt
    gc.collect()
    log(f"  features for {len(cand):,} pairs in {time.time() - t0:.0f}s -> {out_path}")


def feature_columns(df):
    skip = {"s1_row", "pidx", "m_rid", "s1_rid", "label"}
    return [c for c in df.columns if c not in skip]
