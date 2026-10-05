"""M2: exploratory statistics that drive the design. Writes reports/eda_report.md (UTF-8)."""
import gc
import os
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from common import DATA, WORK, log, pq_path

REP = os.path.join(WORK, "reports")
os.makedirs(REP, exist_ok=True)
lines = []


def out(s=""):
    lines.append(str(s))


def md(df):
    """Tiny markdown table writer (avoids the optional 'tabulate' dependency)."""
    df = df.reset_index()
    rows = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    rows += ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join(rows)


CTY = {}  # country string -> small int, shared by all files (keeps RAM low: no 10M Python strings)


def meta(split, src):
    t = pq.read_table(pq_path(split, src), columns=["rid", "country", "name_deva", "name_nonlatin",
                                                    "addr_deva", "addr_nonlatin"], read_dictionary=["country"])
    df = t.to_pandas()
    cats = df.country.cat.categories
    gid = np.array([CTY.setdefault(c, len(CTY)) for c in cats] + [-1], dtype=np.int16)
    df["country"] = gid[df.country.cat.codes.to_numpy()]
    return df


def cname(i):
    return {v: k for k, v in CTY.items()}.get(i, i)


t0 = time.time()
gt1 = pd.read_parquet(os.path.join(DATA, "train_gt_s1.parquet"))
pairs = pd.read_parquet(os.path.join(DATA, "train_gt_pairs.parquet"))
s1 = meta("train", 1)
assert len(s1) == len(gt1) and set(s1.rid) == set(gt1.s1_rid)
s1 = s1.merge(gt1, left_on="rid", right_on="s1_rid").drop(columns="s1_rid")

out("# M2 EDA report (train unless stated)\n")
out(f"S1 entities: {len(s1):,}; true pairs: {len(pairs):,}")
sing = (s1.n_match == 0).mean()
out(f"**Singleton fraction = predict-nothing macro F0.5 = {sing:.4f}**\n")
vc = s1.n_match.clip(upper=10).value_counts().sort_index()
out("## Matches per S1 (10 = 10+)\n\n| n_match | S1 count | share |\n|---|---|---|")
for k, v in vc.items():
    out(f"| {k} | {v:,} | {v / len(s1):.4f} |")
out(f"\nmean {s1.n_match.mean():.3f}, mean among non-singletons "
    f"{s1.n_match[s1.n_match > 0].mean():.3f}, max {s1.n_match.max()}")

pairs["msrc"] = pairs.m_rid // 10**10
out("\n## Source of matched records\n")
out(md(pairs.msrc.map({2: "S2", 3: "S3"}).value_counts(normalize=True).round(4).to_frame()))
per = pd.DataFrame({"s1_rid": pairs.s1_rid, "n2": pairs.msrc == 2, "n3": pairs.msrc == 3}).groupby("s1_rid").sum()
out(f"\nPer non-singleton S1: mean #S2 matches {per.n2.mean():.3f}, mean #S3 matches {per.n3.mean():.3f}; "
    f"share with >=1 S2 {(per.n2 > 0).mean():.4f}, share with >=1 S3 {(per.n3 > 0).mean():.4f}")
out("\nDistribution of #S2 matches per non-singleton S1: " +
    str(per.n2.clip(upper=6).value_counts().sort_index().to_dict()))
out("Distribution of #S3 matches per non-singleton S1: " +
    str(per.n3.clip(upper=6).value_counts().sort_index().to_dict()))
del per

# uniqueness of matched ids
dup = pairs.m_rid.duplicated(keep=False)
out(f"\n## Is each S2/S3 id matched to at most one S1?\n\nS2/S3 ids appearing under >1 S1: "
    f"{pairs.m_rid[dup].nunique():,} (of {pairs.m_rid.nunique():,} matched ids)")

# country
out("\n## Country\n")
s23 = pd.concat([meta("train", 2), meta("train", 3)], ignore_index=True)
s23["src"] = s23.rid // 10**10
g = s1.groupby("country").agg(S1=("rid", "size"), singleton_rate=("n_match", lambda x: (x == 0).mean()),
                              mean_matches=("n_match", "mean"))
g["S2"] = s23[s23.src == 2].country.value_counts()
g["S3"] = s23[s23.src == 3].country.value_counts()
g.index = [cname(i) for i in g.index]
g.index.name = "country"
out(md(g.round(4)))
pm = pairs.merge(s1[["rid", "country", "name_nonlatin", "name_deva", "addr_nonlatin"]].rename(
    columns={"rid": "s1_rid"}), on="s1_rid")
pm = pm.merge(s23[["rid", "country", "name_nonlatin", "name_deva", "addr_nonlatin"]].rename(
    columns={"rid": "m_rid"}), on="m_rid", suffixes=("_1", "_m"))
assert len(pm) == len(pairs)
out(f"\nTrue pairs whose S2/S3 country equals the S1 country: {(pm.country_1 == pm.country_m).mean():.6f}")
matched_share = s23.rid.isin(pairs.m_rid)
out(f"Share of S2/S3 pool records that are a true match of some train S1: {matched_share.mean():.4f} "
    f"(S2 {matched_share[s23.src == 2].mean():.4f}, S3 {matched_share[s23.src == 3].mean():.4f})")

out("\n## Scripts\n\n| file | name Devanagari | name non-Latin | addr Devanagari | addr non-Latin |\n|---|---|---|---|---|")
for sp in ("train", "test"):
    for src in (1, 2, 3):
        if not os.path.exists(pq_path(sp, src)):
            out(f"| {sp} S{src} | (not converted yet) | | | |")
            continue
        m = s1 if (sp, src) == ("train", 1) else (s23[s23.src == src] if sp == "train" else meta(sp, src))
        out(f"| {sp} S{src} | {m.name_deva.mean():.4f} | {m.name_nonlatin.mean():.4f} | "
            f"{m.addr_deva.mean():.4f} | {m.addr_nonlatin.mean():.4f} |")
        if sp == "test":
            out(f"|  countries: {m.country.value_counts().rename(cname).to_dict()} | | | | |")
out(f"\nTrue pairs where the two NAMES are in different scripts (one non-Latin, other Latin): "
    f"{(pm.name_nonlatin_1 != pm.name_nonlatin_m).mean():.4f}; "
    f"S1 name non-Latin: {pm.name_nonlatin_1.mean():.4f}; match name non-Latin: {pm.name_nonlatin_m.mean():.4f}")
out("Cross-script pairs by country: " +
    str(pm.assign(x=pm.name_nonlatin_1 != pm.name_nonlatin_m).groupby("country_1").x.mean().round(4).rename(cname).to_dict()))
out("Non-Latin match names by source: " +
    str(pm.groupby(pm.m_rid // 10**10).name_nonlatin_m.mean().round(4).to_dict()))
del s23, pm
gc.collect()

# ---- string-level statistics of true pairs on a sample (drives blocking design)
rng = np.random.default_rng(0)
samp = pairs.sample(300_000, random_state=0)
cols = ["rid", "country", "name_n", "name_core", "addr_n", "addr_nums", "name_nonlatin"]


def fetch(split, src, rids):
    want = pa.array(np.unique(rids))
    parts = [b.filter(pc.is_in(b.column("rid"), want))
             for b in pq.ParquetFile(pq_path(split, src)).iter_batches(500_000, columns=cols)]
    return pa.Table.from_batches(parts).to_pandas()


a = fetch("train", 1, samp.s1_rid.values)
b = pd.concat([fetch("train", s, samp.m_rid.values) for s in (2, 3)])
d = samp.merge(a.rename(columns={"rid": "s1_rid"}), on="s1_rid").merge(
    b.rename(columns={"rid": "m_rid"}), on="m_rid", suffixes=("_1", "_m"))


def toks(s):
    return set(s.split()) if s else set()


def share(fn):
    return float(np.mean([fn(r) for r in d.itertuples(index=False)]))


out("\n## String agreement of true pairs (300k sample)\n")
lat = d[~d.name_nonlatin_1 & ~d.name_nonlatin_m]
out(f"Latin-Latin pairs: {len(lat) / len(d):.4f} of sample")
stats = {
    "name_n exactly equal": lambda r: r.name_n_1 == r.name_n_m,
    "name_core exactly equal": lambda r: r.name_core_1 == r.name_core_m,
    "share >=1 name_core token": lambda r: bool(toks(r.name_core_1) & toks(r.name_core_m)),
    "share >=1 name_n token": lambda r: bool(toks(r.name_n_1) & toks(r.name_n_m)),
    "share >=1 address number": lambda r: bool(toks(r.addr_nums_1) & toks(r.addr_nums_m)),
    "S1 has no address number": lambda r: not r.addr_nums_1,
    "match has no address number": lambda r: not r.addr_nums_m,
    "match address empty": lambda r: not r.addr_n_m,
    "first address number equal": lambda r: bool(r.addr_nums_1) and r.addr_nums_1.split()[0] == (r.addr_nums_m.split() or [""])[0],
    "share >=1 non-number addr token": lambda r: bool({x for x in toks(r.addr_n_1) if not x.isdigit()} & {x for x in toks(r.addr_n_m) if not x.isdigit()}),
    "share >=1 name token OR >=1 addr number": lambda r: bool(toks(r.name_core_1) & toks(r.name_core_m)) or bool(toks(r.addr_nums_1) & toks(r.addr_nums_m)),
}
out("| statistic | all pairs | by country |\n|---|---|---|")
for k, fn in stats.items():
    v = np.array([fn(r) for r in d.itertuples(index=False)])
    byc = pd.Series(v).groupby(d.country_1.values).mean().round(4).to_dict()  # country strings here
    out(f"| {k} | {v.mean():.4f} | {byc} |")

# examples for the human reader
out("\n## 40 random true pairs\n")
for r in d.sample(40, random_state=1).itertuples(index=False):
    out(f"- `{r.name_n_1}` / `{r.addr_n_1}`  <->  `{r.name_n_m}` / `{r.addr_n_m}` ({r.country_1}, S{r.m_rid // 10**10})")

with open(os.path.join(REP, "eda_report.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
log(f"EDA done in {time.time() - t0:.0f}s -> reports/eda_report.md")
