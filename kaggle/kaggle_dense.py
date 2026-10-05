# Dense retriever for the Amazon ML Challenge entity-resolution data (one file for a Kaggle notebook).
#
#   Settings: GPU T4, Internet ON. Paste this file into one cell (or `%run kaggle_dense.py`).
#   SMOKE = 1 runs every step on 20k training pairs / 50k records (< 10 min) - run it first.
#   SMOKE = 0 is the full run (~3-5 h); it stops cleanly after TIME_LIMIT_H hours, keeping what is done.
#
# What it does
#   1. text = "query: " + business_name + " | " + business_address   (address "" if empty), both sides, 64 tokens.
#   2. evaluation BEFORE fine-tuning: index = all train S1s of the query's country; queries = 200k owned train pool
#      records whose owner S1 is in fold 0 (>= 20k empty-address, >= 20k cross-script); owner recall@1/3/5/10 by segment.
#   3. fine-tune intfloat/multilingual-e5-small with MultipleNegativesRankingLoss on (pool record -> owner S1) pairs
#      of part-1a S1s only (s1_folds.tsv); <= 1.5M pairs, empty-address pairs x3, every batch from one country and
#      never two records of the same S1 in a batch; batch 256, lr 3e-5, 1 epoch, fp16, 5% warmup.
#   4. evaluation AFTER fine-tuning (same queries).  The model is saved to OUT_DIR/e5_small_ft.
#   5. for every pool record of TEST, then TRAIN: its top-10 same-country S1s by cosine ->
#      OUT_DIR/dense_{split}.parquet  (record_id, s1_id, rank 1-10, cosine).
#      Parts are saved as they finish (OUT_DIR/dense_{split}_parts/); the merged file is written when a split is done.
#
# Leakage note: part-1a S1s (and their pool records) were used for fine-tuning, so the TRAIN top-10 lists of
# records owned by part-1a S1s are "in-sample". Use part 1b / fold 0 for anything that is trained or evaluated
# downstream.
import gc
import glob
import os
import re
import time
from collections import deque

import numpy as np
import pandas as pd

DATA_DIR = "/kaggle/input"            # searched recursively for the 7 competition TSVs and s1_folds.tsv
OUT_DIR = "/kaggle/working"
SMOKE = int(os.environ.get("SMOKE", "1"))
# FAST = 1: skip both evaluations, do TRAIN before TEST, and search only the pool records listed in
# v3_open_train.txt / v3_open_test.txt (the records v3 assigns to nobody) if those files are in DATA_DIR
FAST = int(os.environ.get("FAST", "1"))
MODEL_NAME = "intfloat/multilingual-e5-small"
TIME_LIMIT_H = 11.0
MAX_TOKENS = 64
TOPK = 10
BATCH = 256
LR = 3e-5
WARMUP = 0.05
MAX_PAIRS = 1_500_000
EMPTY_REPEAT = 3
N_EVAL = {"empty": 25_000, "cross": 25_000, "other": 150_000}
ENC_BATCH = 512
POOL_CHUNK = 200_000                  # pool records encoded + searched per step
Q_CHUNK = 1024                        # queries per GPU matmul (1024 x 1.3M fp16 sims ~ 2.7 GB)
SEED = 2026
if SMOKE:
    MAX_PAIRS, N_EVAL, POOL_CHUNK = 20_000, {"empty": 1_000, "cross": 1_000, "other": 3_000}, 25_000
SMOKE_RECORDS = 50_000                # smoke: pool records per split and S1 index size
T0 = time.time()
NONLATIN = re.compile(r"[Ͱ-῿　-￿]")
os.makedirs(OUT_DIR, exist_ok=True)


def log(*a):
    print(time.strftime("%H:%M:%S"), f"[{(time.time() - T0) / 60:6.1f} min]", *a, flush=True)


def out_of_time():
    return time.time() - T0 > TIME_LIMIT_H * 3600


# ------------------------------------------------------------------ data
def find(name):
    hits = glob.glob(os.path.join(DATA_DIR, "**", name), recursive=True)
    assert hits, f"{name} not found under {DATA_DIR}"
    return hits[0]


def read_tsv(name):
    import csv
    return pd.read_csv(find(name), sep="\t", quoting=csv.QUOTE_NONE, na_filter=False, dtype=str, encoding="utf-8",
                       nrows=int(os.environ.get("DENSE_NROWS", "0")) or None)


def read_tsv_plain(name):
    """one id per line, no header"""
    with open(find(name), encoding="utf-8") as f:
        return [x.strip() for x in f if x.strip()]


def records(split, src):
    d = read_tsv(f"{split}_source{src}.tsv")
    d.columns = ["id", "name", "addr", "country"]
    return d


def add_text(d):
    """model text + segment flags"""
    d["text"] = "query: " + d["name"] + " | " + d["addr"].where(d["addr"].str.strip() != "", "")
    d["empty_addr"] = ~d["addr"].str.contains(r"[^\W_]", regex=True)
    d["nonlatin"] = d["name"].map(lambda s: bool(NONLATIN.search(s)))
    return d


def load(split):
    s1 = add_text(records(split, 1))
    pool = pd.concat([add_text(records(split, 2)).assign(src="S2"), add_text(records(split, 3)).assign(src="S3")],
                     ignore_index=True)
    log(f"{split}: S1 {len(s1):,}, pool {len(pool):,}")
    return s1, pool


def owners():
    """pool record id -> owner S1 id (train ground truth, exploded)"""
    g = read_tsv("train_ground_truth.tsv")
    g.columns = ["s1", "m"]
    g = g[g.m != ""]
    g = g.assign(m=g.m.str.split(",")).explode("m")
    return g.rename(columns={"m": "id", "s1": "owner"})[["id", "owner"]].reset_index(drop=True)


# ------------------------------------------------------------------ model helpers
def get_model(path=MODEL_NAME):
    import torch
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(path, device="cuda" if torch.cuda.is_available() else "cpu")
    m.max_seq_length = MAX_TOKENS
    return m


def encode(model, texts, chunk=250_000):
    """fp16, L2-normalised embeddings on the GPU. sentence-transformers sorts each call by length."""
    import torch
    out = []
    for a in range(0, len(texts), chunk):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            e = model.encode(list(texts[a:a + chunk]), batch_size=ENC_BATCH, convert_to_tensor=True,
                             normalize_embeddings=True, show_progress_bar=False)
        out.append(e.half())
    return torch.cat(out) if out else None


def search(q, idx, k=TOPK):
    """top-k rows of idx for every row of q (cosine = dot product of normalised vectors)."""
    import torch
    k = min(k, idx.shape[0])
    I, S = [], []
    with torch.inference_mode():
        for a in range(0, q.shape[0], Q_CHUNK):
            s, i = torch.topk(q[a:a + Q_CHUNK] @ idx.T, k, dim=1)
            I.append(i.cpu().numpy())
            S.append(s.float().cpu().numpy())
    return np.concatenate(I), np.concatenate(S)


# ------------------------------------------------------------------ evaluation
def eval_queries(pool, s1, rng):
    """200k owned pool records whose owner is a fold-0 S1: strata empty-address / cross-script / other,
    each with a weight = stratum size / sample size (so 'all' and per-country rows are unbiased)."""
    q = pool[pool.owner_fold == 0].copy()
    q["stratum"] = np.where(q.empty_addr, "empty", np.where(q.nonlatin, "cross", "other"))
    parts = []
    for st, n in N_EVAL.items():
        g = q[q.stratum == st]
        take = g.sample(min(n, len(g)), random_state=int(rng.integers(1 << 30)))
        parts.append(take.assign(w=len(g) / max(len(take), 1)))
    return pd.concat(parts, ignore_index=True)


def evaluate(model, s1, queries, tag):
    t0 = time.time()
    ranks = np.full(len(queries), 99, np.int16)
    for c, g in queries.groupby("country"):
        idx_rows = np.flatnonzero((s1.country == c).values)
        emb_i = encode(model, s1.text.values[idx_rows])
        emb_q = encode(model, g.text.values)
        I, _ = search(emb_q, emb_i)
        own = pd.Index(s1.id.values[idx_rows]).get_indexer(g.owner.values)
        hit = I == own[:, None]
        r = np.where(hit.any(1), hit.argmax(1) + 1, 99)
        ranks[queries.index.get_indexer(g.index)] = r
        del emb_i, emb_q
        gc.collect()
    q = queries.assign(rank=ranks)
    segs = {"all": np.ones(len(q), bool), "empty address": q.empty_addr.values, "address present": ~q.empty_addr.values,
            "cross-script": q.nonlatin.values, "same script": ~q.nonlatin.values}
    for c in sorted(q.country.unique()):
        segs[f"country {c}"] = (q.country == c).values
    segs["S2"], segs["S3"] = (q.src == "S2").values, (q.src == "S3").values
    rows = []
    for name, m in segs.items():
        w, r = q.w.values[m], q["rank"].values[m]
        rows.append({"segment": name, "queries": int(m.sum()),
                     **{f"@{k}": float((w * (r <= k)).sum() / max(w.sum(), 1e-9)) for k in (1, 3, 5, 10)}})
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT_DIR, f"eval_{tag}.csv"), index=False)
    log(f"EVAL {tag} (owner recall@k, weighted to the population; {time.time() - t0:.0f}s)\n" +
        t.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return t


# ------------------------------------------------------------------ fine-tuning
def make_batches(df, rng):
    """Batches of BATCH pairs, each from ONE country, never two pairs of the same S1. Conflicting pairs are
    pushed back to the end of the queue (O(n)); tiny leftover batches (< BATCH/8) are dropped."""
    out = []
    for _, g in df.groupby("country"):
        order = rng.permutation(len(g))
        q = deque(zip(g.index.values[order], g.owner.values[order]))
        while q:
            cur, seen, skipped = [], set(), []
            while q and len(cur) < BATCH:
                i, s = q.popleft()
                if s in seen:
                    skipped.append((i, s))
                else:
                    cur.append(i)
                    seen.add(s)
            q.extend(skipped)
            if len(cur) == BATCH or len(cur) >= BATCH // 8:
                out.append(cur)
    order = rng.permutation(len(out))
    return [out[i] for i in order]


def training_pairs(pool, rng):
    d = pool[pool.owner_part == "1a"][["text", "owner", "owner_text", "country", "empty_addr"]]
    if len(d) > MAX_PAIRS:
        d = d.sample(MAX_PAIRS, random_state=int(rng.integers(1 << 30)))
    d = pd.concat([d] + [d[d.empty_addr]] * (EMPTY_REPEAT - 1), ignore_index=True)
    return d


def finetune(model, pairs, rng):
    import torch
    from sentence_transformers import losses
    from transformers import get_linear_schedule_with_warmup
    batches = make_batches(pairs, rng)
    n = sum(len(b) for b in batches)
    log(f"fine-tuning: {len(pairs):,} pairs -> {len(batches):,} batches ({n:,} pairs used; "
        f"{pairs.empty_addr.mean():.3f} empty-address after x{EMPTY_REPEAT})")
    loss_fn = losses.MultipleNegativesRankingLoss(model)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    sch = get_linear_schedule_with_warmup(opt, int(WARMUP * len(batches)), len(batches))
    try:
        scaler = torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler()
    model.train()
    dev = model.device
    A, P = pairs.text.values, pairs.owner_text.values
    run = 0.0
    prep = getattr(model, "preprocess", None) or model.tokenize  # newer sentence-transformers renamed tokenize
    for step, b in enumerate(batches):
        feats = []
        for texts in (A[b], P[b]):
            f = prep(list(texts))
            # newer versions put non-tensor entries (e.g. strings) in the features: move only the tensors
            feats.append({k: (v.to(dev) if hasattr(v, "to") else v) for k, v in f.items()})
        with torch.autocast("cuda", dtype=torch.float16):
            loss = loss_fn(feats, None)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sch.step()
        run = 0.98 * run + 0.02 * loss.item() if step else loss.item()
        if step % 100 == 0 or step == len(batches) - 1:
            log(f"  step {step + 1}/{len(batches)} loss {run:.4f}")
    model.eval()
    return model


# ------------------------------------------------------------------ top-10 S1s for every pool record
def dense_topk(model, split, s1, pool):
    import pyarrow as pa
    import pyarrow.parquet as pq
    part_dir = os.path.join(OUT_DIR, f"dense_{split}_parts")
    os.makedirs(part_dir, exist_ok=True)
    t0 = time.time()
    n_done = 0
    for c in sorted(pool.country.unique()):
        idx_rows = np.flatnonzero((s1.country == c).values)
        prow = np.flatnonzero((pool.country == c).values)
        emb_i = None
        for i, a in enumerate(range(0, len(prow), POOL_CHUNK)):
            fn = os.path.join(part_dir, f"part_{c}_{i:05d}.parquet")
            if os.path.exists(fn):
                continue
            if out_of_time():
                log(f"TIME LIMIT ({TIME_LIMIT_H} h) reached during {split} / {c} - stopping cleanly")
                return False
            if emb_i is None:
                emb_i = encode(model, s1.text.values[idx_rows])
                ids_i = s1.id.values[idx_rows]
            rows = prow[a:a + POOL_CHUNK]
            I, S = search(encode(model, pool.text.values[rows]), emb_i)
            k = I.shape[1]
            pd.DataFrame({"record_id": np.repeat(pool.id.values[rows], k), "s1_id": ids_i[I.ravel()],
                          "rank": np.tile(np.arange(1, k + 1, dtype=np.int8), len(rows)),
                          "cosine": S.ravel().astype(np.float32)}).to_parquet(fn + ".tmp", index=False)
            os.replace(fn + ".tmp", fn)
            n_done += len(rows)
            log(f"  {split} {c}: {min(a + POOL_CHUNK, len(prow)):,}/{len(prow):,} records "
                f"({n_done / max(time.time() - t0, 1e-9):,.0f} records/s this split)")
        del emb_i
        gc.collect()
    # merge the parts into one file per split
    out = os.path.join(OUT_DIR, f"dense_{split}.parquet")
    files = sorted(glob.glob(os.path.join(part_dir, "part_*.parquet")))
    w = None
    for f in files:
        t = pq.read_table(f)
        if w is None:
            w = pq.ParquetWriter(out + ".tmp", t.schema, compression="zstd")
        w.write_table(t)
    if w is not None:
        w.close()
        os.replace(out + ".tmp", out)
    log(f"{split}: dense top-{TOPK} for {len(pool):,} records -> {out} ({time.time() - t0:.0f}s)")
    return True


# ------------------------------------------------------------------ main
def main():
    rng = np.random.default_rng(SEED)
    folds = read_tsv("s1_folds.tsv")
    folds["fold"] = folds.fold.astype(int)
    s1, pool = load("train")
    own = owners()
    pool = pool.merge(own, on="id", how="left")
    pool["owner"] = pool.owner.fillna("")
    fmap = folds.set_index("entity_id")
    pool["owner_fold"] = fmap.fold.reindex(pool.owner.values).fillna(-1).astype(int).values
    pool["owner_part"] = fmap.part.reindex(pool.owner.values).fillna("").values
    pool["owner_text"] = s1.set_index("id").text.reindex(pool.owner.values).fillna("").values
    log(f"owned pool records {(pool.owner != '').mean():.4f}; part 1a pairs {(pool.owner_part == '1a').sum():,}")
    queries = eval_queries(pool, s1, rng)
    if SMOKE:  # small index: the queries' owners + random S1s, 50k in total
        keep = set(queries.owner) | set(s1.id.sample(SMOKE_RECORDS, random_state=SEED))
        s1_eval = s1[s1.id.isin(keep)].reset_index(drop=True)
    else:
        s1_eval = s1
    model = get_model()
    if FAST:
        log("FAST: 'before' evaluation skipped")
    elif os.path.exists(os.path.join(OUT_DIR, "eval_before.csv")) and not SMOKE:
        log("eval_before.csv already in OUT_DIR (earlier run of this session) - skipping the 'before' evaluation")
    else:
        evaluate(model, s1_eval, queries, "before")
    model = finetune(model, training_pairs(pool, rng), rng)
    model.save(os.path.join(OUT_DIR, "e5_small_ft"))
    log(f"saved fine-tuned model -> {os.path.join(OUT_DIR, 'e5_small_ft')}")
    if not FAST:
        evaluate(model, s1_eval, queries, "after")
    tr = (s1, pool[["id", "text", "country"]])
    del pool, own, queries, s1_eval
    gc.collect()
    for split in (("train", "test") if FAST else ("test", "train")):
        if split == "test":
            s1_, pool_ = load("test")
        else:
            s1_, pool_ = tr
        if FAST:
            try:
                ids = set(read_tsv_plain(f"v3_open_{split}.txt"))
                pool_ = pool_[pool_.id.isin(ids)].reset_index(drop=True)
                log(f"FAST: {split} restricted to the {len(pool_):,} records v3 leaves unassigned")
            except AssertionError:
                log(f"FAST: v3_open_{split}.txt not found - searching all {split} pool records")
        if SMOKE:
            pool_ = pool_.sample(min(SMOKE_RECORDS, len(pool_)), random_state=SEED).reset_index(drop=True)
            s1_ = s1_.sample(min(SMOKE_RECORDS, len(s1_)), random_state=SEED).reset_index(drop=True)
        ok = dense_topk(model, split, s1_, pool_)
        del s1_, pool_
        gc.collect()
        if not ok:
            open(os.path.join(OUT_DIR, "STOPPED_TIME_LIMIT.txt"), "w").write(f"stopped during {split}\n")
            break
    log("done")


if __name__ == "__main__":
    main()
