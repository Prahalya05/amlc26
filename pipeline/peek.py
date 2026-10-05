"""Dump some rows of a parquet file to a UTF-8 text file (console cannot print non-ASCII)."""
import sys

import pandas as pd

path, out = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 40
query = sys.argv[4] if len(sys.argv) > 4 else None
df = pd.read_parquet(path)
if query:
    df = df.query(query)
with open(out, "w", encoding="utf-8") as f:
    for _, r in df.head(n).iterrows():
        f.write(" | ".join(f"{k}={v}" for k, v in r.items()) + "\n")
print("rows", len(df))
