"""M0 safety-net baseline: every test S1 id with an empty match list and an empty candidate list."""
import csv
import os
import time

t0 = time.time()
ROOT = r"C:\amlc"
SRC = os.path.join(ROOT, "student_resource", "dataset", "test", "test_source1.tsv")
OUT = os.path.join(ROOT, "submission", "output_baseline")
os.makedirs(OUT, exist_ok=True)

ids = []
with open(SRC, encoding="utf-8", newline="") as f:
    next(f)
    for line in f:
        if line.strip():
            ids.append(line.split("\t", 1)[0].strip())
assert len(ids) == 1_732_544, len(ids)
assert len(set(ids)) == len(ids)

for name, col in [("matching_results.tsv", "matched_entity_ids"), ("candidate_pairs.tsv", "candidate_entity_ids")]:
    with open(os.path.join(OUT, name), "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{col}\n")
        f.writelines(f"{i}\t\n" for i in ids)
print(f"wrote {len(ids)} rows in {time.time() - t0:.1f}s")
