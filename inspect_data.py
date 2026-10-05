import os

with open("report.txt", "w", encoding="utf-8") as out:
    for root, dirs, files in os.walk("."):
        for name in sorted(files):
            if name == "report.txt":
                continue
            path = os.path.join(root, name)
            size_mb = os.path.getsize(path) / 1e6
            if name.endswith(".tsv"):
                with open(path, encoding="utf-8", errors="replace") as fh:
                    header = fh.readline().strip()
                    first = fh.readline().strip()
                    rows = (1 if first else 0) + sum(1 for _ in fh)
                print(f"{path} | {size_mb:.1f} MB | {rows} rows", file=out)
                print("    header:", header, file=out)
                print("    first :", first[:200], file=out)
            else:
                print(f"{path} | {size_mb:.1f} MB", file=out)

print("Done - open report.txt")