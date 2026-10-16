#!/usr/bin/env python3
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("m*/jgp_pilot_rep001_metrics.csv")):
    with path.open(newline="") as stream:
        row = next(csv.DictReader(stream))
    row["metrics_file"] = str(path)
    rows.append(row)
if not rows:
    raise SystemExit("No JGP tuning metrics found.")
rows.sort(key=lambda row: (float(row["CRPS"]), float(row["RMSE"])))
fields = list(rows[0])
with (root / "training_only_tuning.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader(); writer.writerows(rows)
with (root / "selected_neighbors.txt").open("w") as stream:
    stream.write(str(int(rows[0]["neighbors"])) + "\n")
print(rows[0]["neighbors"])
