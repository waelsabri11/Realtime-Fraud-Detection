"""Turn Locust's CSV into a small JSON (p50/p95/p99, throughput, failures)."""

import csv
import json
import sys
from pathlib import Path

tag = sys.argv[1]
d = Path("work/results")
rows = list(csv.DictReader((d / f"locust_{tag}_stats.csv").open()))
agg = next(r for r in rows if r["Name"] == "Aggregated")
out = {
    "scenario": tag,
    "requests": int(agg["Request Count"]),
    "failures": int(agg["Failure Count"]),
    "throughput_rps": round(float(agg["Requests/s"]), 1),
    "latency_ms": {
        "p50": float(agg["50%"]),
        "p95": float(agg["95%"]),
        "p99": float(agg["99%"]),
        "max": float(agg["Max Response Time"]),
    },
}
(d / f"loadtest_{tag}.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
