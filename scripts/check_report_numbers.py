"""Cross-check: every decimal number printed in the report PDF must match a value
stored in a result file (outputs/**/*.csv|json, artifacts/*.json) at the printed
precision (also as a percentage, a difference of two stored values, or a ratio).

    python scripts/check_report_numbers.py [report.pdf]

Writes outputs/logs/report_number_check.txt and exits 1 if anything is unmatched.
Integers (counts, years, ids) are not checked - only numbers with a decimal point.
"""
import glob
import itertools
import json
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
pdf = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "report", "PathSnap_Technical_Report.pdf")
text = subprocess.run(["pdftotext", "-layout", pdf, "-"], capture_output=True, text=True, check=True).stdout

# ---------------------------------------------------------------- stored values
vals = []


def walk(x):
    if isinstance(x, bool):
        return
    if isinstance(x, (int, float)):
        vals.append(float(x))
    elif isinstance(x, dict):
        for v in x.values():
            walk(v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            walk(v)
    elif isinstance(x, str):
        try:
            walk(json.loads(x))
        except (ValueError, TypeError):
            pass


files = glob.glob(os.path.join(ROOT, "outputs", "**", "*.csv"), recursive=True) + \
    glob.glob(os.path.join(ROOT, "outputs", "**", "*.json"), recursive=True) + \
    glob.glob(os.path.join(ROOT, "artifacts", "*.json"))
for f in files:
    if f.endswith("_point_preds.csv"):
        continue
    if f.endswith(".json"):
        walk(json.load(open(f)))
    else:
        df = pd.read_csv(f)
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                vals.extend(df[c].dropna().astype(float).tolist())
            else:
                for s in df[c].dropna().astype(str):
                    for m in re.findall(r"-?\d+\.\d+", s):
                        vals.append(float(m))
for f in glob.glob(os.path.join(ROOT, "outputs", "screenshots", "*.log")):
    vals.extend(float(m) for m in re.findall(r"(?<![\w.])\d+\.\d+|(?<=[= ])\d+(?= ms)", open(f).read()))
# aggregates the report computes from per-trip inventory (medians, weighted shares)
inv = pd.read_csv(os.path.join(ROOT, "outputs", "eda", "trip_inventory.csv"), dtype={"trip_id": str})
for p, g in inv.groupby("partition"):
    vals.append(float(np.average(g.share_dt_eq_1s, weights=g.n_obs - 1)))
    for c in g.select_dtypes("number").columns:
        vals += [float(g[c].median()), float(g[c].mean())]
    vals.append(float(g[~g.trip_id.isin(["00000033", "00000054"])].gps_to_route_median_m.median()))
V = np.unique(np.round(np.array(vals, dtype=float), 8))
V = V[np.isfinite(V)]
ratios = []

# ---------------------------------------------------------------- printed numbers
# Rules (deliberately strict - a random number must NOT pass):
#   plain number            -> must equal a stored value at the printed precision
#   with % / pp / points    -> may also be 100 x a stored value
#   followed by ' s'        -> may also be a millisecond value from the API log / 1000
# Derived numbers (differences, ratios) are recorded by the report builder in
# outputs/logs/report_derived_values.json, which is indexed like any result file.
nums = re.findall(r"(?<![\w.])([+\u2212-]?)(\d[\d,]*\.\d+)(%| pp| points|x\b| s\b)?", text)
unmatched, checked = [], 0


def near(arr, c, t):
    return bool(len(arr)) and bool(np.any(np.abs(arr - c) <= t + 1e-12))


for sign, raw, suffix in nums:
    s = raw.replace(",", "")
    dec = len(s.split(".")[1])
    x = float(s)
    checked += 1
    tol = 0.5 * 10 ** (-dec)
    is_pct = suffix in ("%", " pp", " points")
    is_change = bool(sign) or suffix in (" pp", " points")
    tests = [(x, tol)] + ([(x / 100, tol / 100)] if is_pct else [])
    ok = any(near(V, c, t) or near(-V, c, t) for c, t in tests)
    if not ok and suffix == " s":
        ok = near(V / 1000, x, tol)
    if not ok:
        unmatched.append(sign + raw + (suffix or ""))

rng = np.random.RandomState(0)
chance = {}
for d in (2, 3, 4):
    r = np.round(rng.uniform(0, 1, 2000), d)
    chance[d] = float(np.mean([near(V, c, 0.5 * 10 ** (-d)) for c in r]))
out = [f"report: {pdf}", f"result values indexed: {len(V):,} from {len(files)} files",
       f"decimal numbers checked: {checked}", f"unmatched: {len(unmatched)}",
       "self-test - share of RANDOM numbers in [0,1] that would match some stored value by chance: "
       + ", ".join(f"{d} decimals {100 * v:.1f}%" for d, v in chance.items())
       + " (the check is a safety net; the primary guarantee is that build_report.py reads every number from the files)"]
if unmatched:
    out.append("UNMATCHED: " + ", ".join(sorted(set(unmatched), key=lambda z: float(re.sub(r'[^\d.+-]', '', z.replace(',', ''))))))
os.makedirs(os.path.join(ROOT, "outputs", "logs"), exist_ok=True)
open(os.path.join(ROOT, "outputs", "logs", "report_number_check.txt"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
sys.exit(1 if unmatched else 0)
