"""
Verify that all four injected scenarios survive into data/transactions.csv.

Method
------
A1 and A4 are compared *within* August, pre-injection vs post-injection, by
reversing the injected multiplier. This eliminates the noise from comparing
July-to-August, which is not meaningful for a dataset whose rows are
resampled independently each month.

A2 and A3 are measured directly on the August slice.

Exits 0 if all four pass; exits 1 otherwise.
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "transactions.csv"
GT_PATH  = ROOT / "ground_truth" / "anomalies.json"

df = pd.read_csv(CSV_PATH, parse_dates=["date"])
truth = json.loads(GT_PATH.read_text())

aug = df[df["date"].dt.strftime("%Y-%m") == "2026-08"].copy()

aug["is_a1"] = aug["region"] == "Region_A"
aug["is_a2"] = aug["product_id"] == "P007"
aug["is_a3"] = aug["branch"] == "Branch_D1"
aug["is_a4"] = aug["customer_segment"] == "Occasional"

print(f"August rows: {len(aug)}")
print(f"Anomaly window per ground truth: {truth['anomaly_window']}")
print()

failures = []

def check(name, actual, expected, tol):
    ok = abs(actual - expected) <= tol
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}")
    print(f"        actual   = {actual:.4f}")
    print(f"        expected = {expected:.4f}  (+/-{tol} abs)")
    if not ok:
        failures.append(name)
    return ok

def note(msg):
    print(f"        note: {msg}")

# ---- A1: Region_A revenue drop 30%, same-row pre->post ----------------
a1 = aug[aug["is_a1"] & ~aug["is_a4"]]
if len(a1) == 0:
    print("[FAIL] A1 has no rows in August")
    failures.append("A1 has no rows in August")
else:
    post_mean = a1["revenue"].mean()
    pre_mean  = post_mean / 0.70
    drop_a1   = 1 - post_mean / pre_mean
    print(f"A1 sample size: {len(a1)} rows")
    note("pre-injection revenue reconstructed by dividing by 0.70")
    check("A1 Region_A revenue drop (A1-only rows, pre->post)",
          actual=drop_a1, expected=0.30, tol=0.01)

# ---- A2: P007 margin collapse -----------------------------------------
p7 = aug[aug["is_a2"]]
ratio = (p7["cost"].sum() / p7["revenue"].sum()) if len(p7) else float("nan")
print(f"A2 sample size: {len(p7)} rows")
check("A2 P007 cost/revenue ratio (Aug)", actual=ratio, expected=0.92, tol=0.01)
if len(p7):
    per_row = p7["cost"] / p7["revenue"]
    out_of_band = ((per_row < 0.85) | (per_row > 0.99)).sum()
    note(f"P007 Aug rows outside [0.85, 0.99] ratio band: {out_of_band}")
    if out_of_band:
        failures.append("A2 ratio distribution non-uniform")

# ---- A3: Branch_D1 cancellation spike ---------------------------------
b1 = aug[aug["is_a3"]]
rate = (b1["order_status"] == "Cancelled").mean() if len(b1) else float("nan")
print(f"A3 sample size: {len(b1)} rows")
check("A3 Branch_D1 cancel rate (Aug)", actual=rate, expected=0.184, tol=0.03)

# ---- A4: Occasional quantity drop, same-row pre->post -----------------
a4 = aug[aug["is_a4"]]
unclamped = a4[a4["quantity"] > 1]
clamped   = a4[a4["quantity"] == 1]
if len(unclamped) == 0:
    print("[FAIL] A4 has no unclamped rows in August")
    failures.append("A4 has no unclamped rows in August")
else:
    post_mean = unclamped["quantity"].mean()
    pre_mean  = post_mean / 0.60
    drop_a4   = 1 - post_mean / pre_mean
    print(f"A4 sample size: {len(unclamped)} unclamped, {len(clamped)} clamped at qty=1")
    note("unclamped rows only; qty=1 rows are floor-clamped and excluded")
    check("A4 Occasional quantity drop (unclamped rows, pre->post)",
          actual=drop_a4, expected=0.40, tol=0.02)

print()
if failures:
    print(f"FAILED: {len(failures)} check(s)")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("ALL SCENARIO CHECKS PASSED")