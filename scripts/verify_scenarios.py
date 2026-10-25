"""
Verify that all four injected scenarios survive into data/transactions.csv.

Method (R1: independent, non-tautological)
------------------------------------------
Loads data/pre_injection_snapshot.csv — a copy of every August row taken
*after* random generation but *before* any anomaly injection — and
compares the final CSV against it per row. No "post / multiplier"
reconstruction: each scenario is verified by comparing the actual final
value to the snapshot's actual pre-injection value.

  A1  Region_A revenue × 0.70 on A1-only rows (per-row check)
  A2  P007 cost / revenue = 0.92 on all P007 August rows
  A3  Branch_D1 cancel rate = baseline + 15pp
  A4  Occasional quantity × 0.60 on unclamped rows (per-row check)

Exits 0 if all four pass; exits 1 otherwise.
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH      = ROOT / "data" / "transactions.csv"
SNAPSHOT_PATH = ROOT / "data" / "pre_injection_snapshot.csv"
META_PATH     = ROOT / "data" / "pre_injection_metadata.json"
GT_PATH       = ROOT / "ground_truth" / "anomalies.json"

# ----------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------
df       = pd.read_csv(CSV_PATH, parse_dates=["date"])
snapshot = pd.read_csv(SNAPSHOT_PATH, parse_dates=["date"])
meta     = json.loads(META_PATH.read_text())
truth    = json.loads(GT_PATH.read_text())

merged = df.merge(snapshot, on="transaction_id", suffixes=("", "_pre"))
aug = merged[merged["date"].dt.strftime("%Y-%m") == "2026-08"].copy()

# Rebuild the row sets from the metadata so we compare exactly the rows
# the generator touched.
a1_ids = set(meta["transaction_ids"]["A1_only"])
a4_ids = set(meta["transaction_ids"]["A4_unclamped"])
a3_newly_cancelled_ids = set(meta["transaction_ids"]["A3_newly_cancelled"])

aug["is_a1"] = aug["transaction_id"].isin(a1_ids)
aug["is_a4_unclamped"] = aug["transaction_id"].isin(a4_ids)
aug["is_a3_newly_cancelled"] = aug["transaction_id"].isin(a3_newly_cancelled_ids)
aug["is_a2"] = aug["product_id"] == "P007"
aug["is_a3_branch"] = aug["branch"] == "Branch_D1"

print(f"August rows: {len(aug)}")
print(f"Pre-injection snapshot: {len(snapshot)} rows")
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

# ---- A1: Region_A revenue, pre-injection vs final (per-row) ----------
a1 = aug[aug["is_a1"]]
if len(a1) == 0:
    print("[FAIL] A1 has no rows in August")
    failures.append("A1 has no rows in August")
else:
    expected_rev = a1["revenue_pre"] * 0.70
    residual = (a1["revenue"] - expected_rev).abs()
    max_residual = float(residual.max())
    print(f"A1 sample size: {len(a1)} rows")
    note(f"max per-row residual vs snapshot×0.70: {max_residual:.4f}")
    check(
        "A1 Region_A revenue drop (A1-only rows, snapshot-based)",
        actual=1 - (a1["revenue"].mean() / a1["revenue_pre"].mean()),
        expected=0.30,
        tol=0.005,
    )
    if max_residual > 0.02:
        print(f"[FAIL] A1 per-row residual exceeds 0.02 (max={max_residual:.4f})")
        failures.append("A1 per-row residual > 0.02")

# ---- A2: P007 margin collapse ----------------------------------------
p7 = aug[aug["is_a2"]]
if len(p7) == 0:
    print("[FAIL] A2 has no rows in August")
    failures.append("A2 has no rows in August")
else:
    ratio = p7["cost"].sum() / p7["revenue"].sum()
    print(f"A2 sample size: {len(p7)} rows")
    check("A2 P007 cost/revenue ratio (Aug)", actual=ratio, expected=0.92, tol=0.01)
    per_row = p7["cost"] / p7["revenue"]
    out_of_band = ((per_row < 0.90) | (per_row > 0.94)).sum()
    note(f"P007 Aug rows outside [0.90, 0.94] ratio band: {out_of_band}")
    if out_of_band > 0:
        failures.append("A2 ratio distribution non-uniform")

# ---- A3: Branch_D1 cancellation spike --------------------------------
b1 = aug[aug["is_a3_branch"]]
if len(b1) == 0:
    print("[FAIL] A3 has no rows in August")
    failures.append("A3 has no rows in August")
else:
    rate = (b1["order_status"] == "Cancelled").mean()
    snapshot_rate = (b1["order_status_pre"] == "Cancelled").mean()
    print(f"A3 sample size: {len(b1)} rows")
    note(f"snapshot cancel rate: {snapshot_rate:.4f}")
    note(f"final cancel rate:    {rate:.4f}")
    expected_rate = min(snapshot_rate + 0.15, 1.0)
    check(
        "A3 Branch_D1 cancel rate (snapshot + 15pp)",
        actual=rate,
        expected=expected_rate,
        tol=0.03,
    )
    if len(a3_newly_cancelled_ids):
        newly = aug[aug["is_a3_newly_cancelled"]]
        missed = (newly["order_status"] != "Cancelled").sum()
        note(f"metadata newly-cancelled rows that aren't Cancelled in CSV: {missed}")
        if missed > 0:
            failures.append(f"A3 {missed} metadata-cancelled rows not Cancelled in CSV")

# ---- A4: Occasional quantity drop ------------------------------------
a4 = aug[aug["is_a4_unclamped"]]
if len(a4) == 0:
    print("[FAIL] A4 has no unclamped rows in August")
    failures.append("A4 has no unclamped rows in August")
else:
    # Per-row: expected final quantity = round(pre × 0.60) with floor 1.
    expected_qty = (a4["quantity_pre"] * 0.60).round().clip(lower=1)
    residuals = (a4["quantity"] - expected_qty).abs()
    max_residual = float(residuals.max())
    print(f"A4 sample size: {len(a4)} unclamped rows")
    note(f"max per-row residual vs snapshot×0.60: {max_residual:.4f}")
    check(
        "A4 Occasional quantity drop (unclamped rows, snapshot-based)",
        actual=1 - (a4["quantity"].mean() / a4["quantity_pre"].mean()),
        expected=0.40,
        tol=0.02,
    )
    if max_residual > 0:
        print(f"[FAIL] A4 per-row residual non-zero (max={max_residual:.4f})")
        failures.append("A4 per-row residual non-zero")

print()
if failures:
    print(f"FAILED: {len(failures)} check(s)")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("ALL SCENARIO CHECKS PASSED")