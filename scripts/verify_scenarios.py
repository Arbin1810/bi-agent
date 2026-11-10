"""
Verify that all four injected scenarios survive into data/transactions.csv.

Method (R1: independent, non-tautological)
------------------------------------------
Loads data/pre_injection_snapshot.csv — a copy of every August row taken
*after* random generation but *before* any anomaly injection — and
compares the final CSV against it per row.

  A1  Region_A revenue × 0.70 on A1-only rows (per-row check)
  A2  P007 cost / revenue = 0.92 on all P007 August rows
  A3  Branch_D1 cancel rate = baseline + 15pp
  A4  Occasional quantity × 0.60 on unclamped rows (per-row check)

R7 extension (headline reproducibility)
---------------------------------------
Also computes July and August revenue totals two independent ways:
  1. Direct from the CSV with pandas
  2. Via the SQL agent's headline query (through DataFrameSQLExecutor)
and asserts they agree to the penny. Prints sha256 of the CSV so any
two runs can be compared by hash alone.

Exits 0 if all checks pass; exits 1 otherwise.
"""

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

# Ensure the project root is on sys.path so `from app...` imports work
# whether the script is invoked as `python scripts/verify_scenarios.py`
# or `python -m scripts.verify_scenarios`. Without this, running the
# script directly sets sys.path[0] to scripts/ and `from app...` fails
# with "No module named 'app'".
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

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

# ======================================================================
# R7 — Headline reproducibility
# ======================================================================
print()
print("=== R7: Headline reproducibility ===")
print()

# (1) Direct from CSV
jul_mask = df["date"].dt.strftime("%Y-%m") == "2026-07"
aug_mask = df["date"].dt.strftime("%Y-%m") == "2026-08"
jul_total_pandas = float(df.loc[jul_mask, "revenue"].sum())
aug_total_pandas = float(df.loc[aug_mask, "revenue"].sum())

# (2) Via the SQL agent's headline SQL (through the same executor path
#     the pipeline uses)
try:
    from app.data.connectors import DataFrameSQLExecutor
    from app.agents.sql_agent import SQLAgent

    ex = DataFrameSQLExecutor(df, table_name="transactions")
    agent = SQLAgent()

    # Build a minimal state with the required intent field for _build_headline
    fake_state = {"intent": {"time_reference": "2026-08"}}
    headline = agent._build_headline(fake_state, ex)

    if headline is None or len(headline.get("rows", [])) < 2:
        print("[FAIL] SQL headline query returned insufficient rows")
        failures.append("SQL headline query returned insufficient rows")
        jul_total_sql = aug_total_sql = float("nan")
    else:
        rows_by_month = {r["month"]: r for r in headline["rows"]}
        jul_total_sql = float(rows_by_month["2026-07"]["total_rev"])
        aug_total_sql = float(rows_by_month["2026-08"]["total_rev"])
except Exception as e:  # noqa: BLE001
    print(f"[FAIL] SQL headline path raised: {e}")
    failures.append(f"SQL headline path raised: {e}")
    jul_total_sql = aug_total_sql = float("nan")

# (3) Assert both methods agree to the penny
print(f"July  totals: pandas={jul_total_pandas:,.2f}  sql={jul_total_sql:,.2f}")
print(f"Aug   totals: pandas={aug_total_pandas:,.2f}  sql={aug_total_sql:,.2f}")

check(
    "July total agrees (pandas vs SQL)",
    actual=jul_total_sql,
    expected=jul_total_pandas,
    tol=0.01,
)
check(
    "August total agrees (pandas vs SQL)",
    actual=aug_total_sql,
    expected=aug_total_pandas,
    tol=0.01,
)

# (4) Print dataset hash so two runs can be compared for identity
csv_bytes = CSV_PATH.read_bytes()
dataset_hash = hashlib.sha256(csv_bytes).hexdigest()
print()
print(f"Dataset sha256: {dataset_hash}")

# ======================================================================
print()
if failures:
    print(f"FAILED: {len(failures)} check(s)")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("ALL SCENARIO CHECKS PASSED")