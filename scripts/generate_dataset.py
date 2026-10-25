"""
Synthetic Business Transactions Dataset Generator
--------------------------------------------------
Produces 50,000 transactions over 20 months (Jan 2025 - Aug 2026)
with 10 regions, 30 branches, 50 products, 5 customer segments,
and intentionally injected anomalies in August 2026.

Anomaly injection is overlap-safe and target-exact. A pre-injection
snapshot is saved before any mutation so scripts/verify_scenarios.py
can compare against actual pre-injection values rather than
reconstructing a baseline from expected multipliers.

  A1 (Region_A) multiplies revenue by exactly 0.70 on A1-only rows
  (rows not also tagged A4). Step 4.6 recompute skips these rows so
  the injection survives.
  A2 (P007) sets cost/revenue = 0.92 in the final recompute pass.
  A3 (Branch_D1) cancels 15% of the currently-not-cancelled August rows.
  A4 (Occasional) multiplies quantity by 0.60 with a floor at 1.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GT_DIR = Path(__file__).resolve().parent.parent / "ground_truth"
GT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# 1. TIME DIMENSION
# ----------------------------------------------------------------------
start_date = pd.Timestamp("2025-01-01")
end_date = pd.Timestamp("2026-08-31")
date_range = pd.date_range(start=start_date, end=end_date, freq="D")
print(f"Date range: {date_range.min().date()} -> {date_range.max().date()} ({len(date_range)} days)")

# ----------------------------------------------------------------------
# 2. MASTER DIMENSIONS
# ----------------------------------------------------------------------
regions = [f"Region_{c}" for c in "ABCDEFGHIJ"]
branches_per_region = 3
branches = [f"Branch_{r[-1]}{i+1}" for r in regions for i in range(branches_per_region)]
segments = ["Premium", "Loyal", "Regular", "Occasional", "New"]
channels = ["Online", "In-Store", "Mobile App", "Partner"]
payment_methods = ["Credit Card", "Debit Card", "Cash", "PayPal", "Bank Transfer"]
order_statuses = ["Completed", "Cancelled", "Returned", "Pending"]
categories = ["Electronics", "Home", "Fashion", "Grocery", "Sports", "Beauty"]

products = []
for i in range(1, 51):
    cat = categories[i % len(categories)]
    products.append({
        "product_id": f"P{i:03d}",
        "product_name": f"{cat}_Item_{i}",
        "product_category": cat,
    })
products_df = pd.DataFrame(products)

base_price = {"Electronics": 450, "Home": 180, "Fashion": 90,
              "Grocery": 25, "Sports": 140, "Beauty": 60}
base_cost_ratio = {"Electronics": 0.78, "Home": 0.72, "Fashion": 0.65,
                   "Grocery": 0.85, "Sports": 0.70, "Beauty": 0.60}

region_weight = {r: np.random.uniform(0.6, 1.4) for r in regions}
segment_weight = {"Premium": 0.20, "Loyal": 0.25, "Regular": 0.30,
                  "Occasional": 0.15, "New": 0.10}

# ----------------------------------------------------------------------
# 3. GENERATE 50,000 TRANSACTIONS
# ----------------------------------------------------------------------
N = 50_000

day_weights = np.linspace(1.0, 1.4, len(date_range))
weekend_boost = np.array([1.25 if d.dayofweek >= 5 else 1.0 for d in date_range])
day_weights = day_weights * weekend_boost
day_weights = day_weights / day_weights.sum()
dates = np.random.choice(date_range, size=N, p=day_weights)

region_probs = np.array([region_weight[r] for r in regions])
region_probs = region_probs / region_probs.sum()
chosen_regions = np.random.choice(regions, size=N, p=region_probs)

branch_map = {r: [b for b in branches if b.startswith(f"Branch_{r[-1]}")] for r in regions}
chosen_branches = [np.random.choice(branch_map[r]) for r in chosen_regions]

prod_idx = np.random.choice(len(products_df), size=N)
product_ids = products_df["product_id"].values[prod_idx]
product_names = products_df["product_name"].values[prod_idx]
product_cats = products_df["product_category"].values[prod_idx]

seg_names = list(segment_weight.keys())
seg_probs = np.array(list(segment_weight.values()))
chosen_segments = np.random.choice(seg_names, size=N, p=seg_probs)
customer_ids = [f"C{np.random.randint(1, 8000):05d}" for _ in range(N)]
customer_age = np.random.randint(18, 70, size=N)
customer_gender = np.random.choice(["Male", "Female", "Other"], size=N, p=[0.48, 0.48, 0.04])

quantity = np.random.choice([1, 2, 3, 4, 5, 8, 10], size=N,
                            p=[0.40, 0.25, 0.15, 0.08, 0.06, 0.04, 0.02])
unit_price = np.array([base_price[product_cats[i]] * np.random.uniform(0.85, 1.15)
                       for i in range(N)])
discount = np.random.choice([0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40],
                            size=N, p=[0.35, 0.15, 0.15, 0.12, 0.10, 0.08, 0.05])
revenue = quantity * unit_price * (1 - discount)
cost_ratio = np.array([base_cost_ratio[product_cats[i]] for i in range(N)])
cost = revenue * cost_ratio * np.random.uniform(0.95, 1.05, size=N)
profit = revenue - cost

payment_method = np.random.choice(payment_methods, size=N, p=[0.35, 0.25, 0.10, 0.20, 0.10])
order_status = np.random.choice(order_statuses, size=N, p=[0.90, 0.04, 0.03, 0.03])
customer_rating = np.where(
    order_status == "Completed",
    np.clip(np.random.normal(4.2, 0.6, size=N), 1, 5),
    np.nan
)

df = pd.DataFrame({
    "transaction_id": [f"T{i+1:06d}" for i in range(N)],
    "date": dates,
    "customer_id": customer_ids,
    "customer_segment": chosen_segments,
    "customer_age": customer_age,
    "customer_gender": customer_gender,
    "region": chosen_regions,
    "branch": chosen_branches,
    "sales_channel": np.random.choice(channels, size=N, p=[0.40, 0.35, 0.20, 0.05]),
    "product_id": product_ids,
    "product_category": product_cats,
    "product_name": product_names,
    "quantity": quantity,
    "unit_price": np.round(unit_price, 2),
    "discount": discount,
    "revenue": np.round(revenue, 2),
    "cost": np.round(cost, 2),
    "profit": np.round(profit, 2),
    "payment_method": payment_method,
    "order_status": order_status,
    "customer_rating": np.round(customer_rating, 1),
})

print(f"Baseline dataset: {df.shape}")

# ----------------------------------------------------------------------
# 4. INJECT ANOMALIES (August 2026) — overlap-safe, target-exact
# ----------------------------------------------------------------------
anomaly_start = pd.Timestamp("2026-08-01")
anomaly_end = pd.Timestamp("2026-08-31")
mask_aug = (df["date"] >= anomaly_start) & (df["date"] <= anomaly_end)
print(f"August 2026 transactions: {mask_aug.sum()}")

# ----------------------------------------------------------------------
# Step 4.0 — Snapshot every August row BEFORE any mutation. The
# verifier compares the final CSV against this snapshot to prove the
# injected effects survived. Do NOT modify these values.
# ----------------------------------------------------------------------
snapshot_cols = [
    "transaction_id", "date", "region", "branch", "product_id",
    "product_category", "customer_segment", "quantity",
    "unit_price", "discount", "revenue", "cost", "profit",
    "order_status",
]
pre_injection = df.loc[mask_aug, snapshot_cols].copy()
snapshot_path = OUTPUT_DIR / "pre_injection_snapshot.csv"
pre_injection.to_csv(snapshot_path, index=False)
print(f"Pre-injection snapshot saved: {snapshot_path} ({len(pre_injection)} rows)")

# Step 4.1 — Tag every August row with the scenarios it belongs to.
df["_scenarios"] = ""
df.loc[mask_aug & (df["region"] == "Region_A"), "_scenarios"] += "A1,"
df.loc[mask_aug & (df["product_id"] == "P007"), "_scenarios"] += "A2,"
df.loc[mask_aug & (df["branch"] == "Branch_D1"), "_scenarios"] += "A3,"
df.loc[mask_aug & (df["customer_segment"] == "Occasional"), "_scenarios"] += "A4,"

# ----------------------------------------------------------------------
# Step 4.2 — A1: Region_A revenue drop exactly 30% on A1-only rows.
#
# CRITICAL: mark these rows so Step 4.6's revenue recompute skips them.
# Without this exclusion, Step 4.6 would overwrite the 0.70x injection
# with the original quantity x unit_price x (1 - discount).
# ----------------------------------------------------------------------
m_a1 = mask_aug & df["_scenarios"].str.contains("A1") & ~df["_scenarios"].str.contains("A4")
a1_only_mask = m_a1.copy()

df.loc[m_a1, "revenue"] = df.loc[m_a1, "revenue"] * 0.70
ratios_a1 = df.loc[m_a1, "product_category"].map(base_cost_ratio).values
df.loc[m_a1, "cost"]   = df.loc[m_a1, "revenue"] * ratios_a1
df.loc[m_a1, "profit"] = df.loc[m_a1, "revenue"] - df.loc[m_a1, "cost"]

# ----------------------------------------------------------------------
# Step 4.3 — A2: P007 margin collapse. Deferred to Step 4.6.
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Step 4.4 — A3: Branch_D1 cancellation spike.
# Sample from the currently-not-cancelled subset so the observed cancel
# rate is baseline + 15pp.
# ----------------------------------------------------------------------
m_a3 = mask_aug & df["_scenarios"].str.contains("A3")
idx_a3 = df[m_a3].index
existing_cancelled = df.loc[idx_a3, "order_status"].eq("Cancelled")
eligible = idx_a3[~existing_cancelled]
n_cancel = min(int(len(idx_a3) * 0.15), len(eligible))
cancel_idx = []
if n_cancel > 0:
    cancel_idx = np.random.choice(eligible, size=n_cancel, replace=False)
    df.loc[cancel_idx, "order_status"] = "Cancelled"
    df.loc[cancel_idx, "customer_rating"] = np.nan

# ----------------------------------------------------------------------
# Step 4.5 — A4: Occasional quantity cut 40%, floor at 1.
# ----------------------------------------------------------------------
m_a4 = mask_aug & df["_scenarios"].str.contains("A4")
df.loc[m_a4, "quantity"] = np.maximum(1, (df.loc[m_a4, "quantity"] * 0.60).round().astype(int))

# ----------------------------------------------------------------------
# Step 4.6 — Final recompute pass for all touched rows, EXCLUDING the
# A1-only rows (whose revenue/cost/profit were set correctly in 4.2).
# ----------------------------------------------------------------------
touched = mask_aug & (df["_scenarios"] != "")
recompute_mask = touched & ~a1_only_mask

df.loc[recompute_mask, "revenue"] = (
    df.loc[recompute_mask, "quantity"]
    * df.loc[recompute_mask, "unit_price"]
    * (1 - df.loc[recompute_mask, "discount"])
)

# A2 rows: cost = revenue * 0.92 (this is the margin collapse).
a2_mask = touched & df["_scenarios"].str.contains("A2")
df.loc[a2_mask, "cost"] = df.loc[a2_mask, "revenue"] * 0.92

# Other recomputed rows (non-A1, non-A2): cost from category ratio.
other_mask = recompute_mask & ~a2_mask
if other_mask.any():
    other_ratios = df.loc[other_mask, "product_category"].map(base_cost_ratio).values
    df.loc[other_mask, "cost"] = df.loc[other_mask, "revenue"] * other_ratios

# All touched rows: profit = revenue - cost.
df.loc[touched, "profit"] = df.loc[touched, "revenue"] - df.loc[touched, "cost"]

# ----------------------------------------------------------------------
# Step 4.7 — Build injection metadata for the verifier. References the
# actual transaction_ids the generator touched, so the verifier can
# match against the snapshot exactly.
# ----------------------------------------------------------------------
a4_unclamped_mask = m_a4 & (df["quantity"] > 1)

metadata = {
    "seed": SEED,
    "anomaly_window": {
        "start": str(anomaly_start.date()),
        "end": str(anomaly_end.date()),
    },
    "injected_multipliers": {
        "A1_revenue_scale": 0.70,
        "A2_cost_revenue_ratio": 0.92,
        "A3_cancel_rate_add": 0.15,
        "A4_quantity_scale": 0.60,
    },
    "row_counts": {
        "mask_aug": int(mask_aug.sum()),
        "A1_only": int(a1_only_mask.sum()),
        "A2_total": int(a2_mask.sum()),
        "A3_total": int(m_a3.sum()),
        "A3_newly_cancelled": int(len(cancel_idx)),
        "A4_total": int(m_a4.sum()),
        "A4_unclamped": int(a4_unclamped_mask.sum()),
    },
    "transaction_ids": {
        "A1_only": df.loc[a1_only_mask, "transaction_id"].tolist(),
        "A4_unclamped": df.loc[a4_unclamped_mask, "transaction_id"].tolist(),
        "A3_newly_cancelled": (
            df.loc[cancel_idx, "transaction_id"].tolist() if len(cancel_idx) else []
        ),
    },
}
metadata_path = OUTPUT_DIR / "pre_injection_metadata.json"
metadata_path.write_text(json.dumps(metadata, indent=2))
print(f"Injection metadata saved: {metadata_path}")

# ----------------------------------------------------------------------
# Step 4.8 — Self-check against the pre-injection snapshot, NOT against
# a reconstructed multiplier reversal.
# ----------------------------------------------------------------------
a1_ids = df.loc[a1_only_mask, "transaction_id"]
a1_snapshot_rev = pre_injection[
    pre_injection["transaction_id"].isin(a1_ids)
]["revenue"].mean()
a1_final_rev = df.loc[a1_only_mask, "revenue"].mean()
achieved_a1 = 1 - (a1_final_rev / a1_snapshot_rev) if a1_snapshot_rev else float("nan")
print(f"  A1 achieved revenue drop (vs snapshot): {achieved_a1:.2%} (target 30%)")
if abs(achieved_a1 - 0.30) > 0.005:
    print(f"  WARNING: A1 drop off-target by {abs(achieved_a1 - 0.30):.4f}")

a4_ids = df.loc[a4_unclamped_mask, "transaction_id"]
a4_snapshot_qty = pre_injection[
    pre_injection["transaction_id"].isin(a4_ids)
]["quantity"].mean()
a4_final_qty = df.loc[a4_unclamped_mask, "quantity"].mean()
achieved_a4 = 1 - (a4_final_qty / a4_snapshot_qty) if a4_snapshot_qty else float("nan")
print(f"  A4 achieved quantity drop (vs snapshot): {achieved_a4:.2%} (target 40%)")
if abs(achieved_a4 - 0.40) > 0.02:
    print(f"  WARNING: A4 drop off-target by {abs(achieved_a4 - 0.40):.4f}")

df.drop(columns="_scenarios", inplace=True)

# ----------------------------------------------------------------------
# 5. VALIDATE
# ----------------------------------------------------------------------
assert len(df) == N, "Row count mismatch"
critical = ["transaction_id", "date", "region", "branch", "product_id",
            "revenue", "cost", "profit", "order_status"]
assert df[critical].isnull().sum().sum() == 0
assert (df["revenue"] >= 0).all()
assert (df["quantity"] > 0).all()
assert df["discount"].between(0, 1).all()

aug = df[df["date"].between(anomaly_start, anomaly_end)]
assert (aug["region"] == "Region_A").any(), "A1 target missing from August"
assert (aug["product_id"] == "P007").any(), "A2 target missing from August"
assert (aug["branch"] == "Branch_D1").any(), "A3 target missing from August"
assert (aug["customer_segment"] == "Occasional").any(), "A4 target missing from August"
print("Validation passed")

# ----------------------------------------------------------------------
# 6. SAVE
# ----------------------------------------------------------------------
csv_path = OUTPUT_DIR / "transactions.csv"
df.to_csv(csv_path, index=False)
print(f"Saved: {csv_path}  ({csv_path.stat().st_size / 1e6:.2f} MB)")

ground_truth = {
    "anomaly_window": {"start": str(anomaly_start.date()), "end": str(anomaly_end.date())},
    "injected_anomalies": [
        {"id": "A1", "type": "regional_revenue_drop", "target": "Region_A",
         "expected_effect": "per-transaction revenue drop 30% on Region_A rows not also Occasional",
         "measurement": "mean revenue per transaction, A1-only rows, pre vs post"},
        {"id": "A2", "type": "product_margin_collapse", "target": "P007",
         "expected_effect": "cost/revenue ratio = 0.92 (margin 25% -> 8%)",
         "measurement": "aggregate cost/revenue ratio, P007 August rows"},
        {"id": "A3", "type": "branch_cancellation_spike", "target": "Branch_D1",
         "expected_effect": "cancel rate 4% -> 18.4% (baseline + 15pp on non-cancelled rows)",
         "measurement": "fraction of August Branch_D1 rows with order_status == Cancelled"},
        {"id": "A4", "type": "segment_decline", "target": "Occasional",
         "expected_effect": "per-transaction quantity cut 40% on unclamped rows (qty > 1)",
         "measurement": "mean quantity per transaction, unclamped Occasional rows, pre vs post"},
    ],
}
gt_path = GT_DIR / "anomalies.json"
gt_path.write_text(json.dumps(ground_truth, indent=2))
print(f"Ground truth saved: {gt_path}")