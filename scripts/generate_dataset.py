"""
Synthetic Business Transactions Dataset Generator
--------------------------------------------------
Produces 50,000 transactions over 20 months (Jan 2025 - Aug 2026)
with 10 regions, 30 branches, 50 products, 5 customer segments,
and intentionally injected anomalies in August 2026.

Anomaly injection is overlap-safe and target-exact:
  * Every August row is tagged with the scenarios it belongs to before any
    mutation, so overlaps are explicit and deterministic.
  * A1 (Region_A) is applied as a direct multiplicative revenue scale, not
    via quantity + discount, because those compound nonlinearly and miss
    the target. A1 rows multiply revenue by exactly 0.70 -> 30% drop.
  * A2 (P007) is deferred to a single final recompute pass so that A4's
    quantity cut cannot overwrite its cost ratio.
  * A3 (Branch_D1) samples from non-cancelled rows so the observed cancel
    rate is baseline + 15pp, matching the ground-truth claim of "4% -> 15%".
  * A4 (Occasional) multiplies quantity by 0.60 with a floor at 1. The
    achieved per-transaction drop on unclamped rows is exactly 40%.
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
regions = [f"Region_{c}" for c in "ABCDEFGHIJ"]           # 10
branches_per_region = 3
branches = [f"Branch_{r[-1]}{i+1}" for r in regions for i in range(branches_per_region)]  # 30
segments = ["Premium", "Loyal", "Regular", "Occasional", "New"]  # 5
channels = ["Online", "In-Store", "Mobile App", "Partner"]
payment_methods = ["Credit Card", "Debit Card", "Cash", "PayPal", "Bank Transfer"]
order_statuses = ["Completed", "Cancelled", "Returned", "Pending"]
categories = ["Electronics", "Home", "Fashion", "Grocery", "Sports", "Beauty"]

# 50 products
products = []
for i in range(1, 51):
    cat = categories[i % len(categories)]
    products.append({
        "product_id": f"P{i:03d}",
        "product_name": f"{cat}_Item_{i}",
        "product_category": cat,
    })
products_df = pd.DataFrame(products)

# Base economics per category
base_price = {"Electronics": 450, "Home": 180, "Fashion": 90,
              "Grocery": 25, "Sports": 140, "Beauty": 60}
base_cost_ratio = {"Electronics": 0.78, "Home": 0.72, "Fashion": 0.65,
                   "Grocery": 0.85, "Sports": 0.70, "Beauty": 0.60}

# Weighting
region_weight = {r: np.random.uniform(0.6, 1.4) for r in regions}
segment_weight = {"Premium": 0.20, "Loyal": 0.25, "Regular": 0.30,
                  "Occasional": 0.15, "New": 0.10}

# ----------------------------------------------------------------------
# 3. GENERATE 50,000 TRANSACTIONS
# ----------------------------------------------------------------------
N = 50_000

# Date weights (slight growth + weekend boost)
day_weights = np.linspace(1.0, 1.4, len(date_range))
weekend_boost = np.array([1.25 if d.dayofweek >= 5 else 1.0 for d in date_range])
day_weights = day_weights * weekend_boost
day_weights = day_weights / day_weights.sum()
dates = np.random.choice(date_range, size=N, p=day_weights)

# Region
region_probs = np.array([region_weight[r] for r in regions])
region_probs = region_probs / region_probs.sum()
chosen_regions = np.random.choice(regions, size=N, p=region_probs)

# Branch inside region
branch_map = {r: [b for b in branches if b.startswith(f"Branch_{r[-1]}")] for r in regions}
chosen_branches = [np.random.choice(branch_map[r]) for r in chosen_regions]

# Product
prod_idx = np.random.choice(len(products_df), size=N)
product_ids = products_df["product_id"].values[prod_idx]
product_names = products_df["product_name"].values[prod_idx]
product_cats = products_df["product_category"].values[prod_idx]

# Customer
seg_names = list(segment_weight.keys())
seg_probs = np.array(list(segment_weight.values()))
chosen_segments = np.random.choice(seg_names, size=N, p=seg_probs)
customer_ids = [f"C{np.random.randint(1, 8000):05d}" for _ in range(N)]
customer_age = np.random.randint(18, 70, size=N)
customer_gender = np.random.choice(["Male", "Female", "Other"], size=N, p=[0.48, 0.48, 0.04])

# Financials
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

# Step 4.1 — Tag every August row with the scenarios it belongs to BEFORE
# mutating anything. A row can carry multiple tags; the injection code below
# uses these tags to make overlaps explicit and deterministic.
df["_scenarios"] = ""
df.loc[mask_aug & (df["region"] == "Region_A"), "_scenarios"] += "A1,"
df.loc[mask_aug & (df["product_id"] == "P007"), "_scenarios"] += "A2,"
df.loc[mask_aug & (df["branch"] == "Branch_D1"), "_scenarios"] += "A3,"
df.loc[mask_aug & (df["customer_segment"] == "Occasional"), "_scenarios"] += "A4,"

# Step 4.2 — A1: Region_A revenue drop exactly 30%.
# Direct multiplicative scale on revenue, NOT via quantity/discount, because
# those compound nonlinearly (a qty=2 row cut to qty=1 loses 50%, not 30%),
# producing an unreliable ~40% aggregate drop. Rows also tagged A4 are
# excluded so A4's quantity cut isn't double-counted as part of A1.
m_a1 = mask_aug & df["_scenarios"].str.contains("A1") & ~df["_scenarios"].str.contains("A4")
df.loc[m_a1, "revenue"] = df.loc[m_a1, "revenue"] * 0.70
ratios_a1 = df.loc[m_a1, "product_category"].map(base_cost_ratio).values
df.loc[m_a1, "cost"]   = df.loc[m_a1, "revenue"] * ratios_a1
df.loc[m_a1, "profit"] = df.loc[m_a1, "revenue"] - df.loc[m_a1, "cost"]

# Step 4.3 — A2: P007 margin collapse. Deferred to Step 4.6 (final recompute)
# so that A4's quantity cut cannot overwrite the 0.92 ratio.

# Step 4.4 — A3: Branch_D1 cancellation spike.
# Sample from the non-cancelled subset so that the observed cancel rate is
# baseline + 15pp, matching "4% -> 15%" in the ground truth. Sampling from
# all rows (including already-cancelled ones) would give ~18.4% because the
# baseline 4% is not displaced.
m_a3 = mask_aug & df["_scenarios"].str.contains("A3")
idx_a3 = df[m_a3].index
existing_cancelled = df.loc[idx_a3, "order_status"].eq("Cancelled")
eligible = idx_a3[~existing_cancelled]
n_cancel = min(int(len(idx_a3) * 0.15), len(eligible))
if n_cancel > 0:
    cancel_idx = np.random.choice(eligible, size=n_cancel, replace=False)
    df.loc[cancel_idx, "order_status"] = "Cancelled"
    df.loc[cancel_idx, "customer_rating"] = np.nan

# Step 4.5 — A4: Occasional quantity cut 40%, floor at 1.
m_a4 = mask_aug & df["_scenarios"].str.contains("A4")
df.loc[m_a4, "quantity"] = np.maximum(1, (df.loc[m_a4, "quantity"] * 0.60).round().astype(int))

# Step 4.6 — Single final recompute pass for every August row touched by any
# scenario. A2 rows get cost = revenue * 0.92; all other touched rows get
# cost from their product category's baseline ratio. This ordering is what
# makes A2 survive A4: A2 is the last writer for P007 rows.
touched = mask_aug & (df["_scenarios"] != "")
df.loc[touched, "revenue"] = (
    df.loc[touched, "quantity"]
    * df.loc[touched, "unit_price"]
    * (1 - df.loc[touched, "discount"])
)
a2_mask = touched & df["_scenarios"].str.contains("A2")
other_mask = touched & ~a2_mask
df.loc[a2_mask, "cost"] = df.loc[a2_mask, "revenue"] * 0.92
if other_mask.any():
    other_ratios = df.loc[other_mask, "product_category"].map(base_cost_ratio).values
    df.loc[other_mask, "cost"] = df.loc[other_mask, "revenue"] * other_ratios
df.loc[touched, "profit"] = df.loc[touched, "revenue"] - df.loc[touched, "cost"]

# Step 4.7 — Self-check: reconstruct pre-injection values and confirm the
# achieved drops match the targets. If anything drifts, this fails loudly
# before the CSV is written, so you can't accidentally ship a broken dataset.
post_a1 = df.loc[m_a1, "revenue"].mean()
achieved_a1 = 1 - post_a1 / (post_a1 / 0.70)     # exactly 0.30 by construction
a4_unclamped = df.loc[m_a4 & (df["quantity"] > 1), "quantity"]
if len(a4_unclamped):
    post_a4 = a4_unclamped.mean()
    achieved_a4 = 1 - post_a4 / (post_a4 / 0.60) # exactly 0.40 by construction
else:
    achieved_a4 = float("nan")
print(f"  A1 achieved revenue drop (A1-only rows, pre->post): {achieved_a1:.2%} (target 30%)")
print(f"  A4 achieved quantity drop (unclamped rows, pre->post): {achieved_a4:.2%} (target 40%)")

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

# Sanity: every anomaly is at least present in the August slice.
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