import json

import numpy as np
import pandas as pd

from app.agents.base_agent import BaseAgent
from config.prompts import ANOMALY_PROMPT
from app.analysis.anomaly import detect_anomalies


# Known scenarios from the dataset generator. Used only to check honesty —
# if a scenario is injected but not detected, we say so, rather than
# claiming "no anomalies found".
KNOWN_SCENARIOS = [
    {"id": "A1", "dimension": "region", "entity": "Region_A",
     "metric": "revenue", "direction": "down"},
    {"id": "A2", "dimension": "product_id", "entity": "P007",
     "metric": "cost_ratio", "direction": "up"},
    {"id": "A3", "dimension": "branch", "entity": "Branch_D1",
     "metric": "cancel_rate", "direction": "up"},
    {"id": "A4", "dimension": "customer_segment", "entity": "Occasional",
     "metric": "quantity", "direction": "down"},
]


class AnomalyAgent(BaseAgent):
    name = "AnomalyAgent"

    def run(self, state: dict) -> dict:
        self.log("Detecting anomalies")

        df = state.get("data")
        if df is None:
            df = state.get("dataframe")

        # ----------------------------------------------------------
        # Monthly totals z-score (kept — still useful for whole-month
        # step changes).
        # ----------------------------------------------------------
        sql_results = state.get("sql_results") or []
        computed = {}

        for r in sql_results:
            rows = r.get("rows", [])
            if not rows:
                continue
            cols = r.get("columns", [])
            if "month" in cols and "total_rev" in cols:
                totals = [row["total_rev"] for row in rows if row.get("total_rev") is not None]
                if len(totals) >= 5:
                    arr = np.array(totals, dtype=float)
                    mean, std = arr.mean(), arr.std()
                    if std > 0:
                        z = (arr - mean) / std
                        anomalies = [
                            {
                                "month": rows[i].get("month"),
                                "total": float(totals[i]),
                                "z_score": float(z[i]),
                            }
                            for i in range(len(arr)) if abs(z[i]) > 2
                        ]
                        computed["monthly_anomalies"] = anomalies
                        computed["baseline_mean"] = float(mean)
                        computed["baseline_std"] = float(std)

        # ----------------------------------------------------------
        # FIX (Step 3.3): cross-entity robust z-scores at the target
        # month.
        #
        # The previous version used sample mean and std. That has two
        # problems on the small dimension populations we see (typically
        # 5–50 entities per dimension):
        #   1. std of 2 entities is undefined behavior for outlier
        #      detection — every point looks like ±1, so nothing flags.
        #      We keep the >= 3 guard as a floor.
        #   2. On any sample size, std is dominated by the very
        #      outlier we're trying to detect (masking effect). A −30%
        #      drop inflates std so much that its own z-score falls
        #      below the threshold.
        #
        # Use median-absolute-deviation (MAD) instead. Median is robust
        # to the outlier, so the scale stays honest. The 0.6745
        # constant rescales MAD to be σ-comparable for normal data,
        # preserving the |z| > 2 interpretation.
        # ----------------------------------------------------------
        dimension_deltas = state.get("dimension_deltas") or {}
        entity_anomalies = []
        MAD_SCALE = 0.6745
        for dim, entities in dimension_deltas.items():
            if len(entities) < 3:
                continue
            pcts = np.array([d["change_pct"] for d in entities], dtype=float)
            med = float(np.median(pcts))
            mad = float(np.median(np.abs(pcts - med)))
            if mad == 0:
                # All entities moved by the same amount — no outlier
                # signal. Falling back to std here would recreate the
                # masking problem, so skip the dimension entirely.
                continue
            for d in entities:
                rz = MAD_SCALE * (d["change_pct"] - med) / mad
                if abs(rz) > 2:
                    entity_anomalies.append({
                        "dimension": dim,
                        "entity": d["entity"],
                        "change_pct": d["change_pct"],
                        "robust_z_score": float(rz),
                        "dimension_median_pct": med,
                        "dimension_mad": mad,
                        "direction": "down" if d["change_pct"] < med else "up",
                        "baseline_value": d["baseline_value"],
                        "target_value": d["target_value"],
                        "method": "cross_entity_mad_zscore",
                    })
        if entity_anomalies:
            computed["entity_anomalies"] = entity_anomalies
            self.log(f"Entity robust z-scores flagged {len(entity_anomalies)} anomalies")

        # ----------------------------------------------------------
        # Known-scenario honesty check: for each scenario, is there any
        # anomaly in our output that matches (dimension, entity)?
        # ----------------------------------------------------------
        detected_keys = {(a["dimension"], a["entity"]) for a in entity_anomalies}
        scenario_check = []
        for s in KNOWN_SCENARIOS:
            hit = (s["dimension"], s["entity"]) in detected_keys
            scenario_check.append({
                "scenario_id": s["id"],
                "dimension": s["dimension"],
                "entity": s["entity"],
                "expected_metric": s["metric"],
                "expected_direction": s["direction"],
                "detected_by_cross_entity_zscore": hit,
            })
        computed["known_scenario_check"] = scenario_check

        # Statistical scan on numeric columns
        stats_anomalies = detect_anomalies(df) if df is not None and not df.empty else []

        # Month-over-month revenue summary
        mom_summary = {}
        if df is not None and not df.empty:
            try:
                d = df.copy()
                if "date" in d.columns and "revenue" in d.columns:
                    d["date"] = pd.to_datetime(d["date"], errors="coerce")
                    d = d.dropna(subset=["date"])
                    d["month"] = d["date"].dt.to_period("M").astype(str)
                    grp = d.groupby("month").agg(
                        total_revenue=("revenue", "sum"),
                        txn_count=("revenue", "size"),
                        avg_order_value=("revenue", "mean"),
                        cancellation_rate=(
                            "order_status",
                            lambda s: (s == "Cancelled").mean()
                            if "order_status" in d.columns else None,
                        ),
                    ).reset_index()
                    mom_summary = grp.to_dict(orient="records")
            except Exception as e:
                self.log(f"MoM summary failed: {e}")

        summary_payload = {
            "computed": computed,
            "statistical_hits": stats_anomalies,
            "monthly": mom_summary,
        }

        prompt = ANOMALY_PROMPT.format(
            question=state["question"],
            summary=json.dumps(summary_payload, default=str)[:4000],
            stats=json.dumps(computed, default=str)[:2000],
        )
        result = self.llm.complete_json(prompt)

        result["computed_statistics"] = computed
        result["statistical_hits"] = stats_anomalies
        result["monthly_summary"] = mom_summary

        # Deduplicate
        seen = set()
        deduped = []
        for a in result.get("anomalies", []):
            key = (a.get("entity"), a.get("period"), a.get("value"))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(a)
        result["anomalies"] = deduped

        state["anomalies"] = result

        if df is None or (hasattr(df, "empty") and df.empty):
            if not computed:
                state["anomalies"] = {"anomalies": [], "method": "none"}
        return state