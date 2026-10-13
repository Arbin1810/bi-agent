import json
import pandas as pd
import numpy as np
from app.agents.base_agent import BaseAgent
from config.prompts import ANOMALY_PROMPT
from app.analysis.anomaly import detect_anomalies


class AnomalyAgent(BaseAgent):
    name = "AnomalyAgent"

    def run(self, state: dict) -> dict:
        self.log("Detecting anomalies")

        # Fix: avoid `or` short-circuit on DataFrames (raises on truth-value check)
        df = state.get("data")
        if df is None:
            df = state.get("dataframe")

        # Fix 5: compute real z-scores from SQL results when available
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

        # Statistical scan on numeric columns (if we have a DataFrame)
        stats_anomalies = detect_anomalies(df) if df is not None and not df.empty else []

        # Build a month-over-month revenue summary if date + revenue exist
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
                            lambda s: (s == "Cancelled").mean() if "order_status" in d.columns else None,
                        ),
                    ).reset_index()
                    mom_summary = grp.to_dict(orient="records")
            except Exception as e:
                self.log(f"MoM summary failed: {e}")

        # Merge computed z-scores + statistical hits into the summary
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

        # Force these to be from real computations, not LLM guesses
        result["computed_statistics"] = computed
        result["statistical_hits"] = stats_anomalies
        result["monthly_summary"] = mom_summary

        # ----------------------------------------------------------
        # Deduplicate: same (entity, period, value) with different labels.
        # The LLM often emits the same data point 3-4 times, once per
        # anomaly_type, producing noise in the report.
        # ----------------------------------------------------------
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
            # No DataFrame — but we may still have real anomalies from SQL
            if not computed:
                state["anomalies"] = {"anomalies": [], "method": "none"}
        return state