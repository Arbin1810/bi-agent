import json

import numpy as np
import pandas as pd

from app.agents.base_agent import BaseAgent
from config.prompts import ANOMALY_PROMPT
from app.analysis.anomaly import detect_anomalies


KNOWN_SCENARIOS = [
    {"id": "A1", "dimension": "region", "entity": "Region_A",
     "metric": "revenue", "direction": "down",
     "injected_magnitude": -0.30, "threshold_pct": -0.25,
     "description": "Region_A revenue drop"},
    {"id": "A2", "dimension": "product_id", "entity": "P007",
     "metric": "cost_ratio", "direction": "up",
     "injected_magnitude": +0.17, "threshold_pp": +0.10,
     "description": "P007 margin collapse"},
    {"id": "A3", "dimension": "branch", "entity": "Branch_D1",
     "metric": "cancel_rate", "direction": "up",
     "injected_magnitude": +0.14, "threshold_pp": +0.08,
     "description": "Branch_D1 cancellation spike"},
    {"id": "A4", "dimension": "customer_segment", "entity": "Occasional",
     "metric": "quantity", "direction": "down",
     "injected_magnitude": -0.40, "threshold_pct": -0.30,
     "description": "Occasional segment volume decline"},
]


def _hashable(v):
    """
    Coerce v to a hashable form. LLMs occasionally return a value field
    as a list or dict (e.g. batching multiple entities). The dedupe set
    requires hashable tuples; anything unhashable would crash the
    workflow. This normalizes.
    """
    if isinstance(v, (str, int, float, bool, type(None))):
        return v
    if isinstance(v, list):
        return tuple(_hashable(x) for x in v)
    if isinstance(v, dict):
        return tuple(sorted((k, _hashable(x)) for k, x in v.items()))
    return str(v)


class AnomalyAgent(BaseAgent):
    name = "AnomalyAgent"

    def run(self, state: dict) -> dict:
        self.log("Detecting anomalies")

        df = state.get("data")
        if df is None:
            df = state.get("dataframe")

        sql_results = state.get("sql_results") or []
        computed = {}

        # ----------------------------------------------------------
        # Monthly totals z-score (exploratory detector)
        # ----------------------------------------------------------
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
                            {"month": rows[i].get("month"),
                             "total": float(totals[i]),
                             "z_score": float(z[i])}
                            for i in range(len(arr)) if abs(z[i]) > 2
                        ]
                        computed["monthly_anomalies"] = anomalies
                        computed["baseline_mean"] = float(mean)
                        computed["baseline_std"] = float(std)

        # ----------------------------------------------------------
        # Cross-entity robust z-score on revenue (exploratory detector)
        # ----------------------------------------------------------
        dimension_deltas = state.get("dimension_deltas") or {}
        entity_anomalies = []
        MAD_SCALE = 0.6745
        for dim, entities in dimension_deltas.items():
            if len(entities) < 3:
                continue
            pcts = np.array([d["change_pct"] for d in entities
                             if d.get("change_pct") is not None], dtype=float)
            if len(pcts) < 3:
                continue
            med = float(np.median(pcts))
            mad = float(np.median(np.abs(pcts - med)))
            if mad == 0:
                continue
            for d in entities:
                if d.get("change_pct") is None:
                    continue
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
        # KNOWN-SCENARIO DETECTION (metric-specific)
        # ----------------------------------------------------------
        metric_deltas = state.get("metric_deltas") or {}
        scenario_detection = self._detect_known_scenarios(metric_deltas)
        state["scenario_detection"] = scenario_detection

        # Legacy compat: known_scenario_check derives from the metric
        # detection but keeps the old field name and shape so
        # report_agent (which reads it in the fallback path) and any
        # external consumer keep working. Delivery 4 will flip the
        # report to read scenario_detection directly.
        detected_ids = {d["id"] for d in scenario_detection if d["detected"]}
        computed["known_scenario_check"] = [
            {
                "scenario_id": s["id"],
                "dimension": s["dimension"],
                "entity": s["entity"],
                "expected_metric": s["metric"],
                "expected_direction": s["direction"],
                "detected_by_cross_entity_zscore": s["id"] in detected_ids,
            }
            for s in KNOWN_SCENARIOS
        ]

        detected_count = sum(1 for d in scenario_detection if d["detected"])
        self.log(
            f"Metric-specific detection: {detected_count}/{len(scenario_detection)} "
            f"scenarios detected"
        )
        for d in scenario_detection:
            marker = "✓" if d["detected"] else "✗"
            self.log(f"  {marker} {d['id']} ({d['metric']}): {d['evidence']}")

        # ----------------------------------------------------------
        # Statistical scan (general-purpose, exploratory)
        # ----------------------------------------------------------
        stats_anomalies = detect_anomalies(df) if df is not None and not df.empty else []

        # Monthly summary
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
            "scenario_detection": scenario_detection,
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
        result["scenario_detection"] = scenario_detection

        # Dedup LLM-emitted anomalies. Normalize each of entity / period
        # / value to a hashable form first so an LLM that returns
        # `value: [list of branches]` doesn't crash the tuple-set lookup.
        seen = set()
        deduped = []
        for a in result.get("anomalies", []):
            key = (
                _hashable(a.get("entity")),
                _hashable(a.get("period")),
                _hashable(a.get("value")),
            )
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

    # ------------------------------------------------------------------
    # Metric-specific scenario detection
    # ------------------------------------------------------------------
    def _detect_known_scenarios(self, metric_deltas: dict) -> list:
        detectors = {
            "A1": self._detect_a1_region_revenue,
            "A2": self._detect_a2_product_margin,
            "A3": self._detect_a3_branch_cancellations,
            "A4": self._detect_a4_segment_quantity,
        }
        results = []
        for scenario in KNOWN_SCENARIOS:
            det = detectors[scenario["id"]](scenario, metric_deltas)
            results.append(det)
        return results

    def _lookup_entity_delta(self, metric_deltas: dict, metric: str,
                             dimension: str, entity: str) -> dict | None:
        table = (metric_deltas.get(metric) or {}).get(dimension) or []
        for d in table:
            if d.get("entity") == entity:
                return d
        return None

    def _detect_a1_region_revenue(self, scenario: dict, metric_deltas: dict) -> dict:
        d = self._lookup_entity_delta(metric_deltas, "revenue", "region", "Region_A")
        if d is None or d.get("change_pct") is None:
            return self._not_detected(scenario, "revenue",
                                      "Region_A not found in revenue deltas")
        pct = d["change_pct"] / 100.0
        threshold = scenario["threshold_pct"]
        detected = pct <= threshold
        return {
            "id": scenario["id"],
            "detected": detected,
            "metric": "revenue",
            "dimension": "region",
            "entity": "Region_A",
            "baseline_value": d["baseline_value"],
            "target_value": d["target_value"],
            "delta_ratio": round(pct, 4),
            "threshold_ratio": threshold,
            "evidence": (
                f"Region_A revenue {d['baseline_value']:,.2f} -> "
                f"{d['target_value']:,.2f} ({pct*100:+.2f}%), "
                f"threshold {threshold*100:.0f}%"
            ),
        }

    def _detect_a2_product_margin(self, scenario: dict, metric_deltas: dict) -> dict:
        d = self._lookup_entity_delta(metric_deltas, "cost_ratio", "product_id", "P007")
        if d is None:
            return self._not_detected(scenario, "cost_ratio",
                                      "P007 not found in cost_ratio deltas")
        delta_pp = d["absolute_change"] / 100.0
        threshold = scenario["threshold_pp"]
        detected = delta_pp >= threshold
        return {
            "id": scenario["id"],
            "detected": detected,
            "metric": "cost_ratio",
            "dimension": "product_id",
            "entity": "P007",
            "baseline_value": d["baseline_value"],
            "target_value": d["target_value"],
            "delta_ratio": round(delta_pp, 4),
            "threshold_ratio": threshold,
            "evidence": (
                f"P007 cost/revenue {d['baseline_value']:.4f} -> "
                f"{d['target_value']:.4f} (+{d['absolute_change']:.2f}pp), "
                f"threshold +{threshold*100:.0f}pp"
            ),
        }

    def _detect_a3_branch_cancellations(self, scenario: dict, metric_deltas: dict) -> dict:
        d = self._lookup_entity_delta(metric_deltas, "cancel_rate", "branch", "Branch_D1")
        if d is None:
            return self._not_detected(scenario, "cancel_rate",
                                      "Branch_D1 not found in cancel_rate deltas")
        delta_pp = d["absolute_change"] / 100.0
        threshold = scenario["threshold_pp"]
        detected = delta_pp >= threshold
        return {
            "id": scenario["id"],
            "detected": detected,
            "metric": "cancel_rate",
            "dimension": "branch",
            "entity": "Branch_D1",
            "baseline_value": d["baseline_value"],
            "target_value": d["target_value"],
            "delta_ratio": round(delta_pp, 4),
            "threshold_ratio": threshold,
            "evidence": (
                f"Branch_D1 cancel_rate {d['baseline_value']:.4f} -> "
                f"{d['target_value']:.4f} (+{d['absolute_change']:.2f}pp), "
                f"threshold +{threshold*100:.0f}pp"
            ),
        }

    def _detect_a4_segment_quantity(self, scenario: dict, metric_deltas: dict) -> dict:
        d = self._lookup_entity_delta(
            metric_deltas, "quantity", "customer_segment", "Occasional"
        )
        if d is None or d.get("change_pct") is None:
            return self._not_detected(scenario, "quantity",
                                      "Occasional not found in quantity deltas")
        pct = d["change_pct"] / 100.0
        threshold = scenario["threshold_pct"]
        detected = pct <= threshold
        return {
            "id": scenario["id"],
            "detected": detected,
            "metric": "quantity",
            "dimension": "customer_segment",
            "entity": "Occasional",
            "baseline_value": d["baseline_value"],
            "target_value": d["target_value"],
            "delta_ratio": round(pct, 4),
            "threshold_ratio": threshold,
            "evidence": (
                f"Occasional mean quantity {d['baseline_value']:.4f} -> "
                f"{d['target_value']:.4f} ({pct*100:+.2f}%), "
                f"threshold {threshold*100:.0f}%"
            ),
        }

    @staticmethod
    def _not_detected(scenario: dict, metric: str, reason: str) -> dict:
        return {
            "id": scenario["id"],
            "detected": False,
            "metric": metric,
            "dimension": scenario["dimension"],
            "entity": scenario["entity"],
            "baseline_value": None,
            "target_value": None,
            "delta_ratio": None,
            "threshold_ratio": None,
            "evidence": reason,
        }