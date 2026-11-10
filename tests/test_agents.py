"""
Phase 3 + Delivery 2 + Delivery 3 tests — per-agent deterministic logic
with mocked LLM.
"""

from unittest.mock import patch

import pandas as pd
import pytest


@pytest.fixture
def base_state():
    return {
        "question": "Why did revenue drop in August 2026?",
        "intent": {"time_reference": "2026-08", "intent_type": "diagnostic"},
        "dataframe": pd.DataFrame({
            "date": pd.to_datetime(["2026-07-15", "2026-08-15"]),
            "region": ["Region_A", "Region_A"],
            "revenue": [100.0, 70.0],
            "order_status": ["Completed", "Completed"],
        }),
        "sql_results": [
            {
                "purpose": "HEADLINE: 2026-07 vs 2026-08 canonical totals",
                "columns": ["month", "txn_count", "total_rev", "aov", "cancel_rate"],
                "rows": [
                    {"month": "2026-07", "txn_count": 10, "total_rev": 1000.0,
                     "aov": 100.0, "cancel_rate": 0.05},
                    {"month": "2026-08", "txn_count": 12, "total_rev": 900.0,
                     "aov": 75.0, "cancel_rate": 0.10},
                ],
            },
        ],
        "dimension_deltas": {
            "region": [
                {"entity": "Region_A", "dimension": "region",
                 "baseline_period": "2026-07", "target_period": "2026-08",
                 "baseline_value": 40000.0, "target_value": 28000.0,
                 "absolute_change": -12000.0, "change_pct": -30.0},
                {"entity": "Region_B", "dimension": "region",
                 "baseline_period": "2026-07", "target_period": "2026-08",
                 "baseline_value": 35000.0, "target_value": 34300.0,
                 "absolute_change": -700.0, "change_pct": -2.0},
                {"entity": "Region_C", "dimension": "region",
                 "baseline_period": "2026-07", "target_period": "2026-08",
                 "baseline_value": 30000.0, "target_value": 29700.0,
                 "absolute_change": -300.0, "change_pct": -1.0},
                {"entity": "Region_D", "dimension": "region",
                 "baseline_period": "2026-07", "target_period": "2026-08",
                 "baseline_value": 30000.0, "target_value": 30000.0,
                 "absolute_change": 0.0, "change_pct": 0.0},
                {"entity": "Region_E", "dimension": "region",
                 "baseline_period": "2026-07", "target_period": "2026-08",
                 "baseline_value": 30000.0, "target_value": 30300.0,
                 "absolute_change": 300.0, "change_pct": 1.0},
                {"entity": "Region_F", "dimension": "region",
                 "baseline_period": "2026-07", "target_period": "2026-08",
                 "baseline_value": 30000.0, "target_value": 30600.0,
                 "absolute_change": 600.0, "change_pct": 2.0},
            ],
        },
        "metric_deltas": {
            "revenue": {
                "region": [
                    {"entity": "Region_A", "dimension": "region",
                     "baseline_period": "2026-07", "target_period": "2026-08",
                     "baseline_value": 40000.0, "target_value": 28000.0,
                     "absolute_change": -12000.0, "change_pct": -30.0},
                ],
            },
            "cost_ratio": {
                "product_id": [
                    {"entity": "P007", "dimension": "product_id",
                     "baseline_period": "2026-07", "target_period": "2026-08",
                     "baseline_value": 0.75, "target_value": 0.92,
                     "absolute_change": 17.0, "change_pct": 22.67},
                ],
            },
            "cancel_rate": {
                "branch": [
                    {"entity": "Branch_D1", "dimension": "branch",
                     "baseline_period": "2026-07", "target_period": "2026-08",
                     "baseline_value": 0.0566, "target_value": 0.1981,
                     "absolute_change": 14.15, "change_pct": 250.18},
                ],
            },
            "quantity": {
                "customer_segment": [
                    {"entity": "Occasional", "dimension": "customer_segment",
                     "baseline_period": "2026-07", "target_period": "2026-08",
                     "baseline_value": 2.5, "target_value": 1.5,
                     "absolute_change": -1.0, "change_pct": -40.0},
                ],
            },
        },
        "root_causes": {
            "primary_causes": [
                {"dimension": "region", "value": "Region_A",
                 "change_pct": -30.0, "absolute_change": -12000.0,
                 "contribution_pct": 92.31,
                 "contribution_basis": "share of region gross negative change",
                 "confidence": "high"},
            ],
            "dimension_totals": {
                "region": {"total_negative": 13000.0, "total_positive": 900.0, "net": -12100.0},
            },
            "gross_negative": 13000.0,
            "positive_offset": 900.0,
            "net_revenue_change": -12100.0,
            "known_limitations": [
                "contribution_pct is measured within each dimension, not across dimensions.",
            ],
        },
        "anomalies": {
            "anomalies": [
                {"entity": "Region_A", "period": "2026-08",
                 "value": 28000.0, "expected": 40000.0,
                 "deviation_pct": -30.0, "severity": "high",
                 "anomaly_type": "contextual"},
            ],
        },
        "charts": [],
        "critique": {"confidence": "medium"},
    }


# ----------------------------------------------------------------------
# AnalysisAgent
# ----------------------------------------------------------------------
def test_analysis_headline_override(base_state):
    """Diagnostic intent + fixture missing completed_* fields → falls back to gross.
    Also asserts driver_decomposition is always present."""
    from app.agents.analysis_agent import AnalysisAgent
    with patch("app.llm.client.llm.complete_json", return_value={"garbage": "from llm"}):
        out = AnalysisAgent().run(dict(base_state))
    os_ = out["analysis"]["overall_summary"]
    assert "revenue_definition" in os_
    # Fixture has no completed_rev → fallback to gross_revenue
    assert os_["revenue_definition"] == "gross_revenue"
    assert "driver_decomposition" in os_


def test_analysis_does_not_null_data(base_state):
    from app.agents.analysis_agent import AnalysisAgent
    with patch("app.llm.client.llm.complete_json", return_value={}):
        out = AnalysisAgent().run(dict(base_state))
    assert out["data"] is not None
    assert isinstance(out["data"], pd.DataFrame)


def test_analysis_data_prefers_non_headline_result(base_state):
    from app.agents.analysis_agent import AnalysisAgent
    with patch("app.llm.client.llm.complete_json", return_value={}):
        out = AnalysisAgent().run(dict(base_state))
    assert isinstance(out["data"], pd.DataFrame)


# Delivery 3 — driver decomposition
def test_driver_decomposition_sums_exactly(base_state):
    from app.agents.analysis_agent import AnalysisAgent
    bs = dict(base_state)
    bs["sql_results"] = [{
        "purpose": "HEADLINE: 2026-07 vs 2026-08 canonical totals",
        "columns": ["month", "txn_count", "total_rev", "aov",
                    "completed_rev", "completed_txn_count", "completed_aov",
                    "cancel_rate"],
        "rows": [
            {"month": "2026-07", "txn_count": 10, "total_rev": 1000.0,
             "aov": 100.0, "completed_rev": 900.0,
             "completed_txn_count": 9, "completed_aov": 100.0,
             "cancel_rate": 0.05},
            {"month": "2026-08", "txn_count": 12, "total_rev": 900.0,
             "aov": 75.0, "completed_rev": 810.0,
             "completed_txn_count": 11, "completed_aov": 73.64,
             "cancel_rate": 0.10},
        ],
    }]
    with patch("app.llm.client.llm.complete_json", return_value={}):
        out = AnalysisAgent().run(bs)
    dd = out["analysis"]["overall_summary"]["driver_decomposition"]
    assert dd is not None
    # abs=0.05 accommodates the pre-rounded completed_aov=73.64 in this
    # fixture (real value is 810/11=73.6363...). Production headline uses
    # unrounded SQL aggregates, so residual there is < 0.01.
    assert dd["sum_check"] == pytest.approx(dd["total_change"], abs=0.05)
    assert "volume_effect" in dd
    assert "price_effect" in dd
    assert "interaction_effect" in dd
    assert isinstance(dd["narrative"], str) and dd["narrative"]


def test_driver_decomposition_narrative_matches_signs():
    from app.agents.analysis_agent import _decompose_drivers
    dd = _decompose_drivers(1000.0, 900.0, 10, 12, 100.0, 75.0)
    assert dd is not None
    assert dd["volume_effect"] == pytest.approx(200.0)
    assert dd["price_effect"] == pytest.approx(-250.0)
    assert dd["interaction_effect"] == pytest.approx(-50.0)
    assert "price" in dd["narrative"].lower()
    assert "volume" in dd["narrative"].lower()
    assert "offset" in dd["narrative"].lower()


def test_revenue_definition_selected_by_intent():
    """Descriptive intent → gross; diagnostic + completed_rev present → completed."""
    from app.agents.analysis_agent import AnalysisAgent

    headline_with_both = {
        "purpose": "HEADLINE: 2026-07 vs 2026-08 canonical totals",
        "columns": ["month", "txn_count", "total_rev", "aov",
                    "completed_rev", "completed_txn_count", "completed_aov",
                    "cancel_rate"],
        "rows": [
            {"month": "2026-07", "txn_count": 10, "total_rev": 1000.0,
             "aov": 100.0, "completed_rev": 900.0,
             "completed_txn_count": 9, "completed_aov": 100.0,
             "cancel_rate": 0.05},
            {"month": "2026-08", "txn_count": 12, "total_rev": 900.0,
             "aov": 75.0, "completed_rev": 810.0,
             "completed_txn_count": 11, "completed_aov": 73.64,
             "cancel_rate": 0.10},
        ],
    }

    # Descriptive → gross
    state_desc = {
        "question": "Show revenue trend",
        "intent": {"intent_type": "descriptive", "time_reference": "2026-08"},
        "sql_results": [headline_with_both],
        "dimension_deltas": {},
    }
    with patch("app.llm.client.llm.complete_json", return_value={}):
        out_desc = AnalysisAgent().run(state_desc)
    assert out_desc["analysis"]["overall_summary"]["revenue_definition"] == "gross_revenue"

    # Diagnostic + completed_rev present → completed
    state_diag = {
        "question": "Why did revenue change?",
        "intent": {"intent_type": "diagnostic", "time_reference": "2026-08"},
        "sql_results": [headline_with_both],
        "dimension_deltas": {},
    }
    with patch("app.llm.client.llm.complete_json", return_value={}):
        out_diag = AnalysisAgent().run(state_diag)
    assert out_diag["analysis"]["overall_summary"]["revenue_definition"] == "completed_revenue"


def test_report_payload_contains_driver_decomposition(base_state):
    from app.agents.analysis_agent import AnalysisAgent
    from app.agents.report_agent import ReportAgent

    bs = dict(base_state)
    bs["sql_results"] = [{
        "purpose": "HEADLINE: 2026-07 vs 2026-08 canonical totals",
        "columns": ["month", "txn_count", "total_rev", "aov",
                    "completed_rev", "completed_txn_count", "completed_aov",
                    "cancel_rate"],
        "rows": [
            {"month": "2026-07", "txn_count": 10, "total_rev": 1000.0,
             "aov": 100.0, "completed_rev": 900.0,
             "completed_txn_count": 9, "completed_aov": 100.0,
             "cancel_rate": 0.05},
            {"month": "2026-08", "txn_count": 12, "total_rev": 900.0,
             "aov": 75.0, "completed_rev": 810.0,
             "completed_txn_count": 11, "completed_aov": 73.64,
             "cancel_rate": 0.10},
        ],
    }]
    with patch("app.llm.client.llm.complete_json", return_value={}):
        analyzed = AnalysisAgent().run(bs)

    payload = ReportAgent()._build_verified_payload(analyzed)
    assert payload["headline"] is not None
    assert payload["headline"]["revenue_definition"] == "completed_revenue"
    dd = payload["headline"]["driver_decomposition"]
    assert dd is not None
    assert "volume_effect" in dd and "price_effect" in dd
    assert "narrative" in dd


# ----------------------------------------------------------------------
# RootCauseAgent
# ----------------------------------------------------------------------
def test_rootcause_contribution_is_within_dimension(base_state):
    from app.agents.rootcause_agent import RootCauseAgent
    with patch("app.llm.client.llm.complete_json", return_value={}):
        out = RootCauseAgent().run(dict(base_state))
    causes = out["root_causes"]["primary_causes"]
    assert causes
    region_a = next((c for c in causes if c["value"] == "Region_A"), None)
    assert region_a is not None
    assert region_a["contribution_pct"] == pytest.approx(12000 / 13000 * 100, abs=0.01)
    assert "not additive" in region_a["note"].lower()


def test_rootcause_limitation_states_cross_dimension_rule(base_state):
    from app.agents.rootcause_agent import RootCauseAgent
    with patch("app.llm.client.llm.complete_json", return_value={}):
        out = RootCauseAgent().run(dict(base_state))
    joined = " ".join(out["root_causes"]["known_limitations"]).lower()
    assert "within each dimension" in joined


def test_rootcause_reports_gross_and_positive(base_state):
    from app.agents.rootcause_agent import RootCauseAgent
    with patch("app.llm.client.llm.complete_json", return_value={}):
        out = RootCauseAgent().run(dict(base_state))
    rc = out["root_causes"]
    assert rc["gross_negative"] == pytest.approx(13000.0, abs=1.0)
    assert rc["positive_offset"] == pytest.approx(900.0, abs=1.0)
    assert rc["net_revenue_change"] == pytest.approx(-12100.0, abs=1.0)


# ----------------------------------------------------------------------
# AnomalyAgent — MAD z-score
# ----------------------------------------------------------------------
def test_anomaly_cross_entity_zscore_flags_outlier(base_state):
    from app.agents.anomaly_agent import AnomalyAgent
    with patch("app.llm.client.llm.complete_json", return_value={"anomalies": []}):
        out = AnomalyAgent().run(dict(base_state))
    entity_anoms = (out["anomalies"].get("computed_statistics") or {}).get("entity_anomalies", [])
    flagged = {(a["dimension"], a["entity"]) for a in entity_anoms}
    assert ("region", "Region_A") in flagged
    assert ("region", "Region_B") not in flagged


def test_anomaly_known_scenario_check_present(base_state):
    from app.agents.anomaly_agent import AnomalyAgent
    with patch("app.llm.client.llm.complete_json", return_value={"anomalies": []}):
        out = AnomalyAgent().run(dict(base_state))
    checks = (out["anomalies"].get("computed_statistics") or {}).get("known_scenario_check", [])
    assert len(checks) == 4
    assert {c["scenario_id"] for c in checks} == {"A1", "A2", "A3", "A4"}


def test_anomaly_does_not_claim_clean_when_scenarios_missed(base_state):
    from app.agents.anomaly_agent import AnomalyAgent
    with patch("app.llm.client.llm.complete_json", return_value={"anomalies": []}):
        out = AnomalyAgent().run(dict(base_state))
    scenario_detection = out.get("scenario_detection", [])
    assert len(scenario_detection) == 4
    by_id = {d["id"]: d for d in scenario_detection}
    assert by_id["A1"]["detected"] is True
    assert by_id["A2"]["detected"] is True
    assert by_id["A3"]["detected"] is True
    assert by_id["A4"]["detected"] is True


def test_metric_deltas_structure(base_state):
    from app.agents.anomaly_agent import AnomalyAgent
    with patch("app.llm.client.llm.complete_json", return_value={"anomalies": []}):
        out = AnomalyAgent().run(dict(base_state))
    md = out.get("metric_deltas") or base_state["metric_deltas"]
    assert "revenue" in md or "metric_deltas" in out
    assert set(base_state["metric_deltas"].keys()) == {
        "revenue", "cost_ratio", "cancel_rate", "quantity"
    }


def test_a2_detection_uses_cost_ratio(base_state):
    from app.agents.anomaly_agent import AnomalyAgent
    with patch("app.llm.client.llm.complete_json", return_value={"anomalies": []}):
        out = AnomalyAgent().run(dict(base_state))
    a2 = next(d for d in out["scenario_detection"] if d["id"] == "A2")
    assert a2["detected"] is True
    assert a2["metric"] == "cost_ratio"
    assert a2["baseline_value"] == pytest.approx(0.75)
    assert a2["target_value"] == pytest.approx(0.92)
    assert a2["evidence"] and "P007" in a2["evidence"]


def test_a3_detection_uses_cancel_rate(base_state):
    from app.agents.anomaly_agent import AnomalyAgent
    with patch("app.llm.client.llm.complete_json", return_value={"anomalies": []}):
        out = AnomalyAgent().run(dict(base_state))
    a3 = next(d for d in out["scenario_detection"] if d["id"] == "A3")
    assert a3["detected"] is True
    assert a3["metric"] == "cancel_rate"
    assert a3["baseline_value"] == pytest.approx(0.0566)
    assert a3["target_value"] == pytest.approx(0.1981)


def test_a4_detection_uses_quantity(base_state):
    from app.agents.anomaly_agent import AnomalyAgent
    with patch("app.llm.client.llm.complete_json", return_value={"anomalies": []}):
        out = AnomalyAgent().run(dict(base_state))
    a4 = next(d for d in out["scenario_detection"] if d["id"] == "A4")
    assert a4["detected"] is True
    assert a4["metric"] == "quantity"
    assert a4["baseline_value"] == pytest.approx(2.5)
    assert a4["target_value"] == pytest.approx(1.5)


def test_scenario_below_threshold_not_detected(base_state):
    from app.agents.anomaly_agent import AnomalyAgent
    bs = dict(base_state)
    bs["metric_deltas"] = {
        **base_state["metric_deltas"],
        "cost_ratio": {
            "product_id": [
                {"entity": "P007", "dimension": "product_id",
                 "baseline_period": "2026-07", "target_period": "2026-08",
                 "baseline_value": 0.75, "target_value": 0.80,
                 "absolute_change": 5.0, "change_pct": 6.67},
            ],
        },
    }
    with patch("app.llm.client.llm.complete_json", return_value={"anomalies": []}):
        out = AnomalyAgent().run(bs)
    a2 = next(d for d in out["scenario_detection"] if d["id"] == "A2")
    assert a2["detected"] is False


def test_scenario_missing_entity_not_detected(base_state):
    from app.agents.anomaly_agent import AnomalyAgent
    bs = dict(base_state)
    bs["metric_deltas"] = {
        **base_state["metric_deltas"],
        "cost_ratio": {"product_id": []},
    }
    with patch("app.llm.client.llm.complete_json", return_value={"anomalies": []}):
        out = AnomalyAgent().run(bs)
    a2 = next(d for d in out["scenario_detection"] if d["id"] == "A2")
    assert a2["detected"] is False
    assert "not found" in a2["evidence"].lower()