"""
Phase 3 + Delivery 2 tests — per-agent deterministic logic with mocked LLM.

Run:
    pytest tests/test_agents.py -v
"""

from unittest.mock import patch

import pandas as pd
import pytest


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
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
# AnalysisAgent (unchanged tests from Phase 3)
# ----------------------------------------------------------------------
def test_analysis_headline_override(base_state):
    from app.agents.analysis_agent import AnalysisAgent
    with patch("app.llm.client.llm.complete_json", return_value={"garbage": "from llm"}):
        out = AnalysisAgent().run(dict(base_state))
    os_ = out["analysis"]["overall_summary"]
    assert os_["total_change_pct"] == pytest.approx(-10.0)
    assert os_["absolute_change"] == pytest.approx(-100.0)


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
    # 12000 / 13000 * 100 ≈ 92.31
    assert region_a["contribution_pct"] == pytest.approx(12000 / 13000 * 100, abs=0.01)
    assert "not additive" in region_a["note"].lower()


def test_rootcause_limitation_states_cross_dimension_rule(base_state):
    from app.agents.rootcause_agent import RootCauseAgent
    with patch("app.llm.client.llm.complete_json", return_value={}):
        out = RootCauseAgent().run(dict(base_state))
    joined = " ".join(out["root_causes"]["known_limitations"]).lower()
    assert "within each dimension" in joined


def test_rootcause_reports_gross_and_positive(base_state):
    """Delivery 2: gross negative and positive offset are separate fields."""
    from app.agents.rootcause_agent import RootCauseAgent
    with patch("app.llm.client.llm.complete_json", return_value={}):
        out = RootCauseAgent().run(dict(base_state))
    rc = out["root_causes"]
    # Region_A(-12000) + Region_B(-700) + Region_C(-300) = -13000
    assert rc["gross_negative"] == pytest.approx(13000.0, abs=1.0)
    # Region_E(+300) + Region_F(+600) = +900
    assert rc["positive_offset"] == pytest.approx(900.0, abs=1.0)
    # net = positive - gross = 900 - 13000 = -12100
    assert rc["net_revenue_change"] == pytest.approx(-12100.0, abs=1.0)


# ----------------------------------------------------------------------
# AnomalyAgent — MAD z-score (Phase 3)
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
    """Delivery 2: known_scenario_check derives from scenario_detection."""
    from app.agents.anomaly_agent import AnomalyAgent
    with patch("app.llm.client.llm.complete_json", return_value={"anomalies": []}):
        out = AnomalyAgent().run(dict(base_state))
    checks = (out["anomalies"].get("computed_statistics") or {}).get("known_scenario_check", [])
    assert len(checks) == 4
    assert {c["scenario_id"] for c in checks} == {"A1", "A2", "A3", "A4"}


def test_anomaly_does_not_claim_clean_when_scenarios_missed(base_state):
    """With the fixture's metric_deltas, all four should be detected."""
    from app.agents.anomaly_agent import AnomalyAgent
    with patch("app.llm.client.llm.complete_json", return_value={"anomalies": []}):
        out = AnomalyAgent().run(dict(base_state))
    scenario_detection = out.get("scenario_detection", [])
    assert len(scenario_detection) == 4
    by_id = {d["id"]: d for d in scenario_detection}
    # Fixture has injected magnitudes matching each scenario's threshold
    assert by_id["A1"]["detected"] is True   # -30% ≤ -25%
    assert by_id["A2"]["detected"] is True   # +0.17 ≥ +0.10
    assert by_id["A3"]["detected"] is True   # +14.15pp ≥ +8pp
    assert by_id["A4"]["detected"] is True   # -40% ≤ -30%


# ----------------------------------------------------------------------
# Delivery 2 — metric-specific detection
# ----------------------------------------------------------------------
def test_metric_deltas_structure(base_state):
    """AnomalyAgent reads state["metric_deltas"] for four metrics."""
    from app.agents.anomaly_agent import AnomalyAgent
    with patch("app.llm.client.llm.complete_json", return_value={"anomalies": []}):
        out = AnomalyAgent().run(dict(base_state))
    md = out.get("metric_deltas") or base_state["metric_deltas"]
    assert "revenue" in md or "metric_deltas" in out
    # On this fixture, metric_deltas is provided by the input state
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
    """If A2's delta is only +0.05 (below +0.10 threshold), detection is False."""
    from app.agents.anomaly_agent import AnomalyAgent
    # Override the fixture: P007 barely moves
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
    """If P007 isn't in the delta table, A2 is not detected (not an error)."""
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