"""
Phase 3 tests — per-agent deterministic logic with mocked LLM.

Run:
    pytest tests/test_agents.py -v

Exit criterion for Phase 3:
    * AnalysisAgent no longer sets state["data"] = None (Step 3.1)
    * RootCauseAgent no longer sums contribution_pct across dimensions (Step 3.2)
    * AnomalyAgent computes cross-entity robust z-scores and reports
      known-scenario detection status honestly (Step 3.3)
    * All tests pass without invoking a real LLM.
"""

from unittest.mock import patch

import pandas as pd
import pytest


# ----------------------------------------------------------------------
# Fixtures: a minimal, deterministic state that each agent can process.
# ----------------------------------------------------------------------
@pytest.fixture
def base_state():
    return {
        "question": "Why did revenue drop in August 2026?",
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
            {
                "purpose": "region MoM",
                "columns": ["region", "month", "total"],
                "rows": [
                    {"region": "Region_A", "month": "2026-07", "total": 400.0},
                    {"region": "Region_A", "month": "2026-08", "total": 280.0},
                    {"region": "Region_B", "month": "2026-07", "total": 350.0},
                    {"region": "Region_B", "month": "2026-08", "total": 340.0},
                ],
            },
        ],
        "dimension_deltas": {
            # Six regions: one extreme outlier (-30%) plus five clustered
            # near zero. MAD of {-30, -2, -1, 0, 1, 2} = 2, median = -0.5,
            # robust z of Region_A ≈ 9.9 → flags. Peers stay under 2.
            "region": [
                {"entity": "Region_A", "dimension": "region",
                 "baseline_period": "2026-07", "target_period": "2026-08",
                 "baseline_value": 400.0, "target_value": 280.0,
                 "absolute_change": -120.0, "change_pct": -30.0},
                {"entity": "Region_B", "dimension": "region",
                 "baseline_period": "2026-07", "target_period": "2026-08",
                 "baseline_value": 350.0, "target_value": 343.0,
                 "absolute_change": -7.0, "change_pct": -2.0},
                {"entity": "Region_C", "dimension": "region",
                 "baseline_period": "2026-07", "target_period": "2026-08",
                 "baseline_value": 300.0, "target_value": 297.0,
                 "absolute_change": -3.0, "change_pct": -1.0},
                {"entity": "Region_D", "dimension": "region",
                 "baseline_period": "2026-07", "target_period": "2026-08",
                 "baseline_value": 300.0, "target_value": 300.0,
                 "absolute_change": 0.0, "change_pct": 0.0},
                {"entity": "Region_E", "dimension": "region",
                 "baseline_period": "2026-07", "target_period": "2026-08",
                 "baseline_value": 300.0, "target_value": 303.0,
                 "absolute_change": 3.0, "change_pct": 1.0},
                {"entity": "Region_F", "dimension": "region",
                 "baseline_period": "2026-07", "target_period": "2026-08",
                 "baseline_value": 300.0, "target_value": 306.0,
                 "absolute_change": 6.0, "change_pct": 2.0},
            ],
            # Four products: P007 extreme (-25%), three peers near zero.
            # MAD of {-25, -1, 0, 1} = 1, median = -0.5,
            # robust z of P007 ≈ 16.5 → flags.
            "product_id": [
                {"entity": "P007", "dimension": "product_id",
                 "baseline_period": "2026-07", "target_period": "2026-08",
                 "baseline_value": 200.0, "target_value": 150.0,
                 "absolute_change": -50.0, "change_pct": -25.0},
                {"entity": "P001", "dimension": "product_id",
                 "baseline_period": "2026-07", "target_period": "2026-08",
                 "baseline_value": 180.0, "target_value": 178.2,
                 "absolute_change": -1.8, "change_pct": -1.0},
                {"entity": "P002", "dimension": "product_id",
                 "baseline_period": "2026-07", "target_period": "2026-08",
                 "baseline_value": 160.0, "target_value": 160.0,
                 "absolute_change": 0.0, "change_pct": 0.0},
                {"entity": "P003", "dimension": "product_id",
                 "baseline_period": "2026-07", "target_period": "2026-08",
                 "baseline_value": 150.0, "target_value": 151.5,
                 "absolute_change": 1.5, "change_pct": 1.0},
            ],
        },
    }


# ----------------------------------------------------------------------
# Step 3.1 — AnalysisAgent
# ----------------------------------------------------------------------
def test_analysis_headline_override(base_state):
    """The Python override must produce correct numbers regardless of LLM."""
    from app.agents.analysis_agent import AnalysisAgent

    with patch("app.llm.client.llm.complete_json", return_value={"garbage": "from llm"}):
        out = AnalysisAgent().run(dict(base_state))

    os_ = out["analysis"]["overall_summary"]
    assert os_["total_change_pct"] == pytest.approx(-10.0)
    assert os_["absolute_change"] == pytest.approx(-100.0)
    assert os_["baseline_period"] == "2026-07"
    assert os_["target_period"] == "2026-08"


def test_analysis_does_not_null_data(base_state):
    """
    Regression test for Step 3.1: state["data"] must be a non-None DataFrame
    after run(), because ChartAgent reads it and bails out on None.
    """
    from app.agents.analysis_agent import AnalysisAgent

    with patch("app.llm.client.llm.complete_json", return_value={}):
        out = AnalysisAgent().run(dict(base_state))

    assert out["data"] is not None, "state['data'] was set to None (breaks ChartAgent)"
    assert isinstance(out["data"], pd.DataFrame)
    assert len(out["data"]) > 0


def test_analysis_data_prefers_non_headline_result(base_state):
    """ChartAgent needs many rows; the headline is only 2 rows."""
    from app.agents.analysis_agent import AnalysisAgent

    with patch("app.llm.client.llm.complete_json", return_value={}):
        out = AnalysisAgent().run(dict(base_state))

    assert len(out["data"]) == 4
    assert "region" in out["data"].columns


# ----------------------------------------------------------------------
# Step 3.2 — RootCauseAgent
# ----------------------------------------------------------------------
def test_rootcause_contribution_is_within_dimension(base_state):
    """
    Regression test for Step 3.2: contribution_pct must be measured within
    each dimension, not summed across dimensions.

    Region_A's -120 is 100% of the region dimension's negative total
    (-130 for {A, B, C} with the fixture above), not a share of a
    cross-dimension total that double-counts Region_A.
    """
    from app.agents.rootcause_agent import RootCauseAgent

    with patch("app.llm.client.llm.complete_json", return_value={}):
        out = RootCauseAgent().run(dict(base_state))

    causes = out["root_causes"]["primary_causes"]
    assert causes, "no primary causes were produced"

    region_a = next((c for c in causes if c["value"] == "Region_A"), None)
    assert region_a is not None
    assert region_a["contribution_pct"] == pytest.approx(120 / 130 * 100, abs=0.01)
    assert "region" in region_a["contribution_basis"].lower()
    assert "not additive" in region_a["note"].lower()


def test_rootcause_each_cause_has_dimension_and_note(base_state):
    """Every cause must carry its dimension and the non-additivity note."""
    from app.agents.rootcause_agent import RootCauseAgent

    with patch("app.llm.client.llm.complete_json", return_value={}):
        out = RootCauseAgent().run(dict(base_state))

    for c in out["root_causes"]["primary_causes"]:
        assert "dimension" in c
        assert "note" in c
        assert "not additive" in c["note"].lower()


def test_rootcause_limitation_states_cross_dimension_rule(base_state):
    """The limitation text must explicitly say dimensions aren't additive."""
    from app.agents.rootcause_agent import RootCauseAgent

    with patch("app.llm.client.llm.complete_json", return_value={}):
        out = RootCauseAgent().run(dict(base_state))

    limitations = out["root_causes"]["known_limitations"]
    joined = " ".join(limitations).lower()
    assert "within each dimension" in joined
    assert "not" in joined and "summed" in joined


# ----------------------------------------------------------------------
# Step 3.3 — AnomalyAgent
# ----------------------------------------------------------------------
def test_anomaly_cross_entity_zscore_flags_outlier(base_state):
    """
    With six regions — one at -30%, five clustered near zero — Region_A
    is a robust cross-entity outlier. MAD-based z-score keeps the std
    from being inflated by the outlier itself.
    """
    from app.agents.anomaly_agent import AnomalyAgent

    with patch("app.llm.client.llm.complete_json", return_value={"anomalies": []}):
        out = AnomalyAgent().run(dict(base_state))

    entity_anoms = out["anomalies"]["computed_statistics"].get("entity_anomalies", [])
    flagged = {(a["dimension"], a["entity"]) for a in entity_anoms}
    assert ("region", "Region_A") in flagged, (
        f"Region_A was not flagged; got {flagged}"
    )
    assert ("product_id", "P007") in flagged, (
        f"P007 was not flagged; got {flagged}"
    )
    # Peer entities must NOT be flagged — robust z-scores of the cluster
    # are tiny. This catches a broken fix that flags everything.
    assert ("region", "Region_B") not in flagged
    assert ("product_id", "P001") not in flagged


def test_anomaly_known_scenario_check_present(base_state):
    """The output must contain a scenario check listing all four known cases."""
    from app.agents.anomaly_agent import AnomalyAgent

    with patch("app.llm.client.llm.complete_json", return_value={"anomalies": []}):
        out = AnomalyAgent().run(dict(base_state))

    checks = out["anomalies"]["computed_statistics"].get("known_scenario_check", [])
    assert len(checks) == 4
    ids = {c["scenario_id"] for c in checks}
    assert ids == {"A1", "A2", "A3", "A4"}


def test_anomaly_does_not_claim_clean_when_scenarios_missed(base_state):
    """
    When known scenarios aren't detected, the check must mark them as
    not-detected. (The reviewer's concern: 'no high-severity anomalies
    detected' should not be the confident answer when we know four were
    injected.)
    """
    from app.agents.anomaly_agent import AnomalyAgent

    with patch("app.llm.client.llm.complete_json", return_value={"anomalies": []}):
        out = AnomalyAgent().run(dict(base_state))

    checks = out["anomalies"]["computed_statistics"]["known_scenario_check"]
    a1 = next(c for c in checks if c["scenario_id"] == "A1")
    assert a1["detected_by_cross_entity_zscore"] is True, (
        "A1 (Region_A) should be detected by robust z-score"
    )

    # A3 (Branch_D1) has no branch dimension in this fixture, so it must
    # honestly report not-detected rather than claiming clean.
    a3 = next(c for c in checks if c["scenario_id"] == "A3")
    assert a3["detected_by_cross_entity_zscore"] is False