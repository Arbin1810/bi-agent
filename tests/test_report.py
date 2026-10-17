"""
Phase 4 tests — report traceability without a real LLM.

Run:
    pytest tests/test_report.py -v

Exit criterion for Phase 4:
    * _build_verified_payload produces a deterministic, complete payload
    * _validate_report flags untraceable numbers
    * _validate_report does NOT flag structural numbers (1, 2, 3, ...)
    * _validate_report flags missing scenario disclaimers
    * Report agent state includes report_validation
"""

from unittest.mock import patch

import pytest


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def base_state():
    return {
        "question": "Why did revenue drop in August 2026?",
        "analysis": {
            "overall_summary": {
                "baseline_period": "2026-07",
                "target_period": "2026-08",
                "baseline_total": 100000.0,
                "target_total": 70000.0,
                "absolute_change": -30000.0,
                "total_change_pct": -30.0,
                "baseline_aov": 100.0,
                "target_aov": 75.0,
                "baseline_txn_count": 1000,
                "target_txn_count": 933,
                "baseline_cancel_rate": 0.05,
                "target_cancel_rate": 0.18,
            },
        },
        "dimension_deltas": {
            "region": [
                {"entity": "Region_A", "dimension": "region",
                 "baseline_period": "2026-07", "target_period": "2026-08",
                 "baseline_value": 40000.0, "target_value": 28000.0,
                 "absolute_change": -12000.0, "change_pct": -30.0},
                {"entity": "Region_B", "dimension": "region",
                 "baseline_period": "2026-07", "target_period": "2026-08",
                 "baseline_value": 35000.0, "target_value": 34000.0,
                 "absolute_change": -1000.0, "change_pct": -2.86},
            ],
            "product_id": [
                {"entity": "P007", "dimension": "product_id",
                 "baseline_period": "2026-07", "target_period": "2026-08",
                 "baseline_value": 5000.0, "target_value": 3750.0,
                 "absolute_change": -1250.0, "change_pct": -25.0},
            ],
        },
        "root_causes": {
            "primary_causes": [
                {"dimension": "region", "value": "Region_A",
                 "change_pct": -30.0, "absolute_change": -12000.0,
                 "contribution_pct": 92.31,
                 "contribution_basis": "share of region negative change",
                 "confidence": "high"},
            ],
            "known_limitations": [
                "contribution_pct is measured within each dimension, not across dimensions.",
                "The anomaly detector z-scores revenue per entity.",
            ],
        },
        "anomalies": {
            "anomalies": [
                {"entity": "Region_A", "period": "2026-08",
                 "value": 28000.0, "expected": 40000.0,
                 "deviation_pct": -30.0, "severity": "high",
                 "anomaly_type": "contextual"},
            ],
            "computed_statistics": {
                "known_scenario_check": [
                    {"scenario_id": "A1", "dimension": "region",
                     "entity": "Region_A",
                     "detected_by_cross_entity_zscore": True},
                    {"scenario_id": "A2", "dimension": "product_id",
                     "entity": "P007",
                     "detected_by_cross_entity_zscore": False},
                    {"scenario_id": "A3", "dimension": "branch",
                     "entity": "Branch_D1",
                     "detected_by_cross_entity_zscore": False},
                    {"scenario_id": "A4", "dimension": "customer_segment",
                     "entity": "Occasional",
                     "detected_by_cross_entity_zscore": False},
                ],
            },
        },
        "charts": [],
        "critique": {"confidence": "medium"},
    }


# ----------------------------------------------------------------------
# 1. Payload construction
# ----------------------------------------------------------------------
def test_payload_built_from_state(base_state):
    from app.agents.report_agent import ReportAgent

    payload = ReportAgent()._build_verified_payload(dict(base_state))

    assert payload["headline"]["total_change_pct"] == -30.0
    assert payload["headline"]["baseline_total"] == 100000.0
    assert payload["headline"]["target_total"] == 70000.0
    assert "region" in payload["deltas"]
    assert "product_id" in payload["deltas"]
    assert payload["deltas"]["region"][0]["entity"] == "Region_A"
    assert len(payload["root_causes"]) == 1
    assert len(payload["anomalies"]) == 1
    assert payload["anomalies"][0]["severity"] == "high"


def test_payload_known_scenarios_carry_detection_status(base_state):
    from app.agents.report_agent import ReportAgent

    payload = ReportAgent()._build_verified_payload(dict(base_state))

    by_id = {s["id"]: s for s in payload["known_scenarios"]}
    assert by_id["A1"]["detected"] is True
    assert by_id["A2"]["detected"] is False
    assert by_id["A3"]["detected"] is False
    assert by_id["A4"]["detected"] is False


# ----------------------------------------------------------------------
# 2. Traceability — untraceable numbers must be caught
# ----------------------------------------------------------------------
def test_traceability_passes_on_clean_report(base_state):
    from app.agents.report_agent import ReportAgent

    agent = ReportAgent()
    payload = agent._build_verified_payload(dict(base_state))

    report = """# Revenue Analysis

## Observed Changes
- Region_A: 2026-07 → 2026-08, -30.0%
- P007: 2026-07 → 2026-08, -25.0%

## Measured Drivers
- AOV: 100.0 → 75.0
- Transaction count: 1000 → 933

## Hypotheses to Investigate
- Cross-tabulate P007 revenue by region to test the margin-collapse hypothesis.
- Branch_D1 cancellation spike — not detected by current analysis.

## Known Limitations
- contribution_pct is measured within each dimension, not across dimensions.
"""
    validation = agent._validate_report(report, payload)
    assert validation["untraceable_numbers"] == [], (
        f"unexpected untraceable: {validation['untraceable_numbers']}"
    )


def test_traceability_catches_hallucinated_number(base_state):
    """
    A currency-formatted, thousands-separated value that isn't in the
    payload must be flagged. The extractor must recognize
    "$999,999.99" as one number, not split it into "999" and "999.99".
    """
    from app.agents.report_agent import ReportAgent

    agent = ReportAgent()
    payload = agent._build_verified_payload(dict(base_state))

    report = """# Revenue Analysis

## Observed Changes
- Region_A: -30.0% ($999,999.99 loss)

## Measured Drivers
- AOV: 100.0 → 75.0
"""
    validation = agent._validate_report(report, payload)

    # Something must be flagged.
    assert validation["untraceable_numbers"], (
        "hallucinated number not caught at all"
    )

    # And the flagged set must contain the hallucinated value. Accept
    # either form (uncomma'd "999999.99" or split "999.99") so the test
    # is robust to extractor-internal tokenization.
    joined = " ".join(validation["untraceable_numbers"]).replace(",", "")
    assert "999999.99" in joined or "999.99" in joined, (
        f"hallucination missed; got {validation['untraceable_numbers']}"
    )


def test_traceability_ignores_structural_numbers(base_state):
    """
    Regression test for the reviewer's regex bug: "1 analysis, 2 dimensions,
    3 charts" should not be flagged as untraceable.
    """
    from app.agents.report_agent import ReportAgent

    agent = ReportAgent()
    payload = agent._build_verified_payload(dict(base_state))

    report = """# Revenue Analysis

This report covers 1 analysis, 2 dimensions, and 3 charts.

## Observed Changes
- Region_A: -30.0%
"""
    validation = agent._validate_report(report, payload)
    assert validation["untraceable_numbers"] == [], (
        f"structural numbers flagged; got {validation['untraceable_numbers']}"
    )


# ----------------------------------------------------------------------
# 3. Scenario disclaimers
# ----------------------------------------------------------------------
def test_report_must_disclaim_undetected_scenarios(base_state):
    from app.agents.report_agent import ReportAgent

    agent = ReportAgent()
    payload = agent._build_verified_payload(dict(base_state))

    report = """# Revenue Analysis

## Observed Changes
- Region_A: -30.0%

## Measured Drivers
- AOV: 100.0 → 75.0

## Hypotheses to Investigate
- Cross-tabulate P007 revenue by region.

## Known Limitations
- Only revenue is analyzed.
"""
    validation = agent._validate_report(report, payload)
    assert set(validation["missing_scenario_disclaimers"]) == {"A2", "A3", "A4"}, (
        f"expected all three, got {validation['missing_scenario_disclaimers']}"
    )


def test_report_with_disclaimers_passes_scenario_check(base_state):
    from app.agents.report_agent import ReportAgent

    agent = ReportAgent()
    payload = agent._build_verified_payload(dict(base_state))

    report = """# Revenue Analysis

## Observed Changes
- Region_A: -30.0%

## Measured Drivers
- AOV: 100.0 → 75.0

## Hypotheses to Investigate
- P007 margin collapse — not detected by current analysis (requires cost-ratio delta).
- Branch_D1 cancellation spike — not detected by current analysis (requires cancel-rate delta).
- Occasional segment volume decline — not detected by current analysis (requires per-segment quantity delta).

## Known Limitations
- Only revenue is analyzed.
"""
    validation = agent._validate_report(report, payload)
    assert validation["missing_scenario_disclaimers"] == [], (
        f"still flagged: {validation['missing_scenario_disclaimers']}"
    )


# ----------------------------------------------------------------------
# 4. Full run — state["report_validation"] is populated
# ----------------------------------------------------------------------
def test_run_populates_report_validation(base_state):
    from app.agents.report_agent import ReportAgent

    mock_report = """# Revenue Analysis — medium confidence

## Observed Changes
- Region_A: 2026-07 → 2026-08, -30.0%

## Measured Drivers
- AOV: 100.0 → 75.0
- Transaction count: 1000 → 933

## Hypotheses to Investigate
- P007 margin collapse — not detected by current analysis.
- Branch_D1 cancellation spike — not detected by current analysis.
- Occasional volume decline — not detected by current analysis.

## Known Limitations
- contribution_pct is measured within each dimension, not across dimensions.
"""
    with patch("app.llm.client.llm.complete", return_value=mock_report):
        out = ReportAgent().run(dict(base_state))

    assert "report_validation" in out
    assert "verified_payload" in out
    assert out["report_validation"]["headline_present"] is True
    assert out["report_validation"]["passed"] is True
    assert out["report_validation"]["untraceable_numbers"] == []


def test_run_flags_bad_report(base_state):
    from app.agents.report_agent import ReportAgent

    mock_report = "# Revenue Analysis\nRegion_A dropped -99.5%.\n"
    with patch("app.llm.client.llm.complete", return_value=mock_report):
        out = ReportAgent().run(dict(base_state))

    v = out["report_validation"]
    assert not v["passed"]
    joined = " ".join(v["untraceable_numbers"])
    assert "99.5" in joined or "99.50" in joined
    assert set(v["missing_scenario_disclaimers"]) == {"A2", "A3", "A4"}