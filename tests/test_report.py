"""
Phase 4 + Delivery 4 tests — report traceability + confidence rules.

Run:
    pytest tests/test_report.py -v
"""

from unittest.mock import patch

import pytest


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
                "revenue_definition": "completed_revenue",
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
        "critique": {"confidence": "medium"},  # legacy — should be ignored
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


def test_payload_known_scenarios_carry_detection_status(base_state):
    from app.agents.report_agent import ReportAgent
    payload = ReportAgent()._build_verified_payload(dict(base_state))
    by_id = {s["id"]: s for s in payload["known_scenarios"]}
    assert by_id["A1"]["detected"] is True
    assert by_id["A2"]["detected"] is False
    assert by_id["A3"]["detected"] is False
    assert by_id["A4"]["detected"] is False


# ----------------------------------------------------------------------
# 2. Traceability
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
- Cross-tabulate P007 revenue by region.
- Branch_D1 cancellation spike — not detected by current analysis.

## Known Limitations
- contribution_pct is measured within each dimension, not across dimensions.
"""
    validation = agent._validate_report(report, payload)
    assert validation["untraceable_numbers"] == []


def test_traceability_catches_hallucinated_number(base_state):
    from app.agents.report_agent import ReportAgent
    agent = ReportAgent()
    payload = agent._build_verified_payload(dict(base_state))
    report = """# Revenue Analysis

## Observed Changes
- Region_A: -30.0% ($999,999.99 loss)
"""
    validation = agent._validate_report(report, payload)
    assert validation["untraceable_numbers"]
    joined = " ".join(validation["untraceable_numbers"]).replace(",", "")
    assert "999999.99" in joined or "999.99" in joined


def test_traceability_ignores_structural_numbers(base_state):
    from app.agents.report_agent import ReportAgent
    agent = ReportAgent()
    payload = agent._build_verified_payload(dict(base_state))
    report = """# Revenue Analysis

This report covers 1 analysis, 2 dimensions, and 3 charts.

## Observed Changes
- Region_A: -30.0%
"""
    validation = agent._validate_report(report, payload)
    assert validation["untraceable_numbers"] == []


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

## Hypotheses to Investigate
- Cross-tabulate P007 revenue by region.

## Known Limitations
- Only revenue is analyzed.
"""
    validation = agent._validate_report(report, payload)
    assert set(validation["missing_scenario_disclaimers"]) == {"A2", "A3", "A4"}


def test_report_with_disclaimers_passes_scenario_check(base_state):
    from app.agents.report_agent import ReportAgent
    agent = ReportAgent()
    payload = agent._build_verified_payload(dict(base_state))
    report = """# Revenue Analysis

## Observed Changes
- Region_A: -30.0%

## Hypotheses to Investigate
- P007 margin collapse — not detected by current analysis.
- Branch_D1 cancellation spike — not detected by current analysis.
- Occasional segment volume decline — not detected by current analysis.

## Known Limitations
- Only revenue is analyzed.
"""
    validation = agent._validate_report(report, payload)
    assert validation["missing_scenario_disclaimers"] == []


# ----------------------------------------------------------------------
# 4. Full run — payload, validation, confidence all populated
# ----------------------------------------------------------------------
def test_run_populates_report_validation(base_state):
    """Verifies report_validation + report_confidence are populated.
    The mock report intentionally omits a ## Driver Decomposition section
    with fabricated numbers — those are tested separately in
    test_driver_decomposition_sums_exactly and by the live pipeline.
    """
    from app.agents.report_agent import ReportAgent
    mock_report = """# Revenue Analysis — high confidence

## Observed Changes
- Region_A: 2026-07 → 2026-08, -30.0%

## Measured Drivers
- Revenue (completed_revenue) changed by -30.00% (-30000.00 dollars, from $100000.00 to $70000.00).
- AOV changed by -25.00%
- Transaction count changed by -6.70%
- Cancellation rate changed by 260.00%

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
    assert "report_confidence" in out
    assert out["report_validation"]["headline_present"] is True
    assert out["report_validation"]["passed"] is True


def test_run_flags_bad_report(base_state):
    from app.agents.report_agent import ReportAgent
    mock_report = "# Revenue Analysis\nRegion_A dropped -99.5%.\n"
    with patch("app.llm.client.llm.complete", return_value=mock_report):
        out = ReportAgent().run(dict(base_state))
    v = out["report_validation"]
    assert not v["passed"]
    assert any("99.5" in n or "99.50" in n for n in v["untraceable_numbers"])


# ----------------------------------------------------------------------
# 5. Delivery 4 — confidence derived from validation + detection
# ----------------------------------------------------------------------
def test_confidence_high_when_all_detected_and_validation_passed(base_state):
    """Fixture detects only A1 (1/4), so validation passes but this is NOT high."""
    from app.agents.report_agent import ReportAgent
    payload = ReportAgent()._build_verified_payload(dict(base_state))
    validation = {"passed": True}
    conf = ReportAgent()._compute_confidence(dict(base_state), payload, validation=validation)
    assert conf == "medium"


def test_confidence_high_when_all_scenarios_detected():
    """Explicit test: all four detected + validation passes → high."""
    from app.agents.report_agent import ReportAgent
    payload = {
        "known_scenarios": [
            {"id": "A1", "detected": True},
            {"id": "A2", "detected": True},
            {"id": "A3", "detected": True},
            {"id": "A4", "detected": True},
        ],
    }
    conf = ReportAgent()._compute_confidence({}, payload, validation={"passed": True})
    assert conf == "high"


def test_confidence_low_when_no_detections():
    from app.agents.report_agent import ReportAgent
    payload = {
        "known_scenarios": [
            {"id": "A1", "detected": False},
            {"id": "A2", "detected": False},
            {"id": "A3", "detected": False},
            {"id": "A4", "detected": False},
        ],
    }
    conf = ReportAgent()._compute_confidence({}, payload, validation={"passed": True})
    assert conf == "low"


def test_confidence_low_when_validation_failed():
    from app.agents.report_agent import ReportAgent
    payload = {
        "known_scenarios": [
            {"id": "A1", "detected": True},
            {"id": "A2", "detected": True},
            {"id": "A3", "detected": True},
            {"id": "A4", "detected": True},
        ],
    }
    conf = ReportAgent()._compute_confidence({}, payload, validation={"passed": False})
    assert conf == "low"


def test_detection_summary_rebuilt_from_payload(base_state):
    """The rebuilt Detection Summary lists all four scenarios with markers."""
    from app.agents.report_agent import ReportAgent
    payload = ReportAgent()._build_verified_payload(dict(base_state))
    section = ReportAgent()._rebuild_detection_summary(payload)

    assert "## Detection Summary" in section
    assert "A1" in section
    assert "A2" in section
    assert "A3" in section
    assert "A4" in section
    assert "\u2713" in section  # checkmark for detected (A1)
    assert "\u2717" in section  # cross for undetected (A2/A3/A4)


def test_known_limitations_rebuilt_from_payload(base_state):
    """The rebuilt Known Limitations contains the methodological notes
    even though the LLM wrote disclaimer text."""
    from app.agents.report_agent import ReportAgent
    payload = ReportAgent()._build_verified_payload(dict(base_state))
    section = ReportAgent()._rebuild_known_limitations(payload)

    assert "## Known Limitations" in section
    assert "contribution_pct is measured within each dimension" in section
    assert "anomaly detector z-scores" in section


def test_title_confidence_rewritten_by_postprocessing(base_state):
    """Even if the LLM writes 'high confidence', the post-process rewrites
    it based on validation + detection (which for this fixture is medium)."""
    from app.agents.report_agent import ReportAgent
    mock_report = """# Revenue Analysis — high confidence

## Observed Changes
- placeholder

## Measured Drivers
- placeholder

## Driver Decomposition
- placeholder

## Detection Summary
- placeholder

## Hypotheses to Investigate
- placeholder

## Known Limitations
- placeholder
"""
    with patch("app.llm.client.llm.complete", return_value=mock_report):
        out = ReportAgent().run(dict(base_state))

    assert "medium confidence" in out["report"]
    assert "high confidence" not in out["report"]