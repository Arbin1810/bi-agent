"""
R8 — Full-workflow integration test with mocked LLM.

Runs BIWorkflow.run() end-to-end against the real data/transactions.csv
with the LLM mocked to return canned responses that mirror actual
qwen2.5:7b output. Asserts:

  * All four scenarios detected (4/4)
  * report_validation passes
  * report_confidence == "high"
  * ## Detection Summary has all four scenarios
  * Driver decomposition sums exactly
  * Title says "high confidence"

No live LLM calls. Catches integration regressions the unit tests miss
(agent signature changes, state key mismatches, planner/schema wiring,
chart/agent ordering).
"""

import json
import re
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "transactions.csv"


# ----------------------------------------------------------------------
# Canned LLM responses for each agent
# ----------------------------------------------------------------------
INTENT_RESPONSE = {
    "intent_type": "diagnostic",
    "business_domain": "sales",
    "time_reference": "2026-08",
    "entities": [],
    "requires_multi_step": True,
}

PLANNER_RESPONSE = {
    "steps": [
        {"step": 1, "action": "identify_data", "description": "Load transactions"},
        {"step": 2, "action": "query_data", "description": "Decompose revenue"},
        {"step": 3, "action": "anomaly_detection", "description": "Detect"},
        {"step": 4, "action": "root_cause", "description": "Rank causes"},
    ],
    "expected_outputs": ["revenue delta", "root causes"],
    "success_criteria": "All four scenarios identified",
}

# SQL agent: return a minimal multi-query plan that produces the
# dimensions the pipeline needs.
SQL_RESPONSE = {
    "queries": [
        {"purpose": "Revenue by region for Jul/Aug 2026",
         "sql": (
             "SELECT region, strftime('%Y-%m', date) AS month, "
             "SUM(revenue) AS total FROM transactions "
             "WHERE strftime('%Y-%m', date) IN ('2026-07', '2026-08') "
             "GROUP BY region, month ORDER BY region, month;"
         )},
        {"purpose": "Revenue by product for Jul/Aug 2026",
         "sql": (
             "SELECT product_id, strftime('%Y-%m', date) AS month, "
             "SUM(revenue) AS total FROM transactions "
             "WHERE strftime('%Y-%m', date) IN ('2026-07', '2026-08') "
             "GROUP BY product_id, month ORDER BY product_id, month;"
         )},
        {"purpose": "Revenue by branch for Jul/Aug 2026",
         "sql": (
             "SELECT branch, strftime('%Y-%m', date) AS month, "
             "SUM(revenue) AS total FROM transactions "
             "WHERE strftime('%Y-%m', date) IN ('2026-07', '2026-08') "
             "GROUP BY branch, month ORDER BY branch, month;"
         )},
        {"purpose": "Revenue by customer_segment for Jul/Aug 2026",
         "sql": (
             "SELECT customer_segment, strftime('%Y-%m', date) AS month, "
             "SUM(revenue) AS total FROM transactions "
             "WHERE strftime('%Y-%m', date) IN ('2026-07', '2026-08') "
             "GROUP BY customer_segment, month "
             "ORDER BY customer_segment, month;"
         )},
    ],
    "explanation": "Multi-dimensional revenue decomposition",
}

ANALYSIS_RESPONSE = {
    "overall_summary": {},
    "key_findings": [],
    "quantitative_evidence": [],
    "dimension_breakdown": [],
    "driver_decomposition": {},
    "limitations": [],
}

ANOMALY_RESPONSE = {
    "anomalies": [],
    "method": "mocked",
    "baseline_period": "2026-07",
    "notes": [],
}

ROOT_CAUSE_RESPONSE = {
    "primary_causes": [],
    "secondary_causes": [],
    "hypotheses_to_test": [],
    "known_limitations": [],
}

CRITIC_RESPONSE = {
    "is_complete": True,
    "gaps": [],
    "weak_evidence": [],
    "additional_analysis_needed": [],
}

# The report agent is a freeform call (complete, not complete_json).
# Return a minimal report with all the sections; the pipeline will
# overwrite Observed Changes / Detection Summary / Known Limitations
# deterministically.
REPORT_MARKDOWN = """# Revenue Analysis — high confidence

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


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def mock_responses():
    """Configured patchers for llm.complete_json and llm.complete."""
    from app.llm.client import llm

    original_complete_json = llm.complete_json
    original_complete = llm.complete

    def _complete_json_stub(prompt, system="", temperature=0.1):
        p = prompt.lower()
        if "intent classification" in p:
            return dict(INTENT_RESPONSE)
        if "planning agent" in p:
            return dict(PLANNER_RESPONSE)
        if "expert sql analyst" in p:
            return dict(SQL_RESPONSE)
        if "senior business analyst" in p:
            return dict(ANALYSIS_RESPONSE)
        if "anomaly detection expert" in p:
            return dict(ANOMALY_RESPONSE)
        if "root cause analysis expert" in p:
            return dict(ROOT_CAUSE_RESPONSE)
        if "critic agent" in p:
            return dict(CRITIC_RESPONSE)
        return {}

    def _complete_stub(prompt, system="", temperature=0.1, max_tokens=4096):
        return REPORT_MARKDOWN

    llm.complete_json = _complete_json_stub
    llm.complete = _complete_stub
    try:
        yield
    finally:
        llm.complete_json = original_complete_json
        llm.complete = original_complete


# ----------------------------------------------------------------------
# Skip if the dataset isn't present
# ----------------------------------------------------------------------
@pytest.mark.skipif(
    not CSV_PATH.exists(),
    reason="data/transactions.csv not found; run scripts/generate_dataset.py first",
)
def test_full_workflow_end_to_end(mock_responses):
    """
    Full pipeline run. Asserts all detection, validation, and confidence
    outcomes that D2–D4 established.
    """
    from app.orchestration.workflow import workflow

    df = pd.read_csv(CSV_PATH, parse_dates=["date"])
    state = workflow.run(
        "Why did revenue change in August 2026?",
        dataframe=df,
    )

    # --- no error ---
    assert state.get("error") is None, f"workflow error: {state.get('error')}"

    # --- scenarios detected ---
    scenario_detection = state.get("scenario_detection") or []
    assert len(scenario_detection) == 4, (
        f"expected 4 scenarios evaluated, got {len(scenario_detection)}"
    )
    by_id = {d["id"]: d for d in scenario_detection}
    for sid in ("A1", "A2", "A3", "A4"):
        assert by_id[sid]["detected"] is True, (
            f"{sid} not detected: {by_id[sid].get('evidence')}"
        )

    # --- report validation passes ---
    validation = state.get("report_validation") or {}
    assert validation.get("passed") is True, (
        f"validation did not pass: {validation}"
    )
    assert validation.get("untraceable_numbers") == []
    assert validation.get("untraceable_entities") == []
    assert validation.get("missing_scenario_disclaimers") == []
    assert validation.get("headline_present") is True

    # --- confidence is high (all detected + validation passed) ---
    assert state.get("report_confidence") == "high"

    # --- driver decomposition sums exactly ---
    os_ = (state.get("analysis") or {}).get("overall_summary") or {}
    dd = os_.get("driver_decomposition")
    assert dd is not None, "driver_decomposition missing from overall_summary"
    assert abs(dd["sum_check"] - dd["total_change"]) < 0.05, (
        f"sum_check {dd['sum_check']} != total_change {dd['total_change']}"
    )

    # --- KPI definition present ---
    assert os_.get("revenue_definition") in ("completed_revenue", "gross_revenue")

    # --- report has all six sections ---
    report = state.get("report") or ""
    for section in (
        "## Observed Changes",
        "## Measured Drivers",
        "## Driver Decomposition",
        "## Detection Summary",
        "## Hypotheses to Investigate",
        "## Known Limitations",
    ):
        assert section in report, f"report missing section: {section}"

    # --- Detection Summary has all four scenarios with checkmarks ---
    det_section = report.split("## Detection Summary", 1)[1].split("##", 1)[0]
    for sid in ("A1", "A2", "A3", "A4"):
        assert sid in det_section, f"Detection Summary missing {sid}"
    assert "\u2713" in det_section  # ✓

    # --- title says high confidence ---
    assert re.search(r"# Revenue Analysis\s+[—-]\s+high\s+confidence", report), (
        f"title does not say 'high confidence': {report.split(chr(10), 1)[0]}"
    )


@pytest.mark.skipif(
    not CSV_PATH.exists(),
    reason="data/transactions.csv not found; run scripts/generate_dataset.py first",
)
def test_workflow_populates_expected_state_keys(mock_responses):
    """
    Verifies the workflow populates every state key the report agent
    and downstream consumers read. Catches state-key regressions.
    """
    from app.orchestration.workflow import workflow

    df = pd.read_csv(CSV_PATH, parse_dates=["date"])
    state = workflow.run("Why did revenue change in August 2026?", dataframe=df)

    for key in (
        "analysis",
        "anomalies",
        "root_causes",
        "scenario_detection",
        "metric_deltas",
        "dimension_deltas",
        "sql_results",
        "report",
        "report_path",
        "report_validation",
        "verified_payload",
        "report_confidence",
    ):
        assert key in state, f"workflow did not populate state[{key!r}]"