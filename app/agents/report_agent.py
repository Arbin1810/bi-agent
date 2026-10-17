import json
import re
from datetime import datetime, timezone

from app.agents.base_agent import BaseAgent
from config.prompts import REPORT_PROMPT
from config.settings import settings


# Known scenarios from the generator. Used to force the report to name
# any scenario that wasn't detected, rather than silently omitting it.
KNOWN_SCENARIOS = [
    {"id": "A1", "dimension": "region", "entity": "Region_A",
     "description": "Region_A revenue drop"},
    {"id": "A2", "dimension": "product_id", "entity": "P007",
     "description": "P007 margin collapse (cost/revenue → 0.92)"},
    {"id": "A3", "dimension": "branch", "entity": "Branch_D1",
     "description": "Branch_D1 cancellation spike"},
    {"id": "A4", "dimension": "customer_segment", "entity": "Occasional",
     "description": "Occasional segment volume decline"},
]


class ReportAgent(BaseAgent):
    name = "ReportAgent"

    # ----------------------------------------------------------
    # Public entry point
    # ----------------------------------------------------------
    def run(self, state: dict) -> dict:
        self.log("Writing report")

        payload = self._build_verified_payload(state)
        prompt = REPORT_PROMPT.format(
            question=state["question"],
            verified_payload=self._serialize_payload(payload),
            charts=[c["path"] for c in state.get("charts", [])],
            confidence=state.get("critique", {}).get("confidence", "medium"),
        )

        report = self.llm.complete(prompt, temperature=0.3)

        # ----------------------------------------------------------
        # Post-generation validation: every number must be traceable.
        # ----------------------------------------------------------
        validation = self._validate_report(report, payload)
        if validation["untraceable_numbers"]:
            self.log(
                f"WARNING: report contains untraceable numbers: "
                f"{sorted(validation['untraceable_numbers'])}"
            )
        if validation["missing_scenario_disclaimers"]:
            self.log(
                f"WARNING: report omits undetected scenarios: "
                f"{validation['missing_scenario_disclaimers']}"
            )
        if validation["passed"]:
            self.log("Report traceability: PASSED")

        # Persist
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out = settings.reports_dir / f"report_{ts}.md"
        out.write_text(report, encoding="utf-8")

        state["report"] = report
        state["report_path"] = str(out)
        state["report_validation"] = validation
        state["verified_payload"] = payload
        return state

    # ----------------------------------------------------------
    # Step 4.1 — Build the structured, deterministic payload.
    # ----------------------------------------------------------
    def _build_verified_payload(self, state: dict) -> dict:
        """Every number the report may cite must live in here."""
        payload = {
            "headline": None,
            "deltas": {},
            "root_causes": [],
            "anomalies": [],
            "known_scenarios": [],
            "known_limitations": [],
            "confidence": state.get("critique", {}).get("confidence", "medium"),
        }

        # 1. Headline
        os_ = (state.get("analysis") or {}).get("overall_summary") or {}
        if os_:
            payload["headline"] = {
                "baseline_period": os_.get("baseline_period"),
                "target_period": os_.get("target_period"),
                "baseline_total": os_.get("baseline_total"),
                "target_total": os_.get("target_total"),
                "absolute_change": os_.get("absolute_change"),
                "total_change_pct": os_.get("total_change_pct"),
                "baseline_aov": os_.get("baseline_aov"),
                "target_aov": os_.get("target_aov"),
                "baseline_txn_count": os_.get("baseline_txn_count"),
                "target_txn_count": os_.get("target_txn_count"),
                "baseline_cancel_rate": os_.get("baseline_cancel_rate"),
                "target_cancel_rate": os_.get("target_cancel_rate"),
            }

        # 2. Deltas — top 5 most-negative per dimension
        for dim, entities in (state.get("dimension_deltas") or {}).items():
            payload["deltas"][dim] = [
                {
                    "entity": d.get("entity"),
                    "baseline_period": d.get("baseline_period"),
                    "target_period": d.get("target_period"),
                    "baseline_value": d.get("baseline_value"),
                    "target_value": d.get("target_value"),
                    "absolute_change": d.get("absolute_change"),
                    "change_pct": d.get("change_pct"),
                }
                for d in entities[:5]
            ]

        # 3. Root causes
        for c in (state.get("root_causes") or {}).get("primary_causes", [])[:5]:
            payload["root_causes"].append({
                "dimension": c.get("dimension"),
                "value": c.get("value"),
                "change_pct": c.get("change_pct"),
                "absolute_change": c.get("absolute_change"),
                "contribution_pct": c.get("contribution_pct"),
                "contribution_basis": c.get("contribution_basis"),
                "confidence": c.get("confidence"),
            })

        # 4. Anomalies — high/medium only
        raw_anoms = (state.get("anomalies") or {}).get("anomalies", [])
        payload["anomalies"] = [
            {
                "entity": a.get("entity"),
                "period": a.get("period"),
                "value": a.get("value"),
                "expected": a.get("expected"),
                "deviation_pct": a.get("deviation_pct"),
                "severity": a.get("severity"),
                "anomaly_type": a.get("anomaly_type"),
            }
            for a in raw_anoms
            if a.get("severity") in ("high", "medium")
        ]

        # 5. Known scenario check
        computed = (state.get("anomalies") or {}).get("computed_statistics") or {}
        computed_check = {
            c.get("scenario_id"): c.get("detected_by_cross_entity_zscore")
            for c in computed.get("known_scenario_check", [])
        }
        payload["known_scenarios"] = [
            {
                "id": s["id"],
                "dimension": s["dimension"],
                "entity": s["entity"],
                "description": s["description"],
                "detected": computed_check.get(s["id"]),
            }
            for s in KNOWN_SCENARIOS
        ]

        # 6. Known limitations
        payload["known_limitations"] = (
            (state.get("root_causes") or {}).get("known_limitations", [])
        )

        return payload

    # ----------------------------------------------------------
    # Serialization with a size cap that NEVER truncates mid-token.
    # ----------------------------------------------------------
    def _serialize_payload(self, payload: dict, max_chars: int = 6000) -> str:
        s = json.dumps(payload, default=str)
        if len(s) <= max_chars:
            return s

        for field in ("anomalies", "known_limitations", "root_causes"):
            trimmed = dict(payload)
            trimmed.pop(field, None)
            s = json.dumps(trimmed, default=str)
            if len(s) <= max_chars:
                return s

        minimal = {
            "headline": payload.get("headline"),
            "deltas": payload.get("deltas"),
        }
        return json.dumps(minimal, default=str)

    # ----------------------------------------------------------
    # Step 4.1 — Post-generation traceability check.
    # ----------------------------------------------------------
    def _validate_report(self, report: str, payload: dict) -> dict:
        report_nums = self._extract_claim_numbers(report)
        payload_nums = self._extract_payload_numbers(payload)
        untraceable = report_nums - payload_nums

        missing_disclaimers = []
        report_lower = report.lower()
        for s in payload.get("known_scenarios", []):
            if s.get("detected") is False:
                entity_lower = str(s["entity"]).lower()
                has_entity = entity_lower in report_lower
                has_phrase = ("not detected" in report_lower
                              or "not flagged" in report_lower
                              or "missed by" in report_lower)
                if not (has_entity and has_phrase):
                    missing_disclaimers.append(s["id"])

        headline_pct = (payload.get("headline") or {}).get("total_change_pct")
        headline_present = (
            headline_pct is not None
            and self._number_in_report(headline_pct, report)
        )

        return {
            "untraceable_numbers": sorted(untraceable),
            "missing_scenario_disclaimers": missing_disclaimers,
            "headline_present": headline_present,
            "passed": (
                not untraceable
                and not missing_disclaimers
                and headline_present
            ),
        }

    @staticmethod
    def _extract_claim_numbers(text: str) -> set:
        """
        Extract numbers that look like real numerical claims from markdown.

        A claim number is a number with one of:
          * a decimal point     → 3.5, -30.00, 999,999.99
          * a percent sign      → 30%   (captured as "30")
          * a currency prefix   → $70   (captured as "70")

        We ignore bare integers like "1" in "1 analysis" or "2" in
        "2 dimensions" — they are structural, not claims.

        Thousands separators (,, non-breaking space, thin space, narrow
        no-break space) are stripped so "$999,999.99" is extracted as
        "999999.99", not split into "999" and "999.99".
        """
        claims = set()

        def _normalize(s: str) -> str:
            return (
                s.replace(",", "")
                 .replace("\u00a0", "")
                 .replace("\u2009", "")
                 .replace("\u202f", "")
            )

        # Percent claims: 30%, -12.5%, +2.4%  (sign optional, % required)
        for m in re.finditer(r"([+-]?)\s*\$?\s*([\d,\.]+)\s*%", text):
            num = _normalize(m.group(2))
            if num.replace(".", "").isdigit():
                claims.add(f"{m.group(1)}{num}")

        # Currency claims: $999,999.99, -$70, +$1,000
        for m in re.finditer(r"([+-]?)\s*\$\s*([\d,\.]+)", text):
            num = _normalize(m.group(2))
            if num.replace(".", "").isdigit():
                claims.add(f"{m.group(1)}{num}")

        # Bare decimals: 3.5, -30.00, 999,999.99 (must contain a '.')
        for m in re.finditer(r"([+-]?\d[\d,]*\.\d+)", text):
            claims.add(_normalize(m.group(1)))

        return claims

    @staticmethod
    def _extract_payload_numbers(payload: dict) -> set:
        """Extract the same claim-number format from the payload."""
        s = json.dumps(payload, default=str)
        nums = set()
        for m in re.finditer(r"-?\d+\.\d+", s):
            nums.add(m.group(0))
        for m in re.finditer(r":\s*(-?\d+)(?=[,\s}])", s):
            nums.add(m.group(1))
        return nums

    @staticmethod
    def _number_in_report(value, report: str) -> bool:
        try:
            f = float(value)
        except (TypeError, ValueError):
            return False
        candidates = {
            f"{f:.1f}", f"{f:.2f}", f"{abs(f):.1f}", f"{abs(f):.2f}",
            f"{f:g}", f"{abs(f):g}",
        }
        return any(c in report for c in candidates)