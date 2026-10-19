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
     "description": "P007 margin collapse (cost/revenue -> 0.92)"},
    {"id": "A3", "dimension": "branch", "entity": "Branch_D1",
     "description": "Branch_D1 cancellation spike"},
    {"id": "A4", "dimension": "customer_segment", "entity": "Occasional",
     "description": "Occasional segment volume decline"},
]


def _pct_change(base, target):
    """Percentage change from base to target, rounded to 2 decimals."""
    try:
        b = float(base)
        t = float(target)
        return round((t - b) / b * 100, 2) if b else None
    except (TypeError, ValueError):
        return None


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
            mandatory_disclaimers=self._format_disclaimers(payload),
            charts=[c["path"] for c in state.get("charts", [])],
            confidence=state.get("critique", {}).get("confidence", "medium"),
        )

        report = self.llm.complete(prompt, temperature=0.3)

        # ----------------------------------------------------------
        # Post-generation validation: every number must be traceable
        # AND every named entity must exist in the payload.
        # ----------------------------------------------------------
        validation = self._validate_report(report, payload)
        if validation["untraceable_numbers"]:
            self.log(
                f"WARNING: report contains untraceable numbers: "
                f"{validation['untraceable_numbers']}"
            )
        if validation["untraceable_entities"]:
            self.log(
                f"WARNING: report names entities not in payload: "
                f"{validation['untraceable_entities']}"
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
    # Build the structured, deterministic payload.
    # ----------------------------------------------------------
    def _build_verified_payload(self, state: dict) -> dict:
        payload = {
            "headline": None,
            "deltas": {},
            "root_causes": [],
            "anomalies": [],
            "known_scenarios": [],
            "mandatory_disclaimers": [],
            "known_limitations": [],
            "confidence": state.get("critique", {}).get("confidence", "medium"),
        }

        # 1. Headline — with derived change-pct fields
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
                "aov_change_pct": _pct_change(
                    os_.get("baseline_aov"), os_.get("target_aov")
                ),
                "baseline_txn_count": os_.get("baseline_txn_count"),
                "target_txn_count": os_.get("target_txn_count"),
                "txn_change_pct": _pct_change(
                    os_.get("baseline_txn_count"), os_.get("target_txn_count")
                ),
                "baseline_cancel_rate": os_.get("baseline_cancel_rate"),
                "target_cancel_rate": os_.get("target_cancel_rate"),
                "cancel_rate_change_pct": _pct_change(
                    os_.get("baseline_cancel_rate"), os_.get("target_cancel_rate")
                ),
            }

        # 2. Deltas — top-5 most-negative PLUS bottom-3 most-positive per
        # dimension, so the LLM has real values for both over- and
        # under-performers. Without the bottom-3, an entity like Region_G
        # (which actually GREW 30%) wouldn't be in the payload, and the
        # LLM would invent values for it if it mentioned it.
        for dim, entities in (state.get("dimension_deltas") or {}).items():
            if not entities:
                continue
            sorted_ents = sorted(entities, key=lambda d: d["absolute_change"])
            selected = sorted_ents[:5] + sorted_ents[-3:]
            seen = set()
            deduped = []
            for d in selected:
                key = d.get("entity")
                if key in seen:
                    continue
                seen.add(key)
                deduped.append({
                    "entity": d.get("entity"),
                    "baseline_period": d.get("baseline_period"),
                    "target_period": d.get("target_period"),
                    "baseline_value": d.get("baseline_value"),
                    "target_value": d.get("target_value"),
                    "absolute_change": d.get("absolute_change"),
                    "change_pct": d.get("change_pct"),
                })
            payload["deltas"][dim] = deduped

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

        # 5b. Pre-built mandatory disclaimers
        payload["mandatory_disclaimers"] = [
            f"{s['entity']} \u2014 {s['description']}. Not detected by current analysis."
            for s in payload["known_scenarios"]
            if s["detected"] is False
        ]

        # 6. Known limitations
        payload["known_limitations"] = (
            (state.get("root_causes") or {}).get("known_limitations", [])
        )

        return payload

    # ----------------------------------------------------------
    # Serialization with a size cap that NEVER truncates mid-token.
    # ----------------------------------------------------------
    def _serialize_payload(self, payload: dict, max_chars: int = 8000) -> str:
        s = json.dumps(payload, default=str)
        if len(s) <= max_chars:
            return s

        # Drop optional sections in priority order, keeping the derived
        # headline fields (aov/txn/cancel change_pct) and deltas since
        # those are the ones the report needs to cite.
        for field in ("anomalies", "known_limitations", "root_causes"):
            trimmed = dict(payload)
            trimmed.pop(field, None)
            s = json.dumps(trimmed, default=str)
            if len(s) <= max_chars:
                return s

        # Last resort: keep only headline + mandatory_disclaimers.
        minimal = {
            "headline": payload.get("headline"),
            "mandatory_disclaimers": payload.get("mandatory_disclaimers"),
        }
        return json.dumps(minimal, default=str)

    @staticmethod
    def _format_disclaimers(payload: dict) -> str:
        bullets = payload.get("mandatory_disclaimers") or []
        if not bullets:
            return "(none \u2014 all known scenarios were detected)"
        return "\n".join(f"- {b}" for b in bullets)

    # ----------------------------------------------------------
    # Post-generation traceability check.
    # ----------------------------------------------------------
    def _validate_report(self, report: str, payload: dict) -> dict:
        report_nums = self._extract_claim_numbers(report)
        payload_nums = self._extract_payload_numbers(payload)
        untraceable = {
            n for n in report_nums
            if not self._matches_any(n, payload_nums)
        }

        # Semantic check: any entity named in an Observed Changes bullet
        # that carries dollar amounts must appear in payload["deltas"].
        # This catches cases like "Region_G: from $642,603.22 to $499,342.92"
        # where Region_G isn't in the payload at all and the values are
        # invented, even if the numbers happen to be fabricated rather
        # than invented (both are hallucinations).
        untraceable_entities = self._find_untraceable_entities(report, payload)

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
            "untraceable_entities": sorted(untraceable_entities),
            "missing_scenario_disclaimers": missing_disclaimers,
            "headline_present": headline_present,
            "passed": (
                not untraceable
                and not untraceable_entities
                and not missing_disclaimers
                and headline_present
            ),
        }

    def _find_untraceable_entities(self, report: str, payload: dict) -> list:
        """
        Extract entity names from Observed Changes bullets that carry
        dollar amounts, and return any that don't appear in payload["deltas"].

        A bullet like:
            "**Region_G:** from 642,603.22 (2026-07) to 499,342.92 (2026-08), -26.09% change"
        names Region_G and cites two dollar values. If Region_G isn't in
        any dimension of payload["deltas"], the values are hallucinated.
        """
        if "## Observed Changes" not in report:
            return []
        # Isolate the Observed Changes section
        observed = report.split("## Observed Changes", 1)[1]
        observed = observed.split("##", 1)[0]

        # Collect every entity name that appears in payload["deltas"]
        known_entities = set()
        for dim_entities in payload.get("deltas", {}).values():
            for e in dim_entities:
                ent = e.get("entity")
                if ent:
                    known_entities.add(str(ent).lower())

        untraceable = []
        for line in observed.split("\n"):
            # Only consider bullet lines with a bold entity and a $
            if "$" not in line and not re.search(r"\d+,\d+\.\d+", line):
                continue
            m = re.search(r"\*\*([^*]+)\*\*", line)
            if not m:
                continue
            entity_name = m.group(1).strip().rstrip(":").strip()
            if not entity_name:
                continue
            # Skip bullet headers that aren't entity names (e.g. "Total")
            if entity_name.lower() in ("total", "overall", "summary"):
                continue
            if entity_name.lower() not in known_entities:
                untraceable.append(entity_name)
        return untraceable

    # ----------------------------------------------------------
    # Number extractors — both return ABSOLUTE VALUES.
    # ----------------------------------------------------------
    @staticmethod
    def _extract_claim_numbers(text: str) -> set:
        """
        Extract numbers from markdown as ABSOLUTE VALUES.

        Sign is inferred from context words ("dropped", "fell", "up"),
        so -1.56 in the payload matches 1.56 in the report.
        """
        claims = set()

        def _normalize(s: str) -> str:
            s = (s.replace(",", "")
                 .replace("\u00a0", "")
                 .replace("\u2009", "")
                 .replace("\u202f", ""))
            return s.lstrip("+-")

        # Percent claims
        for m in re.finditer(r"[+-]?\s*\$?\s*([\d,\.]+)\s*%", text):
            num = _normalize(m.group(1))
            if num and num.replace(".", "").isdigit():
                claims.add(num)

        # Currency claims
        for m in re.finditer(r"[+-]?\s*\$\s*([\d,\.]+)", text):
            num = _normalize(m.group(1))
            if num and num.replace(".", "").isdigit():
                claims.add(num)

        # Bare decimals
        for m in re.finditer(r"[+-]?\d[\d,]*\.\d+", text):
            claims.add(_normalize(m.group(0)))

        return claims

    @staticmethod
    def _extract_payload_numbers(payload: dict) -> set:
        """
        Extract numbers from the payload in the same absolute-value form
        as _extract_claim_numbers.
        """
        s = json.dumps(payload, default=str)
        nums = set()
        for m in re.finditer(r"-?\d+\.\d+", s):
            nums.add(m.group(0).lstrip("+-"))
        for m in re.finditer(r":\s*(-?\d+)(?=[,\s}])", s):
            nums.add(m.group(1).lstrip("+-"))
        return nums

    @staticmethod
    def _matches_any(num_str: str, candidate_set: set, tol: float = 0.05) -> bool:
        """
        True if num_str matches any candidate numerically, within tolerance.

        Also tries candidate * 100 and candidate / 100 so that a payload
        value stored as a decimal (0.036) matches a report value written
        as a percent (3.60).
        """
        try:
            target = float(num_str)
        except ValueError:
            return num_str in candidate_set
        for c in candidate_set:
            try:
                val = float(c)
            except ValueError:
                if c == num_str:
                    return True
                continue
            for scaled in (val, val * 100, val / 100):
                if abs(scaled - target) <= tol:
                    return True
        return False

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