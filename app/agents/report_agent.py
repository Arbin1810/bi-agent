import json
import re
from datetime import datetime, timezone

from app.agents.base_agent import BaseAgent
from config.prompts import REPORT_PROMPT
from config.settings import settings


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


PLACEHOLDER_KEYS = [
    "baseline_value", "target_value", "change_pct", "absolute_change",
    "total_change_pct", "baseline_total", "target_total",
    "baseline_period", "target_period",
    "aov_change_pct", "txn_change_pct", "cancel_rate_change_pct",
    "baseline_aov", "target_aov",
    "baseline_txn_count", "target_txn_count",
    "baseline_cancel_rate", "target_cancel_rate",
]


def _pct_change(base, target):
    try:
        b = float(base)
        t = float(target)
        return round((t - b) / b * 100, 2) if b else None
    except (TypeError, ValueError):
        return None


class ReportAgent(BaseAgent):
    name = "ReportAgent"

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

        # Deterministic rewrite: replace the entire Observed Changes section
        # with a payload-derived version. Guarantees every number in that
        # section comes from the payload, not from the LLM.
        deterministic_observed = self._rebuild_observed_changes(payload)
        report = self._replace_section(
            report, "## Observed Changes", deterministic_observed
        )

        validation = self._validate_report(report, payload)
        if validation["placeholder_echoes"]:
            self.log(
                f"WARNING: report echoes placeholder keys: "
                f"{validation['placeholder_echoes']}"
            )
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

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out = settings.reports_dir / f"report_{ts}.md"
        out.write_text(report, encoding="utf-8")

        state["report"] = report
        state["report_path"] = str(out)
        state["report_validation"] = validation
        state["verified_payload"] = payload
        return state

    # ----------------------------------------------------------
    # Deterministic Observed Changes section (bypasses the LLM)
    # ----------------------------------------------------------
    def _rebuild_observed_changes(self, payload: dict) -> str:
        """
        Build the Observed Changes section from payload["deltas"] only.
        Every number is a verbatim payload value.

        Iterates all dimensions present in payload["deltas"] and emits the
        top-3 most-negative entity per dimension (per the payload's sort
        order, which is most-negative first).
        """
        lines = ["## Observed Changes"]
        any_emitted = False

        for dim, entities in payload.get("deltas", {}).items():
            # Take top 3 (already sorted most-negative first)
            for e in entities[:3]:
                entity = e.get("entity")
                bv = e.get("baseline_value")
                tv = e.get("target_value")
                pct = e.get("change_pct")
                bp = e.get("baseline_period", "?")
                tp = e.get("target_period", "?")
                if entity is None or bv is None or tv is None or pct is None:
                    continue
                # Label the dimension so a reader knows which grouping
                # each entity belongs to
                lines.append(
                    f"- **{entity}** ({dim}): revenue went from "
                    f"${bv:,.2f} in {bp} to ${tv:,.2f} in {tp}, "
                    f"a {pct:+.2f}% change."
                )
                any_emitted = True

        if not any_emitted:
            lines.append("- No entity-level deltas were computed.")
        return "\n".join(lines)

    @staticmethod
    def _replace_section(report: str, header: str, replacement: str) -> str:
        """
        Replace a whole `## Header` section (up to the next `##` or end of
        report) with new content. Preserves everything before and after.
        """
        if header not in report:
            # Section missing — prepend after the title line (first #) or
            # append at end.
            return report.rstrip() + "\n\n" + replacement + "\n"

        before, rest = report.split(header, 1)
        # Find the next `## ` on its own line
        lines = rest.split("\n")
        next_idx = None
        for i, line in enumerate(lines):
            if i > 0 and line.strip().startswith("## "):
                next_idx = i
                break

        if next_idx is None:
            after = ""
        else:
            after = "\n".join(lines[next_idx:])

        result = before + replacement
        if after:
            result += "\n\n" + after
        return result

    # ----------------------------------------------------------
    # Payload builder
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

        payload["mandatory_disclaimers"] = [
            f"{s['entity']} \u2014 {s['description']}. Not detected by current analysis."
            for s in payload["known_scenarios"]
            if s["detected"] is False
        ]

        payload["known_limitations"] = (
            (state.get("root_causes") or {}).get("known_limitations", [])
        )

        return payload

    def _serialize_payload(self, payload: dict, max_chars: int = 8000) -> str:
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
    # Validation
    # ----------------------------------------------------------
    def _validate_report(self, report: str, payload: dict) -> dict:
        report_lower = report.lower()
        placeholder_echoes = [
            key for key in PLACEHOLDER_KEYS
            if re.search(rf"\b{re.escape(key)}\b", report_lower)
        ]

        report_nums = self._extract_claim_numbers(report)
        payload_nums = self._extract_payload_numbers(payload)
        untraceable = {
            n for n in report_nums
            if not self._matches_any(n, payload_nums)
        }

        untraceable_entities = self._find_untraceable_entities(report, payload)

        missing_disclaimers = []
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
            "placeholder_echoes": sorted(placeholder_echoes),
            "untraceable_numbers": sorted(untraceable),
            "untraceable_entities": sorted(untraceable_entities),
            "missing_scenario_disclaimers": missing_disclaimers,
            "headline_present": headline_present,
            "passed": (
                not placeholder_echoes
                and not untraceable
                and not untraceable_entities
                and not missing_disclaimers
                and headline_present
            ),
        }

    def _find_untraceable_entities(self, report: str, payload: dict) -> list:
        if "## Observed Changes" not in report:
            return []
        observed = report.split("## Observed Changes", 1)[1]
        observed = observed.split("##", 1)[0]

        known_entities = set()
        for dim_entities in payload.get("deltas", {}).values():
            for e in dim_entities:
                ent = e.get("entity")
                if ent:
                    known_entities.add(str(ent).lower())

        untraceable = []
        for line in observed.split("\n"):
            if "$" not in line and not re.search(r"\d+,\d+\.\d+", line):
                continue
            m = re.search(r"\*\*([^*]+)\*\*", line)
            if not m:
                continue
            entity_name = m.group(1).strip().rstrip(":").strip()
            # The rebuilt section wraps entity name in parens for dimension,
            # but the bold text is only the entity name. Strip trailing
            # "(dim)" if present.
            entity_name = re.sub(r"\s*\([^)]*\)\s*$", "", entity_name)
            if not entity_name:
                continue
            if entity_name.lower() in ("total", "overall", "summary"):
                continue
            if entity_name.lower() not in known_entities:
                untraceable.append(entity_name)
        return untraceable

    # ----------------------------------------------------------
    # Number extractors
    # ----------------------------------------------------------
    @staticmethod
    def _extract_claim_numbers(text: str) -> set:
        claims = set()

        def _normalize(s: str) -> str:
            s = (s.replace(",", "")
                 .replace("\u00a0", "")
                 .replace("\u2009", "")
                 .replace("\u202f", ""))
            return s.lstrip("+-")

        for m in re.finditer(r"[+-]?\s*\$?\s*([\d,\.]+)\s*%", text):
            num = _normalize(m.group(1))
            if num and num.replace(".", "").isdigit():
                claims.add(num)

        for m in re.finditer(r"[+-]?\s*\$\s*([\d,\.]+)", text):
            num = _normalize(m.group(1))
            if num and num.replace(".", "").isdigit():
                claims.add(num)

        for m in re.finditer(r"[+-]?\d[\d,]*\.\d+", text):
            claims.add(_normalize(m.group(0)))

        return claims

    @staticmethod
    def _extract_payload_numbers(payload: dict) -> set:
        s = json.dumps(payload, default=str)
        nums = set()
        for m in re.finditer(r"-?\d+\.\d+", s):
            nums.add(m.group(0).lstrip("+-"))
        for m in re.finditer(r":\s*(-?\d+)(?=[,\s}])", s):
            nums.add(m.group(1).lstrip("+-"))
        return nums

    @staticmethod
    def _matches_any(num_str: str, candidate_set: set, tol: float = 0.05) -> bool:
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