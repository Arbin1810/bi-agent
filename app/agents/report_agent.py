import json
import re
from datetime import UTC, datetime

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
    "revenue_definition", "driver_decomposition",
    "volume_effect", "price_effect", "interaction_effect",
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

        # R5: confidence derived from validation + detection, not from critic.
        confidence_pre = self._compute_confidence(state, payload)

        prompt = REPORT_PROMPT.format(
            question=state["question"],
            verified_payload=self._serialize_payload(payload),
            mandatory_disclaimers=self._format_disclaimers(payload),
            charts=[c["path"] for c in state.get("charts", [])],
            confidence=confidence_pre,
        )

        report = self.llm.complete(prompt, temperature=0.3)

        # R9: deterministic rewrites of three sections (LLM cannot fabricate)
        report = self._replace_section(
            report, "## Observed Changes", self._rebuild_observed_changes(payload)
        )
        report = self._replace_section(
            report, "## Detection Summary", self._rebuild_detection_summary(payload)
        )
        report = self._replace_section(
            report, "## Known Limitations", self._rebuild_known_limitations(payload)
        )

        # R5: derive confidence from the actual report+state and force
        # the title to say the right thing (even if the LLM wrote a
        # different value).
        validation = self._validate_report(report, payload)
        confidence = self._compute_confidence(state, payload, validation=validation)
        report = re.sub(
            r"# Revenue Analysis\s+[—-]\s+\w+\s+confidence",
            f"# Revenue Analysis \u2014 {confidence} confidence",
            report,
            count=1,
        )

        if validation["placeholder_echoes"]:
            self.log(f"WARNING: report echoes placeholder keys: {validation['placeholder_echoes']}")
        if validation["untraceable_numbers"]:
            self.log(f"WARNING: report contains untraceable numbers: {validation['untraceable_numbers']}")
        if validation["untraceable_entities"]:
            self.log(f"WARNING: report names entities not in payload: {validation['untraceable_entities']}")
        if validation["missing_scenario_disclaimers"]:
            self.log(f"WARNING: report omits undetected scenarios: {validation['missing_scenario_disclaimers']}")
        if validation["passed"]:
            self.log(f"Report traceability: PASSED (confidence: {confidence})")

        ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        out = settings.reports_dir / f"report_{ts}.md"
        out.write_text(report, encoding="utf-8")

        state["report"] = report
        state["report_path"] = str(out)
        state["report_validation"] = validation
        state["verified_payload"] = payload
        state["report_confidence"] = confidence
        return state

    def _compute_confidence(self, state: dict, payload: dict, validation: dict | None = None) -> str:
        if validation is None:
            validation = state.get("report_validation") or {}

        scenarios = payload.get("known_scenarios", [])
        n_total = len(scenarios)
        n_detected = sum(1 for s in scenarios if s.get("detected"))
        validation_passed = bool(validation.get("passed", False))

        if n_total == 0:
            return "high" if validation_passed else "low"

        if not validation_passed:
            return "low"
        if n_detected == n_total:
            return "high"
        if n_detected == 0:
            return "low"
        return "medium"

    def _rebuild_observed_changes(self, payload: dict) -> str:
        lines = ["## Observed Changes"]
        any_emitted = False

        for dim, entities in payload.get("deltas", {}).items():
            for e in entities[:3]:
                entity = e.get("entity")
                bv = e.get("baseline_value")
                tv = e.get("target_value")
                pct = e.get("change_pct")
                bp = e.get("baseline_period", "?")
                tp = e.get("target_period", "?")
                if entity is None or bv is None or tv is None or pct is None:
                    continue
                lines.append(
                    f"- **{entity}** ({dim}): revenue went from "
                    f"${bv:,.2f} in {bp} to ${tv:,.2f} in {tp}, "
                    f"a {pct:+.2f}% change."
                )
                any_emitted = True

        if not any_emitted:
            lines.append("- No entity-level deltas were computed.")
        return "\n".join(lines)

    def _rebuild_detection_summary(self, payload: dict) -> str:
        lines = ["## Detection Summary"]
        scenarios = payload.get("known_scenarios", [])
        if not scenarios:
            lines.append("- No scenarios evaluated.")
            return "\n".join(lines)

        for s in scenarios:
            detected = s.get("detected")
            marker = "\u2713" if detected else "\u2717"
            sid = s.get("id", "?")
            entity = s.get("entity", "?")
            description = s.get("description", "")
            metric = s.get("metric")
            baseline = s.get("baseline_value")
            target = s.get("target_value")
            evidence = s.get("evidence")

            if evidence:
                detail = evidence
            elif metric and baseline is not None and target is not None:
                detail = f"{metric}: {baseline} \u2192 {target}"
            else:
                detail = description

            lines.append(f"- {marker} **{sid}** ({entity}) \u2014 {detail}")

        return "\n".join(lines)

    def _rebuild_known_limitations(self, payload: dict) -> str:
        lines = ["## Known Limitations"]
        limitations = payload.get("known_limitations") or []
        disclaimers = payload.get("mandatory_disclaimers") or []

        if not limitations and not disclaimers:
            lines.append("- No limitations recorded.")
            return "\n".join(lines)

        for lim in limitations:
            lines.append(f"- {lim}")
        for disc in disclaimers:
            lines.append(f"- {disc}")

        return "\n".join(lines)

    @staticmethod
    def _replace_section(report: str, header: str, replacement: str) -> str:
        if header not in report:
            return report.rstrip() + "\n\n" + replacement + "\n"

        before, rest = report.split(header, 1)
        lines = rest.split("\n")
        next_idx = None
        for i, line in enumerate(lines):
            if i > 0 and line.strip().startswith("## "):
                next_idx = i
                break

        after = "\n".join(lines[next_idx:]) if next_idx is not None else ""
        result = before + replacement
        if after:
            result += "\n\n" + after
        return result

    def _build_verified_payload(self, state: dict) -> dict:
        payload = {
            "headline": None,
            "deltas": {},
            "root_causes": [],
            "anomalies": [],
            "known_scenarios": [],
            "mandatory_disclaimers": [],
            "known_limitations": [],
            "gross_negative": None,
            "positive_offset": None,
            "net_revenue_change": None,
        }

        os_ = (state.get("analysis") or {}).get("overall_summary") or {}
        if os_:
            payload["headline"] = {
                "revenue_definition": os_.get("revenue_definition"),
                "baseline_period": os_.get("baseline_period"),
                "target_period": os_.get("target_period"),
                "baseline_total": os_.get("baseline_total"),
                "target_total": os_.get("target_total"),
                "absolute_change": os_.get("absolute_change"),
                "total_change_pct": os_.get("total_change_pct"),
                "gross_baseline_total": os_.get("gross_baseline_total"),
                "gross_target_total": os_.get("gross_target_total"),
                "baseline_aov": os_.get("baseline_aov"),
                "target_aov": os_.get("target_aov"),
                "aov_change_pct": _pct_change(os_.get("baseline_aov"), os_.get("target_aov")),
                "baseline_txn_count": os_.get("baseline_txn_count"),
                "target_txn_count": os_.get("target_txn_count"),
                "txn_change_pct": _pct_change(os_.get("baseline_txn_count"), os_.get("target_txn_count")),
                "baseline_cancel_rate": os_.get("baseline_cancel_rate"),
                "target_cancel_rate": os_.get("target_cancel_rate"),
                "cancel_rate_change_pct": _pct_change(os_.get("baseline_cancel_rate"), os_.get("target_cancel_rate")),
                "driver_decomposition": os_.get("driver_decomposition"),
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

        scenario_detection = state.get("scenario_detection") or []
        detection_by_id = {d["id"]: d for d in scenario_detection}

        if not detection_by_id:
            computed = (state.get("anomalies") or {}).get("computed_statistics") or {}
            legacy = {
                c.get("scenario_id"): c.get("detected_by_cross_entity_zscore")
                for c in computed.get("known_scenario_check", [])
            }
            for s in KNOWN_SCENARIOS:
                detection_by_id[s["id"]] = {
                    "detected": legacy.get(s["id"], False),
                    "metric": None,
                    "baseline_value": None,
                    "target_value": None,
                    "delta_ratio": None,
                    "evidence": None,
                }

        payload["known_scenarios"] = [
            {
                "id": s["id"],
                "dimension": s["dimension"],
                "entity": s["entity"],
                "description": s["description"],
                "detected": detection_by_id.get(s["id"], {}).get("detected", False),
                "metric": detection_by_id.get(s["id"], {}).get("metric"),
                "baseline_value": detection_by_id.get(s["id"], {}).get("baseline_value"),
                "target_value": detection_by_id.get(s["id"], {}).get("target_value"),
                "delta_ratio": detection_by_id.get(s["id"], {}).get("delta_ratio"),
                "evidence": detection_by_id.get(s["id"], {}).get("evidence"),
            }
            for s in KNOWN_SCENARIOS
        ]

        payload["mandatory_disclaimers"] = [
            f"{s['entity']} \u2014 {s['description']}. Not detected by current analysis."
            for s in payload["known_scenarios"]
            if s["detected"] is False
        ]

        rc = state.get("root_causes") or {}
        payload["known_limitations"] = rc.get("known_limitations", [])
        payload["gross_negative"] = rc.get("gross_negative")
        payload["positive_offset"] = rc.get("positive_offset")
        payload["net_revenue_change"] = rc.get("net_revenue_change")

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
            entity_name = re.sub(r"\s*\([^)]*\)\s*$", "", entity_name)
            if not entity_name:
                continue
            if entity_name.lower() in ("total", "overall", "summary"):
                continue
            if entity_name.lower() not in known_entities:
                untraceable.append(entity_name)
        return untraceable

    # ------------------------------------------------------------------
    # DELIVERY 4 FIX — extractors (strip trailing punctuation, comma-
    # normalize payload, extract percent-form from payload strings)
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_claim_numbers(text: str) -> set:
        claims = set()

        def _normalize(s: str) -> str:
            s = (s.replace(",", "")
                 .replace("\u00a0", "")
                 .replace("\u2009", "")
                 .replace("\u202f", ""))
            # Strip trailing sentence punctuation the regex may have
            # captured (e.g. "198990.93." → "198990.93"). Does NOT strip
            # the decimal point inside the number.
            s = s.rstrip(".;:!?,)]}\u00a0 \t")
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
        # Strip commas that separate thousands so "347,241.06" is one
        # token. Pattern requires a digit on both sides, so JSON
        # structural commas are untouched.
        s = re.sub(r"(?<=\d),(?=\d)", "", s)

        nums = set()
        for m in re.finditer(r"-?\d+\.\d+", s):
            nums.add(m.group(0).lstrip("+-"))
        for m in re.finditer(r":\s*(-?\d+)(?=[,\s}])", s):
            nums.add(m.group(1).lstrip("+-"))
        # Percent-form numbers inside strings (e.g. "threshold -25%")
        for m in re.finditer(r"([+-]?\d+(?:\.\d+)?)\s*%", s):
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