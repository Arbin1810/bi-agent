import json
from datetime import datetime
from pathlib import Path
from app.agents.base_agent import BaseAgent
from config.prompts import REPORT_PROMPT
from config.settings import settings


class ReportAgent(BaseAgent):
    name = "ReportAgent"

    def run(self, state: dict) -> dict:
        self.log("Writing report")

        analysis = state.get("analysis") or {}

        # ----------------------------------------------------------
        # Strip llm_raw_excerpt — it's debug-only and contains
        # hallucinated numbers the report writer was quoting.
        # ----------------------------------------------------------
        analysis_clean = {k: v for k, v in analysis.items() if k != "llm_raw_excerpt"}

        # ----------------------------------------------------------
        # Inject headline numbers as dedicated prompt slots so the
        # writer copies them verbatim instead of paraphrasing.
        # ----------------------------------------------------------
        os_ = analysis.get("overall_summary") or {}
        headline_pct = os_.get("total_change_pct")
        headline_abs = os_.get("absolute_change")
        headline_target = os_.get("target_period", "n/a")
        headline_baseline = os_.get("baseline_period", "n/a")
        headline_target_total = os_.get("target_total")
        headline_baseline_total = os_.get("baseline_total")

        # ----------------------------------------------------------
        # Filter anomalies to high/medium severity only.
        # ----------------------------------------------------------
        raw_anomalies = (state.get("anomalies") or {}).get("anomalies", [])
        kept_anomalies = [
            a for a in raw_anomalies
            if a.get("severity") in ("high", "medium")
        ]

        # ----------------------------------------------------------
        # Compress root causes to only what the report needs.
        # ----------------------------------------------------------
        rc = state.get("root_causes") or {}
        rc_compact = [
            {
                "dimension": c.get("dimension"),
                "value": c.get("value"),
                "change_pct": c.get("change_pct"),
                "absolute_change": c.get("absolute_change"),
                "contribution_pct": c.get("contribution_pct"),
                "confidence": c.get("confidence"),
            }
            for c in rc.get("primary_causes", [])
        ]

        prompt = REPORT_PROMPT.format(
            question=state["question"],
            headline_target=headline_target,
            headline_baseline=headline_baseline,
            headline_pct=headline_pct,
            headline_abs=headline_abs,
            headline_target_total=headline_target_total,
            headline_baseline_total=headline_baseline_total,
            findings=json.dumps(analysis_clean, default=str)[:2500],
            root_causes=json.dumps(rc_compact, default=str)[:2000],
            anomalies=json.dumps(kept_anomalies, default=str)[:1500],
            charts=[c["path"] for c in state.get("charts", [])],
            confidence=state.get("critique", {}).get("confidence", "medium"),
        )

        # Freeform markdown — call without JSON mode
        report = self.llm.complete(prompt, temperature=0.3)

        # ----------------------------------------------------------
        # Sanity check: warn if the headline number the LLM wrote
        # doesn't match the one we gave it.
        # ----------------------------------------------------------
        if headline_pct is not None:
            expected = f"{abs(headline_pct):.1f}"
            if expected not in report:
                self.log(
                    f"WARNING: report does not contain expected headline "
                    f"{headline_pct}% (searched for '{expected}')"
                )

        # Persist
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out = settings.reports_dir / f"report_{ts}.md"
        out.write_text(report, encoding="utf-8")

        state["report"] = report
        state["report_path"] = str(out)
        return state