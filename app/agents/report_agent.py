from datetime import datetime
from pathlib import Path
from app.agents.base_agent import BaseAgent
from config.prompts import REPORT_PROMPT
from config.settings import settings

class ReportAgent(BaseAgent):
    name = "ReportAgent"

    def run(self, state: dict) -> dict:
        self.log("Writing report")
        prompt = REPORT_PROMPT.format(
            question=state["question"],
            findings=str(state.get("analysis", {}))[:2500],
            root_causes=str(state.get("root_causes", {}))[:2000],
            anomalies=str(state.get("anomalies", {}))[:1500],
            charts=[c["path"] for c in state.get("charts", [])],
            confidence=state.get("critique", {}).get("confidence", "medium"),
        )
        # Freeform markdown — call without JSON mode
        report = self.llm.complete(prompt, temperature=0.3)

        # Persist
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out = settings.reports_dir / f"report_{ts}.md"
        out.write_text(report, encoding="utf-8")

        state["report"] = report
        state["report_path"] = str(out)
        return state