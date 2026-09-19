from app.agents.base_agent import BaseAgent
from config.prompts import ANOMALY_PROMPT
from app.analysis.anomaly import detect_anomalies

class AnomalyAgent(BaseAgent):
    name = "AnomalyAgent"

    def run(self, state: dict) -> dict:
        self.log("Detecting anomalies")
        df = state.get("data")
        if df is None or df.empty:
            state["anomalies"] = {"anomalies": [], "method": "none"}
            return state

        # Statistical detection (numeric columns)
        stats_anomalies = detect_anomalies(df)

        prompt = ANOMALY_PROMPT.format(
            question=state["question"],
            summary=str(stats_anomalies)[:3000],
            stats=str(state.get("stats", {}))[:1500],
        )
        result = self.llm.complete_json(prompt)
        # merge
        result["statistical_hits"] = stats_anomalies
        state["anomalies"] = result
        return state