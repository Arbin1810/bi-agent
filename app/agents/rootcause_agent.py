from app.agents.base_agent import BaseAgent
from config.prompts import ROOT_CAUSE_PROMPT

class RootCauseAgent(BaseAgent):
    name = "RootCauseAgent"

    def run(self, state: dict) -> dict:
        self.log("Analyzing root causes")
        prompt = ROOT_CAUSE_PROMPT.format(
            question=state["question"],
            findings=str(state.get("analysis", {}))[:3000],
            anomalies=str(state.get("anomalies", {}))[:2000],
            additional=str(state.get("stats", {}))[:1500],
        )
        rc = self.llm.complete_json(prompt)
        state["root_causes"] = rc
        return state