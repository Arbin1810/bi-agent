from app.agents.base_agent import BaseAgent
from config.prompts import CRITIC_PROMPT

class CriticAgent(BaseAgent):
    name = "CriticAgent"

    def run(self, state: dict) -> dict:
        self.log("Critiquing analysis")
        prompt = CRITIC_PROMPT.format(
            question=state["question"],
            findings=str(state.get("analysis", {}))[:2500],
            root_causes=str(state.get("root_causes", {}))[:2000],
            plan=str(state.get("plan", {}))[:1500],
        )
        critique = self.llm.complete_json(prompt)
        state["critique"] = critique
        return state