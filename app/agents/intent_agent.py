from app.agents.base_agent import BaseAgent
from config.prompts import INTENT_PROMPT

class IntentAgent(BaseAgent):
    name = "IntentAgent"

    def run(self, state: dict) -> dict:
        q = state["question"]
        self.log(f"Analyzing intent: {q[:80]}")
        result = self.llm.complete_json(INTENT_PROMPT.format(question=q))
        state["intent"] = result
        return state