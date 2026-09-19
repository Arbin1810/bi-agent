from app.agents.base_agent import BaseAgent
from config.prompts import PLANNER_PROMPT
from app.data.schema_introspector import introspector

class PlannerAgent(BaseAgent):
    name = "PlannerAgent"

    def run(self, state: dict) -> dict:
        self.log("Building analysis plan")
        schema_ctx = state.get("schema_context") or introspector.build_context(
            state.get("uploaded_files")
        )
        state["schema_context"] = schema_ctx

        prompt = PLANNER_PROMPT.format(
            question=state["question"],
            intent=state["intent"],
            domain=state["intent"].get("business_domain", "general"),
            time_reference=state["intent"].get("time_reference"),
            schema_context=schema_ctx[:6000],
        )
        plan = self.llm.complete_json(prompt)
        state["plan"] = plan
        return state