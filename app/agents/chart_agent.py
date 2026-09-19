from app.agents.base_agent import BaseAgent
from app.visualization.charts import generate_charts

class ChartAgent(BaseAgent):
    name = "ChartAgent"

    def run(self, state: dict) -> dict:
        self.log("Generating charts")
        df = state.get("data")
        if df is None or df.empty:
            state["charts"] = []
            return state
        charts = generate_charts(df, question=state["question"])
        state["charts"] = charts
        return state