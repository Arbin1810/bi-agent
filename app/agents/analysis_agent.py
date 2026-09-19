import pandas as pd
from app.agents.base_agent import BaseAgent
from config.prompts import ANALYSIS_PROMPT
from app.data.connectors import db
from app.analysis.statistics import summarize_dataframe

class AnalysisAgent(BaseAgent):
    name = "AnalysisAgent"

    def run(self, state: dict) -> dict:
        self.log("Running analysis")

        # For DB queries
        if state.get("sql") and db.engine:
            try:
                df = db.execute(state["sql"])
                state["data"] = df
            except Exception as e:
                self.log(f"Query failed: {e}")
                state["error"] = str(e)
                return state

        # For uploaded CSV-only workflows
        elif state.get("dataframe") is not None:
            df = state["dataframe"]
        else:
            self.log("No data source to query")
            state["analysis"] = {"key_findings": ["No data available"], "quantitative_evidence": []}
            return state

        stats = summarize_dataframe(df)
        sample = df.head(10).to_dict(orient="records")
        columns = list(df.columns)

        prompt = ANALYSIS_PROMPT.format(
            question=state["question"],
            sql=state.get("sql", "N/A (file-based)"),
            stats=str(stats)[:3000],
            sample=str(sample)[:2000],
            columns=columns,
        )
        analysis = self.llm.complete_json(prompt)
        state["analysis"] = analysis
        state["stats"] = stats
        return state