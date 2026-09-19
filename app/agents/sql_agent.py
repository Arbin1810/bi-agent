from app.agents.base_agent import BaseAgent
from config.prompts import SQL_GENERATION_PROMPT, SQL_VALIDATION_PROMPT
from app.data.connectors import db
from app.utils.sql_safety import is_safe_sql

class SQLAgent(BaseAgent):
    name = "SQLAgent"

    def run(self, state: dict) -> dict:
        self.log("Generating SQL")
        dialect = db.dialect

        prompt = SQL_GENERATION_PROMPT.format(
            dialect=dialect,
            schema=state["schema_context"][:6000],
            question=state["question"],
            plan_context=str(state.get("plan", ""))[:1500],
        )
        gen = self.llm.complete_json(prompt)
        sql = gen.get("sql", "").strip()

        # Local safety check
        if not is_safe_sql(sql):
            self.log(f"Unsafe SQL blocked: {sql[:200]}")
            state["sql"] = None
            state["sql_error"] = "Unsafe SQL detected"
            return state

        # LLM validation
        val = self.llm.complete_json(SQL_VALIDATION_PROMPT.format(
            dialect=dialect,
            schema=state["schema_context"][:4000],
            sql=sql,
        ))

        if not val.get("valid", False) and val.get("corrected_sql"):
            sql = val["corrected_sql"]
            self.log("SQL corrected by validator")

        state["sql"] = sql
        state["sql_explanation"] = gen.get("explanation", "")
        state["sql_validation"] = val
        return state