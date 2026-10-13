from app.agents.base_agent import BaseAgent
from config.prompts import PLANNER_PROMPT
from app.data.schema_introspector import introspector


class PlannerAgent(BaseAgent):
    name = "PlannerAgent"

    def run(self, state: dict) -> dict:
        self.log("Building analysis plan")

        # Build schema context
        schema_ctx = state.get("schema_context")

        if not schema_ctx:
            df = state.get("dataframe")
            if df is not None:
                # Fix 1.4: Build schema directly from the in-memory DataFrame
                cols = [f"{c} ({df[c].dtype})" for c in df.columns]
                schema_ctx = (
                    "=== Uploaded Table: `transactions` ===\n"
                    f"Columns: {', '.join(cols)}\n"
                    f"Rows: {len(df)}\n"
                    "Sample:\n" + df.head(3).to_string()
                )
            else:
                schema_ctx = introspector.build_context(state.get("uploaded_files"))

        # Fix 4: Make the planner schema-aware by surfacing dimension/KPI hints
        # at the top of schema_context so the model anchors on real columns
        # instead of inventing its own groupings.
        dimension_hint = (
            "IMPORTANT: The table is named `transactions`. "
            "Dimension columns: region, branch, product_id, product_category, "
            "customer_segment, sales_channel, payment_method, order_status. "
            "Time column: date (ISO format YYYY-MM-DD). "
            "KPI columns: revenue, cost, profit, quantity, unit_price, discount."
        )
        state["schema_context"] = dimension_hint + "\n\n" + schema_ctx

        prompt = PLANNER_PROMPT.format(
            question=state["question"],
            intent=state["intent"],
            domain=state["intent"].get("business_domain", "general"),
            time_reference=state["intent"].get("time_reference"),
            schema_context=state["schema_context"][:6000],
        )
        plan = self.llm.complete_json(prompt)
        state["plan"] = plan
        return state