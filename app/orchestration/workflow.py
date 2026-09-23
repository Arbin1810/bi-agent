from app.orchestration.state import WorkflowState
from app.agents.intent_agent import IntentAgent
from app.agents.planner_agent import PlannerAgent
from app.agents.sql_agent import SQLAgent
from app.agents.analysis_agent import AnalysisAgent
from app.agents.anomaly_agent import AnomalyAgent
from app.agents.rootcause_agent import RootCauseAgent
from app.agents.critic_agent import CriticAgent
from app.agents.chart_agent import ChartAgent
from app.agents.report_agent import ReportAgent
from app.data.connectors import db
from app.utils.logger import logger

MAX_ITERATIONS = 2


class BIWorkflow:
    def __init__(self):
        self.intent = IntentAgent()
        self.planner = PlannerAgent()
        self.sql = SQLAgent()
        self.analysis = AnalysisAgent()
        self.anomaly = AnomalyAgent()
        self.rootcause = RootCauseAgent()
        self.critic = CriticAgent()
        self.chart = ChartAgent()
        self.report = ReportAgent()

    def run(self, question: str, uploaded_files: list[dict] | None = None,
            dataframe=None) -> WorkflowState:
        state: WorkflowState = {
            "question": question,
            "uploaded_files": uploaded_files or [],
            "trace": [],
            "iterations": 0,
        }
        if dataframe is not None:
            state["dataframe"] = dataframe

        try:
            state = self.intent.run(state)
            state["trace"].append("intent")

            state = self.planner.run(state)
            state["trace"].append("plan")

            # Only run SQL path if we have a DB AND no dataframe provided
            if db.engine is not None and state.get("dataframe") is None:
                state = self.sql.run(state)
                state["trace"].append("sql")

            state = self.analysis.run(state)
            state["trace"].append("analysis")

            state = self.anomaly.run(state)
            state["trace"].append("anomaly")

            state = self.rootcause.run(state)
            state["trace"].append("rootcause")

            while state.get("iterations", 0) < MAX_ITERATIONS:
                state = self.critic.run(state)
                state["trace"].append(f"critic#{state['iterations']}")
                critique = state.get("critique", {})
                if critique.get("is_complete", True):
                    break
                state["iterations"] += 1

            state = self.chart.run(state)
            state["trace"].append("charts")

            state = self.report.run(state)
            state["trace"].append("report")

        except Exception as e:
            logger.exception("Workflow failed")
            state["error"] = str(e)

        return state


workflow = BIWorkflow()