from typing import TypedDict, Any, Optional
import pandas as pd

class WorkflowState(TypedDict, total=False):
    question: str
    uploaded_files: list[dict]
    schema_context: str
    intent: dict
    plan: dict
    sql: Optional[str]
    sql_explanation: str
    sql_validation: dict
    sql_error: str
    data: Optional[pd.DataFrame]
    dataframe: Optional[pd.DataFrame]
    stats: dict
    analysis: dict
    anomalies: dict
    root_causes: dict
    critique: dict
    charts: list[dict]
    report: str
    report_path: str
    error: str
    iterations: int
    trace: list[str]