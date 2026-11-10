from typing import TypedDict

import pandas as pd


class WorkflowState(TypedDict, total=False):
    question: str
    uploaded_files: list[dict]
    schema_context: str
    intent: dict
    plan: dict
    sql: str | None
    sql_explanation: str
    sql_validation: dict
    sql_error: str
    data: pd.DataFrame | None
    dataframe: pd.DataFrame | None
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