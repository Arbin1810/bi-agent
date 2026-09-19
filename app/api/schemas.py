from pydantic import BaseModel
from typing import Any

class QueryRequest(BaseModel):
    question: str
    use_database: bool = True

class QueryResponse(BaseModel):
    question: str
    intent: dict | None = None
    sql: str | None = None
    analysis: dict | None = None
    anomalies: dict | None = None
    root_causes: dict | None = None
    critique: dict | None = None
    charts: list[dict] = []
    report: str | None = None
    report_path: str | None = None
    trace: list[str] = []
    error: str | None = None