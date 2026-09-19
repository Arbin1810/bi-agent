import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from app.api.schemas import QueryRequest, QueryResponse
from app.orchestration.workflow import workflow
from app.data.file_handler import file_handler
from config.settings import settings
import pandas as pd

router = APIRouter()

@router.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    state = workflow.run(question=req.question)
    return _serialize(state)

@router.post("/query-with-files", response_model=QueryResponse)
async def query_with_files(
    question: str = Form(...),
    files: list[UploadFile] = File(default=[]),
):
    uploaded = []
    df_for_analysis = None

    for f in files:
        dest = settings.uploads_dir / f.filename
        with dest.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        uploaded.append({"path": str(dest), "name": f.filename})

        # If CSV, load as dataframe for direct analysis
        if f.filename.lower().endswith(".csv") and df_for_analysis is None:
            df_for_analysis = pd.read_csv(dest)

    state = workflow.run(question=question, uploaded_files=uploaded, dataframe=df_for_analysis)
    return _serialize(state)

def _serialize(state: dict) -> QueryResponse:
    df = state.get("data")
    return QueryResponse(
        question=state.get("question", ""),
        intent=state.get("intent"),
        sql=state.get("sql"),
        analysis=state.get("analysis"),
        anomalies=state.get("anomalies"),
        root_causes=state.get("root_causes"),
        critique=state.get("critique"),
        charts=state.get("charts", []),
        report=state.get("report"),
        report_path=state.get("report_path"),
        trace=state.get("trace", []),
        error=state.get("error"),
    )