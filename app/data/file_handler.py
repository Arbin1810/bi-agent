import pandas as pd
from pathlib import Path
from typing import Any
from pypdf import PdfReader
from config.settings import settings
from app.utils.logger import logger

class FileHandler:
    """Loads uploaded files into pandas DataFrames or text."""

    def load(self, file_path: str | Path) -> dict[str, Any]:
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext in (".xlsx", ".xls"):
            sheets = pd.read_excel(path, sheet_name=None)
            return {"type": "excel", "sheets": {k: v for k, v in sheets.items()}}

        if ext == ".csv":
            df = pd.read_csv(path)
            return {"type": "csv", "df": df, "name": path.stem}

        if ext == ".pdf":
            reader = PdfReader(str(path))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            return {"type": "pdf", "text": text, "name": path.stem}

        if ext in (".txt", ".md"):
            return {"type": "text", "text": path.read_text(), "name": path.stem}

        raise ValueError(f"Unsupported file type: {ext}")

    def to_schema_description(self, loaded: dict[str, Any]) -> str:
        """Convert loaded file data into a textual schema for the LLM."""
        if loaded["type"] == "csv":
            df = loaded["df"]
            cols = [f"{c} ({df[c].dtype})" for c in df.columns]
            return f"Table: {loaded['name']}\nColumns: {', '.join(cols)}\nSample:\n{df.head(5).to_string()}"

        if loaded["type"] == "excel":
            parts = []
            for name, df in loaded["sheets"].items():
                cols = [f"{c} ({df[c].dtype})" for c in df.columns]
                parts.append(f"Sheet: {name}\nColumns: {', '.join(cols)}\nSample:\n{df.head(5).to_string()}")
            return "\n\n".join(parts)

        if loaded["type"] in ("pdf", "text"):
            return f"Document: {loaded['name']}\nContent preview:\n{loaded['text'][:2000]}"

        return ""

file_handler = FileHandler()