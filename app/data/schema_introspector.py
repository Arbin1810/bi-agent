from app.data.connectors import db
from app.data.file_handler import file_handler, FileHandler
from pathlib import Path
from typing import Any

class SchemaIntrospector:
    """Combines DB schema + uploaded files into unified schema context."""

    def build_context(self, uploaded_files: list[dict[str, Any]] | None = None) -> str:
        parts = []

        # DB schema
        db_schema = db.get_schema()
        if db_schema:
            parts.append("=== Database ===")
            for table, cols in db_schema.items():
                col_str = ", ".join(f"{c['name']} ({c['type']})" for c in cols)
                parts.append(f"Table `{table}`: {col_str}")

        # Uploaded files
        if uploaded_files:
            parts.append("\n=== Uploaded Files ===")
            for f in uploaded_files:
                loaded = file_handler.load(f["path"])
                parts.append(file_handler.to_schema_description(loaded))

        return "\n".join(parts) if parts else "No data sources available."

introspector = SchemaIntrospector()