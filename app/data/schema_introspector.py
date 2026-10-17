from app.data.connectors import db
from app.data.file_handler import file_handler
import pandas as pd


class SchemaIntrospector:
    """Combines DB schema + uploaded files into unified schema context."""

    def build_context(self, uploaded_files=None):
        parts = []

        # DB schema
        db_schema = db.get_schema()
        if db_schema:
            parts.append("=== Database ===")
            for table, cols in db_schema.items():
                col_str = ", ".join(f"{c['name']} ({c['type']})" for c in cols)
                parts.append(f"Table `{table}`: {col_str}")

        # Uploaded files (CSV mode) — synthesize a SQL-visible table schema
        if uploaded_files:
            parts.append("\n=== Uploaded Files ===")
            for f in uploaded_files:
                loaded = file_handler.load(f["path"])
                # Human-readable description (kept from previous behavior)
                parts.append(file_handler.to_schema_description(loaded))

                # FIX (Step 2.1): file_handler.load() returns a dict of the
                # form {"df": DataFrame, "name": str, "type": str}. The previous
                # `hasattr(loaded, "columns")` check was always False because
                # dicts don't have a `.columns` attribute, so the "Queryable
                # table" line never reached the SQL agent — it then invented
                # table names like `sales` or `orders` and every query failed
                # with `no such table`.
                df = loaded.get("df") if isinstance(loaded, dict) else None
                if isinstance(df, pd.DataFrame) and len(df.columns) > 0:
                    col_str = ", ".join(f"{c} ({df[c].dtype})" for c in df.columns)
                    parts.append(
                        f"Queryable table `transactions` (aliases: "
                        f"`sales_data`, `df`, `data`): {col_str}"
                    )

        return "\n".join(parts) if parts else "No data sources available."


introspector = SchemaIntrospector()