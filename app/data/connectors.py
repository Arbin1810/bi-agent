import pandas as pd
from sqlalchemy import create_engine, text, inspect
from config.settings import settings
from app.utils.logger import logger


class DatabaseConnector:
    def __init__(self, url: str | None = None):
        self.url = url or settings.database_url
        self.engine = create_engine(self.url) if self.url else None

    def get_schema(self) -> dict:
        if not self.engine:
            return {}
        inspector = inspect(self.engine)
        schema = {}
        for table in inspector.get_table_names():
            cols = inspector.get_columns(table)
            schema[table] = [{"name": c["name"], "type": str(c["type"])} for c in cols]
        return schema

    def execute(self, sql: str, params: dict | None = None) -> pd.DataFrame:
        logger.info(f"Executing SQL: {sql[:200]}")
        with self.engine.connect() as conn:
            return pd.read_sql(text(sql), conn, params=params or {})

    @property
    def dialect(self) -> str:
        if not self.engine:
            return "sqlite"
        return self.engine.dialect.name


db = DatabaseConnector()


# ---------- SQL dialect normalization ----------
def normalize_for_sqlite(sql: str) -> str:
    """Convert common Postgres/MySQL syntax to SQLite-compatible."""
    import re

    # DATE_TRUNC('month', date) → strftime('%Y-%m', date)
    sql = re.sub(
        r"DATE_TRUNC\s*\(\s*'month'\s*,\s*([^)]+)\)",
        r"strftime('%Y-%m', \1)",
        sql,
        flags=re.IGNORECASE,
    )
    # EXTRACT(YEAR FROM date) → CAST(strftime('%Y', date) AS INTEGER)
    sql = re.sub(
        r"EXTRACT\s*\(\s*YEAR\s+FROM\s+([^)]+)\)",
        r"CAST(strftime('%Y', \1) AS INTEGER)",
        sql,
        flags=re.IGNORECASE,
    )
    # EXTRACT(MONTH FROM date) → CAST(strftime('%m', date) AS INTEGER)
    sql = re.sub(
        r"EXTRACT\s*\(\s*MONTH\s+FROM\s+([^)]+)\)",
        r"CAST(strftime('%m', \1) AS INTEGER)",
        sql,
        flags=re.IGNORECASE,
    )
    # Remove ::type casts (Postgres-only)
    sql = re.sub(r"::\w+", "", sql)
    return sql


# ---------- In-memory SQL executor for CSV mode ----------
class DataFrameSQLExecutor:
    """Runs SQL against an in-memory pandas DataFrame via pandasql."""

    def __init__(self, df: pd.DataFrame, table_name: str = "transactions"):
        self.df = df
        self.table_name = table_name

    def execute(self, sql: str) -> pd.DataFrame:
        import re
        from pandasql import sqldf

        # Register the DataFrame under multiple names so ANY of these work:
        scope = {
            self.table_name: self.df,
            "transactions": self.df,
            "sales_data": self.df,
            "df": self.df,
            "data": self.df,
        }

        # Strip any "<alias>." prefixes — pandasql doesn't support them
        cleaned = re.sub(
            r"\b(transactions|sales_data|df|data)\.",
            "",
            sql,
            flags=re.IGNORECASE,
        )

        # Fix 1.3: translate Postgres/MySQL-only syntax to SQLite
        cleaned = normalize_for_sqlite(cleaned)

        logger.info(f"DataFrame SQL (cleaned): {cleaned[:300]}")
        try:
            result = sqldf(cleaned, scope)
            logger.info(f"DataFrame SQL returned {len(result)} rows")
            return result
        except Exception as e:
            logger.error(f"DataFrame SQL failed on: {cleaned[:500]}")
            raise

    def get_schema(self) -> dict:
        cols = [{"name": c, "type": str(self.df[c].dtype)} for c in self.df.columns]
        return {self.table_name: cols}

    @property
    def dialect(self) -> str:
        return "sqlite"