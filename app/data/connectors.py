import pandas as pd
from sqlalchemy import create_engine, text, inspect
from typing import Any
from config.settings import settings
from app.utils.logger import logger

class DatabaseConnector:
    def __init__(self, url: str | None = None):
        self.url = url or settings.database_url
        self.engine = create_engine(self.url) if self.url else None

    def get_schema(self) -> dict[str, list[dict]]:
        if not self.engine:
            return {}
        inspector = inspect(self.engine)
        schema = {}
        for table in inspector.get_table_names():
            cols = inspector.get_columns(table)
            schema[table] = [
                {"name": c["name"], "type": str(c["type"])} for c in cols
            ]
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