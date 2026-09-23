from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    openai_api_key: str
    openai_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"

    database_url: str = ""
    storage_path: str = "./storage"
    max_upload_size_mb: int = 100
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        extra = "ignore"

    @property
    def uploads_dir(self) -> Path:
        p = Path(self.storage_path) / "uploads"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def reports_dir(self) -> Path:
        p = Path(self.storage_path) / "reports"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def charts_dir(self) -> Path:
        p = Path(self.storage_path) / "charts"
        p.mkdir(parents=True, exist_ok=True)
        return p


settings = Settings()