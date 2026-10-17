from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # ----- LLM -----
    openai_api_key: str = "ollama-local"          # ignored by Ollama; required by SDK
    openai_model: str = "llama3.1"                # must match `ollama list`
    openai_embedding_model: str = "nomic-embed-text"  # Ollama embedding model (see note)
    openai_base_url: str = "http://localhost:11434/v1"  # default to Ollama

    # ----- Database -----
    database_url: str = ""                        # empty => CSV mode

    # ----- Storage -----
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