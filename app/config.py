from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application configuration"""

    # Vector Database
    qdrant_url: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: Optional[str]  = None

    # LLM
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "deepseek-r1:7b"
    openrouter_api_key: Optional[str] = None

    # Embedding
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_device: Optional[str] = None

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 9000

    # Logging
    log_level: str = "INFO"

    class Config:
        env_file = ".env"

def get_settings() -> Settings:
    return Settings()