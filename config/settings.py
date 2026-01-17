"""Application settings using Pydantic Settings."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Paths
    project_root: Path = Field(default=Path(__file__).parent.parent)
    data_dir: Path = Field(default=Path(__file__).parent.parent / "data")
    documents_dir: Path = Field(default=Path(__file__).parent.parent / "data" / "documents")
    chroma_db_path: Path = Field(default=Path(__file__).parent.parent / "data" / "chroma_db")

    # Ollama settings
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="mistral:7b")
    ollama_embedding_model: str = Field(default="nomic-embed-text")
    ollama_timeout: int = Field(default=120)

    # ChromaDB settings
    chroma_collection_name: str = Field(default="knowledge_base")

    # RAG settings
    retriever_k: int = Field(default=4)
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)
    relevance_threshold: float = Field(default=0.7)
    max_rewrite_attempts: int = Field(default=2)

    # External provider API keys (optional)
    anthropic_api_key: str | None = Field(default=None)
    openai_api_key: str | None = Field(default=None)
    google_api_key: str | None = Field(default=None)
    xai_api_key: str | None = Field(default=None)

    # Provider settings
    provider_config_path: Path = Field(
        default=Path(__file__).parent / "providers.yaml"
    )
    fallback_enabled: bool = Field(default=True)

    # Tool execution settings
    bash_timeout: int = Field(default=30)
    bash_require_approval: bool = Field(default=True)

    # API settings
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")


settings = Settings()
