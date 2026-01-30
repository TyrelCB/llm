"""Application settings using Pydantic Settings."""

import os
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
    ollama_num_ctx: int = Field(default=8192)

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

    # Agentic loop settings
    agentic_state_dir: Path = Field(default=Path(__file__).parent.parent / ".agent")
    agentic_max_steps: int = Field(default=8)
    agentic_facts_k: int = Field(default=6)
    agentic_fact_max_chars: int = Field(default=200)
    agentic_state_max_chars: int = Field(default=1200)
    agentic_last_result_max_chars: int = Field(default=800)
    agentic_tool_output_max_chars: int = Field(default=2000)
    agentic_default_approval_mode: Literal["step", "auto"] = Field(default="step")
    agentic_repeat_limit: int = Field(default=2)
    agentic_action_retries: int = Field(default=1)
    agentic_no_match_limit: int = Field(default=2)

    # API settings
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")

    # Mode settings
    default_mode: str = Field(default="chat")

    # Stable Diffusion settings
    comfyui_url: str = Field(default="http://localhost:8188")
    automatic1111_url: str = Field(default="http://localhost:7860")
    image_output_dir: Path = Field(default=Path(__file__).parent.parent / "data" / "images")

    # Web search settings
    searxng_url: str | None = Field(default=None)

    # TTS settings
    tts_service_url: str = Field(default="http://127.0.0.1:8123")
    tts_output_dir: Path = Field(default=Path(__file__).parent.parent / "data" / "tts")
    tts_output_format: Literal["mp3", "wav"] = Field(default="mp3")
    tts_device: Literal["auto", "cpu", "cuda", "mps"] = Field(default="auto")
    tts_dtype: Literal["float32", "float16", "bfloat16"] = Field(default="float32")
    tts_model_custom: str = Field(default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    tts_model_design: str = Field(default="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    tts_model_clone: str = Field(default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")


settings = Settings()


def configure_project_root(root: Path) -> None:
    """Set project root and rebase derived paths when not explicitly configured."""
    resolved_root = root.expanduser().resolve()
    settings.project_root = resolved_root

    data_dir = settings.data_dir
    if os.getenv("DATA_DIR") is None:
        data_dir = resolved_root / "data"
        settings.data_dir = data_dir

    if os.getenv("DOCUMENTS_DIR") is None:
        settings.documents_dir = data_dir / "documents"

    if os.getenv("CHROMA_DB_PATH") is None:
        settings.chroma_db_path = data_dir / "chroma_db"

    if os.getenv("AGENTIC_STATE_DIR") is None:
        settings.agentic_state_dir = resolved_root / ".agent"

    if os.getenv("IMAGE_OUTPUT_DIR") is None:
        settings.image_output_dir = data_dir / "images"

    if os.getenv("TTS_OUTPUT_DIR") is None:
        settings.tts_output_dir = data_dir / "tts"
