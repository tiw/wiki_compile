"""Configuration management for kb-compiler."""

import os
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and config file."""

    model_config = SettingsConfigDict(
        env_prefix="KB_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM Provider Selection
    llm_provider: str = Field(
        default="kimi",
        description="LLM provider: 'kimi' or 'local'",
    )

    # Kimi API Configuration
    kimi_api_key: str = Field(default="", description="Kimi API key")
    kimi_base_url: str = Field(
        default="https://api.moonshot.cn/v1",
        description="Kimi API base URL",
    )
    kimi_model: str = Field(
        default="moonshot-v1-128k",
        description="Kimi model to use (moonshot-v1-8k/32k/128k)",
    )
    # Kimi Code specific settings
    kimi_code_mode: bool = Field(
        default=False,
        description="Use Kimi Code API format (anthropic-messages)",
    )

    # Local LLM Configuration (MLX, Ollama, etc.)
    local_llm_base_url: str = Field(
        default="http://127.0.0.1:8017/v1",
        description="Local LLM OpenAI-compatible API base URL",
    )
    local_llm_api_key: str = Field(
        default="dandan",
        description="Local LLM API key",
    )
    local_llm_model: str = Field(
        default="Qwen3___5-27B-Claude-4___6-Opus-Distilled-MLX-6bit",
        description="Local LLM model name",
    )

    # Knowledge Base Paths
    kb_root: Path = Field(
        default=Path.home() / "KnowledgeBase",
        description="Root directory for knowledge base",
    )
    raw_dir: Optional[Path] = None
    wiki_dir: Optional[Path] = None
    output_dir: Optional[Path] = None
    meta_dir: Optional[Path] = None

    # Obsidian Configuration
    obsidian_vault: Optional[str] = Field(
        default=None,
        description="Obsidian vault name (if using obsidian-cli)",
    )
    obsidian_vault_path: Optional[Path] = Field(
        default=None,
        description="Direct path to Obsidian vault",
    )

    # Compilation Settings
    max_concurrent_requests: int = 3
    chunk_size: int = 8000
    chunk_overlap: int = 500
    incremental_compile: bool = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Resolve derived paths
        if self.raw_dir is None:
            self.raw_dir = self.kb_root / "raw"
        if self.wiki_dir is None:
            self.wiki_dir = self.kb_root / "wiki"
        if self.output_dir is None:
            self.output_dir = self.kb_root / "output"
        if self.meta_dir is None:
            self.meta_dir = self.kb_root / "_meta"

    def ensure_directories(self) -> None:
        """Create all necessary directories if they don't exist."""
        for path in [self.raw_dir, self.wiki_dir, self.output_dir, self.meta_dir]:
            if path:
                path.mkdir(parents=True, exist_ok=True)

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors."""
        errors = []

        # Validate LLM provider
        if self.llm_provider not in ["kimi", "local"]:
            errors.append(f"Invalid llm_provider: {self.llm_provider}. Use 'kimi' or 'local'")

        # Validate based on provider
        if self.llm_provider == "kimi" and not self.kimi_api_key:
            errors.append("KIMI_API_KEY environment variable is not set (required for kimi provider)")

        # Local LLM doesn't require API key validation (it's optional)

        if not self.obsidian_vault and not self.obsidian_vault_path:
            errors.append("Either OBSIDIAN_VAULT or OBSIDIAN_VAULT_PATH must be set")
        return errors


def get_settings() -> Settings:
    """Get or create settings singleton."""
    # Map KIMI_API_KEY to KB_KIMI_API_KEY for pydantic-settings
    if "KIMI_API_KEY" in os.environ and "KB_KIMI_API_KEY" not in os.environ:
        os.environ["KB_KIMI_API_KEY"] = os.environ["KIMI_API_KEY"]

    # Set default Kimi Code configuration if not set
    if "KB_KIMI_BASE_URL" not in os.environ:
        os.environ["KB_KIMI_BASE_URL"] = "https://api.kimi.com/coding/"
    if "KB_KIMI_CODE_MODE" not in os.environ:
        os.environ["KB_KIMI_CODE_MODE"] = "true"
    if "KB_KIMI_MODEL" not in os.environ:
        os.environ["KB_KIMI_MODEL"] = "kimi-k2-5"

    return Settings()
