"""Configuration for Milestone 1 ingest + knowledge model."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LiteLLM (single model access layer)
    litellm_api_base: str | None = Field(default=None, alias="LITELLM_API_BASE")
    litellm_api_key: str | None = Field(default=None, alias="LITELLM_API_KEY")
    # Provider keys — only needed when calling LiteLLM SDK without a proxy
    mistral_api_key: str | None = Field(default=None, alias="MISTRAL_API_KEY")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    mistral_ocr_model: str = Field(
        default="mistral/mistral-ocr-latest",
        alias="MISTRAL_OCR_MODEL",
    )
    llm_model: str = Field(
        default="claude-haiku",
        alias="LLM_MODEL",
    )
    # claude = vision OCR via LiteLLM chat (no Mistral key)
    # mistral = LiteLLM /ocr endpoint
    # skip = store page images only for diagram pages
    ocr_backend: str = Field(default="claude", alias="OCR_BACKEND")
    ocr_vision_model: str = Field(default="claude-haiku", alias="OCR_VISION_MODEL")
    agentic_enabled: bool = Field(default=True, alias="AGENTIC_ENABLED")
    # Ingest page router: heuristic first, LLM only on genuinely ambiguous pages.
    page_router_llm_enabled: bool = Field(default=True, alias="PAGE_ROUTER_LLM_ENABLED")
    page_router_llm_max_pages: int = Field(default=60, alias="PAGE_ROUTER_LLM_MAX_PAGES")
    docling_device: str = Field(default="auto", alias="DOCLING_DEVICE")
    docling_batch_size: int = Field(default=10, alias="DOCLING_BATCH_SIZE")

    # Postgres
    postgres_host: str = Field(default="localhost", alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5433, alias="POSTGRES_PORT")
    postgres_db: str = Field(default="marine_docs", alias="POSTGRES_DB")
    postgres_user: str = Field(default="marine", alias="POSTGRES_USER")
    postgres_password: str = Field(default="marine_dev", alias="POSTGRES_PASSWORD")

    # MinIO
    minio_endpoint: str = Field(default="localhost:9000", alias="MINIO_ENDPOINT")
    minio_access_key: str = Field(default="minioadmin", alias="MINIO_ACCESS_KEY")
    minio_secret_key: str = Field(default="minioadmin", alias="MINIO_SECRET_KEY")
    minio_bucket: str = Field(default="marine-figures", alias="MINIO_BUCKET")
    minio_secure: bool = Field(default=False, alias="MINIO_SECURE")

    # Ingest
    pdf_path: Path = Field(
        default=ROOT / "690988979-MAN-B-W-6S50MC-Maintenance.pdf",
        alias="PDF_PATH",
    )
    docling_enabled: bool = Field(default=True, alias="DOCLING_ENABLED")
    artifacts_dir: Path = Field(default=ROOT / "artifacts")

    @property
    def resolved_pdf_path(self) -> Path:
        """Always resolve PDF against project root when relative."""
        p = Path(self.pdf_path)
        if not p.is_absolute():
            p = ROOT / p
        return p.resolve()

    @property
    def database_url(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def ocr_ready(self) -> bool:
        """Hard-page OCR ready depending on OCR_BACKEND."""
        backend = (self.ocr_backend or "claude").strip().lower()
        if backend == "skip":
            return True
        if backend == "claude":
            return bool(self.litellm_api_base and self.litellm_api_key)
        if self.litellm_api_base and self.litellm_api_key:
            return True
        return bool(self.mistral_api_key or self.litellm_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
