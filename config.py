"""Loads .env and exposes all config as a single settings object."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    GEMINI_API_KEY: str
    QDRANT_URL: str
    QDRANT_API_KEY: str
    QDRANT_COLLECTION_NAME: str = "medical_reports"
    EMBEDDING_MODEL_NAME: str = "google/gemma-2-2b"
    INGESTION_EMBEDDING_MODEL: str = "gemini-embedding-001"  # Report ingestion only; preserves the original Gemma model.
    OCR_MODEL_NAME: str = "gemini-2.5-flash"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200


settings = Settings()
