from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):

    CHUNK_SIZE: int = 300
    CHUNK_OVERLAP: int = 30
    EMBED_MODEL: str = "nomic-embed-text"
    CHROMA_PATH: str = "chroma_db"
    DATA_FOLDER: str = "data"
    GEMINI_API_KEY: str = ""
    OLLAMA_MODEL_NAME: str = ""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()
