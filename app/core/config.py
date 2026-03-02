from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):

    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    EMBED_MODEL: str = "nomic-embed-text"
    CHROMA_PATH: str = "chroma_db"
    DATA_FOLDER: str = "data"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()
