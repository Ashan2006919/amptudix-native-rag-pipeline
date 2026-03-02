from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):

    CHUNK_SIZE: int = 200
    CHUNK_OVERLAP: int = 20
    EMBED_MODEL: str = "mxbai-embed-large"
    CHROMA_PATH: str = "chroma_db"
    DATA_FOLDER: str = "data"
    GEMINI_API_KEY: str = ""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()
