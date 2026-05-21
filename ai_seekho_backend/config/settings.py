import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Ensure .env is loaded in context
BACKEND_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BACKEND_DIR / ".env")

class Settings(BaseSettings):
    GEMINI_API_KEY: str = ""
    MAPS_API_KEY: str = ""
    FIREBASE_CREDENTIALS_PATH: str = "../ai-seekho--01-firebase-adminsdk-fbsvc-ff54490668.json"
    port: int = 8000
    host: str = "0.0.0.0"

    model_config = SettingsConfigDict(
        env_file=str(BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
