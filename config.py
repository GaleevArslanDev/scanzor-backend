from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Настройки приложения"""
    APP_NAME: str = "Scanzor API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    MAX_IMAGE_SIZE_MB: int = 10
    ALLOWED_EXTENSIONS: set = {"jpg", "jpeg", "png"}

    class Config:
        env_file = ".env"


settings = Settings()