"""
Configuration management for the LLM-NMT Translation System.
Handles environment variables and settings.
"""

import os
from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Keys
    OPENAI_API_KEY: str = ""
    
    # OpenAI Settings
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_TIMEOUT: int = 30
    OPENAI_MAX_RETRIES: int = 3
    
    # NMT Model Settings
    NMT_MODEL_NAME: str = "Helsinki-NLP/opus-mt-fr-en"
    NMT_USE_QUANTIZATION: bool = True
    NMT_BEAM_SIZE: int = 5
    NMT_MAX_LENGTH: int = 128
    
    # Cache Settings
    CACHE_BACKEND: Literal["memory", "redis"] = "memory"
    CACHE_TTL_SECONDS: int = 86400  # 24 hours
    CACHE_MAX_SIZE: int = 100000
    CACHE_FUZZY_THRESHOLD: float = 0.85
    
    # Redis Settings (if using Redis)
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    
    # Data Source Settings
    CATALOG_SOURCE: Literal["json", "postgres", "api"] = "json"
    
    # LLM Provider Settings
    LLM_PROVIDER: Literal["openai", "local"] = "openai"
    
    # Paths
    DATA_DIR: Path = Path(__file__).parent / "data"
    
    # Batch Processing
    BATCH_MAX_SIZE: int = 32
    BATCH_TIMEOUT: int = 30
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Convenience function to get settings
settings = get_settings()
