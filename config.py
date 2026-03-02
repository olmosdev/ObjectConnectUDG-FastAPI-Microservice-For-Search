import os
import logging
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    """
    Application settings and environment variables management.
    """
    # --- Supabase Config ---
    SUPABASE_URL: str = Field(..., validation_alias="SUPABASE_URL")
    SUPABASE_KEY: str = Field(..., validation_alias="SUPABASE_APIKEY_SERVICE_ROLE")
    
    # --- ML Config (SBERT) ---
    SBERT_MODEL_NAME: str = "paraphrase-multilingual-MiniLM-L12-v2"
    FIXED_DIM: int = 384 # SBERT mini multilingual output dimension
    K_CLUSTERS_DEFAULT: int = 4
    SYNC_INTERVAL_SECONDS: int = 300 # 5 minutes
    MODEL_FOLDER: str = "data"
    
    # --- Logging Config ---
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

# Global settings instance
settings = Settings()

# Global Logger configuration
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper()),
    format=settings.LOG_FORMAT,
    datefmt="%Y-%m-%d %H:%M:%S"
)

def get_logger(name: str):
    """
    Returns a configured logger for a specific module.
    """
    return logging.getLogger(name)
