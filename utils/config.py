"""Configuration management for HireFlow project."""

from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class AppConfig(BaseSettings):
    """Minimal application configuration (only what's used)."""

    PINECONE_API_KEY: str = Field(default="", env="PINECONE_API_KEY")
    PINECONE_INDEX_NAME: str = Field(default="hireflow", env="PINECONE_INDEX_NAME")
    PINECONE_DIMENSION: int = Field(default=768, env="PINECONE_DIMENSION")
    PINECONE_METRIC: str = Field(default="cosine", env="PINECONE_METRIC")

    GOOGLE_API_KEY: str = Field(default="", env="GOOGLE_API_KEY")
    # Free tier: gemini-3.5-flash-lite allows ~500 requests/day but only 5 per MINUTE.
    LLM_MODEL: str = Field(default="gemini-3.5-flash-lite", env="LLM_MODEL")
    LLM_REQUESTS_PER_MINUTE: int = Field(default=5, env="LLM_REQUESTS_PER_MINUTE")

    MAX_TEXT_LENGTH: int = Field(default=4000, env="MAX_TEXT_LENGTH")

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"

config = AppConfig()

GOOGLE_API_KEY = config.GOOGLE_API_KEY
LLM_MODEL = config.LLM_MODEL
LLM_REQUESTS_PER_MINUTE = config.LLM_REQUESTS_PER_MINUTE

PINECONE_API_KEY = config.PINECONE_API_KEY
PINECONE_INDEX_NAME = config.PINECONE_INDEX_NAME
PINECONE_DIMENSION = config.PINECONE_DIMENSION
PINECONE_METRIC = config.PINECONE_METRIC

MAX_TEXT_LENGTH = config.MAX_TEXT_LENGTH
