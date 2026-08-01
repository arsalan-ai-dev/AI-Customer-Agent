from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    APP_NAME: str = "Enterprise AI Customer Support Agent"
    VERSION: str = "2.0.0"
    
    # Storage & DB Paths
    DATA_DIR: str = "./data"
    CHROMA_DIR: str = "./chroma_db"
    
    # RAG & Embedding Configurations
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100
    RETRIEVAL_TOP_K: int = 3
    
    # LLM Provider Configuration
    GROQ_API_KEY: str = ""
    GROQ_MODEL_NAME: str = "llama-3.1-8b-instant"
    LLM_TEMPERATURE: float = 0.2
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()