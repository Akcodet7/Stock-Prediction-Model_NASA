import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PROJECT_NAME: str = "Stock Price Prediction & Sentiment API"
    VERSION: str = "2.0.0"
    API_V1_PREFIX: str = "/api/v1"
    
    # Supabase PostgreSQL / Local SQLite fallback
    DATABASE_URL: str = os.getenv("DATABASE_URL", "").strip()
    
    # Normalize postgres:// to postgresql:// for SQLAlchemy compatibility
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
        
    # If no DATABASE_URL provided, fallback to SQLite for local development
    if not DATABASE_URL:
        DATABASE_URL = "sqlite:///./stock_predictor.db"
        
    DEFAULT_STOCK_SYMBOL: str = "^NSEI"
    DEFAULT_START_DATE: str = "2018-01-01"
    DEFAULT_TIME_STEP: int = 60
    DEFAULT_EPOCHS: int = 15
    DEFAULT_BATCH_SIZE: int = 32

settings = Settings()
