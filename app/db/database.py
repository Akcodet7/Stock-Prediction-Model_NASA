import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from app.config import settings

logger = logging.getLogger(__name__)

Base = declarative_base()

def _create_db_engine():
    db_url = settings.DATABASE_URL
    if db_url.startswith("sqlite"):
        logger.info("Using local SQLite database: stock_predictor.db")
        return create_engine(
            db_url,
            connect_args={"check_same_thread": False},
            echo=False
        )
    
    # Supabase PostgreSQL configuration with resilient error handling
    try:
        engine = create_engine(
            db_url,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
            echo=False
        )
        logger.info("Configured Supabase PostgreSQL engine.")
        return engine
    except Exception as e:
        logger.error(f"Failed to initialize PostgreSQL engine ({e}). Falling back to local SQLite.")
        return create_engine(
            "sqlite:///./stock_predictor.db",
            connect_args={"check_same_thread": False},
            echo=False
        )

engine = _create_db_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """
    FastAPI dependency that yields a SQLAlchemy database session.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """
    Creates all database tables defined in models.
    """
    try:
        from . import models
        Base.metadata.create_all(bind=engine)
        logger.info("Database schema synchronized.")
    except Exception as e:
        logger.warning(f"Database schema sync notice: {e}")
