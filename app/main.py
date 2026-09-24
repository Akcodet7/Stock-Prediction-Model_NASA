import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional

from app.config import settings
from app.db.database import get_db, init_db, engine
from app.db.models import PredictionRun, StockPriceRecord
from app.schemas.stock_schemas import (
    StockHistoryResponse,
    StockHistoryItem,
    SentimentResponse,
    SentimentSummary,
    SentimentItem,
    PredictRequest,
    PredictResponse,
    PredictionHistoryResponse,
    HealthResponse
)
from app.services.stock_service import stock_service
from app.services.sentiment_service import sentiment_service
from app.services.predictor_service import predictor_service, DEVICE

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Initialize database tables
try:
    init_db()
except Exception as e:
    logger.warning(f"Initial DB sync notice: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensures database tables are synced on server startup."""
    try:
        init_db()
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
    yield

# FastAPI Application Definition
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    lifespan=lifespan,
    description="""
    ## Stock Price Prediction & Sentiment Analysis API
    Production-grade financial engineering backend combining:
    - **Multivariate Stacked LSTM** (fusing historical closing price + news sentiment).
    - **Live Sentiment Analysis** with TextBlob on RSS & Yahoo Finance news feeds.
    - **Supabase PostgreSQL** cloud persistence for predictions and stock cache.
    - **Financial Risk Metrics**: Directional Accuracy (%) and Annualized Sharpe Ratio Judging Score.
    """
)

# CORS Middleware (allows web dashboards / Swagger UI access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.api_route("/", methods=["GET", "HEAD"], tags=["General"])
def root():
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs_url": "/docs",
        "health_check": "/health"
    }

@app.api_route("/health", methods=["GET", "HEAD"], response_model=HealthResponse, tags=["General"])
def health_check():
    """Checks the health of the API, Database connection, and ML compute engine."""
    db_type = "Supabase PostgreSQL" if "postgresql" in settings.DATABASE_URL else "Local SQLite"
    return HealthResponse(
        status="healthy",
        database=f"{db_type} ({engine.url.database})",
        version=settings.VERSION
    )

@app.get(
    f"{settings.API_V1_PREFIX}/stocks/{{symbol}}/history",
    response_model=StockHistoryResponse,
    tags=["Stock Data"]
)
def get_stock_history(
    symbol: str,
    limit: int = Query(default=100, ge=10, le=1000),
    db: Session = Depends(get_db)
):
    """
    Fetches historical stock prices and caches the most recent data points.
    """
    try:
        df = stock_service.get_stock_data(symbol=symbol, db=db)
        recent_df = df.tail(limit)

        data_items = []
        for idx, row in recent_df.iterrows():
            data_items.append(StockHistoryItem(
                date=idx.strftime("%Y-%m-%d"),
                close=round(float(row["close"]), 2),
                open=round(float(row.get("open", row["close"])), 2),
                high=round(float(row.get("high", row["close"])), 2),
                low=round(float(row.get("low", row["close"])), 2),
                volume=float(row.get("volume", 0))
            ))

        return StockHistoryResponse(
            symbol=symbol,
            count=len(data_items),
            data=data_items
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error fetching history for {symbol}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error fetching stock history.")

@app.get(
    f"{settings.API_V1_PREFIX}/stocks/{{symbol}}/sentiment",
    response_model=SentimentResponse,
    tags=["Sentiment Analysis"]
)
def get_stock_sentiment(symbol: str):
    """
    Scrapes current headlines and calculates market sentiment using TextBlob.
    """
    try:
        news = sentiment_service.fetch_news(symbol)
        articles, summary = sentiment_service.analyze_sentiment(news)

        article_items = [
            SentimentItem(
                title=a["title"],
                published_date=a.get("published_date"),
                sentiment=a["sentiment"],
                polarity=a["polarity"]
            )
            for a in articles
        ]

        summary_model = SentimentSummary(**summary)

        return SentimentResponse(
            symbol=symbol,
            summary=summary_model,
            articles=article_items
        )
    except Exception as e:
        logger.error(f"Error analyzing sentiment for {symbol}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve sentiment analysis.")

@app.post(
    f"{settings.API_V1_PREFIX}/stocks/predict",
    response_model=PredictResponse,
    tags=["Prediction Engine"]
)
def predict_stock(request: PredictRequest, db: Session = Depends(get_db)):
    """
    Executes the Multivariate LSTM model training and generates a 30-day forecast.
    Fuses historical price sequences with news sentiment signals.
    Persists the run record to the Supabase PostgreSQL database.
    """
    try:
        results = predictor_service.train_and_forecast(
            symbol=request.symbol,
            time_step=request.time_step,
            epochs=request.epochs,
            use_sentiment=request.use_sentiment,
            db=db
        )
        return PredictResponse(**results)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction failed for {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction pipeline encountered an error: {str(e)}")

@app.get(
    f"{settings.API_V1_PREFIX}/predictions/history",
    response_model=List[PredictionHistoryResponse],
    tags=["Prediction Engine"]
)
def get_prediction_history(
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """
    Retrieves previous model training runs and their performance metrics from the database.
    """
    runs = db.query(PredictionRun).order_by(PredictionRun.created_at.desc()).limit(limit).all()
    return runs

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
