from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
from datetime import datetime

class StockHistoryItem(BaseModel):
    date: str
    close: float
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    volume: Optional[float] = None

class StockHistoryResponse(BaseModel):
    symbol: str
    count: int
    data: List[StockHistoryItem]

class SentimentItem(BaseModel):
    title: str
    published_date: Optional[str] = None
    sentiment: str
    polarity: float

class SentimentSummary(BaseModel):
    total_articles: int
    positive_count: int
    negative_count: int
    neutral_count: int
    average_polarity: float
    market_mood: str

class SentimentResponse(BaseModel):
    symbol: str
    summary: SentimentSummary
    articles: List[SentimentItem]

class PredictRequest(BaseModel):
    symbol: str = Field(default="^NSEI", description="Stock or Index ticker symbol")
    time_step: int = Field(default=60, ge=10, le=120, description="Lookback window (days)")
    epochs: int = Field(default=15, ge=1, le=50, description="Training epochs for LSTM")
    use_sentiment: bool = Field(default=True, description="Fuse news sentiment as a feature in LSTM")

class ForecastPoint(BaseModel):
    date: str
    predicted_close: float

class PredictResponse(BaseModel):
    symbol: str
    model_type: str
    train_rmse: float
    test_rmse: float
    directional_accuracy: float = Field(..., description="Percentage of days model correctly predicted price direction")
    judging_score: float = Field(..., description="Annualized Sharpe ratio of strategy vs benchmark")
    recent_actual: List[float]
    test_actual: List[float]
    test_predicted: List[float]
    forecast_30_days: List[ForecastPoint]
    saved_to_database: bool = True

class PredictionHistoryResponse(BaseModel):
    id: int
    symbol: str
    model_type: str
    train_rmse: float
    test_rmse: float
    directional_accuracy: float
    judging_score: float
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class HealthResponse(BaseModel):
    status: str
    database: str
    version: str
