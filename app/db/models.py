import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON
from .database import Base

class StockPriceRecord(Base):
    __tablename__ = "stock_prices"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), index=True, nullable=False)
    date = Column(String(30), index=True, nullable=False)
    open = Column(Float, nullable=True)
    high = Column(Float, nullable=True)
    low = Column(Float, nullable=True)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class SentimentRecord(Base):
    __tablename__ = "sentiment_records"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), index=True, nullable=False)
    title = Column(Text, nullable=False)
    published_date = Column(String(50), nullable=True)
    sentiment = Column(String(20), nullable=False)
    polarity = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class PredictionRun(Base):
    __tablename__ = "prediction_runs"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), index=True, nullable=False)
    model_type = Column(String(50), default="Multivariate-LSTM (Price+Sentiment)")
    time_step = Column(Integer, default=60)
    epochs = Column(Integer, default=15)
    train_rmse = Column(Float, nullable=False)
    test_rmse = Column(Float, nullable=False)
    directional_accuracy = Column(Float, nullable=False)
    judging_score = Column(Float, nullable=False)  # Annualized Sharpe ratio / excess risk-adjusted return
    forecast_30_days = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
