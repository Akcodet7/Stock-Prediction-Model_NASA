import logging
import yfinance as yf
import pandas as pd
from typing import Optional, List
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from app.db.models import StockPriceRecord

logger = logging.getLogger(__name__)

class StockService:
    def __init__(self):
        pass

    def get_stock_data(self, symbol: str, start_date: str = "2018-01-01", db: Optional[Session] = None) -> pd.DataFrame:
        """
        Downloads historical stock data from Yahoo Finance and formats it as a clean DataFrame.
        Optionally persists prices to the database (Supabase PostgreSQL / SQLite).
        """
        logger.info(f"Downloading historical data for {symbol} starting {start_date}")
        
        # Download from yfinance
        df = yf.download(symbol, start=start_date, auto_adjust=True, progress=False)

        if df.empty:
            # Try appending exchange suffix if Indian stock without suffix
            if not symbol.startswith("^") and not ("." in symbol):
                alt_symbol = f"{symbol}.NS"
                logger.info(f"Retrying download with alternate symbol: {alt_symbol}")
                df = yf.download(alt_symbol, start=start_date, auto_adjust=True, progress=False)
                if not df.empty:
                    symbol = alt_symbol

        if df.empty:
            raise ValueError(f"No price data found for ticker '{symbol}'. Please verify the symbol.")

        # Handle multi-level columns if returned by newer yfinance versions
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Standardize column names
        df = df.rename(columns={
            "Close": "close",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Volume": "volume"
        })

        # Fill any missing values forward then backward
        df = df.ffill().bfill()

        # Cache top 100 most recent records to database if db session provided
        if db is not None:
            try:
                self._cache_to_db(df, symbol, db)
            except Exception as e:
                logger.warning(f"Database caching skipped: {e}")

        return df

    def _cache_to_db(self, df: pd.DataFrame, symbol: str, db: Session, limit: int = 50):
        """
        Saves recent stock price records to the database without duplicates.
        """
        recent = df.tail(limit)
        for idx, row in recent.iterrows():
            date_str = idx.strftime("%Y-%m-%d")
            exists = db.query(StockPriceRecord).filter(
                StockPriceRecord.symbol == symbol,
                StockPriceRecord.date == date_str
            ).first()

            if not exists:
                rec = StockPriceRecord(
                    symbol=symbol,
                    date=date_str,
                    close=float(row["close"]),
                    open=float(row.get("open", row["close"])),
                    high=float(row.get("high", row["close"])),
                    low=float(row.get("low", row["close"])),
                    volume=float(row.get("volume", 0))
                )
                db.add(rec)
        db.commit()

stock_service = StockService()
