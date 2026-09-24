import pytest
import pandas as pd
import numpy as np
from app.services.sentiment_service import sentiment_service

def test_analyze_sentiment_polarity():
    sample_news = [
        {"title": "Company reports record profit and stellar revenue growth", "date": "2026-09-01"},
        {"title": "Stock market suffers terrible decline and bad losses", "date": "2026-09-02"},
        {"title": "Federal reserve maintains policy rate unchanged at meeting", "date": "2026-09-03"}
    ]
    
    articles, summary = sentiment_service.analyze_sentiment(sample_news)
    
    assert len(articles) == 3
    assert articles[0]["sentiment"] == "positive"
    assert articles[1]["sentiment"] == "negative"
    assert summary["positive_count"] == 1
    assert summary["negative_count"] == 1
    assert summary["neutral_count"] == 1

def test_build_aligned_sentiment_series():
    dates = pd.date_range("2026-09-01", periods=10, freq="D")
    sample_news = [
        {"title": "Great rally for tech stocks", "date": "2026-09-05", "polarity": 0.8}
    ]
    
    series = sentiment_service.build_aligned_sentiment_series(dates, sample_news)
    assert len(series) == 10
    assert isinstance(series, np.ndarray)
