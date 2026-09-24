#!/usr/bin/env python3
"""
Stock Price Prediction with Stacked LSTM & Sentiment Analysis (CLI Runner)
Author: Team NASA (Refactored for SDE Portfolio)
"""

import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Ensure app package is discoverable
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.services.predictor_service import predictor_service
from app.services.sentiment_service import sentiment_service

def main():
    stock_symbol = "^NSEI"  # Default symbol: Nifty 50 Index
    if len(sys.argv) > 1:
        stock_symbol = sys.argv[1]

    print("=" * 60)
    print(f"Executing Multivariate LSTM Prediction Pipeline for: {stock_symbol}")
    print("=" * 60)

    # 1. News Sentiment Analysis
    print("\n[Step 1] Fetching live news headlines and analyzing market sentiment...")
    news = sentiment_service.fetch_news(stock_symbol)
    articles, summary = sentiment_service.analyze_sentiment(news)
    print(f"Total Headlines Analyzed: {summary['total_articles']}")
    print(f"Market Mood: {summary['market_mood']} (Average Polarity: {summary['average_polarity']:+.3f})")
    print(f"Breakdown -> Positive: {summary['positive_count']}, Negative: {summary['negative_count']}, Neutral: {summary['neutral_count']}")

    # 2. Train Multivariate LSTM and generate forecast
    print("\n[Step 2] Training Stacked Multivariate LSTM (Price + Sentiment)...")
    results = predictor_service.train_and_forecast(
        symbol=stock_symbol,
        time_step=60,
        epochs=15,
        use_sentiment=True
    )

    print("\n[Step 3] Evaluation Metrics (Strict Train/Test Split Without Leakage):")
    print(f"  * Train RMSE            : {results['train_rmse']:.2f}")
    print(f"  * Test RMSE             : {results['test_rmse']:.2f}")
    print(f"  * Directional Accuracy  : {results['directional_accuracy']:.2f}%")
    print(f"  * Judging Score (Sharpe): {results['judging_score']:.2f}")

    print("\n[Step 4] 30-Day Forward Forecast:")
    forecast_df = pd.DataFrame(results['forecast_30_days'])
    print(forecast_df.head(10).to_string(index=False))
    print(f"... ({len(forecast_df)} days projected)")

    # 3. Save Summary Plot
    try:
        plt.figure(figsize=(12, 6))
        actual = results['test_actual']
        pred = results['test_predicted']
        plt.plot(actual, label='Actual Price', color='#2b5c8f', linewidth=2)
        plt.plot(pred, label='Predicted Price', color='#e05d44', linestyle='--', linewidth=2)
        plt.title(f"{stock_symbol} - Multivariate LSTM (Price + Sentiment) Validation", fontsize=14)
        plt.xlabel("Test Days", fontsize=12)
        plt.ylabel("Price", fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plot_path = "prediction_result.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n[Info] Evaluation chart saved to: {plot_path}")
    except Exception as e:
        print(f"Could not generate plot: {e}")

    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()