import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

from .metrics_service import calculate_metrics
from .stock_service import stock_service
from .sentiment_service import sentiment_service
from app.db.models import PredictionRun

logger = logging.getLogger(__name__)

# Device configuration (CUDA GPU if available, else CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class StockLSTM(nn.Module):
    """
    Stacked 2-Layer LSTM with Dropout and Dense Linear head for stock regression.
    Supports multivariate input (Price + Sentiment).
    """
    def __init__(self, input_size: int = 2, hidden_size: int = 50, num_layers: int = 2, dropout: float = 0.2):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        # Pass the last time-step hidden representation into the linear head
        return self.fc(out[:, -1, :])

class PredictorService:
    def __init__(self):
        pass

    def _create_sequences(self, data: np.ndarray, time_step: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates sliding window sequences for LSTM input.
        X shape: (N, time_step, num_features)
        y shape: (N, 1) -> target is next day's price (feature index 0)
        """
        X, y = [], []
        for i in range(len(data) - time_step):
            X.append(data[i : i + time_step, :])
            y.append(data[i + time_step, 0])
        return np.array(X), np.array(y).reshape(-1, 1)

    def train_and_forecast(
        self,
        symbol: str = "^NSEI",
        time_step: int = 60,
        epochs: int = 15,
        use_sentiment: bool = True,
        db: Optional[Session] = None
    ) -> Dict[str, Any]:
        """
        End-to-end pipeline:
        1. Fetch price & news sentiment.
        2. Preprocess features without data leakage.
        3. Train Multivariate LSTM in PyTorch.
        4. Evaluate with RMSE, Directional Accuracy, and Annualized Sharpe (Judging Score).
        5. Generate 30-day forward forecast.
        6. Persist execution results to Supabase/SQLite.
        """
        # 1. Download stock prices
        df = stock_service.get_stock_data(symbol, start_date="2018-01-01", db=db)
        prices = df["close"].values.astype(np.float32).reshape(-1, 1)
        dates = df.index

        # 2. Fetch and align news sentiment
        news_items = sentiment_service.fetch_news(symbol)
        articles, sentiment_summary = sentiment_service.analyze_sentiment(news_items)

        if use_sentiment:
            sentiment_series = sentiment_service.build_aligned_sentiment_series(dates, articles)
            feature_matrix = np.column_stack([prices, sentiment_series.reshape(-1, 1)])
            input_size = 2
            model_type = "Multivariate-LSTM (Price + Sentiment)"
        else:
            feature_matrix = prices
            input_size = 1
            model_type = "Univariate-LSTM (Price Only)"

        # 3. Train/Test split without data leakage
        train_ratio = 0.70
        train_len = int(len(feature_matrix) * train_ratio)

        train_raw = feature_matrix[:train_len]
        test_raw = feature_matrix[train_len - time_step:]  # include lookback window for continuous test predictions

        # Scaler is fitted STRICTLY on the training split
        price_scaler = MinMaxScaler(feature_range=(0, 1))
        train_scaled_price = price_scaler.fit_transform(train_raw[:, 0:1])
        test_scaled_price = price_scaler.transform(test_raw[:, 0:1])

        if use_sentiment:
            train_features = np.column_stack([train_scaled_price, train_raw[:, 1:2]])
            test_features = np.column_stack([test_scaled_price, test_raw[:, 1:2]])
        else:
            train_features = train_scaled_price
            test_features = test_scaled_price

        # 4. Generate sequences
        X_train, y_train = self._create_sequences(train_features, time_step)
        X_test, y_test = self._create_sequences(test_features, time_step)

        # Convert to PyTorch Tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32)
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        y_test_t = torch.tensor(y_test, dtype=torch.float32)

        train_loader = DataLoader(
            TensorDataset(X_train_t, y_train_t),
            batch_size=32,
            shuffle=True
        )

        # 5. Build and Train Model
        model = StockLSTM(input_size=input_size, hidden_size=50, num_layers=2, dropout=0.2).to(DEVICE)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-5)

        model.train()
        for epoch in range(epochs):
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                optimizer.zero_grad()
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()

        # 6. Evaluation and Inversion of Scaling
        model.eval()
        with torch.no_grad():
            train_pred_scaled = model(X_train_t.to(DEVICE)).cpu().numpy()
            test_pred_scaled = model(X_test_t.to(DEVICE)).cpu().numpy()

        train_pred = price_scaler.inverse_transform(train_pred_scaled)
        test_pred = price_scaler.inverse_transform(test_pred_scaled)
        y_train_actual = price_scaler.inverse_transform(y_train)
        y_test_actual = price_scaler.inverse_transform(y_test)

        train_metrics = calculate_metrics(y_train_actual, train_pred)
        test_metrics = calculate_metrics(y_test_actual, test_pred)

        # 7. Autoregressive 30-Day Forward Forecast
        forecast_points = []
        last_window = test_features[-time_step:].copy()
        current_date = dates[-1]

        avg_sentiment = float(sentiment_summary["average_polarity"])

        model.eval()
        with torch.no_grad():
            for day in range(1, 31):
                cur_input = torch.tensor(last_window.reshape(1, time_step, input_size), dtype=torch.float32).to(DEVICE)
                next_pred_scaled = model(cur_input).cpu().numpy()[0, 0]
                
                # Unscale price
                next_price = float(price_scaler.inverse_transform([[next_pred_scaled]])[0, 0])
                next_date = (current_date + timedelta(days=day)).strftime("%Y-%m-%d")

                forecast_points.append({
                    "date": next_date,
                    "predicted_close": round(next_price, 2)
                })

                # Roll forward window
                if use_sentiment:
                    # Rolling sentiment slowly decays toward neutral
                    decayed_sentiment = avg_sentiment * (0.95 ** day)
                    next_feature = np.array([next_pred_scaled, decayed_sentiment])
                else:
                    next_feature = np.array([next_pred_scaled])

                last_window = np.vstack([last_window[1:], next_feature])

        # Subsample actual vs predicted for frontend visualization (last 60 test points)
        sample_size = min(60, len(test_pred))
        recent_actual = [round(float(v), 2) for v in y_test_actual[-sample_size:].flatten()]
        test_predicted = [round(float(v), 2) for v in test_pred[-sample_size:].flatten()]

        # 8. Persist Prediction Run to Database
        saved_db = False
        if db is not None:
            try:
                run_rec = PredictionRun(
                    symbol=symbol,
                    model_type=model_type,
                    time_step=time_step,
                    epochs=epochs,
                    train_rmse=train_metrics["rmse"],
                    test_rmse=test_metrics["rmse"],
                    directional_accuracy=test_metrics["directional_accuracy"],
                    judging_score=test_metrics["judging_score"],
                    forecast_30_days=forecast_points
                )
                db.add(run_rec)
                db.commit()
                saved_db = True
            except Exception as e:
                logger.warning(f"Failed to persist prediction run to database: {e}")

        return {
            "symbol": symbol,
            "model_type": model_type,
            "train_rmse": train_metrics["rmse"],
            "test_rmse": test_metrics["rmse"],
            "directional_accuracy": test_metrics["directional_accuracy"],
            "judging_score": test_metrics["judging_score"],
            "recent_actual": recent_actual,
            "test_actual": recent_actual,
            "test_predicted": test_predicted,
            "forecast_30_days": forecast_points,
            "saved_to_database": saved_db
        }

predictor_service = PredictorService()
