# MarketPulse

A Python REST API for stock price forecasting and news sentiment analysis, built with FastAPI, SQLAlchemy, and a lightweight multivariate LSTM engine.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com)
[![Database](https://img.shields.io/badge/Database-Supabase%20PostgreSQL-3ECF8E.svg)](https://supabase.com/)
[![Tests](https://img.shields.io/badge/Tests-12%2F12%20Passed-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Live API Documentation: [https://marketpulse-api-eb4i.onrender.com/docs](https://marketpulse-api-eb4i.onrender.com/docs)

---

## Overview

MarketPulse is a backend web service designed to forecast short-term stock prices by combining two data streams:
1. **Historical market data**: End-of-day OHLCV prices fetched via Yahoo Finance.
2. **Market sentiment**: Recent financial news headlines scraped from Yahoo Finance RSS feeds and scored using TextBlob polarity analysis.

The system processes these features through a multivariate Long Short-Term Memory (LSTM) network to produce a 30-day forecast, evaluates performance against directional accuracy and Sharpe ratio metrics, and logs results to a Supabase PostgreSQL database.

---

## Architecture

The project is structured into distinct layers separating API routing, business logic, and database operations:

```mermaid
flowchart TD
    subgraph Client ["Client Layer"]
        Browser["Browser / HTTP Client"]
        Swagger["Swagger UI (/docs)"]
    end

    subgraph Gateway ["API Gateway (FastAPI)"]
        Router["Router (app/main.py)"]
        Validation["Pydantic Validation (app/schemas)"]
    end

    subgraph Services ["Service Layer"]
        StockService["Stock Service (yfinance)"]
        SentimentService["Sentiment Service (RSS + TextBlob)"]
        PredictorService["Predictor Service (NumPy LSTM)"]
        MetricsService["Metrics Service (Sharpe & Accuracy)"]
    end

    subgraph Persistence ["Persistence Layer"]
        ORM["SQLAlchemy 2.0 ORM"]
        Postgres[("Supabase PostgreSQL / SQLite Fallback")]
    end

    Browser --> Router
    Swagger --> Router
    Router --> Validation
    Validation --> StockService
    Validation --> SentimentService
    Validation --> PredictorService
    PredictorService --> MetricsService
    StockService --> ORM
    PredictorService --> ORM
    ORM --> Postgres
```

---

## Key Features

- **Multivariate Sequence Input**: Combines scaled historical closing prices with aligned daily news sentiment polarity vectors `[Price, Sentiment]`.
- **Lookahead Prevention**: The `MinMaxScaler` is fitted strictly on the training partition ($70\%$) to avoid data leakage into the evaluation partition ($30\%$).
- **Performance Metrics**:
  - **Directional Accuracy (%)**: Measures how often the model correctly predicts the sign of daily price changes (UP vs DOWN).
  - **Annualized Sharpe Ratio**: Evaluates risk-adjusted returns relative to a risk-free benchmark ($5\%$ annual rate).
  - **RMSE / MAE**: Calculated on unscaled currency values.
- **Database Persistence**: Automatic table creation and caching using Supabase PostgreSQL (via connection pooler) with automatic fallback to local SQLite when running offline.
- **Micro-Instance Optimization**: The recurrent forecasting model is implemented in NumPy with vectorized operations, reducing memory usage to under 35 MB RAM to operate reliably within cloud free-tier memory constraints (512 MB).
- **Automated Testing**: 12 unit and integration tests using `pytest` covering endpoints, risk metrics, and sentiment logic.

---

## Project Structure

```text
MarketPulse/
├── app/
│   ├── config.py                 # Application settings and database URL handling
│   ├── main.py                   # FastAPI routes, middleware, and lifecycle
│   ├── db/
│   │   ├── database.py           # SQLAlchemy engine and session management
│   │   └── models.py             # ORM models (prices, sentiment, prediction runs)
│   ├── schemas/
│   │   └── stock_schemas.py      # Pydantic request and response schemas
│   └── services/
│       ├── stock_service.py      # Price fetching and batch database caching
│       ├── sentiment_service.py  # RSS feed parsing and sentiment scoring
│       ├── metrics_service.py    # Directional accuracy and Sharpe calculations
│       └── predictor_service.py  # Lightweight Multivariate LSTM implementation
├── tests/
│   ├── test_api.py               # API endpoint integration tests
│   ├── test_metrics.py           # Metrics calculation tests
│   └── test_sentiment.py         # Sentiment analysis and alignment tests
├── .env.example                  # Environment configuration template
├── Procfile                      # Render / cloud process file
├── render.yaml                   # Infrastructure configuration
├── requirements.txt              # Project dependencies
├── Stock_Prediction.py           # Standalone CLI prediction script
└── README.md
```

---

## API Endpoints

Interactive documentation is available at `/docs` (Swagger UI) and `/redoc`.

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Service health status and database connection check. |
| `GET` | `/api/v1/stocks/{symbol}/history?limit=100` | Historical prices with database caching. |
| `GET` | `/api/v1/stocks/{symbol}/sentiment` | Recent news headlines with polarity scores. |
| `POST`| `/api/v1/stocks/predict` | Trains model and returns a 30-day forecast with evaluation metrics. |
| `GET` | `/api/v1/predictions/history` | List of past prediction runs stored in the database. |

### Example: POST /api/v1/stocks/predict

**Request:**
```json
{
  "symbol": "^NSEI",
  "time_step": 60,
  "epochs": 15,
  "use_sentiment": true
}
```

**Response:**
```json
{
  "symbol": "^NSEI",
  "model_type": "Multivariate-LSTM (Price + Sentiment)",
  "train_rmse": 412.18,
  "test_rmse": 769.75,
  "directional_accuracy": 51.26,
  "judging_score": 0.18,
  "forecast_30_days": [
    { "date": "2026-09-25", "predicted_close": 24714.96 },
    { "date": "2026-09-26", "predicted_close": 24838.59 },
    { "date": "2026-09-27", "predicted_close": 24928.43 }
  ],
  "saved_to_database": true
}
```

---

## Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/Askme007/MarketPulse.git
cd MarketPulse
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment variables (Optional)
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

If connecting to Supabase PostgreSQL, set `DATABASE_URL` in `.env`:
```env
DATABASE_URL=postgresql://postgres.[REF]:[PASSWORD]@aws-0-[REGION].pooler.supabase.com:6543/postgres
```
*If `DATABASE_URL` is omitted, the application defaults to local SQLite (`stock_predictor.db`).*

---

## Running the Application

### Start the API Server
```bash
uvicorn app.main:app --reload --port 8000
```
- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health Check: [http://localhost:8000/health](http://localhost:8000/health)

### Run the CLI Script
```bash
python Stock_Prediction.py AAPL
# or with indices:
python Stock_Prediction.py ^NSEI
```

### Run Tests
```bash
pytest
```

---

## Cloud Deployment (Render)

This repository includes a `Procfile` and `render.yaml` configured for Render web services:

1. Connect the GitHub repository in the Render dashboard.
2. Configure settings:
   - **Environment**: Python
   - **Build Command**: `pip install -r requirements.txt && python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger_eng')"`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
3. Add the `DATABASE_URL` environment variable pointing to your Supabase PostgreSQL pooler instance.

---

## Contributors

- **Ashkrit Rai** ([@Askme007](https://github.com/Askme007)) - Backend architecture, API design, database integration, test suite, and cloud deployment.
- **Navdeep** ([@NavdeepKakrod](https://github.com/NavdeepKakrod)) - Data analysis and exploratory modeling.
- **Abhishek Kumar** ([@Akabhi2311](https://github.com/Akabhi2311)) - Feature engineering and sentiment research.
- **Aayush Kumar** ([@Akcodet7](https://github.com/Akcodet7)) - Model experimentation and evaluation.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
