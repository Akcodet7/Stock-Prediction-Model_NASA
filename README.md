# 📈 AlphaForecast: Multivariate Stock Prediction & Sentiment Analysis Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![PostgreSQL](https://img.shields.io/badge/Database-Supabase%20PostgreSQL-3ECF8E.svg)](https://supabase.com/)
[![Tests](https://img.shields.io/badge/Tests-12%2F12%20Passed-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

A production-ready financial engineering platform combining **Deep Learning (Multivariate Stacked LSTM)** with **Natural Language Processing (TextBlob Sentiment Analysis)**, backed by **Supabase PostgreSQL** and served via **FastAPI** with interactive Swagger documentation.

---

## 🏛 System Architecture

The platform separates concerns across a layered microservice architecture:

```mermaid
flowchart TD
    subgraph Client Layer
        Web["Web Client / Browser"]
        Docs["Interactive Swagger UI (/docs)"]
    end

    subgraph API & Routing Layer (FastAPI)
        Main["FastAPI Gateway (app/main.py)"]
        CORS["CORS Middleware"]
        Schemas["Pydantic V2 DTO Validation"]
    end

    subgraph Service Layer (Business Logic)
        StockSvc["Stock Service (yfinance data pipeline)"]
        SentSvc["Sentiment Service (RSS / TextBlob Polarity)"]
        PredictSvc["Predictor Service (Multivariate LSTM in PyTorch)"]
        MetricSvc["Metrics Service (Sharpe Ratio & Directional Accuracy)"]
    end

    subgraph Persistence Layer
        SQLA["SQLAlchemy ORM 2.0"]
        SupaDB[("Supabase PostgreSQL Cloud\n(Fallback: Local SQLite)")]
    end

    Web --> Main
    Docs --> Main
    Main --> CORS
    CORS --> Schemas
    Schemas --> StockSvc
    Schemas --> SentSvc
    Schemas --> PredictSvc
    PredictSvc --> MetricSvc
    StockSvc --> SQLA
    PredictSvc --> SQLA
    SQLA --> SupaDB
```

---

## 🚀 Key Features & Engineering Highlights

- **True Multivariate Feature Fusion**: Fuses historical closing prices with aligned daily news sentiment polarity vectors `[Scaled_Close, Sentiment_Polarity]` into a 2-layer Stacked LSTM.
- **Zero Lookahead Data Leakage**: `MinMaxScaler` is fit *strictly* on the historical training partition ($70\%$), preventing future data leakage into training features.
- **Defensible Financial Metrics**:
  - **Directional Accuracy (%)**: Measures the percentage of trading sessions where the model correctly anticipates price direction (UP vs DOWN).
  - **Annualized Sharpe Ratio (Judging Score)**: Calculates excess return over a benchmark risk-free rate divided by return volatility:
    $$\text{Judging Score} = \frac{\bar{R}_{\text{strategy}} - R_f}{\sigma_{\text{strategy}}} \times \sqrt{252}$$
  - **RMSE & MAE**: Unscaled error in base currency units ($/₹).
- **Cloud Database (Supabase PostgreSQL)**:
  - Automated table provisioning (`stocks_prices`, `sentiment_records`, `prediction_runs`).
  - Graceful zero-config fallback to local SQLite for offline development.
- **Automated Testing Suite**: 12 unit and integration tests using `pytest` covering endpoints, sequence math, and NLP scoring.
- **Deployment Ready (No Docker Required)**: Out-of-the-box configuration for instant 1-click cloud deployment on **Render** or **Railway**.

---

## 📁 Project Directory Structure

```text
Stock-Prediction-Model_NASA/
├── app/
│   ├── config.py                 # Environment configurations & Supabase credentials
│   ├── main.py                   # FastAPI application & REST route definitions
│   ├── db/
│   │   ├── database.py           # SQLAlchemy connection pool (Supabase / SQLite)
│   │   └── models.py             # ORM models (StockPriceRecord, PredictionRun, etc.)
│   ├── schemas/
│   │   └── stock_schemas.py      # Pydantic V2 request & response schemas
│   └── services/
│       ├── stock_service.py      # Yahoo Finance fetcher & DB price cache
│       ├── sentiment_service.py  # RSS feedparser & TextBlob sentiment pipeline
│       ├── metrics_service.py    # Directional Accuracy & Sharpe Ratio calculations
│       └── predictor_service.py  # PyTorch Multivariate LSTM train & forecast loop
├── tests/
│   ├── test_api.py               # FastAPI TestClient endpoint integration tests
│   ├── test_metrics.py           # Quantitative metrics unit tests
│   └── test_sentiment.py         # TextBlob sentiment & series alignment tests
├── .env.example                  # Environment template for Supabase
├── Procfile                      # Cloud process definition (Render / Railway)
├── render.yaml                   # 1-click Render blueprint specification
├── requirements.txt              # Production dependency specifications
├── Stock_Prediction.py           # Clean CLI runner
└── README.md
```

---

## 🛠 Installation & Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Askme007/Stock-Prediction-Model_NASA.git
cd Stock-Prediction-Model_NASA
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Supabase PostgreSQL (Optional but Recommended)
1. Create a free database on [Supabase](https://supabase.com).
2. Go to **Project Settings** $\rightarrow$ **Database** $\rightarrow$ **Connection String** $\rightarrow$ **URI**.
3. Create a `.env` file from the example:
   ```bash
   cp .env.example .env
   ```
4. Paste your connection URI:
   ```env
   DATABASE_URL=postgresql://postgres:[YOUR-PASSWORD]@db.[PROJECT-REF].supabase.co:5432/postgres
   ```
> *Note: If left empty, the application automatically uses a local `stock_predictor.db` SQLite database.*

---

## 🖥 Running the Application

### Option A: Run the FastAPI REST Server
```bash
uvicorn app.main:app --reload --port 8000
```
Open your browser and navigate to:
- **Interactive Swagger Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **API Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

### Option B: Run the CLI Prediction Script
```bash
python Stock_Prediction.py ^NSEI
```
*(You can pass any ticker: `^NSEI`, `AAPL`, `MSFT`, `RELIANCE.NS`, etc.)*

### Option C: Run the Automated Test Suite
```bash
pytest
```

---

## 📡 REST API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Check API status, database engine, and compute availability. |
| `GET` | `/api/v1/stocks/{symbol}/history?limit=100` | Fetch historical OHLCV quotes with DB caching. |
| `GET` | `/api/v1/stocks/{symbol}/sentiment` | Scrape recent news headlines and compute sentiment polarity. |
| `POST`| `/api/v1/stocks/predict` | Train Multivariate LSTM, compute metrics, and return 30-day forecast. |
| `GET` | `/api/v1/predictions/history` | Retrieve historical model runs and performance records from Supabase. |

#### Example Request: `POST /api/v1/stocks/predict`
```json
{
  "symbol": "^NSEI",
  "time_step": 60,
  "epochs": 15,
  "use_sentiment": true
}
```

#### Example Response:
```json
{
  "symbol": "^NSEI",
  "model_type": "Multivariate-LSTM (Price + Sentiment)",
  "train_rmse": 392.48,
  "test_rmse": 1333.17,
  "directional_accuracy": 50.85,
  "judging_score": -1.10,
  "forecast_30_days": [
    {"date": "2026-09-25", "predicted_close": 22266.14},
    {"date": "2026-09-26", "predicted_close": 22164.93}
  ],
  "saved_to_database": true
}
```

---

## ☁️ Zero-Docker Cloud Deployment (Render / Railway)

This repository includes a `Procfile` and `render.yaml` for instant cloud deployment without needing Docker:

### Deploying to Render:
1. Push your repository to **GitHub**.
2. Log into [Render.com](https://render.com) and click **New +** $\rightarrow$ **Web Service**.
3. Connect your GitHub repository.
4. Select runtime **Python 3**.
5. Set:
   - **Build Command**: `pip install -r requirements.txt && python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger_eng')"`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
6. Under **Environment Variables**, add:
   - `DATABASE_URL`: *(Your Supabase PostgreSQL URI)*
7. Click **Create Web Service**. Your live interactive API and Swagger UI will be deployed with free automatic HTTPS!

---

## 👥 Contributors

- **Ashkrit Rai** ([@Askme007](https://github.com/Askme007))
- **Navdeep** ([@NavdeepKakrod](https://github.com/NavdeepKakrod))
- **Abhishek Kumar** ([@Akabhi2311](https://github.com/Akabhi2311))
- **Aayush Kumar** ([@Akcodet7](https://github.com/Akcodet7))

---

## 📄 License
This project is licensed under the MIT License.
