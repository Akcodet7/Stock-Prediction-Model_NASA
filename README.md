# 📊 MarketPulse: Real-Time Financial Analytics & Multivariate Forecasting Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com)
[![NumPy](https://img.shields.io/badge/Engine-Zero--Allocation%20LSTM%20(NumPy)-013243.svg)](https://numpy.org/)
[![PostgreSQL](https://img.shields.io/badge/Database-Supabase%20PostgreSQL-3ECF8E.svg)](https://supabase.com/)
[![Tests](https://img.shields.io/badge/Tests-12%2F12%20Passed-brightgreen.svg)]()
[![Cloud Deployment](https://img.shields.io/badge/Cloud-Render%20Live-46E3B7.svg)](https://marketpulse-api-eb4i.onrender.com/docs)
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

> **Production Deployment**: [https://marketpulse-api-eb4i.onrender.com/docs](https://marketpulse-api-eb4i.onrender.com/docs)  
> Interactive OpenAPI / Swagger UI documentation deployed live with automated HTTPS.

---

## 📌 Executive Overview

**MarketPulse** is an enterprise-grade financial analytics and algorithmic forecasting backend service. It fuses **multivariate market sequences (OHLCV prices)** with **real-time financial news sentiment signals** to generate 30-day autoregressive forward projections.

The platform was architected to bridge the gap between academic data science prototypes and production software engineering:
* **Zero-Allocation Recurrent Engine**: Custom-engineered, vectorized Multivariate Long Short-Term Memory (LSTM) network in pure NumPy, operating at **<35 MB peak RAM** (a **94.5% memory reduction** over monolithic deep-learning runtimes).
* **High-Throughput Data Pipeline**: Resilient Yahoo Finance data extraction, sub-second RSS news scraping with timeout guards, and batch database caching.
* **Cloud Database Persistence**: Cloud-native persistence using **Supabase PostgreSQL** via IPv4 connection pooling, paired with **SQLAlchemy 2.0 ORM** and automatic local SQLite fallback.
* **Defensible Financial Metrics**: Evaluation via Directional Hit Rate (%) and Annualized Sharpe Ratio judging scores, with **strict mathematical separation between train and test splits** to eliminate lookahead data leakage.
* **Exhaustive Automated Testing**: Complete test coverage via `pytest` testing endpoints, sequence mathematics, and NLP polarity scoring.

---

## 🏛 System Architecture

The service adheres to clean layered architecture principles, ensuring strict separation of concerns between HTTP transport, domain validation, numerical computation, and database persistence:

```mermaid
flowchart TD
    subgraph Client Layer
        Web["Web Dashboards / Traders"]
        Docs["Interactive Swagger UI (/docs)"]
    end

    subgraph API Gateway Layer (FastAPI)
        Main["FastAPI Router (app/main.py)"]
        CORS["CORS Middleware"]
        Schemas["Pydantic V2 DTO Validation"]
    end

    subgraph Domain & Service Layer
        StockSvc["Stock Service\n(yfinance pipeline & batch cache)"]
        SentSvc["Sentiment Service\n(Yahoo RSS + TextBlob NLP)"]
        PredictSvc["Predictor Engine\n(Zero-Allocation NumPy LSTM)"]
        MetricSvc["Risk Metrics Engine\n(Sharpe Ratio & Directional Hit Rate)"]
    end

    subgraph Persistence Layer
        SQLA["SQLAlchemy 2.0 ORM"]
        Pooler["Supabase Session Pooler (IPv4:6543)"]
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
    SQLA --> Pooler
    Pooler --> SupaDB
```

---

## ⚡ Cloud Micro-Instance Optimization & Engineering Challenges

Deploying machine-learning backends to cloud micro-instances (such as Render's free tier with **0.1 vCPU and 512 MB RAM**) presents critical systems constraints that typically cause container terminations. MarketPulse was specifically engineered to overcome these bottlenecks:

### 1. Eliminating 512 MB RAM OOM (Out Of Memory) Crashes
* **The Problem**: Standard deep learning frameworks (PyTorch, TensorFlow) allocate heavy BLAS/MKL shared-library buffers and Adam optimizer momentum states. Profiling revealed baseline imports consumed **443 MB**, spiking past **550–960 MB** during training and triggering Linux kernel `SIGKILL (OOM 137)` termination.
* **The Solution**: We engineered a custom, vectorized **Multivariate LSTM in pure NumPy** (`NumpyMultivariateLSTM`) featuring 4-gate recurrent mechanics ($f_t, i_t, \tilde{C}_t, o_t$), Momentum SGD with Backpropagation Through Time (BPTT), and gradient clipping.
* **Result**: Peak RAM dropped from **552 MB to 30.2 MB** (**94.5% memory reduction**), training completes in **~3.2 seconds**, and the monolithic 900 MB wheel was removed from dependencies—slashing deployment build times from 4 minutes to 25 seconds.

### 2. Eliminating 100-Second Gateway Timeouts (HTTP 502)
* **The Problem**: Synchronous news RSS scrapers lacked socket timeouts, hanging indefinitely on external rate limits. Additionally, historical price caching ran 50 sequential SQL queries in a loop over public internet connections to Supabase ($N+1$ query latency).
* **The Solution**:
  1. Wrapped RSS requests in `requests.get(..., timeout=2.5)` with instant fallback to neutral sentiment if external feeds degrade.
  2. Replaced loop queries with a single batch `IN` query to check and insert missing dates in one round-trip.
  3. Windowed historical training to the most recent 400 trading days (~1.5 years of market momentum).

| Metric | Monolithic Framework (PyTorch) | Engineered Engine (NumPy LSTM) | Improvement |
| :--- | :--- | :--- | :--- |
| **Peak RAM Allocation** | 552 MB – 962 MB *(OOM Crash)* | **30.2 MB** | **94.5% reduction** |
| **Cloud Sizing Feasibility** | Exceeds 512 MB Free Tier | Fits comfortably with **400+ MB headroom** | **100% stable** |
| **Execution Latency** | Gateway Timeout (>100s) | **3.5 – 5.2 seconds** | **~25x faster** |
| **Cloud Build Duration** | ~4 minutes (900MB wheel download) | **~25 seconds** | **90% build speedup** |

---

## 🚀 Key Features & Implementation Rigor

### 1. True Multivariate Feature Fusion
The model does not rely on price alone. It fuses daily normalized closing prices with aligned news sentiment polarity vectors:
$$\mathbf{X}_t = \begin{bmatrix} \text{Scaled\_Close}_t \\ \text{Sentiment\_Polarity}_t \end{bmatrix}$$
During 30-day autoregressive forward forecasting, sentiment scores decay smoothly toward neutral baseline ($0.95^{\text{day}}$), accurately reflecting market information half-life.

### 2. Zero Lookahead Data Leakage
Unlike naive implementations that normalize entire datasets prior to splitting, MarketPulse enforces strict temporal discipline:
* The `MinMaxScaler` is fitted **strictly on the historical training partition** ($70\%$).
* The test partition ($30\%$) is transformed using training distribution parameters.
* Test sequences preserve the lookback window without leaking future target prices.

### 3. Quantitative Financial Metrics
* **Directional Hit Rate (%)**: Evaluates actual trade utility by calculating the percentage of sessions where predicted price direction matches market movement:
  $$\text{Directional Accuracy} = \frac{1}{N-1} \sum_{t=1}^{N-1} \mathbb{I}\left(\operatorname{sgn}(\hat{y}_{t+1} - y_t) == \operatorname{sgn}(y_{t+1} - y_t)\right) \times 100\%$$
* **Annualized Sharpe Ratio (Judging Score)**: Calculates risk-adjusted excess returns over an annualized risk-free rate ($5\%$):
  $$\text{Sharpe Ratio} = \frac{\bar{R}_{\text{strategy}} - R_f}{\sigma_{\text{strategy}}} \times \sqrt{252}$$
* **RMSE & MAE**: Unscaled error measures returned in native currency units ($ / ₹).

---

## 📁 Project Directory Structure

```text
MarketPulse/
├── app/
│   ├── config.py                 # Pydantic Settings & DB URL normalization
│   ├── main.py                   # FastAPI application, CORS, Swagger UI & lifecycle
│   ├── db/
│   │   ├── database.py           # SQLAlchemy 2.0 pool config (Supabase / SQLite)
│   │   └── models.py             # ORM models (StockPriceRecord, PredictionRun, etc.)
│   ├── schemas/
│   │   └── stock_schemas.py      # Pydantic V2 validation DTOs & response contracts
│   └── services/
│       ├── stock_service.py      # Yahoo Finance fetcher & batch DB cache
│       ├── sentiment_service.py  # RSS feedparser, timeout guards & TextBlob polarity
│       ├── metrics_service.py    # Directional Hit Rate & Annualized Sharpe Ratio
│       └── predictor_service.py  # Zero-Allocation Multivariate LSTM Engine (<35MB RAM)
├── tests/
│   ├── test_api.py               # FastAPI TestClient endpoint integration tests
│   ├── test_metrics.py           # Quantitative metrics & financial math unit tests
│   └── test_sentiment.py         # NLP scoring & temporal series alignment tests
├── .env.example                  # Environment configuration template
├── Procfile                      # Cloud process worker specification
├── render.yaml                   # 1-click cloud infrastructure blueprint
├── requirements.txt              # Production dependency specifications
├── Stock_Prediction.py           # Clean CLI terminal runner
└── README.md                     # Comprehensive project documentation
```

---

## 📡 REST API Reference

The interactive OpenAPI documentation is accessible at `/docs`.

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Cloud health check verifying PostgreSQL database connectivity. |
| `GET` | `/api/v1/stocks/{symbol}/history?limit=100` | Historical OHLCV market data with automated database caching. |
| `GET` | `/api/v1/stocks/{symbol}/sentiment` | Live RSS headline extraction with TextBlob polarity scoring. |
| `POST`| `/api/v1/stocks/predict` | Executes Multivariate LSTM training & 30-day forecast generation. |
| `GET` | `/api/v1/predictions/history` | Historical prediction runs, risk metrics, and audit records from DB. |

---

### Request & Response Examples

#### 1. Execute Multivariate Prediction: `POST /api/v1/stocks/predict`
```bash
curl -X 'POST' \
  'https://marketpulse-api-eb4i.onrender.com/api/v1/stocks/predict' \
  -H 'Content-Type: application/json' \
  -d '{
    "symbol": "^NSEI",
    "time_step": 60,
    "epochs": 15,
    "use_sentiment": true
  }'
```

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

#### 2. Query Financial News Sentiment: `GET /api/v1/stocks/{symbol}/sentiment`
```json
{
  "symbol": "AAPL",
  "summary": {
    "total_articles": 8,
    "average_polarity": 0.184,
    "overall_sentiment": "Positive"
  },
  "articles": [
    {
      "title": "Apple Expands AI Integration Across Ecosystem",
      "published_date": "Thu, 24 Sep 2026 14:30:00 GMT",
      "sentiment": "Positive",
      "polarity": 0.35
    }
  ]
}
```

#### 3. Service Health Check: `GET /health`
```json
{
  "status": "healthy",
  "database": "Supabase PostgreSQL (postgres)",
  "version": "2.0.0"
}
```

---

## 🛠 Installation & Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Askme007/MarketPulse.git
cd MarketPulse
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Database Credentials (Supabase PostgreSQL)
1. Create a free project on [Supabase](https://supabase.com).
2. Retrieve your **Connection Pooler** URI (**Session mode**, Port `6543`, IPv4 compatible):
   ```env
   DATABASE_URL=postgresql://postgres.[REF]:[PASSWORD]@aws-0-[REGION].pooler.supabase.com:6543/postgres
   ```
3. Copy `.env.example` to `.env` and assign your connection string:
   ```bash
   cp .env.example .env
   ```
> *Zero-Config Fallback*: If `DATABASE_URL` is omitted, MarketPulse automatically initializes a local `stock_predictor.db` SQLite database with the identical schema.

---

## 🖥 Running Locally

### Option A: Launch the FastAPI REST Server
```bash
uvicorn app.main:app --reload --port 8000
```
* **Interactive Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
* **Alternative Redoc Documentation**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### Option B: Execute Terminal CLI Runner
```bash
python Stock_Prediction.py AAPL
# Or test indices:
python Stock_Prediction.py ^NSEI
```

### Option C: Execute Automated Test Suite
```bash
pytest
```
*Executes all 12 unit and integration tests across endpoints, metrics, and sentiment pipelines.*

---

## ☁️ Cloud Deployment Guide (Render / Railway)

This repository includes a production `Procfile` and `render.yaml` for zero-Docker cloud hosting:

1. Push your repository to **GitHub**.
2. On [Render](https://render.com), create a new **Web Service** connected to your repository.
3. Configure the service:
   * **Runtime**: `Python 3`
   * **Build Command**: `pip install -r requirements.txt && python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger_eng')"`
   * **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. Under **Environment Variables**, provide your Supabase session pooler connection string:
   * `DATABASE_URL`: `postgresql://postgres.[REF]:[PASSWORD]@aws-0-[REGION].pooler.supabase.com:6543/postgres`
5. Click **Deploy**. The service will build in ~25 seconds and launch with automated HTTPS.

---

## 👥 Engineering & Collaboration Breakdown

This project was developed through collaborative engineering:

* **Ashkrit Rai ([@Askme007](https://github.com/Askme007))** — *Lead Software Development Engineer (SDE)*  
  Engineered the FastAPI production backend, zero-allocation NumPy LSTM recurrent engine, database architecture with Supabase connection pooling, automated `pytest` suite, latency/timeout optimizations, and cloud deployment.
* **Collaborators ([@NavdeepKakrod](https://github.com/NavdeepKakrod), [@Akabhi2311](https://github.com/Akabhi2311), [@Akcodet7](https://github.com/Akcodet7))** — *Data Science & Research*  
  Exploratory financial data analysis, feature engineering experiments, sentiment lexicon research, and initial notebook prototyping.

---

## 📄 License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
