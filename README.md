# Air Quality Index (AQI) Predictor

An end-to-end machine learning system that forecasts the Air Quality Index for **Karachi** up to **3 days ahead**, built as part of the **10Pearls Shine Internship Program**.

The system collects live and historical air quality data, engineers features into a feature store, trains forecasting models, serves predictions through a REST API, and visualises them on a dashboard — with the data and training pipelines running on an automated schedule.

---

## 1. What Needs To Be Built (Full Scope)

| # | Component | Requirement |
|---|-----------|-------------|
| 1 | **Data collection** | Pull current + historical AQI, pollutant and weather readings from public APIs |
| 2 | **Feature store** | Persist model-ready feature rows in MongoDB, appended continuously over time |
| 3 | **EDA** | Explore trends, distributions, seasonality, missing data and correlations |
| 4 | **Feature engineering** | Lag features, rolling statistics, cyclical time encoding, derived ratios |
| 5 | **Model training** | Train and compare multiple regressors, evaluate with R² / RMSE / MAE, persist the best model |
| 6 | **Forecasting** | Iterative multi-step forecasting to produce hourly predictions for the next 72 hours, aggregated into 3 daily forecasts |
| 7 | **REST API** | FastAPI endpoints for current AQI, 3-day forecast, model metadata and health |
| 8 | **Dashboard** | Interactive frontend showing current AQI, category, history and forecast charts |
| 9 | **Explainability** | Feature importance / SHAP values so predictions are interpretable |
| 10 | **Automation** | GitHub Actions: hourly feature pipeline + scheduled retraining pipeline |
| 11 | **Deployment** | Host the API and dashboard publicly |

---

## 2. What Has Been Done So Far

### ✅ Backend skeleton (FastAPI)
- `app/main.py` — FastAPI app (`AQI Predictor API`, v1.0.0) with a root route.
- `app/routes/health.py` — `GET /health/` health check router.

### ✅ Live data ingestion (WAQI)
- `app/src/features/fetch_waqi.py` — fetches current AQI, pollutants (PM2.5, PM10, O₃, NO₂, SO₂, CO) and weather (temperature, humidity, pressure, wind speed) for any city from the WAQI API, with error handling for non-`ok` responses.

### ✅ Historical data ingestion (OpenWeather)
- `app/src/features/fetch_openweather.py` — geocodes a city name to lat/lon and pulls hourly historical pollutant readings from OpenWeather's Air Pollution History API.
- `app/src/features/aqi.py` — converts raw PM2.5 concentrations to a standard **US EPA AQI (0–500)** using official breakpoints, since OpenWeather returns concentrations rather than AQI.

### ✅ Feature building
- `app/src/features/build_features.py` — turns one raw reading into a model-ready row:
  - identifiers: `city`, `timestamp`
  - time features: `hour`, `day`, `month`, `day_of_week`
  - target: `aqi`
  - derived: `aqi_change_rate`
  - pollutants + weather: `pm25`, `pm10`, `o3`, `no2`, `so2`, `co`, `temperature`, `humidity`, `pressure`, `wind_speed`

### ✅ Feature store (MongoDB)
- `app/src/features/feature_store.py` — MongoDB Atlas connection with a lazily-initialised client; inserts feature rows into the `aqi_features` collection.

### ✅ Pipelines
- `app/src/features/pipeline.py` — CLI feature pipeline: fetch → build row → append to local CSV cache → insert into MongoDB.
- `app/src/features/backfill.py` — CLI backfill: geocode → fetch N days of history → compute AQI → insert all rows.
- **30 days of Karachi history has already been backfilled** into the feature store.

### ✅ Exploratory Data Analysis
- `app/notebooks/eda.ipynb` — loads the feature store into a DataFrame and covers:
  - missing-data fractions per column
  - AQI over time, per city
  - AQI distribution (histogram + KDE)
  - average AQI by hour of day and by day of week
  - correlation heatmap between pollutants/weather and AQI

---

## 3. What Is Remaining

### 🔜 Phase 1 — Data quality fixes (do before modelling)
- [ ] **AQI definition mismatch**: live WAQI rows use WAQI's own AQI, while backfilled rows compute AQI from PM2.5 only. Pick one definition and apply it consistently — otherwise the target variable has two different meanings.
- [ ] **Missing weather in backfilled rows**: `temperature`, `humidity`, `pressure`, `wind_speed` are `None` for all historical rows (OpenWeather's free tier has no historical weather API). Either source weather history elsewhere (e.g. Open-Meteo archive) or drop those columns from training.
- [ ] **Duplicate protection**: add a unique index on `(city, timestamp)` in MongoDB — re-running `backfill.py` currently inserts duplicate rows.
- [ ] **`aqi_change_rate` correctness**: it reads the *last* row of the local CSV for the city rather than the chronologically previous row; recompute it from sorted feature-store data instead of the CSV cache.
- [ ] Fix the stale comment in `pipeline.py` that still refers to a Hopsworks feature group (the store is MongoDB).

### 🔜 Phase 2 — Feature engineering
- [ ] Cyclical encoding of `hour` / `day` / `month` (sine/cosine) instead of raw integers.
- [ ] Lag features (AQI at t-1, t-3, t-24 hours).
- [ ] Rolling statistics (24h / 36h / 72h rolling mean and std for AQI and key pollutants).
- [ ] Derived features: PM2.5 / PM10 ratio, rush-hour flag, weekend/workday flag.
- [ ] Move feature engineering into a reusable module shared by training and inference (avoids train/serve skew).

### 🔜 Phase 3 — Model training
- [ ] `model_training.ipynb` — baseline (naive/persistence) plus Linear Regression, Random Forest, XGBoost / LightGBM.
- [ ] Time-based train/test split (not random) and time-series cross validation.
- [ ] Evaluate with R², RMSE, MAE; compare models and select the best.
- [ ] Persist the trained model + metrics + feature list (MongoDB or a `models/` artifact).
- [ ] Compute SHAP values at training time for dashboard explainability.

### 🔜 Phase 4 — Forecasting + API
- [ ] Iterative 72-hour forecasting function (predict → feed prediction back as input → repeat), aggregated into 3 daily forecasts.
- [ ] AQI category classification (Good / Moderate / Unhealthy / Hazardous) with colour codes.
- [ ] New endpoints:
  - `GET /predict` — current AQI, category, 3-day forecast, last 24h history, model metadata
  - `GET /shap-values` — feature contributions
  - `GET /debug` — model availability and data-range diagnostics
- [ ] Pydantic response models for each endpoint.

### 🔜 Phase 5 — Frontend
- [ ] Dashboard (Streamlit) with: current AQI card, 3-day forecast, hourly history chart, model metrics and SHAP plot.

### 🔜 Phase 6 — Automation & deployment
- [ ] `.github/workflows/feature_pipeline.yml` — run `pipeline.py` hourly, credentials via GitHub Secrets.
- [ ] `.github/workflows/training_pipeline.yml` — retrain on a daily schedule.
- [ ] Deploy the API and dashboard (e.g. Render) and document the live URLs.

### 🔜 Phase 7 — Engineering hygiene
- [ ] Add `.env.example`, tests for `aqi.py` / `build_features.py`, and logging in place of `print`.
- [ ] Add `scikit-learn`, `xgboost`, `shap`, `numpy`, `plotly`, `streamlit` to `requirements.txt` as those phases land.
- [ ] Remove the vendored `Affan Project/` reference repo from the working tree before final submission.

---

## 4. Project Structure

```
Air-Quality-Index-Predictor/
├── app/
│   ├── main.py                       # FastAPI application entry point
│   ├── routes/
│   │   └── health.py                 # /health endpoint
│   ├── notebooks/
│   │   └── eda.ipynb                 # Exploratory data analysis
│   └── src/
│       └── features/
│           ├── aqi.py                # PM2.5 -> EPA AQI conversion
│           ├── build_features.py     # Raw reading -> feature row
│           ├── fetch_waqi.py         # Live data source
│           ├── fetch_openweather.py  # Historical data source + geocoding
│           ├── feature_store.py      # MongoDB feature store writes
│           ├── pipeline.py           # Hourly feature pipeline (CLI)
│           └── backfill.py           # Historical backfill (CLI)
├── data/                             # Local CSV cache (git-ignored)
├── requirements.txt
├── .env                              # Secrets (git-ignored)
└── README.md
```

---

## 5. Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI + Uvicorn |
| Data sources | WAQI API (live), OpenWeather Air Pollution History (historical) |
| Feature store | MongoDB Atlas (`aqi_features` collection) |
| Data processing | Pandas |
| EDA / visualisation | Jupyter, Matplotlib, Seaborn |
| Config | python-dotenv |
| Planned | scikit-learn, XGBoost, SHAP, Streamlit, GitHub Actions |

---

## 6. Getting Started

### Prerequisites
- Python 3.10+
- A MongoDB Atlas cluster
- API keys for [WAQI](https://aqicn.org/data-platform/token/) and [OpenWeather](https://openweathermap.org/api)

### Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt
```

### Environment variables

Create a `.env` file in the project root:

```
WAQI_API_KEY=your_waqi_token
OPENWEATHER_API_KEY=your_openweather_key
MONGODB_URI=your_mongodb_connection_string
MONGODB_DB_NAME=aqi_predictor
```

### Usage

```bash
# Run the API (from the project root)
uvicorn app.main:app --reload
# -> http://127.0.0.1:8000        (Swagger docs at /docs)

# Fetch the current reading for a city and store it
python -m app.src.features.pipeline --city karachi

# Backfill the last 30 days of history for a city
python -m app.src.features.backfill --city karachi --days 30

# Open the EDA notebook
jupyter notebook app/notebooks/eda.ipynb
```

---

## 7. Feature Schema

| Field | Description |
|-------|-------------|
| `city` | City name as queried |
| `timestamp` | ISO-8601 reading time |
| `hour`, `day`, `month`, `day_of_week` | Time features derived from the timestamp |
| `aqi` | **Target** — Air Quality Index (0–500) |
| `aqi_change_rate` | Difference from the previous reading's AQI |
| `pm25`, `pm10`, `o3`, `no2`, `so2`, `co` | Pollutant readings |
| `temperature`, `humidity`, `pressure`, `wind_speed` | Weather readings (live rows only) |

---

## 8. Progress Summary

| Stage | Status |
|-------|--------|
| Backend skeleton | ✅ Done |
| Live data ingestion (WAQI) | ✅ Done |
| Historical ingestion + backfill (OpenWeather) | ✅ Done |
| MongoDB feature store | ✅ Done |
| Feature pipeline (CLI) | ✅ Done |
| Exploratory data analysis | ✅ Done |
| Data quality fixes | 🔜 Pending |
| Advanced feature engineering | 🔜 Pending |
| Model training & evaluation | 🔜 Pending |
| 3-day forecasting + prediction API | 🔜 Pending |
| Dashboard | 🔜 Pending |
| Explainability (SHAP) | 🔜 Pending |
| GitHub Actions automation | 🔜 Pending |
| Deployment | 🔜 Pending |
