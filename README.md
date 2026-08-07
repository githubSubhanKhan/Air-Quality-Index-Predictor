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

### ✅ Historical weather backfill (Open-Meteo)
- `app/src/features/fetch_openmeteo.py` — pulls hourly historical `temperature`, `humidity`, `pressure` and `wind_speed` from Open-Meteo's free Archive API (keyed by UNIX timestamp).
- `app/src/features/backfill.py` — now merges Open-Meteo weather into every historical row, fixing the previous gap where backfilled rows had no weather data.
- **365 days of Karachi history** has been backfilled into the feature store (up from the initial 30).

### ✅ Feature engineering (lag + rolling)
- `app/notebooks/lag_feature_engineering.ipynb` builds the model-ready feature set on top of the raw feature store:
  - AQI lags: `aqi_lag_1/3/6/12/24`
  - PM2.5 lags: `pm25_lag_1/6/24`
  - Rolling AQI stats: `aqi_roll_mean_6/12/24`, `aqi_roll_std_24`
  - Combined with calendar, pollutant and weather fields into a 26-column `FEATURES` set used for training.

### ✅ Model training (24h-ahead AQI)
- `app/notebooks/lag_feature_engineering.ipynb` trains and compares 5 regressors — Linear Regression, Ridge, Random Forest, XGBoost, LightGBM — on a time-based (non-shuffled) 80/20 split, evaluated with MAE / RMSE / R².
- Trained models are persisted with `joblib` to `app/models/` (git-ignored): `linear_regression_lag.pkl`, `ridge_regression.pkl`, `random_forest_lag.pkl`, `xgboost.pkl`, `lightgbm.pkl`.
- Best performer: **XGBoost** (MAE 8.88, RMSE 11.85, R² 0.46), followed by LightGBM and Random Forest; the linear models underperform (negative R²).

### ✅ Day-wise 3-day forecasting
- `app/notebooks/lag_feature_engineering_3_days.ipynb` — instead of iterative single-step forecasting, trains **three independent XGBoost regressors**, each predicting the *mean* AQI over a future window: `target_day1` (next 1–24h), `target_day2` (25–48h), `target_day3` (49–72h).
- Saved to `app/models/xgboost_day1.pkl`, `xgboost_day2.pkl`, `xgboost_day3.pkl`.
- Accuracy degrades sharply with horizon: Day 1 R² = 0.70 (MAE 6.52), Day 2 R² = 0.22 (MAE 10.88), Day 3 R² = -0.05 (MAE 12.96) — the 3-day-ahead model currently performs no better than predicting the mean.

---

## 3. What Is Remaining

### 🔜 Phase 1 — Data quality fixes (do before modelling)
- [ ] **AQI definition mismatch**: live WAQI rows use WAQI's own AQI, while backfilled rows compute AQI from PM2.5 only. Pick one definition and apply it consistently — otherwise the target variable has two different meanings.
- [x] ~~Missing weather in backfilled rows~~ — fixed via Open-Meteo (`fetch_openmeteo.py`); historical rows now carry `temperature`, `humidity`, `pressure`, `wind_speed`.
- [ ] **Duplicate protection**: add a unique index on `(city, timestamp)` in MongoDB — re-running `backfill.py` currently inserts duplicate rows.
- [ ] **`aqi_change_rate` correctness**: it reads the *last* row of the local CSV for the city rather than the chronologically previous row; recompute it from sorted feature-store data instead of the CSV cache.
- [ ] Fix the stale comment in `pipeline.py` that still refers to a Hopsworks feature group (the store is MongoDB).

### 🔜 Phase 2 — Feature engineering
- [ ] Cyclical encoding of `hour` / `day` / `month` (sine/cosine) instead of raw integers.
- [x] ~~Lag features~~ — AQI lags (1/3/6/12/24h) and PM2.5 lags (1/6/24h) added in `lag_feature_engineering.ipynb`.
- [x] ~~Rolling statistics~~ — 6/12/24h rolling mean and 24h rolling std for AQI added.
- [ ] Derived features: PM2.5 / PM10 ratio, rush-hour flag, weekend/workday flag.
- [ ] Move feature engineering into a reusable module shared by training and inference (currently only exists inline in the notebooks — training/serving will skew until this lands).

### 🔜 Phase 3 — Model training
- [x] ~~Train and compare multiple regressors~~ — Linear Regression, Ridge, Random Forest, XGBoost and LightGBM trained and compared in `lag_feature_engineering.ipynb`.
- [x] ~~Time-based train/test split~~ — 80/20 chronological split (no shuffling); no time-series cross-validation yet.
- [x] ~~Evaluate with R², RMSE, MAE~~ — best is XGBoost (MAE 8.88, RMSE 11.85, R² 0.46).
- [x] ~~Persist the trained model~~ — saved as `.pkl` files via `joblib` to `app/models/` (git-ignored, not yet uploaded anywhere durable).
- [ ] No baseline (naive/persistence) model has been benchmarked against the trained models yet.
- [ ] Compute SHAP values at training time for dashboard explainability.
- [ ] Track metrics/feature list alongside the model artifact (currently only visible inside the notebook output).

### 🔜 Phase 4 — Forecasting + API
- [x] ~~72-hour forecasting~~ — implemented as three independent XGBoost models predicting the mean AQI for day 1 / day 2 / day 3 (`lag_feature_engineering_3_days.ipynb`), rather than iterative single-step forecasting. Accuracy drops off fast: Day 1 R² 0.70 → Day 3 R² -0.05, so the day-3 model needs more work (more history, better features, or a different approach) before it's usable.
- [ ] AQI category classification (Good / Moderate / Unhealthy / Hazardous) with colour codes.
- [ ] Wire the trained models into the FastAPI app — `app/main.py` / `app/routes/` still only expose the root and `/health` routes; no inference endpoint exists yet.
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
- [ ] `requirements.txt` still doesn't list `scikit-learn`, `xgboost`, `lightgbm`, `joblib` or `numpy`, even though the training notebooks now depend on them — add them so a fresh clone can run the notebooks.
- [ ] Add `shap`, `plotly`, `streamlit` to `requirements.txt` as those phases land.
- [ ] Remove the vendored `Affan Project/` reference repo from the working tree before final submission.

---

## 4. Project Structure

```
Air-Quality-Index-Predictor/
├── app/
│   ├── main.py                                     # FastAPI application entry point
│   ├── routes/
│   │   └── health.py                               # /health endpoint
│   ├── models/                                      # Trained model artifacts (.pkl, git-ignored)
│   ├── notebooks/
│   │   ├── eda.ipynb                                # Exploratory data analysis
│   │   ├── train_data.ipynb                         # First-pass Linear Regression / Random Forest baseline
│   │   ├── lag_feature_engineering.ipynb            # Lag/rolling features + 24h-ahead model comparison (5 models)
│   │   └── lag_feature_engineering_3_days.ipynb     # Day-wise 3-day (72h) forecasting with per-horizon XGBoost models
│   └── src/
│       └── features/
│           ├── aqi.py                # PM2.5 -> EPA AQI conversion
│           ├── build_features.py     # Raw reading -> feature row
│           ├── fetch_waqi.py         # Live data source
│           ├── fetch_openweather.py  # Historical pollutant source + geocoding
│           ├── fetch_openmeteo.py    # Historical weather source
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
| Data sources | WAQI API (live AQI + pollutants), OpenWeather Air Pollution History (historical pollutants), Open-Meteo Archive API (historical weather) |
| Feature store | MongoDB Atlas (`aqi_features` collection) |
| Data processing | Pandas, NumPy |
| Modelling | scikit-learn (Linear/Ridge/Random Forest), XGBoost, LightGBM, joblib (persistence) |
| EDA / visualisation | Jupyter, Matplotlib, Seaborn |
| Config | python-dotenv |
| Planned | SHAP, Streamlit, GitHub Actions |

---

## 6. Getting Started

### Prerequisites
- Python 3.10+
- A MongoDB Atlas cluster
- API keys for [WAQI](https://aqicn.org/data-platform/token/) and [OpenWeather](https://openweathermap.org/api) (Open-Meteo needs no key)

### Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. requirements.txt doesn't yet include the modelling libraries used by the
#    notebooks (see Phase 7 in "What Is Remaining") — install them too if you
#    plan to run the training notebooks:
pip install scikit-learn xgboost lightgbm joblib numpy
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

# Backfill the last 365 days of history for a city (pollutants + weather)
python -m app.src.features.backfill --city karachi --days 365

# Open the notebooks
jupyter notebook app/notebooks/eda.ipynb
jupyter notebook app/notebooks/lag_feature_engineering.ipynb          # 24h-ahead model comparison
jupyter notebook app/notebooks/lag_feature_engineering_3_days.ipynb   # 3-day day-wise forecasting
```

---

## 7. Feature Schema

### Stored in the feature store (`aqi_features` collection)

| Field | Description |
|-------|-------------|
| `city` | City name as queried |
| `timestamp` | ISO-8601 reading time |
| `hour`, `day`, `month`, `day_of_week` | Time features derived from the timestamp |
| `aqi` | **Target** — Air Quality Index (0–500) |
| `aqi_change_rate` | Difference from the previous reading's AQI |
| `pm25`, `pm10`, `o3`, `no2`, `so2`, `co` | Pollutant readings |
| `temperature`, `humidity`, `pressure`, `wind_speed` | Weather readings (backfilled via Open-Meteo, live via WAQI) |

### Engineered at training time (notebooks only, not yet in the store)

| Field | Description |
|-------|-------------|
| `aqi_lag_1/3/6/12/24` | AQI value 1/3/6/12/24 hours ago |
| `pm25_lag_1/6/24` | PM2.5 value 1/6/24 hours ago |
| `aqi_roll_mean_6/12/24` | Rolling mean AQI over the trailing 6/12/24 hours |
| `aqi_roll_std_24` | Rolling std of AQI over the trailing 24 hours |
| `target_day1/2/3` | **3-day forecast targets** — mean AQI over hours 1–24 / 25–48 / 49–72 ahead |

---

## 8. Progress Summary

| Stage | Status |
|-------|--------|
| Backend skeleton | ✅ Done |
| Live data ingestion (WAQI) | ✅ Done |
| Historical ingestion + backfill (OpenWeather + Open-Meteo, 365 days) | ✅ Done |
| MongoDB feature store | ✅ Done |
| Feature pipeline (CLI) | ✅ Done |
| Exploratory data analysis | ✅ Done |
| Lag / rolling feature engineering | ✅ Done (notebook-only) |
| Model training & comparison (24h-ahead, 5 models) | ✅ Done |
| Day-wise 3-day forecasting (per-horizon XGBoost) | ✅ Done — Day 3 accuracy still weak (R² -0.05) |
| Data quality fixes (AQI definition, dedup, etc.) | 🔜 Pending |
| Feature engineering module shared by train/serve | 🔜 Pending |
| Explainability (SHAP) | 🔜 Pending |
| Prediction API endpoints | 🔜 Pending |
| Dashboard | 🔜 Pending |
| GitHub Actions automation | 🔜 Pending |
| Deployment | 🔜 Pending |
