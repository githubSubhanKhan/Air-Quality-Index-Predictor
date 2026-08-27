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

### ✅ Automated hourly feature pipeline (GitHub Actions)
- `.github/workflows/data_pipeline.yml` — runs `app.src.features.pipeline` every hour with API keys and the MongoDB URI supplied through GitHub Secrets, so the feature store keeps growing without manual runs.

### ✅ Automated daily retraining (GitHub Actions)
- `app/src/training/train.py` — the scripted equivalent of the 3-day notebook, so retraining can run unattended: loads the city's rows from the feature store, dedupes on timestamp, rebuilds the lag/rolling features via the shared `feature_engineering` module, recreates the `target_day1/2/3` windows, trains one XGBoost regressor per horizon on a time-based 80/20 split, and writes the models only after all three have trained (a mid-run failure leaves the shipped models untouched).
- Guardrails: refuses to train on fewer than `--min-rows` usable rows (default 500), and scores every horizon against a **persistence baseline** (assume the next three days look like now) so a horizon that adds no value is visible in the metrics.
- Alongside the `.pkl` files it writes `app/models/training_metadata.json` — trained-at timestamp, city, hyperparameters, row counts, data range, missing hourly rows, and MAE / RMSE / R² (plus baseline R²) per horizon.
- `.github/workflows/daily_training.yml` — runs daily at 02:30 UTC (07:30 PKT) or on demand via *Run workflow*: retrain → publish each horizon to the model registry → smoke-test whatever is now in `production` through the real serving path → print the registry table to the run summary. Nothing is committed back to the repo, so the workflow only needs `contents: read`.

### ✅ Model registry (MongoDB + GridFS)
- `app/src/registry/model_registry.py` — models are no longer just `.pkl` files in the repo. Every training run publishes each horizon to a registry that lives in the same MongoDB database as the feature store, so there is no extra service to host and no credentials beyond `MONGODB_URI`:
  - `model_registry` — one document per `(name, version)`: the **metrics** it earned (MAE / RMSE / R² + persistence-baseline R²), the **hyperparameters**, the **feature list**, the **data lineage** (row counts, date range, missing hours), the **library versions** it was trained with, the commit SHA when run from CI, and a SHA-256 checksum of the artifact.
  - `model_artifacts.*` — the pickled models themselves in GridFS, compressed (~380 KB per model instead of 1.2 MB).
  - Stages: `staging` → `production` → `archived`, one production version per model name.
- **Promotion is a metadata change, not a redeploy.** Serving asks the registry for the production version, so `promote` / `rollback` take effect without touching code or committing a binary.
- Promotion gate: a retrain is auto-promoted unless it is more than 25% worse than the incumbent on MAE. Each run is scored on its own hold-out window, so run-to-run metrics are not strictly comparable — the gate is a guard against a broken run, not a fine-grained comparison. Override per run with `--promotion always|never`.
- Retention: the newest 5 versions (and always production) keep their GridFS binary; older binaries are pruned while their **metric history is kept forever**, so the free-tier Atlas cluster does not fill up with a year of daily models.
- Integrity: every load re-checks the SHA-256 and refuses to serve a mismatch, and a version whose binary has been pruned cannot be promoted or rolled back into.
- `app/src/registry/cli.py` — `list`, `show`, `promote`, `rollback`, `prune`, `download`.
- `app/routes/models.py` — `GET /models` (what is serving now, with metrics), `GET /models/versions` (metric history per horizon), `GET /models/names`. Read-only on purpose: promotion changes what production serves, so it stays in the CLI rather than on an unauthenticated endpoint.
- `app/src/prediction/predictor.py` — loads the production version of each horizon from the registry and uses **that version's own feature list**, so one horizon can be retrained on different features without breaking the others. The git-ignored `.pkl` files in `app/models/` are only a local-development fallback.
- The dashboard sidebar shows which versions are serving and what they scored.

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
- [x] `.github/workflows/data_pipeline.yml` — runs `pipeline.py` hourly, credentials via GitHub Secrets.
- [x] `.github/workflows/daily_training.yml` — retrains daily and publishes the new versions to the model registry.
- [ ] Deploy the API and dashboard (e.g. Render) and document the live URLs.

### 🔜 Phase 7 — Engineering hygiene
- [ ] Add `.env.example`, tests for `aqi.py` / `build_features.py`, and logging in place of `print`.
- [x] `requirements.txt` now lists `scikit-learn`, `xgboost`, `joblib` and `numpy`, which the retraining pipeline needs in CI (`lightgbm` is still notebook-only).
- [ ] Add `shap`, `plotly`, `streamlit` to `requirements.txt` as those phases land.
- [ ] Remove the vendored `Affan Project/` reference repo from the working tree before final submission.

---

## 4. Project Structure

```
Air-Quality-Index-Predictor/
├── app/
│   ├── main.py                                     # FastAPI application entry point
│   ├── routes/
│   │   ├── health.py                               # /health endpoint
│   │   └── models.py                               # /models registry endpoints
│   ├── models/                                      # Local dev copy of the models (git-ignored; registry is authoritative)
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
│       ├── prediction/
│       │   ├── build_prediction_features.py   # Latest feature row for serving
│       │   └── predictor.py                   # Loads the .pkl models, returns the 3-day forecast
│       ├── training/
│       │   └── train.py              # Daily retraining pipeline (CLI)
│       └── registry/
│           ├── model_registry.py     # MongoDB/GridFS model registry
│           └── cli.py                # Registry CLI (list/show/promote/rollback/prune)
├── .github/workflows/
│   ├── data_pipeline.yml             # Hourly feature pipeline
│   └── daily_training.yml            # Daily model retraining
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

# 3. lightgbm is only used by the comparison notebook, so it isn't pinned:
pip install lightgbm
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

# Retrain the 3-day forecast models (what the daily workflow runs)
python -m app.src.training.train --city karachi

# Train and report metrics without publishing or overwriting anything
python -m app.src.training.train --city karachi --no-publish --no-save

# Register the new versions but leave promotion to a human
python -m app.src.training.train --city karachi --promotion never

# Inspect the model registry
python -m app.src.registry.cli list
python -m app.src.registry.cli show aqi_xgboost_day3
python -m app.src.registry.cli promote aqi_xgboost_day3 4
python -m app.src.registry.cli rollback aqi_xgboost_day3
python -m app.src.registry.cli prune --keep 5

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

### Stored in the model registry (`model_registry` collection + `model_artifacts` GridFS bucket)

| Field | Description |
|-------|-------------|
| `name`, `version` | Registry identity, e.g. `aqi_xgboost_day3` v4 — unique and monotonic per name |
| `stage` | `staging`, `production` or `archived`; one production version per name |
| `metrics` | `mae`, `rmse`, `r2`, `baseline_r2` on that run's hold-out window |
| `params`, `features`, `environment` | Hyperparameters, feature list and library versions used |
| `data` | Lineage: row counts, train/test split, first/last timestamp, missing hourly rows |
| `run_id`, `git_sha`, `created_at`, `promoted_at` | Which training run and commit produced it, and when it shipped |
| `artifact` | GridFS file id, size, SHA-256 checksum, and whether the binary is still retained |

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
| GitHub Actions automation | ✅ Done — hourly feature pipeline + daily retraining |
| Model registry with metrics (MongoDB + GridFS) | ✅ Done — versioning, stages, promotion gate, rollback, retention |
| Deployment | 🔜 Pending |
