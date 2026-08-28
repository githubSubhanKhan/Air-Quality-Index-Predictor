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
- Accuracy degrades with horizon, as expected. These notebook numbers are superseded by the reworked pipeline documented under *Day-3 accuracy rework* below.

### ✅ Automated hourly feature pipeline (GitHub Actions)
- `.github/workflows/data_pipeline.yml` — runs `app.src.features.pipeline` every hour with API keys and the MongoDB URI supplied through GitHub Secrets, so the feature store keeps growing without manual runs.

### ✅ Automated daily retraining (GitHub Actions)
- `app/src/training/train.py` — retraining runs unattended: loads the city's rows from the feature store, rebuilds features via the shared `feature_engineering` module, recreates the `target_day1/2/3` windows, trains the candidate slate per horizon on a purged chronological split, and publishes only after all three have trained (a mid-run failure leaves the production models serving).

### ✅ Automated model selection per retrain
Previously the pipeline trained one XGBoost per horizon and took the five-model comparison in `lag_feature_engineering.ipynb` as justification — but that comparison was run once, against a different target (24h-ahead *absolute* AQI on 26 features), and nothing re-checked it after the horizons moved to a damped correction on 14 features.

Every retrain now trains a **candidate slate** per horizon and keeps the winner:

| Candidate | Family | Estimator |
|---|---|---|
| `persistence` | constant | `DummyRegressor` — the null model, reference only |
| `ridge` | linear | `StandardScaler` + `Ridge` |
| `random_forest` | tree | `RandomForestRegressor` |
| `hist_gbm` | tree | `HistGradientBoostingRegressor` |
| `xgboost` | tree | `XGBRegressor` — the incumbent |

- **Defined in** `app/src/training/candidates.py`; evaluation and the selection rule in `app/src/training/selection.py`.
- **Selection is on the validation block only.** Train fits, validation fits each candidate's `alpha` *and* ranks the candidates, test is scored once and never consulted before the choice. Test metrics are recorded for every candidate for the write-up, but took no part in the decision — on day 3, `random_forest` in fact edges `hist_gbm` on test while losing on validation, which is exactly the honesty this protocol buys.
- **A 2% margin guards the incumbent.** A challenger must beat `xgboost` by 2% on validation MAE to take the slot, so the served family does not flip between daily retrains on differences inside the noise.
- **`persistence` is scored but not eligible** unless `--allow-baseline` is passed: a constant has no SHAP attribution, and the dashboard's explanation panel is a deliverable.
- **Every candidate's metrics are stored with the version** (`selection.comparison` on the registry document), so the comparison behind a served model survives the run that produced it. `python -m app.src.registry.cli show <name>` prints it; the daily workflow renders it into the job summary via `app/src/training/report.py`.
- **Explanations follow the winner.** `explainer.py` picks its method from what the fitted estimator exposes — exact linear contributions for `coef_` models, `shap.TreeExplainer` for trees, XGBoost's own TreeSHAP as a fallback — so switching family keeps the SHAP panel working. For every method, `base_value + sum(contributions)` reconstructs the prediction exactly.

Result on the 2025-08 → 2026-08 Karachi data (7,749 usable rows), against the XGBoost-only pipeline it replaces:

| Horizon | Was (XGBoost) | Now | Winner | Test MAE | Test R² | Skill vs persistence |
|---|---|---|---|---|---|---|
| Day 1 | MAE 3.76, R² 0.861, skill **−0.025** | MAE 3.71, R² 0.866, skill **+0.007** | `random_forest` | 3.71 | +0.866 | +0.007 |
| Day 2 | MAE 7.08, R² 0.568, skill +0.024 | MAE 6.95, R² 0.580, skill +0.052 | `random_forest` | 6.95 | +0.580 | +0.052 |
| Day 3 | MAE 9.32, R² 0.275, skill +0.027 | MAE 9.06, R² 0.315, skill +0.081 | `hist_gbm` | 9.06 | +0.315 | +0.081 |

Day 1's skill against persistence crossed from negative to positive — the previous model was very slightly worse than simply carrying the current reading forward. Day 1 remains close to persistence (3.71 vs 3.71 MAE); the honest reading is that the model adds little at 24 hours and most of its value is at 48–72 hours.

The registry slot names (`aqi_xgboost_day1`, …) are deliberately unchanged: version numbers, promotion history and rollback are keyed on them, so renaming per family would restart numbering and leave the promotion gate with no incumbent. The `candidate`, `model_family` and `model_type` fields on each document say what is actually inside.
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

### ✅ Explainability (SHAP)
- `app/src/explain/explainer.py` — TreeSHAP contributions in AQI points, in two flavours:
  - **Local** — why *this* forecast came out where it did. `base_value` plus the sum of all contributions reconstructs the prediction exactly, so the explanation can be checked against `/predict/{city}`.
  - **Global** — mean |SHAP| per feature over the evaluation rows. Training computes this per horizon and stores it **in the registry document next to the metrics**, so every registered version carries its own explanation.
- `shap.TreeExplainer` is the primary path. XGBoost implements the same TreeSHAP algorithm internally (`pred_contribs=True`), so if `shap` is missing or misbehaves, explanations fall back to the booster instead of the dashboard losing them — and a failed explanation never fails a retrain.
- `app/routes/explain.py` — `GET /explain/{city}` (per-prediction contributions, optional `horizon` and `top`), `GET /explain/global` (the stored global ranking for the production versions).
- Dashboard — a **Why This Forecast?** section with one tab per horizon: a diverging bar chart of each feature's push on the forecast (red up, blue down, signed labels), the baseline → forecast arithmetic, and an expander with that version's overall drivers from the registry.

### ✅ Day-3 accuracy rework

Day 3's R² was **-0.03** — worse than predicting the mean. Diagnosis first, in the data rather than the model:

| Problem | Evidence |
|---|---|
| Severe train/test distribution shift | Between the training and evaluation windows of the 2025-26 data, o3 fell **54%**, no2 **83%**, co **58%**, pm2.5 **45%**. Trees cannot extrapolate, and SHAP confirmed day 3 was leaning on exactly these absolute levels (o3 was its top driver) with no AQI lag near the top. |
| A calendar value never seen in training | Training covered months 1-6 and 8-12; the test window covered 6, 7, 8. **Month 7 never appeared in training**, so every split on `month` sent July down an untrained branch. |
| Lags that were not really lags | 221 readings sat off the hour (the live pipeline records the observation time) and there were 21 gaps over 2h — all inside the evaluation window — so `shift(24)` meant "24 rows ago", not "24 hours ago". |
| Optimistic evaluation | A row's target window reaches 72 hours ahead, so training targets overlapped the test window. Removing that leakage made the old configuration's day-3 score swing between +0.19 and **-1.80** depending on which window it was measured on. |

What changed:

- **A real hourly grid** — `normalise_hourly` floors every timestamp and reindexes onto a complete hourly grid (forward-filling gaps up to 6h, never backwards), so lag, rolling and target windows are true time offsets.
- **Stationary features** — the model set is now 14 *relative* features (deviations from trailing means, cyclical season and time-of-day) instead of absolute pollutant levels. Raw `day` and `month` are gone from the model set.
- **Persistence plus a damped correction** — each model predicts the *deviation* from the current AQI, and serving computes `forecast = current AQI + alpha * model output`. A model with no skill now collapses toward persistence instead of toward something worse, which is what made day 3 negative.
- **`alpha` is fitted, not guessed** — it is the least-squares weight of the correction on a validation window between train and test, then halved. The shrinkage beat the unshrunk fit in **11 of 12** window × horizon combinations tested, including windows it was not chosen on.
- **Purged, nested evaluation** — train 65% | purge 72 rows | validation 15% | purge 72 rows | test 20%. The test block is still the final 20% of rows, so the reported numbers stay comparable with the previous pipeline.
- **Skill score** — every horizon also records `skill_vs_persistence` (`1 - MSE/MSE_persistence`), positive only when the model genuinely beats persistence. R² alone is misleading here: in a strongly trending window even a perfect persistence forecast scores below zero.

Result — all three horizons positive, and every one better than before:

| Horizon | R² before | R² after | MAE before | MAE after | Persistence R² | Skill vs persistence |
|---------|-----------|----------|------------|-----------|----------------|----------------------|
| Day 1 | +0.7240 | **+0.8572** | 5.27 | 3.84 | +0.8639 | -0.049 |
| Day 2 | +0.2083 | **+0.5497** | 9.30 | 7.29 | +0.5429 | +0.015 |
| Day 3 | **-0.0287** | **+0.2575** | 9.92 | 9.51 | +0.2497 | +0.010 |

Two things worth saying plainly, since the skill column sits close to zero:

- **Persistence is most of the forecast.** The learned correction adds a little at day 2 and day 3 and costs a little at day 1. The damping is what keeps it honest — the pipeline can no longer ship a model dramatically worse than doing nothing.
- **Forecast weather is not the missing ingredient.** This was measured rather than assumed: feeding the models the weather *actually observed* over each target window — a flawless 3-day forecast — was worth only +0.06 skill at day 1 and day 2, and was negative at day 3. The limit is how predictable a 24h-mean AQI is 49-72 hours out from one station's own history, not the weather inputs.

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
- [x] ~~Train and compare multiple regressors~~ — done twice over: a one-off five-model comparison in `lag_feature_engineering.ipynb`, and now a candidate slate (persistence / Ridge / Random Forest / HistGradientBoosting / XGBoost) evaluated per horizon on **every** retrain, with the winner chosen on a validation block and the full comparison stored on the registry document.
- [x] ~~Time-based train/test split~~ — 80/20 chronological split (no shuffling); no time-series cross-validation yet.
- [x] ~~Evaluate with R², RMSE, MAE~~ — best is XGBoost (MAE 8.88, RMSE 11.85, R² 0.46).
- [x] ~~Persist the trained model~~ — saved as `.pkl` files via `joblib` to `app/models/` (git-ignored, not yet uploaded anywhere durable).
- [x] ~~Benchmark a naive/persistence baseline~~ — every retrain records the persistence baseline's R² and the model's skill score against it, per horizon.
- [x] ~~Compute SHAP values at training time for dashboard explainability~~ — global mean |SHAP| per horizon, stored with each registry version.
- [x] ~~Track metrics/feature list alongside the model artifact~~ — the model registry stores metrics, feature list, params, lineage and SHAP importance per version.

### 🔜 Phase 4 — Forecasting + API
- [x] ~~72-hour forecasting~~ — three independent XGBoost models predicting the mean AQI for day 1 / day 2 / day 3, each as a damped correction to persistence. Day 1 R² +0.86 → Day 3 R² +0.26 after the day-3 rework.
- [ ] AQI category classification (Good / Moderate / Unhealthy / Hazardous) with colour codes.
- [ ] Wire the trained models into the FastAPI app — `app/main.py` / `app/routes/` still only expose the root and `/health` routes; no inference endpoint exists yet.
- [ ] New endpoints:
  - `GET /predict` — current AQI, category, 3-day forecast, last 24h history, model metadata
  - [x] ~~`GET /explain/{city}` — feature contributions (SHAP)~~
  - `GET /debug` — model availability and data-range diagnostics
- [ ] Pydantic response models for each endpoint.

### 🔜 Phase 5 — Frontend
- [x] ~~Dashboard (Streamlit) with: current AQI card, 3-day forecast, hourly history chart, model metrics and SHAP plot~~ — all present; model metrics and versions come from the registry.

### 🔜 Phase 6 — Automation & deployment
- [x] `.github/workflows/data_pipeline.yml` — runs `pipeline.py` hourly, credentials via GitHub Secrets.
- [x] `.github/workflows/daily_training.yml` — retrains daily and publishes the new versions to the model registry.
- [ ] Deploy the API and dashboard (e.g. Render) and document the live URLs.

### 🔜 Phase 7 — Engineering hygiene
- [ ] Add `.env.example`, tests for `aqi.py` / `build_features.py`, and logging in place of `print`.
- [x] `requirements.txt` now lists `scikit-learn`, `xgboost`, `joblib` and `numpy`, which the retraining pipeline needs in CI (`lightgbm` is still notebook-only).
- [x] ~~Add `shap`, `plotly`, `streamlit` to `requirements.txt`~~ — all three are pinned.
- [ ] Remove the vendored `Affan Project/` reference repo from the working tree before final submission.

---

## 4. Project Structure

```
Air-Quality-Index-Predictor/
├── app/
│   ├── main.py                                     # FastAPI application entry point
│   ├── routes/
│   │   ├── health.py                               # /health endpoint
│   │   ├── models.py                               # /models registry endpoints
│   │   └── explain.py                              # /explain SHAP endpoints
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
│       │   ├── train.py              # Daily retraining pipeline (CLI)
│       │   ├── candidates.py         # Candidate model slate (persistence/ridge/RF/HistGBM/XGBoost)
│       │   ├── selection.py          # Per-horizon evaluation + winner selection
│       │   └── report.py             # Run metadata -> Markdown (CI job summary)
│       ├── registry/
│       │   ├── model_registry.py     # MongoDB/GridFS model registry
│       │   └── cli.py                # Registry CLI (list/show/promote/rollback/prune)
│       └── explain/
│           └── explainer.py          # SHAP explanations (local + global)
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
| Explainability | SHAP (TreeSHAP), with XGBoost's `pred_contribs` as the fallback |
| Frontend / automation | Streamlit, Plotly, GitHub Actions |

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

# Retrain the 3-day forecast models (what the daily workflow runs).
# Trains the full candidate slate per horizon and publishes the winners.
python -m app.src.training.train --city karachi

# Train and report metrics without publishing or overwriting anything
python -m app.src.training.train --city karachi --no-publish --no-save

# Register the new versions but leave promotion to a human
python -m app.src.training.train --city karachi --promotion never

# Model selection controls
python -m app.src.training.train --candidates xgboost                 # skip the bake-off
python -m app.src.training.train --candidates all --allow-baseline    # let persistence compete
python -m app.src.training.train --select-metric rmse --select-margin 0.05
python -m app.src.training.train --default-model random_forest        # change the incumbent

# Render a run's candidate comparison as Markdown (what CI puts in the job summary)
python -m app.src.training.train --city karachi --no-publish --metadata-out run.json
python -m app.src.training.report run.json

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
| `explanations` | Global SHAP ranking (mean |SHAP| per feature) for that version, with the sample size and method |
| `target_transform` | How to turn the model output into a forecast — `{mode: delta_from_anchor, anchor: aqi, alpha: 0.37}`. Absent on versions registered before the day-3 rework, which predicted the level directly |
| `artifact` | GridFS file id, size, SHA-256 checksum, and whether the binary is still retained |

### Engineered at training time (notebooks only, not yet in the store)

| Field | Description |
|-------|-------------|
| `aqi_lag_1/3/6/12/24` | AQI value 1/3/6/12/24 hours ago |
| `aqi_rel_24/72/168` | AQI minus its trailing 24h / 72h / 7-day mean — the model set uses these instead of absolute levels |
| `aqi_trend_24_72`, `aqi_trend_24_168` | 24h mean minus the 3-day / 7-day mean (which way the level is moving) |
| `<pollutant>_rel_24/168` | Each pollutant and weather field minus its own trailing mean |
| `hour_sin/cos`, `doy_sin/cos` | Cyclical time-of-day and season, replacing raw `day` / `month` in the model set |
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
| Automated model selection per retrain (5-candidate slate, validation-ranked) | ✅ Done — Day 1 skill vs persistence −0.025 → +0.007, Day 3 R² 0.275 → 0.315 |
| Day-wise 3-day forecasting (per-horizon model, selected per retrain) | ✅ Done |
| Day-3 accuracy rework (negative R² fixed) | ✅ Done — Day 3 R² -0.03 → +0.26, all horizons positive |
| Data quality fixes (AQI definition, dedup, etc.) | 🔜 Pending |
| Feature engineering module shared by train/serve | 🔜 Pending |
| Explainability (SHAP) | ✅ Done — global (in the registry) + per-prediction, API + dashboard |
| Prediction API endpoints | 🔜 Pending |
| Dashboard | 🔜 Pending |
| GitHub Actions automation | ✅ Done — hourly feature pipeline + daily retraining |
| Model registry with metrics (MongoDB + GridFS) | ✅ Done — versioning, stages, promotion gate, rollback, retention |
| Deployment | 🔜 Pending |
