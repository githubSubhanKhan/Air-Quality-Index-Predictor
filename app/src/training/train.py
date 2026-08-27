"""
Retraining pipeline for the 3-day AQI forecast models.

Reads the MongoDB feature store, rebuilds the lag / rolling feature set,
trains one XGBoost regressor per forecast horizon (day 1, 2 and 3) on a
time-based split, then **publishes each model to the model registry** with
the metrics it earned. The registry — not the local ``.pkl`` files — is what
serving reads from, so a retrain ships by promoting a version rather than by
committing a binary.

This is the scripted equivalent of
``app/notebooks/lag_feature_engineering_3_days.ipynb`` so the same training
can run unattended from GitHub Actions.

Usage:
    python -m app.src.training.train --city karachi
    python -m app.src.training.train --city karachi --no-publish --no-save
    python -m app.src.training.train --city karachi --promotion never
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from dotenv import load_dotenv
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from xgboost import XGBRegressor

from app.src.features.feature_engineering import build_training_features
from app.src.features.feature_store import get_collection
from app.src.registry import model_registry as registry

load_dotenv()

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"

METADATA_FILENAME = "training_metadata.json"

FEATURE_COLUMNS = [

    # Calendar
    "hour",
    "day",
    "month",
    "day_of_week",

    # Pollutants
    "pm25",
    "pm10",
    "o3",
    "no2",
    "so2",
    "co",

    # Weather
    "temperature",
    "humidity",
    "pressure",
    "wind_speed",

    # AQI lag
    "aqi_lag_1",
    "aqi_lag_3",
    "aqi_lag_6",
    "aqi_lag_12",
    "aqi_lag_24",

    # PM2.5 lag
    "pm25_lag_1",
    "pm25_lag_6",
    "pm25_lag_24",

    # Rolling
    "aqi_roll_mean_6",
    "aqi_roll_mean_12",
    "aqi_roll_mean_24",
    "aqi_roll_std_24",
]

# Forecast horizon -> hours ahead the target window ends at.
# day1 = mean AQI over hours 1-24, day2 = 25-48, day3 = 49-72.
HORIZONS = {
    "day1": 24,
    "day2": 48,
    "day3": 72,
}

XGB_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": -1,
}

DEFAULT_TEST_SIZE = 0.2

# Training on less than this many usable rows is not worth shipping;
# roughly three weeks of hourly readings after lags and targets are dropped.
DEFAULT_MIN_ROWS = 500

# Which metric the registry promotion gate compares runs on.
PROMOTION_METRIC = "mae"


def target_column(horizon: str) -> str:
    return f"target_{horizon}"


def environment_snapshot() -> dict:
    """Library versions the models were trained with, recorded per version."""

    return {
        "python": sys.version.split()[0],
        "xgboost": xgboost.__version__,
        "scikit_learn": sklearn.__version__,
        "pandas": pd.__version__,
    }


def load_feature_store(city: str) -> pd.DataFrame:
    """Load every stored reading for a city, oldest first."""

    collection = get_collection()

    cursor = (
        collection
        .find({"city": city.lower()}, {"_id": 0})
        .sort("timestamp", 1)
    )

    df = pd.DataFrame(list(cursor))

    if df.empty:
        raise RuntimeError(
            f"No rows in the feature store for city '{city}'."
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    df = df.drop_duplicates(subset="timestamp", keep="last")

    return df.sort_values("timestamp").reset_index(drop=True)


def add_forecast_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach the day-wise targets: the mean AQI over each 24-hour window
    ahead of the current row.
    """

    df = df.copy()

    # rolling(24).mean() at position i + offset is the mean of the 24 rows
    # ending there, so shifting it back by `offset` lands that window
    # immediately ahead of row i.
    trailing_mean = df["aqi"].rolling(24).mean()

    for horizon, offset in HORIZONS.items():
        df[target_column(horizon)] = trailing_mean.shift(-offset)

    return df


def count_missing_hours(df: pd.DataFrame) -> int:
    """
    How many hourly slots are absent between the first and last reading.

    The lag, rolling and target windows all assume contiguous hourly rows,
    so gaps are worth surfacing in the training report.
    """

    span = df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]

    expected = int(span.total_seconds() // 3600) + 1

    return max(expected - len(df), 0)


def score(y_true, y_pred) -> dict:
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def train_horizon(X_train, y_train) -> XGBRegressor:
    model = XGBRegressor(**XGB_PARAMS)

    model.fit(X_train, y_train)

    return model


def run(
    city: str = "karachi",
    test_size: float = DEFAULT_TEST_SIZE,
    min_rows: int = DEFAULT_MIN_ROWS,
    model_dir: Path = MODEL_DIR,
    save: bool = True,
    publish: bool = True,
    promotion: str = "auto",
    keep: int = registry.DEFAULT_ARTIFACTS_KEPT,
    tolerance: float = registry.DEFAULT_DEGRADATION_TOLERANCE,
) -> dict:
    """
    Retrain every forecast horizon and return the run's metadata.

    Nothing is published until all three horizons have trained, so a failure
    part-way through leaves the current production models serving.
    """

    raw = load_feature_store(city)

    rows_in_store = len(raw)

    missing_hours = count_missing_hours(raw)

    df = build_training_features(raw)

    df = add_forecast_targets(df)

    required = FEATURE_COLUMNS + [
        target_column(horizon) for horizon in HORIZONS
    ]

    df = df.dropna(subset=required).reset_index(drop=True)

    if len(df) < min_rows:
        raise RuntimeError(
            f"Only {len(df)} usable rows for '{city}' "
            f"(minimum {min_rows}); skipping retrain."
        )

    X = df[FEATURE_COLUMNS]

    split = int(len(df) * (1 - test_size))

    X_train, X_test = X.iloc[:split], X.iloc[split:]

    print(
        f"City          : {city.lower()}\n"
        f"Rows in store : {rows_in_store}\n"
        f"Usable rows   : {len(df)}\n"
        f"Train / test  : {len(X_train)} / {len(X_test)}\n"
        f"Missing hours : {missing_hours}\n"
    )

    # Persistence baseline: assume the next three days look like now.
    # A horizon that cannot beat this is not adding value.
    baseline_pred = df["aqi"].iloc[split:]

    models = {}
    metrics = {}

    for horizon in HORIZONS:
        y = df[target_column(horizon)]

        y_train, y_test = y.iloc[:split], y.iloc[split:]

        model = train_horizon(X_train, y_train)

        models[horizon] = model

        metrics[horizon] = {
            **score(y_test, model.predict(X_test)),
            "baseline_r2": score(y_test, baseline_pred)["r2"],
        }

        print(
            f"{horizon}: "
            f"MAE {metrics[horizon]['mae']:.2f}  "
            f"RMSE {metrics[horizon]['rmse']:.2f}  "
            f"R2 {metrics[horizon]['r2']:.4f}  "
            f"(persistence baseline R2 {metrics[horizon]['baseline_r2']:.4f})"
        )

    trained_at = datetime.now(timezone.utc)

    metadata = {
        "run_id": trained_at.strftime("%Y%m%dT%H%M%SZ"),
        "trained_at": trained_at.isoformat(timespec="seconds"),
        "city": city.lower(),
        "model_type": "XGBRegressor",
        "model_params": XGB_PARAMS,
        "test_size": test_size,
        "features": FEATURE_COLUMNS,
        "environment": environment_snapshot(),
        "data": {
            "rows_in_store": rows_in_store,
            "usable_rows": len(df),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "first_timestamp": raw["timestamp"].iloc[0].isoformat(),
            "last_timestamp": raw["timestamp"].iloc[-1].isoformat(),
            "missing_hourly_rows": missing_hours,
        },
        "metrics": metrics,
    }

    if publish:
        metadata["registry"] = publish_to_registry(
            models,
            metadata,
            promotion=promotion,
            keep=keep,
            tolerance=tolerance,
        )
    else:
        print("\n--no-publish set: models were not sent to the registry.")

    if save:
        save_artifacts(models, metadata, model_dir)
    else:
        print("--no-save set: models were not written to disk.")

    return metadata


def publish_to_registry(
    models: dict,
    metadata: dict,
    promotion: str = "auto",
    keep: int = registry.DEFAULT_ARTIFACTS_KEPT,
    tolerance: float = registry.DEFAULT_DEGRADATION_TOLERANCE,
) -> dict:
    """
    Register every horizon's model and apply the promotion policy.

    ``promotion`` is one of:
      auto   — promote unless the candidate is materially worse than the
               incumbent on the promotion metric (the default),
      always — promote regardless,
      never  — register as ``staging`` only, for a human to promote.
    """

    print("\nPublishing to the model registry")

    published = {}

    for horizon, model in models.items():
        name = registry.model_name(horizon)

        incumbent = registry.get_production(name)

        document = registry.register_model(
            name,
            model,
            metrics=metadata["metrics"][horizon],
            params=metadata["model_params"],
            features=metadata["features"],
            city=metadata["city"],
            horizon=horizon,
            run_id=metadata["run_id"],
            data=metadata["data"],
            environment=metadata["environment"],
        )

        version = document["version"]

        if promotion == "never":
            promote, reason = False, "promotion disabled for this run"

        elif promotion == "always":
            promote, reason = True, "promotion forced for this run"

        else:
            promote, reason = registry.passes_promotion_gate(
                metadata["metrics"][horizon],
                incumbent,
                metric=PROMOTION_METRIC,
                tolerance=tolerance,
            )

        if promote:
            document = registry.promote(name, version)

        pruned = registry.prune_artifacts(name, keep=keep)

        published[horizon] = {
            "name": name,
            "version": version,
            "stage": document["stage"],
            "promoted": promote,
            "reason": reason,
            "previous_production_version": (
                incumbent["version"] if incumbent else None
            ),
            "artifacts_pruned": pruned,
        }

        print(
            f"  {name} v{version} -> {document['stage']} ({reason})"
            + (f"; pruned {pruned} old artifact(s)" if pruned else "")
        )

    return {
        "collection": registry.REGISTRY_COLLECTION,
        "artifact_bucket": registry.ARTIFACT_BUCKET,
        "promotion_policy": promotion,
        "promotion_metric": PROMOTION_METRIC,
        "published": published,
    }


def save_artifacts(
    models: dict,
    metadata: dict,
    model_dir: Path = MODEL_DIR,
) -> None:
    """
    Write a local copy of the models, the feature list and the metrics.

    ``app/models/`` is git-ignored — these are a development convenience and
    an offline fallback, not the artifacts serving depends on.
    """

    model_dir = Path(model_dir)

    model_dir.mkdir(parents=True, exist_ok=True)

    artifacts = {}

    for horizon, model in models.items():
        filename = f"xgboost_{horizon}.pkl"

        joblib.dump(model, model_dir / filename)

        artifacts[horizon] = filename

    joblib.dump(FEATURE_COLUMNS, model_dir / "feature_columns.pkl")

    metadata["local_artifacts"] = artifacts

    metadata_path = model_dir / METADATA_FILENAME

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved a local copy of the models to {model_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrain the 3-day AQI forecast models",
    )

    parser.add_argument(
        "--city",
        default="karachi",
        help="City to train for, as stored in the feature store",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help="Fraction of the most recent rows held out for evaluation",
    )

    parser.add_argument(
        "--min-rows",
        type=int,
        default=DEFAULT_MIN_ROWS,
        help="Refuse to train on fewer usable rows than this",
    )

    parser.add_argument(
        "--model-dir",
        default=str(MODEL_DIR),
        help="Directory the local model copy is written to",
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip the local .pkl copy",
    )

    parser.add_argument(
        "--no-publish",
        action="store_true",
        help="Skip publishing to the model registry",
    )

    parser.add_argument(
        "--promotion",
        choices=["auto", "always", "never"],
        default="auto",
        help="Promotion policy for the newly registered versions",
    )

    parser.add_argument(
        "--keep",
        type=int,
        default=registry.DEFAULT_ARTIFACTS_KEPT,
        help="Model artifacts to keep per horizon (metrics are kept forever)",
    )

    parser.add_argument(
        "--tolerance",
        type=float,
        default=registry.DEFAULT_DEGRADATION_TOLERANCE,
        help=(
            "How much worse than production a candidate may be on "
            f"{PROMOTION_METRIC} and still be promoted, e.g. 0.25 for 25%%"
        ),
    )

    args = parser.parse_args()

    run(
        city=args.city,
        test_size=args.test_size,
        min_rows=args.min_rows,
        model_dir=Path(args.model_dir),
        save=not args.no_save,
        publish=not args.no_publish,
        promotion=args.promotion,
        keep=args.keep,
        tolerance=args.tolerance,
    )


if __name__ == "__main__":
    main()
