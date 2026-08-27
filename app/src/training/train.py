"""
Retraining pipeline for the 3-day AQI forecast models.

The forecast is built as **persistence plus a damped learned correction**:

    forecast = current AQI + alpha * model(features)

The model is trained on the *deviation* from the current AQI rather than on
the absolute level, and ``alpha`` is fitted on a validation window that sits
between training and test. Two things follow, both of which the earlier
absolute-target setup lacked:

* a model with no skill collapses toward persistence instead of toward
  something worse than persistence — which is what drove day 3 negative;
* nothing depends on absolute pollutant levels, which drift far enough
  across a year to put the far horizons outside the training range.

Evaluation is chronological and **purged**: a training row's target window
reaches 72 hours ahead, so the 72 rows before each boundary are dropped.
Without that, training targets overlap the test window and the reported
numbers are optimistic.

Metrics recorded per horizon: MAE / RMSE / R2 on the reconstructed forecast,
the persistence baseline's R2 on the same window, and the skill score against
persistence (positive only when the model genuinely beats it).

Usage:
    python -m app.src.training.train --city karachi
    python -m app.src.training.train --city karachi --no-publish --no-save
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

from app.src.explain.explainer import global_importance
from app.src.features.feature_engineering import build_training_features
from app.src.features.feature_store import get_collection
from app.src.registry import model_registry as registry

load_dotenv()

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"

METADATA_FILENAME = "training_metadata.json"

# Deliberately small and entirely relative, apart from the AQI level itself:
# the AQI index is bounded 0-500, while raw pollutant concentrations drift.
FEATURE_COLUMNS = [

    # AQI level and recent averages
    "aqi",
    "aqi_roll_mean_24",
    "aqi_roll_mean_168",

    # Deviation from those averages, and the trend between them
    "aqi_rel_24",
    "aqi_rel_168",
    "aqi_trend_24_168",

    # Volatility
    "aqi_roll_std_24",

    # Season and time of day, cyclically encoded
    "doy_sin",
    "doy_cos",
    "hour_sin",
    "hour_cos",

    # One pollutant and two weather terms, as deviations from their own
    # weekly level rather than absolute values
    "pm25_rel_168",
    "humidity",
    "wind_speed",
]

# Forecast horizon -> hours ahead the target window ends at.
# day1 = mean AQI over hours 1-24, day2 = 25-48, day3 = 49-72.
HORIZONS = {
    "day1": 24,
    "day2": 48,
    "day3": 72,
}

# The forecast the model corrects. Predicting the deviation from this makes
# the target stationary and bounds how wrong the model can be.
ANCHOR_COLUMN = "aqi"

TRANSFORM_MODE = "delta_from_anchor"

XGB_PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.03,
    "max_depth": 2,
    "subsample": 0.8,
    "colsample_bytree": 0.6,
    "min_child_weight": 20,
    "reg_lambda": 5.0,
    # Absolute error: ~32% of consecutive AQI readings repeat exactly and the
    # series has spikes, both of which squared error chases.
    "objective": "reg:absoluteerror",
    "random_state": 42,
    "n_jobs": -1,
}

# Chronological split. The test block stays the final 20% of rows, unchanged
# from the previous pipeline, so the reported metrics remain comparable; the
# validation block that fits alpha is carved out of the training portion.
TRAIN_FRACTION = 0.65

VALIDATION_FRACTION = 0.15

# A row's target reaches this many hours ahead, so this many rows are dropped
# before each split boundary.
PURGE_ROWS = 72

# alpha is a single coefficient fitted on ~1000 autocorrelated rows, so the
# least-squares estimate is noisy and optimistic. Halving it beat the
# unshrunk fit in 11 of 12 window x horizon combinations tested, including
# windows the rule was not chosen on.
ALPHA_SHRINKAGE = 0.5

# Rolling windows tolerate gaps: 24-hour statistics need 17 of 24 hours.
TARGET_MIN_PERIODS = 17

DEFAULT_MIN_ROWS = 500

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

    # rolling(24) at position i + offset is the mean of the 24 rows ending
    # there, so shifting it back by `offset` lands that window immediately
    # ahead of row i. The row spacing is a true hour after normalisation.
    trailing_mean = (
        df["aqi"]
        .rolling(24, min_periods=TARGET_MIN_PERIODS)
        .mean()
    )

    for horizon, offset in HORIZONS.items():
        df[target_column(horizon)] = trailing_mean.shift(-offset)

    return df


def count_missing_hours(df: pd.DataFrame) -> int:
    """Hourly slots with no real reading between the first and last row."""

    span = df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]

    expected = int(span.total_seconds() // 3600) + 1

    return max(expected - len(df), 0)


def fit_alpha(actual, anchor, predicted_delta) -> float:
    """
    Least-squares weight for the model's correction, clipped to [0, 1].

    This is the regression of what persistence got wrong on what the model
    says it got wrong. If the two are unrelated — or point in opposite
    directions — the weight goes to zero and the forecast is persistence.
    """

    residual = np.asarray(actual, dtype=float) - np.asarray(anchor, dtype=float)

    predicted_delta = np.asarray(predicted_delta, dtype=float)

    denominator = float((predicted_delta ** 2).sum())

    if denominator <= 1e-9:
        return 0.0

    return float(np.clip((predicted_delta * residual).sum() / denominator, 0.0, 1.0))


def score(y_true, y_pred, anchor) -> dict:
    """Accuracy of a forecast, plus its skill against persistence."""

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    anchor = np.asarray(anchor, dtype=float)

    mse = float(((y_true - y_pred) ** 2).mean())

    mse_persistence = float(((y_true - anchor) ** 2).mean())

    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "baseline_r2": round(float(r2_score(y_true, anchor)), 4),
        "skill_vs_persistence": round(
            1 - mse / mse_persistence if mse_persistence > 0 else 0.0, 4
        ),
    }


def train_horizon(X_train, y_train) -> XGBRegressor:
    model = XGBRegressor(**XGB_PARAMS)

    model.fit(X_train, y_train)

    return model


def run(
    city: str = "karachi",
    min_rows: int = DEFAULT_MIN_ROWS,
    model_dir: Path = MODEL_DIR,
    save: bool = True,
    publish: bool = True,
    promotion: str = "auto",
    keep: int = registry.DEFAULT_ARTIFACTS_KEPT,
    tolerance: float = registry.DEFAULT_DEGRADATION_TOLERANCE,
    shrinkage: float = ALPHA_SHRINKAGE,
    refit_full: bool = False,
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

    required = FEATURE_COLUMNS + [ANCHOR_COLUMN] + [
        target_column(horizon) for horizon in HORIZONS
    ]

    df = df.dropna(subset=required).reset_index(drop=True)

    if len(df) < min_rows:
        raise RuntimeError(
            f"Only {len(df)} usable rows for '{city}' "
            f"(minimum {min_rows}); skipping retrain."
        )

    total = len(df)

    train_end = int(total * TRAIN_FRACTION)

    validation_end = int(total * (TRAIN_FRACTION + VALIDATION_FRACTION))

    # Purged boundaries: training stops 72 rows short of validation, and the
    # alpha fit stops 72 rows short of the test block.
    fit_end = train_end - PURGE_ROWS

    validation = slice(train_end, validation_end - PURGE_ROWS)

    test = slice(validation_end, total)

    X = df[FEATURE_COLUMNS]

    anchor = df[ANCHOR_COLUMN]

    print(
        f"City          : {city.lower()}\n"
        f"Rows in store : {rows_in_store}\n"
        f"Usable rows   : {total}\n"
        f"Train / val / test : {fit_end} / "
        f"{validation.stop - validation.start} / {total - test.start}"
        f"   (purge {PURGE_ROWS} rows per boundary)\n"
        f"Missing hours : {missing_hours}\n"
    )

    models = {}
    metrics = {}
    transforms = {}
    explanations = {}

    for horizon in HORIZONS:
        y_absolute = df[target_column(horizon)]

        # The model learns the deviation from persistence, not the level.
        y_delta = y_absolute - anchor

        model = train_horizon(X.iloc[:fit_end], y_delta.iloc[:fit_end])

        unshrunk = fit_alpha(
            y_absolute.iloc[validation],
            anchor.iloc[validation],
            model.predict(X.iloc[validation]),
        )

        alpha = round(unshrunk * shrinkage, 4)

        predicted = (
            anchor.iloc[test].to_numpy()
            + alpha * model.predict(X.iloc[test])
        )

        metrics[horizon] = {
            **score(y_absolute.iloc[test], predicted, anchor.iloc[test]),
            "alpha": alpha,
            "alpha_unshrunk": round(unshrunk, 4),
        }

        transforms[horizon] = {
            "mode": TRANSFORM_MODE,
            "anchor": ANCHOR_COLUMN,
            "alpha": alpha,
        }

        if refit_full:
            # More recent data at the cost of alpha having been fitted for a
            # model trained on less of it.
            model = train_horizon(X, y_delta)

        models[horizon] = model

        entry = metrics[horizon]

        print(
            f"{horizon}: "
            f"MAE {entry['mae']:.2f}  "
            f"RMSE {entry['rmse']:.2f}  "
            f"R2 {entry['r2']:+.4f}  "
            f"(persistence R2 {entry['baseline_r2']:+.4f}, "
            f"skill {entry['skill_vs_persistence']:+.3f}, "
            f"alpha {alpha:.2f})"
        )

        # Global SHAP view of the correction, stored with the version. An
        # explanation failing is not a reason to fail the retrain.
        try:
            explanations[horizon] = {
                **global_importance(model, X.iloc[test], FEATURE_COLUMNS),
                "explains": "correction applied to the persistence anchor",
            }

            drivers = ", ".join(
                f"{item['feature']} {item['mean_abs_shap']:.2f}"
                for item in explanations[horizon]["features"][:3]
            )

            print(f"        top SHAP drivers: {drivers}")

        except Exception as exc:
            print(f"        SHAP importance unavailable: {exc}")

            explanations[horizon] = {}

    trained_at = datetime.now(timezone.utc)

    metadata = {
        "run_id": trained_at.strftime("%Y%m%dT%H%M%SZ"),
        "trained_at": trained_at.isoformat(timespec="seconds"),
        "city": city.lower(),
        "model_type": "XGBRegressor",
        "model_params": XGB_PARAMS,
        "features": FEATURE_COLUMNS,
        "environment": environment_snapshot(),
        "evaluation": {
            "scheme": "chronological, purged, nested",
            "train_fraction": TRAIN_FRACTION,
            "validation_fraction": VALIDATION_FRACTION,
            "test_fraction": round(1 - TRAIN_FRACTION - VALIDATION_FRACTION, 4),
            "purge_rows": PURGE_ROWS,
            "alpha_shrinkage": shrinkage,
            "refit_on_all_rows": refit_full,
            "metrics_scope": (
                "held-out test block; the published model was refit on all "
                "rows" if refit_full else
                "held-out test block, measured on the published model itself"
            ),
        },
        "data": {
            "rows_in_store": rows_in_store,
            "usable_rows": total,
            "train_rows": fit_end,
            "validation_rows": validation.stop - validation.start,
            "test_rows": total - test.start,
            "first_timestamp": raw["timestamp"].iloc[0].isoformat(),
            "last_timestamp": raw["timestamp"].iloc[-1].isoformat(),
            "missing_hourly_rows": missing_hours,
        },
        "metrics": metrics,
        "target_transforms": transforms,
        "explanations": explanations,
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
            explanations=metadata["explanations"].get(horizon, {}),
            target_transform=metadata["target_transforms"][horizon],
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
        "--shrinkage",
        type=float,
        default=ALPHA_SHRINKAGE,
        help=(
            "Factor applied to the fitted correction weight; 0 forecasts pure "
            "persistence, 1 uses the unshrunk least-squares fit"
        ),
    )

    parser.add_argument(
        "--refit-full",
        action="store_true",
        help=(
            "After evaluating, refit on every row before publishing. Uses "
            "more recent data, but the metrics then describe a model trained "
            "on less of it"
        ),
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
        min_rows=args.min_rows,
        model_dir=Path(args.model_dir),
        save=not args.no_save,
        publish=not args.no_publish,
        promotion=args.promotion,
        keep=args.keep,
        tolerance=args.tolerance,
        shrinkage=args.shrinkage,
        refit_full=args.refit_full,
    )


if __name__ == "__main__":
    main()
