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

**Which** model provides that correction is decided per retrain, not assumed.
Every horizon trains the candidate slate in ``candidates.py`` — persistence,
Ridge, Random Forest, HistGradientBoosting, XGBoost — and keeps whichever wins
on the validation window, subject to a margin that stops the served family
flip-flopping on noise. All five candidates' metrics are recorded with the
version, so the comparison behind the choice is auditable after the fact
rather than living in a notebook that was run once.

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
    python -m app.src.training.train --candidates xgboost --no-publish
    python -m app.src.training.train --candidates all --allow-baseline
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
import xgboost
from dotenv import load_dotenv

from app.src.explain.explainer import global_importance
from app.src.features.feature_engineering import build_training_features
from app.src.features.feature_store import get_collection
from app.src.registry import model_registry as registry
from app.src.training import candidates as zoo
from app.src.training import selection

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

# Chronological split. The test block stays the final 20% of rows, unchanged
# from the previous pipeline, so the reported metrics remain comparable; the
# validation block that fits alpha and ranks the candidates is carved out of
# the training portion.
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


def artifact_filename(horizon: str) -> str:
    return f"model_{horizon}.pkl"


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
    slate=None,
    select_metric: str = selection.DEFAULT_SELECTION_METRIC,
    select_margin: float = selection.DEFAULT_SELECTION_MARGIN,
    default_model: str = zoo.DEFAULT_MODEL,
    allow_baseline: bool = False,
) -> dict:
    """
    Retrain every forecast horizon and return the run's metadata.

    Nothing is published until all three horizons have trained, so a failure
    part-way through leaves the current production models serving.
    """

    candidate_slate = zoo.resolve_slate(slate)

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
    # alpha fit and candidate ranking stop 72 rows short of the test block.
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
        f"Candidates    : {', '.join(c.name for c in candidate_slate)}\n"
        f"Selection     : validation {select_metric}, "
        f"{select_margin:.1%} margin over '{default_model}'"
        f"{'' if allow_baseline else ', reference models excluded'}\n"
    )

    models = {}
    metrics = {}
    transforms = {}
    explanations = {}
    chosen = {}
    comparisons = {}

    for horizon in HORIZONS:
        print(f"{horizon}")

        y_absolute = df[target_column(horizon)]

        trials = selection.evaluate_slate(
            candidate_slate,
            X,
            y_absolute,
            anchor,
            fit_end,
            validation,
            test,
            shrinkage,
        )

        winner, reason = selection.select(
            trials,
            metric=select_metric,
            margin=select_margin,
            default_model=default_model,
            allow_baseline=allow_baseline,
        )

        winner.selected = True

        print(selection.comparison_table(trials, metric=select_metric))

        print(f"        selected {winner.candidate.name}: {reason}")

        comparisons[horizon] = [trial.record() for trial in trials]

        model = winner.model

        alpha = winner.alpha

        if refit_full:
            # More recent data at the cost of alpha having been fitted for a
            # model trained on less of it.
            model = winner.candidate.build()

            model.fit(X, y_absolute - anchor)

        models[horizon] = model

        metrics[horizon] = {
            **winner.test,
            "alpha": alpha,
            "alpha_unshrunk": winner.alpha_unshrunk,
            "validation": winner.validation,
        }

        transforms[horizon] = {
            "mode": TRANSFORM_MODE,
            "anchor": ANCHOR_COLUMN,
            "alpha": alpha,
        }

        chosen[horizon] = {
            "candidate": winner.candidate.name,
            "label": winner.candidate.label,
            "family": winner.candidate.family,
            "model_type": type(model).__name__,
            "params": dict(winner.candidate.params),
            "alpha": alpha,
            "reason": reason,
            "notes": winner.notes,
        }

        entry = winner.test

        print(
            f"        winner: "
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

            print(f"        top SHAP drivers: {drivers}\n")

        except Exception as exc:
            print(f"        SHAP importance unavailable: {exc}\n")

            explanations[horizon] = {}

    trained_at = datetime.now(timezone.utc)

    metadata = {
        "run_id": trained_at.strftime("%Y%m%dT%H%M%SZ"),
        "trained_at": trained_at.isoformat(timespec="seconds"),
        "city": city.lower(),
        "models": chosen,
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
            "selection": {
                "slate": [c.name for c in candidate_slate],
                "metric": select_metric,
                "scope": "validation block",
                "margin": select_margin,
                "default_model": default_model,
                "reference_models_eligible": allow_baseline,
                "note": (
                    "Candidates were ranked on the validation block only. "
                    "Test metrics are reported for every candidate but took "
                    "no part in the choice."
                ),
            },
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
        "candidates": comparisons,
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
        print("--no-publish set: models were not sent to the registry.")

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

        chosen = metadata["models"][horizon]

        document = registry.register_model(
            name,
            model,
            metrics=metadata["metrics"][horizon],
            params=chosen["params"],
            features=metadata["features"],
            city=metadata["city"],
            horizon=horizon,
            run_id=metadata["run_id"],
            data=metadata["data"],
            environment=metadata["environment"],
            explanations=metadata["explanations"].get(horizon, {}),
            target_transform=metadata["target_transforms"][horizon],
            candidate=chosen["candidate"],
            model_family=chosen["family"],
            model_type=chosen["model_type"],
            selection={
                **metadata["evaluation"]["selection"],
                "reason": chosen["reason"],
                "comparison": metadata["candidates"][horizon],
            },
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
            "candidate": chosen["candidate"],
            "promoted": promote,
            "reason": reason,
            "previous_production_version": (
                incumbent["version"] if incumbent else None
            ),
            "artifacts_pruned": pruned,
        }

        print(
            f"  {name} v{version} [{chosen['candidate']}] "
            f"-> {document['stage']} ({reason})"
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
        filename = artifact_filename(horizon)

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
        "--candidates",
        default=",".join(zoo.DEFAULT_SLATE),
        help=(
            "Comma-separated candidates to evaluate per horizon, or 'all'. "
            f"Available: {', '.join(sorted(zoo.CANDIDATES))}"
        ),
    )

    parser.add_argument(
        "--select-metric",
        default=selection.DEFAULT_SELECTION_METRIC,
        choices=["mae", "rmse", "r2", "skill_vs_persistence"],
        help="Validation metric the winning candidate is chosen on",
    )

    parser.add_argument(
        "--select-margin",
        type=float,
        default=selection.DEFAULT_SELECTION_MARGIN,
        help=(
            "Relative improvement a challenger needs over "
            f"'{zoo.DEFAULT_MODEL}' before it takes the slot, e.g. 0.02 for 2%%"
        ),
    )

    parser.add_argument(
        "--default-model",
        default=zoo.DEFAULT_MODEL,
        help=(
            "The incumbent candidate: it wins ties and keeps the slot unless "
            "a challenger clears --select-margin"
        ),
    )

    parser.add_argument(
        "--allow-baseline",
        action="store_true",
        help=(
            "Let reference candidates such as 'persistence' win a slot. They "
            "are always evaluated; by default they cannot be selected, "
            "because a constant model has no SHAP explanation to show"
        ),
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
            "After evaluating, refit the winner on every row before "
            "publishing. Uses more recent data, but the metrics then describe "
            "a model trained on less of it"
        ),
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip the local .pkl copy",
    )

    parser.add_argument(
        "--metadata-out",
        default=None,
        help=(
            "Also write this run's metadata as JSON to this path, regardless "
            "of --no-save. Feeds `python -m app.src.training.report`"
        ),
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

    metadata = run(
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
        slate=args.candidates,
        select_metric=args.select_metric,
        select_margin=args.select_margin,
        default_model=args.default_model,
        allow_baseline=args.allow_baseline,
    )

    if args.metadata_out:
        path = Path(args.metadata_out)

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"Wrote run metadata to {path}")


if __name__ == "__main__":
    main()
