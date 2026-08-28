"""
Serving-side model loading and forecasting.

Predictions come from whatever is in the **production** stage of the model
registry, so promoting or rolling back a version changes what serving uses
without a redeploy. Each horizon carries its own feature list *and its own
target transform*, so a horizon can be retrained on different features, or
switched between predicting the AQI level and predicting a correction to
persistence, without touching the other two.

The git-ignored ``.pkl`` files in ``app/models`` are a fallback for local
development and for the moment before the first version is promoted; the
metadata written next to them keeps their transforms intact.
"""

import json
import time
from pathlib import Path

import joblib

from app.src.registry import model_registry as registry

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"

METADATA_FILENAME = "training_metadata.json"

HORIZONS = ("day1", "day2", "day3")

# Downloaded models are cached for the life of the process, but the production
# version numbers are re-checked this often (a small metadata query) so a
# promotion or rollback is picked up without restarting the API or dashboard.
CACHE_TTL_SECONDS = 900

_bundle = None

_checked_at = 0.0


def _load_from_registry() -> dict:
    models = {}
    features = {}
    documents = {}

    for horizon in HORIZONS:
        model, document = registry.load_production_model(
            registry.model_name(horizon)
        )

        models[horizon] = model
        features[horizon] = document["features"]
        documents[horizon] = document

    return {
        "source": "registry",
        "models": models,
        "features": features,
        "documents": documents,
    }


def _local_artifact(horizon: str) -> Path:
    """
    The local .pkl for a horizon.

    Training writes ``model_<horizon>.pkl``, since the estimator inside is
    whichever candidate won and is not necessarily XGBoost. The older
    ``xgboost_<horizon>.pkl`` name is still accepted so a working tree that
    has not been retrained since keeps serving.
    """

    current = MODEL_DIR / f"model_{horizon}.pkl"

    if current.exists():
        return current

    return MODEL_DIR / f"xgboost_{horizon}.pkl"


def _local_metadata() -> dict:
    path = MODEL_DIR / METADATA_FILENAME

    if not path.exists():
        return {}

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_from_disk() -> dict:
    columns = joblib.load(MODEL_DIR / "feature_columns.pkl")

    metadata = _local_metadata()

    transforms = metadata.get("target_transforms", {})

    models = {}
    features = {}
    documents = {}

    chosen = metadata.get("models", {})

    for horizon in HORIZONS:
        models[horizon] = joblib.load(_local_artifact(horizon))

        # Per-horizon first: a statistical winner reads the raw history block
        # rather than the curated feature list, exactly as it does when loaded
        # from the registry. Older metadata has only the global list.
        features[horizon] = (
            chosen.get(horizon, {}).get("features")
            or metadata.get("features")
            or columns
        )

        documents[horizon] = {
            "name": f"xgboost_{horizon}",
            "version": None,
            "stage": "local file",
            "features": features[horizon],
            "target_transform": transforms.get(horizon, {}),
            "metrics": metadata.get("metrics", {}).get(horizon, {}),
            "candidate": chosen.get(horizon, {}).get("candidate"),
            "model_family": chosen.get(horizon, {}).get("family"),
            "model_type": chosen.get(horizon, {}).get("model_type"),
            "created_at": metadata.get("trained_at"),
        }

    return {
        "source": "local",
        "models": models,
        "features": features,
        "documents": documents,
    }


def _serving_versions() -> dict:
    return {
        document["name"]: document["version"]
        for document in _bundle["documents"].values()
    }


def _reload() -> dict:
    global _bundle, _checked_at

    _checked_at = time.monotonic()

    try:
        _bundle = _load_from_registry()

    except Exception as exc:
        # Registry unreachable, or no version promoted yet. Fall back to the
        # local copy so development still works; if that is missing too the
        # error surfaces to the caller.
        print(
            f"Model registry unavailable ({exc}); "
            f"falling back to local models in {MODEL_DIR}"
        )

        _bundle = _load_from_disk()

    return _bundle


def get_bundle(refresh: bool = False) -> dict:
    """
    The models currently serving predictions, with their feature lists.

    Cached per process. Once ``CACHE_TTL_SECONDS`` has passed the production
    version numbers are re-checked and the artifacts are only re-downloaded
    if something was promoted or rolled back. Pass ``refresh=True`` to force
    a reload immediately.
    """

    global _checked_at

    if refresh or _bundle is None:
        return _reload()

    if time.monotonic() - _checked_at < CACHE_TTL_SECONDS:
        return _bundle

    _checked_at = time.monotonic()

    try:
        promoted = registry.production_versions(
            registry.model_name(horizon) for horizon in HORIZONS
        )

    except Exception:
        # Registry unreachable — keep serving the models already loaded.
        return _bundle

    if _bundle["source"] != "registry" or promoted != _serving_versions():
        return _reload()

    return _bundle


def anchor_value(document: dict, features_df) -> float:
    """The persistence value a horizon's correction is applied to."""

    transform = (document or {}).get("target_transform") or {}

    column = transform.get("anchor", "aqi")

    return float(features_df[column].iloc[0])


def forecast_horizon(horizon: str, features_df, bundle=None) -> float:
    """One horizon's AQI forecast, with its target transform applied."""

    bundle = bundle or get_bundle()

    document = bundle["documents"].get(horizon)

    X = features_df[bundle["features"][horizon]]

    raw = float(bundle["models"][horizon].predict(X)[0])

    return float(registry.transform_forecast(
        (document or {}).get("target_transform"),
        anchor_value(document, features_df),
        raw,
    ))


def predict(features_df, refresh: bool = False) -> dict:
    """
    Predict AQI for the next 3 days.
    """

    bundle = get_bundle(refresh=refresh)

    forecast = {
        "current_aqi": round(float(features_df["aqi"].iloc[0]), 2),
    }

    for index, horizon in enumerate(HORIZONS, start=1):
        value = forecast_horizon(horizon, features_df, bundle)

        forecast[f"day_{index}"] = round(value, 2)

    return forecast


def model_info(refresh: bool = False) -> dict:
    """Provenance of the models behind the current forecast."""

    bundle = get_bundle(refresh=refresh)

    horizons = {}

    for horizon in HORIZONS:
        document = bundle["documents"].get(horizon)

        horizons[horizon] = (
            registry.summarise(document)
            if document is not None
            else {
                "name": f"xgboost_{horizon}",
                "version": None,
                "stage": "unknown",
                "metrics": {},
            }
        )

    return {
        "source": bundle["source"],
        "horizons": horizons,
    }
