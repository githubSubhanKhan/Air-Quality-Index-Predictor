"""
Serving-side model loading.

Predictions are served from whatever is in the **production** stage of the
model registry, so promoting or rolling back a version changes what serving
uses without a redeploy. Each horizon carries its own feature list from the
registry, which means a horizon can be retrained on a different feature set
without breaking the other two.

The git-ignored ``.pkl`` files in ``app/models`` are only a fallback for
local development and for the moment before the first version is promoted.
"""

import time
from pathlib import Path

import joblib

from app.src.registry import model_registry as registry

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"

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


def _load_from_disk() -> dict:
    columns = joblib.load(MODEL_DIR / "feature_columns.pkl")

    models = {
        horizon: joblib.load(MODEL_DIR / f"xgboost_{horizon}.pkl")
        for horizon in HORIZONS
    }

    return {
        "source": "local",
        "models": models,
        "features": {horizon: columns for horizon in HORIZONS},
        "documents": {},
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


def predict(features_df, refresh: bool = False) -> dict:
    """
    Predict AQI for the next 3 days.
    """

    bundle = get_bundle(refresh=refresh)

    forecast = {
        "current_aqi": round(float(features_df["aqi"].iloc[0]), 2),
    }

    for index, horizon in enumerate(HORIZONS, start=1):
        X = features_df[bundle["features"][horizon]]

        value = float(bundle["models"][horizon].predict(X)[0])

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
                "stage": "local file",
                "metrics": {},
            }
        )

    return {
        "source": bundle["source"],
        "horizons": horizons,
    }
