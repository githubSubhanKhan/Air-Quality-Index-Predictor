"""
SHAP explanations for the AQI forecasts.

Two kinds of explanation come out of the same contributions:

* **Local** — why *this* forecast came out where it did. Each feature gets a
  signed contribution in AQI points, and ``base_value + sum(contributions)``
  reconstructs the model's prediction exactly.
* **Global** — which features drive a model version overall, as mean |SHAP|
  over the evaluation rows. Training records this in the model registry, so
  every registered version carries its own explanation next to its metrics.

Because training picks a winner from a candidate slate (see
``app/src/training/candidates.py``), the estimator behind a horizon is not
necessarily a tree. The right method is chosen from what the fitted estimator
exposes rather than from anything recorded alongside it, so artifacts
registered before selection existed — and the local ``.pkl`` fallback — are
explained the same way:

* **linear** (anything whose final step has ``coef_``) — exact contributions,
  ``coef_j * z_j``, computed after the pipeline's own preprocessing. The
  candidate slate standardises features before Ridge, and ``StandardScaler``
  centres on the training mean, so ``z = 0`` *is* the training mean: the
  contributions are measured against the average row and ``intercept_`` is
  the baseline. (A linear model fitted without a scaler still reconstructs
  its prediction exactly, but its baseline is the origin rather than the
  data mean.)
* **tree** — ``shap.TreeExplainer``, which covers XGBoost, Random Forest and
  HistGradientBoosting. XGBoost implements the same TreeSHAP algorithm
  internally (``pred_contribs=True``), so if ``shap`` is not installed — or a
  version of it misbehaves — XGBoost explanations still work from the booster
  instead of the whole dashboard losing them.
* **anything else**, including the ``persistence`` reference model — zero
  contributions against a baseline equal to the prediction. The identity
  above still holds, and the reported method says plainly that no feature
  attribution was possible, rather than the caller raising.
"""

import numpy as np
from xgboost import DMatrix

from app.src.prediction.predictor import HORIZONS, anchor_value, get_bundle
from app.src.registry import model_registry as registry

METHOD_SHAP = "shap.TreeExplainer"

METHOD_BOOSTER = "xgboost.pred_contribs"

METHOD_LINEAR = "linear.exact"

METHOD_NONE = "unavailable.no_attribution"

# Features shown per explanation before the rest are folded away.
DEFAULT_TOP_N = 8

# Rows used for the global (mean |SHAP|) view. TreeSHAP is exact and fast on
# these models, but the cap keeps the training job's runtime predictable.
DEFAULT_GLOBAL_SAMPLE = 1000

_explainers = {}

_shap_available = None


def _final_estimator(model):
    """The estimator at the end of a pipeline, or the model itself."""

    steps = getattr(model, "steps", None)

    return steps[-1][1] if steps else model


def _preprocess(model, X):
    """``X`` as the final estimator sees it, with pipeline steps applied."""

    steps = getattr(model, "steps", None)

    if not steps or len(steps) < 2:
        return X

    return model[:-1].transform(X)


def _tree_explainer(model):
    """
    A cached ``shap.TreeExplainer``, or None if it cannot explain this model.

    Failures are cached too: the dispatch below calls this on every row of
    every request, and re-raising inside shap for a model it will never
    support is not free.
    """

    global _shap_available

    if _shap_available is False:
        return None

    key = id(model)

    if key in _explainers:
        # The model is kept in the cache value too, so its id cannot be
        # recycled by another object while the explainer is alive.
        return _explainers[key][1]

    try:
        import shap

    except ImportError:
        _shap_available = False

        return None

    _shap_available = True

    try:
        explainer = shap.TreeExplainer(model)

    except Exception:
        explainer = None

    _explainers[key] = (model, explainer)

    return explainer


def _linear_contributions(model, X):
    """Exact per-feature contributions for a linear model."""

    estimator = _final_estimator(model)

    transformed = np.asarray(_preprocess(model, X), dtype=float)

    coefficients = np.asarray(estimator.coef_, dtype=float).ravel()

    intercept = float(np.asarray(estimator.intercept_, dtype=float).ravel()[0])

    values = transformed * coefficients

    return values, np.full(len(X), intercept), METHOD_LINEAR


def _no_contributions(model, X):
    """
    Zero attribution, with the prediction itself as the baseline.

    Used for models with no notion of feature effect — the ``persistence``
    reference candidate, in practice. Keeps the reconstruction identity
    ``base_value + sum(contributions) == prediction`` true for callers.
    """

    predictions = np.asarray(model.predict(X), dtype=float).ravel()

    values = np.zeros((len(X), X.shape[1]), dtype=float)

    return values, predictions, METHOD_NONE


def _booster_contributions(model, X):
    """XGBoost's own TreeSHAP, used when shap is unavailable or fails."""

    contributions = np.asarray(
        model.get_booster().predict(DMatrix(X), pred_contribs=True),
        dtype=float,
    )

    # The last column is the bias term, i.e. the expected value.
    return contributions[:, :-1], contributions[:, -1], METHOD_BOOSTER


def shap_contributions(model, X):
    """
    Per-feature contributions for every row of ``X``.

    Returns ``(values, base_values, method)`` where ``values`` has shape
    ``(rows, features)`` and ``base_values`` has one entry per row. For every
    method, ``base_values[i] + values[i].sum()`` is the model's prediction for
    row ``i``.
    """

    estimator = _final_estimator(model)

    if hasattr(estimator, "coef_"):
        return _linear_contributions(model, X)

    explainer = _tree_explainer(model)

    if explainer is not None:
        try:
            values = np.asarray(explainer.shap_values(X), dtype=float)

            expected = np.asarray(explainer.expected_value, dtype=float)

            base = np.full(len(X), float(expected.ravel()[0]))

            return values, base, METHOD_SHAP

        except Exception as exc:
            print(
                f"shap.TreeExplainer failed ({exc}); "
                f"trying XGBoost's own TreeSHAP"
            )

    if hasattr(estimator, "get_booster"):
        try:
            return _booster_contributions(estimator, X)

        except Exception as exc:
            print(f"XGBoost TreeSHAP failed ({exc})")

    print(
        f"No feature attribution available for "
        f"{type(estimator).__name__}; reporting zero contributions"
    )

    return _no_contributions(model, X)


def explain_row(model, X, columns, top_n: int = DEFAULT_TOP_N) -> dict:
    """
    Explain a single prediction: what pushed it up, what pulled it down.

    Contributions are in AQI points, ordered by absolute effect. Anything
    past ``top_n`` is summed into ``other_contribution`` so the parts still
    add up to the prediction.
    """

    values, base, method = shap_contributions(model, X.iloc[:1])

    row = values[0]

    contributions = [
        {
            "feature": column,
            "feature_value": round(float(X.iloc[0][column]), 4),
            "contribution": round(float(value), 4),
            "effect": "increases" if value > 0 else "decreases",
        }
        for column, value in zip(columns, row)
    ]

    contributions.sort(key=lambda item: abs(item["contribution"]), reverse=True)

    shown = contributions[:top_n] if top_n else contributions

    hidden = contributions[len(shown):]

    base_value = float(base[0])

    return {
        "method": method,
        "base_value": round(base_value, 4),
        "prediction": round(base_value + float(row.sum()), 4),
        "contributions": shown,
        "other_contribution": round(
            sum(item["contribution"] for item in hidden), 4
        ),
        "features_hidden": len(hidden),
    }


def _to_forecast_space(explanation: dict, alpha: float, anchor: float) -> dict:
    """
    Move an explanation from correction space into AQI-forecast space.

    The model predicts a deviation that serving damps by ``alpha`` and adds
    to the anchor, so every contribution scales by the same alpha and the
    baseline shifts by the anchor. ``base_value`` plus the contributions then
    reconstructs the forecast the dashboard displays, not the raw deviation.
    """

    explanation = dict(explanation)

    for contribution in explanation["contributions"]:
        contribution["contribution"] = round(
            contribution["contribution"] * alpha, 4
        )

    explanation["other_contribution"] = round(
        explanation["other_contribution"] * alpha, 4
    )

    explanation["base_value"] = round(
        anchor + alpha * explanation["base_value"], 4
    )

    explanation["prediction"] = round(
        anchor + alpha * explanation["prediction"], 4
    )

    explanation["anchor"] = round(anchor, 4)
    explanation["alpha"] = alpha
    explanation["explains"] = (
        "damped correction to a persistence forecast of "
        f"{anchor:.1f} AQI"
    )

    return explanation


def explain_prediction(
    features_df,
    horizons=None,
    top_n: int = DEFAULT_TOP_N,
) -> dict:
    """
    Explain the 3-day forecast for one prepared feature row.

    Each horizon is explained with its own production model and that
    version's own feature list.
    """

    bundle = get_bundle()

    horizons = horizons or list(HORIZONS)

    explanations = {}

    for horizon in horizons:
        columns = bundle["features"][horizon]

        document = bundle["documents"].get(horizon)

        explanation = explain_row(
            bundle["models"][horizon],
            features_df[columns],
            columns,
            top_n=top_n,
        )

        transform = (document or {}).get("target_transform") or {}

        if transform.get("mode") == registry.TRANSFORM_DELTA_FROM_ANCHOR:
            explanation = _to_forecast_space(
                explanation,
                float(transform.get("alpha", 1.0)),
                anchor_value(document, features_df),
            )

        explanation["model"] = {
            "name": document["name"] if document else f"xgboost_{horizon}",
            "version": document["version"] if document else None,
            "candidate": (document or {}).get("candidate"),
            "model_type": (document or {}).get("model_type"),
        }

        explanations[horizon] = explanation

    return {
        "source": bundle["source"],
        "horizons": explanations,
    }


def global_importance(
    model,
    X,
    columns,
    sample: int = DEFAULT_GLOBAL_SAMPLE,
) -> dict:
    """
    Mean |SHAP| per feature — the global view stored with each version.

    Computed on the most recent ``sample`` evaluation rows, so the ranking
    describes the data the model was actually scored on.
    """

    if len(X) > sample:
        X = X.iloc[-sample:]

    values, base, method = shap_contributions(model, X)

    ranking = [
        {
            "feature": column,
            "mean_abs_shap": round(float(value), 4),
        }
        for column, value in zip(columns, np.abs(values).mean(axis=0))
    ]

    ranking.sort(key=lambda item: item["mean_abs_shap"], reverse=True)

    return {
        "method": method,
        "base_value": round(float(base.mean()), 4),
        "sample_rows": int(len(X)),
        "features": ranking,
    }
