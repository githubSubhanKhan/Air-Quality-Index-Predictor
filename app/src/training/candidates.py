"""
The candidate model families the training pipeline chooses between.

The brief asks for the best model *for this data*, evaluated with RMSE, MAE
and R2. Until now the pipeline trained one XGBoost per horizon and assumed it
was that model; the five-way comparison that justified the choice lives in
``lag_feature_engineering.ipynb`` and was run once, against a different target
(24-hour-ahead *absolute* AQI on 26 features). Nothing re-checked the choice
after the horizons switched to a damped correction on 14 features, and nothing
re-checks it as a year of new readings arrives.

So the slate is trained per horizon on every retrain and the winner is picked
on a validation window it was not fitted on. Every candidate learns the same
thing — the deviation of the forecast window's mean AQI from the current
reading (see ``train.py``) — so they are interchangeable: serving only ever
calls ``predict``, and day 3 may end up served by a different family than
day 1.

Two things a new candidate must get right:

* **``family``** is not decoration. ``app/src/explain/explainer.py`` picks its
  SHAP path from what the fitted estimator exposes, and the family recorded
  here is what the registry and the dashboard report. A family with no
  explanation path loses the dashboard's "Why this forecast?" panel.
* **No internal validation split.** Anything that carves its own hold-out
  (early stopping, in particular) would cut it at random out of a time series,
  which leaks the future into the fit. Where a candidate supports early
  stopping it is turned off explicitly.
"""

from dataclasses import dataclass, field
from typing import Callable

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# How a fitted model can be explained. The explainer infers this from the
# estimator itself so old artifacts keep working, but it is recorded per
# version so the registry says which path a model version used.
FAMILY_TREE = "tree"

FAMILY_LINEAR = "linear"

FAMILY_CONSTANT = "constant"


# ---------------------------------------------------------------------------
# Hyperparameters
#
# The target is a deviation from persistence, which is mostly noise: the
# signal that survives 24-72 hours out is small. Every candidate is therefore
# regularised hard — shallow trees, large leaves, strong penalties. Left
# looser, each one fits the training block's noise and scores worse on
# validation than doing nothing at all.
# ---------------------------------------------------------------------------

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

RANDOM_FOREST_PARAMS = {
    "n_estimators": 400,
    "max_depth": 6,
    "min_samples_leaf": 25,
    "max_features": 0.5,
    # Squared error, unlike the boosters: a forest with absolute-error splits
    # costs roughly an order of magnitude more to fit for no gain seen here,
    # and the daily retrain has to finish inside a GitHub Actions runner.
    "random_state": 42,
    "n_jobs": -1,
}

HIST_GBM_PARAMS = {
    "loss": "absolute_error",
    "max_iter": 400,
    "learning_rate": 0.03,
    "max_depth": 3,
    "min_samples_leaf": 25,
    "l2_regularization": 5.0,
    # Default is "auto", which switches on above 10k samples and would take
    # its hold-out at random from a time series.
    "early_stopping": False,
    "random_state": 42,
}

RIDGE_PARAMS = {
    "alpha": 10.0,
}


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _xgboost(params: dict):
    return XGBRegressor(**params)


def _random_forest(params: dict):
    return RandomForestRegressor(**params)


def _hist_gbm(params: dict):
    return HistGradientBoostingRegressor(**params)


def _ridge(params: dict):
    """
    Ridge on standardised features.

    The features are on wildly different scales — AQI points, sine/cosine
    terms in [-1, 1], wind speed in m/s — so an unscaled penalty would fall
    almost entirely on the trigonometric terms. ``StandardScaler`` keeps the
    column order, which is what lets the explainer map coefficients back to
    feature names.
    """

    return Pipeline([
        ("scale", StandardScaler()),
        ("model", Ridge(**params)),
    ])


def _persistence(params: dict):
    """
    The null model: predict zero deviation, i.e. tomorrow looks like now.

    Kept as a real candidate rather than a number in a table so it is fitted,
    scored and reported through exactly the same code path as the rest — the
    comparison is then honest about how much any model actually adds. It is
    excluded from *selection* by default (see ``is_baseline``): a constant has
    no explanation to show, and the dashboard's SHAP panel is a deliverable.
    """

    return DummyRegressor(strategy="constant", constant=0.0)


@dataclass(frozen=True)
class Candidate:
    """One entry on the slate, and how to build a fresh copy of it."""

    name: str
    label: str
    family: str
    builder: Callable[[dict], object]
    params: dict = field(default_factory=dict)

    # Reference models: always evaluated, never promoted unless asked for.
    is_baseline: bool = False

    def build(self):
        return self.builder(dict(self.params))


CANDIDATES = {
    candidate.name: candidate
    for candidate in (
        Candidate(
            name="persistence",
            label="Persistence (no model)",
            family=FAMILY_CONSTANT,
            builder=_persistence,
            is_baseline=True,
        ),
        Candidate(
            name="ridge",
            label="Ridge Regression",
            family=FAMILY_LINEAR,
            builder=_ridge,
            params=RIDGE_PARAMS,
        ),
        Candidate(
            name="random_forest",
            label="Random Forest",
            family=FAMILY_TREE,
            builder=_random_forest,
            params=RANDOM_FOREST_PARAMS,
        ),
        Candidate(
            name="hist_gbm",
            label="HistGradientBoosting",
            family=FAMILY_TREE,
            builder=_hist_gbm,
            params=HIST_GBM_PARAMS,
        ),
        Candidate(
            name="xgboost",
            label="XGBoost",
            family=FAMILY_TREE,
            builder=_xgboost,
            params=XGB_PARAMS,
        ),
    )
}

# Evaluated on every retrain, cheapest first so a failure surfaces early.
DEFAULT_SLATE = (
    "persistence",
    "ridge",
    "random_forest",
    "hist_gbm",
    "xgboost",
)

# The incumbent. It wins ties and keeps the slot unless a challenger clears
# the selection margin, which stops the served family flip-flopping between
# daily retrains on differences that are inside the noise.
DEFAULT_MODEL = "xgboost"


def get(name: str) -> Candidate:
    if name not in CANDIDATES:
        raise KeyError(
            f"Unknown candidate '{name}'. "
            f"Available: {', '.join(sorted(CANDIDATES))}"
        )

    return CANDIDATES[name]


def resolve_slate(names=None) -> list:
    """
    The candidates to evaluate, de-duplicated and in a stable order.

    ``names`` may be a comma-separated string (from the CLI) or a sequence.
    ``"all"`` expands to every registered candidate.
    """

    if names is None:
        names = DEFAULT_SLATE

    if isinstance(names, str):
        names = [part.strip() for part in names.split(",") if part.strip()]

    if list(names) == ["all"]:
        names = list(CANDIDATES)

    seen = []

    for name in names:
        candidate = get(name)

        if candidate.name not in seen:
            seen.append(candidate.name)

    if not seen:
        raise ValueError("The candidate slate is empty.")

    return [CANDIDATES[name] for name in seen]
