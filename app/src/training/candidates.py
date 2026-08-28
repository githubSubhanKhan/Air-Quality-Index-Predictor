"""
The candidate model families the training pipeline chooses between.

The brief asks for the best model *for this data*, evaluated with RMSE, MAE
and R2, and for a variety of approaches "from statistical modelling to deep
learning". Until recently the pipeline trained one XGBoost per horizon and
assumed it was that model; the five-model comparison that justified the choice
lives in ``lag_feature_engineering.ipynb``, was run once, and was run against
a different target (24-hour-ahead *absolute* AQI on 26 features). Nothing
re-checked the choice after the horizons switched to a damped correction on 14
features, and nothing re-checked it as a year of new readings arrived.

So the slate is trained per horizon on every retrain and the winner is picked
on a validation window it was not fitted on. It spans three kinds of model:

* **Reference** — ``persistence`` (tomorrow equals now) and ``seasonal_naive``
  (this hour tomorrow equals this hour today). The bars everything else has
  to clear.
* **Statistical** — ``holt_winters`` (exponential smoothing / ETS) and
  ``seasonal_ar`` (seasonal differencing plus an autoregressive model, i.e.
  the ARIMA family). These describe the AQI *series*; see
  ``statistical.py``.
* **Machine learning** — ``ridge``, ``random_forest``, ``hist_gbm`` and
  ``xgboost``, which learn a mapping from engineered features to the target.

Every candidate predicts the same quantity — the deviation of the forecast
window's mean AQI from the current reading (see ``train.py``) — so they are
interchangeable and directly comparable: serving only ever calls ``predict``,
and day 3 may end up served by a different family than day 1.

Four things a new candidate must get right:

* **``feature_set``** decides which columns it sees. The ML models take the
  curated list in ``train.FEATURE_COLUMNS``; the series models take the raw
  hourly history block. The registry stores a feature list per horizon, which
  is what lets these coexist without serving needing to know the difference.
* **``family``** is reported and grouped on, but is not what picks the
  explanation method — ``explain/explainer.py`` infers that from the fitted
  estimator, so old artifacts keep working.
* **``selectable``** says whether the candidate may take a production slot.
  Reference and statistical models are always trained and scored but not
  promoted by default, because neither offers per-feature attribution and the
  dashboard's SHAP panel is a deliverable. ``--allow-reference`` overrides it,
  and the run reports loudly when a non-selectable candidate would have won.
* **No internal validation split.** Anything that carves its own hold-out
  (early stopping, in particular) would cut it at random out of a time
  series, which leaks the future into the fit. Where a candidate supports
  early stopping it is turned off explicitly.
"""

from dataclasses import dataclass, field
from typing import Callable

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from app.src.training.statistical import (
    HoltWintersForecaster,
    SeasonalArForecaster,
    SeasonalNaiveForecaster,
)

# Which columns a candidate is given.
FEATURE_SET_MODEL = "model"

FEATURE_SET_HISTORY = "history"

# Broad grouping, for reporting.
FAMILY_TREE = "tree"

FAMILY_LINEAR = "linear"

FAMILY_CONSTANT = "constant"

FAMILY_SEASONAL = "seasonal"

FAMILY_STATISTICAL = "statistical"

# Why a candidate is scored but not promoted.
NO_ATTRIBUTION = (
    "no per-feature attribution, so the dashboard's SHAP panel would be empty"
)


# ---------------------------------------------------------------------------
# Hyperparameters
#
# The target is a deviation from persistence, which is mostly noise: the
# signal that survives 24-72 hours out is small. Every ML candidate is
# therefore regularised hard — shallow trees, large leaves, strong penalties.
# Left looser, each one fits the training block's noise and scores worse on
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

# The series models take their period from here. 24 is not a guess: the EDA
# notebook's average-AQI-by-hour plot is a clean daily cycle.
SEASONAL_PARAMS = {
    "season": 24,
}


# ---------------------------------------------------------------------------
# Builders
#
# Each takes the candidate's configured params plus the horizon it is being
# built for, in hours. The feature-based models ignore the horizon — they get
# a different target column instead — while the series models need it, since
# they forecast a path and average the slice it names.
# ---------------------------------------------------------------------------

def _xgboost(params: dict, horizon_hours: int):
    return XGBRegressor(**params)


def _random_forest(params: dict, horizon_hours: int):
    return RandomForestRegressor(**params)


def _hist_gbm(params: dict, horizon_hours: int):
    return HistGradientBoostingRegressor(**params)


def _ridge(params: dict, horizon_hours: int):
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


def _persistence(params: dict, horizon_hours: int):
    """
    The null model: predict zero deviation, i.e. tomorrow looks like now.

    Kept as a real candidate rather than a number in a table so it is fitted,
    scored and reported through exactly the same code path as the rest — the
    comparison is then honest about how much any model actually adds.
    """

    return DummyRegressor(strategy="constant", constant=0.0)


def _seasonal_naive(params: dict, horizon_hours: int):
    return SeasonalNaiveForecaster(horizon_hours=horizon_hours, **params)


def _holt_winters(params: dict, horizon_hours: int):
    return HoltWintersForecaster(horizon_hours=horizon_hours, **params)


def _seasonal_ar(params: dict, horizon_hours: int):
    return SeasonalArForecaster(horizon_hours=horizon_hours, **params)


@dataclass(frozen=True)
class Candidate:
    """One entry on the slate, and how to build a fresh copy of it."""

    name: str
    label: str
    family: str
    builder: Callable[[dict, int], object]
    params: dict = field(default_factory=dict)
    feature_set: str = FEATURE_SET_MODEL

    # May this candidate take a production slot?
    selectable: bool = True
    reference_note: str = ""

    def build(self, horizon_hours: int):
        return self.builder(dict(self.params), horizon_hours)


CANDIDATES = {
    candidate.name: candidate
    for candidate in (
        Candidate(
            name="persistence",
            label="Persistence (no model)",
            family=FAMILY_CONSTANT,
            builder=_persistence,
            selectable=False,
            reference_note=f"a constant has {NO_ATTRIBUTION}",
        ),
        Candidate(
            name="seasonal_naive",
            label="Seasonal naive (24h)",
            family=FAMILY_SEASONAL,
            builder=_seasonal_naive,
            params=SEASONAL_PARAMS,
            feature_set=FEATURE_SET_HISTORY,
            selectable=False,
            reference_note=f"a benchmark with {NO_ATTRIBUTION}",
        ),
        Candidate(
            name="holt_winters",
            label="Holt-Winters ETS (damped, seasonal)",
            family=FAMILY_STATISTICAL,
            builder=_holt_winters,
            params=SEASONAL_PARAMS,
            feature_set=FEATURE_SET_HISTORY,
            selectable=False,
            reference_note=f"a series model with {NO_ATTRIBUTION}",
        ),
        Candidate(
            name="seasonal_ar",
            label="Seasonal AR / SARIMA(p,0,0)(0,1,0)[24]",
            family=FAMILY_STATISTICAL,
            builder=_seasonal_ar,
            params=SEASONAL_PARAMS,
            feature_set=FEATURE_SET_HISTORY,
            selectable=False,
            reference_note=f"a series model with {NO_ATTRIBUTION}",
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

# Evaluated on every retrain: references, then statistical, then ML.
DEFAULT_SLATE = (
    "persistence",
    "seasonal_naive",
    "holt_winters",
    "seasonal_ar",
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


def feature_sets_used(slate) -> list:
    """The distinct feature sets a slate needs, so the caller can build them."""

    used = []

    for candidate in slate:
        if candidate.feature_set not in used:
            used.append(candidate.feature_set)

    return used
