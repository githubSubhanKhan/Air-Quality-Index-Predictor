"""
Evaluating the candidate slate for one horizon, and picking the winner.

The split is nested and chronological: **train** fits each candidate,
**validation** fits its correction weight and decides which candidate wins,
**test** is scored once at the end and never consulted before the choice is
made. Boundaries are purged by 72 rows in ``train.py``, because a row's target
window reaches that far ahead.

Two consequences worth being explicit about, since both flatter the numbers if
they go unsaid:

* The validation block does double duty — it fits ``alpha`` *and* ranks the
  candidates. The winner's validation metrics are therefore mildly optimistic
  and are reported for the ranking only. Test metrics are what the model is
  judged on, and every candidate's are recorded so the losers can be compared
  too.
* Test metrics are computed for candidates that were *not* selected purely for
  the write-up. They played no part in the choice. Selecting on them would
  turn the hold-out into another validation set.

The correction weight makes the comparison fair in a way that scoring raw
model output would not. Each candidate gets its own ``alpha``, fitted on
validation, so a family whose predictions are mostly noise is damped towards
persistence automatically rather than being punished for its variance. It also
puts the feature-based and series-based models on the same footing: both are
scored on the reconstructed AQI forecast, not on their own internal quantity.
"""

import time
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from app.src.registry.model_registry import LOWER_IS_BETTER
from app.src.training import candidates as zoo

# Ranked on absolute error, matching the registry's promotion gate: the series
# is spiky and MAE is what the dashboard's error figure means to a reader.
DEFAULT_SELECTION_METRIC = "mae"

# A challenger must beat the incumbent by this much, relatively, on the
# selection metric before it takes the slot. Differences smaller than this are
# inside what a 1,000-row autocorrelated validation window can resolve, and
# acting on them would swap the served family between daily retrains for no
# real gain — while resetting the SHAP explanations a reader has got used to.
DEFAULT_SELECTION_MARGIN = 0.02


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


@dataclass
class Trial:
    """One candidate, fitted and scored for one horizon."""

    candidate: zoo.Candidate
    model: object
    alpha: float
    alpha_unshrunk: float
    validation: dict
    test: dict
    feature_count: int = 0
    fit_seconds: float = 0.0
    predict_seconds: float = 0.0
    selected: bool = False
    notes: str = ""

    def value(self, metric: str, scope: str = "validation") -> float:
        metrics = self.validation if scope == "validation" else self.test

        return metrics[metric]

    @property
    def model_type(self) -> str:
        return type(self.model).__name__

    def record(self) -> dict:
        """JSON-safe row for the metadata and the registry document."""

        return {
            "candidate": self.candidate.name,
            "label": self.candidate.label,
            "family": self.candidate.family,
            "model_type": self.model_type,
            "feature_set": self.candidate.feature_set,
            "feature_count": self.feature_count,
            "params": dict(self.candidate.params),
            # What estimation actually produced, where the candidate fits
            # parameters of its own (smoothing constants, AR order and
            # coefficients). Empty for the ML models, whose fitted state is
            # the artifact itself.
            "fitted_params": getattr(self.model, "fitted_params_", {}) or {},
            "selectable": self.candidate.selectable,
            "reference_only": not self.candidate.selectable,
            "reference_note": self.candidate.reference_note,
            "alpha": self.alpha,
            "alpha_unshrunk": self.alpha_unshrunk,
            "fit_seconds": round(self.fit_seconds, 2),
            "predict_seconds": round(self.predict_seconds, 2),
            "validation": self.validation,
            "test": self.test,
            "selected": self.selected,
        }


def evaluate_candidate(
    candidate: zoo.Candidate,
    frames: dict,
    y_absolute,
    anchor,
    fit_end: int,
    validation: slice,
    test: slice,
    shrinkage: float,
    horizon_hours: int,
) -> Trial:
    """
    Fit one candidate, fit its alpha on validation, score both blocks.

    ``frames`` maps a feature-set name to the frame of those columns, so a
    series model can read raw hourly history while the feature-based models
    read the curated list. ``y_absolute`` is the AQI level the horizon
    targets; the candidate is fitted on its deviation from ``anchor``.
    """

    X = frames[candidate.feature_set]

    y_delta = y_absolute - anchor

    model = candidate.build(horizon_hours)

    started = time.perf_counter()

    model.fit(X.iloc[:fit_end], y_delta.iloc[:fit_end])

    fit_seconds = time.perf_counter() - started

    started = time.perf_counter()

    validation_delta = np.asarray(model.predict(X.iloc[validation]), dtype=float)

    test_delta = np.asarray(model.predict(X.iloc[test]), dtype=float)

    predict_seconds = time.perf_counter() - started

    unshrunk = fit_alpha(
        y_absolute.iloc[validation],
        anchor.iloc[validation],
        validation_delta,
    )

    alpha = round(unshrunk * shrinkage, 4)

    validation_metrics = score(
        y_absolute.iloc[validation],
        anchor.iloc[validation].to_numpy() + alpha * validation_delta,
        anchor.iloc[validation],
    )

    test_metrics = score(
        y_absolute.iloc[test],
        anchor.iloc[test].to_numpy() + alpha * test_delta,
        anchor.iloc[test],
    )

    return Trial(
        candidate=candidate,
        model=model,
        alpha=alpha,
        alpha_unshrunk=round(unshrunk, 4),
        validation=validation_metrics,
        test=test_metrics,
        feature_count=X.shape[1],
        fit_seconds=fit_seconds,
        predict_seconds=predict_seconds,
    )


def evaluate_slate(
    slate,
    frames: dict,
    y_absolute,
    anchor,
    fit_end: int,
    validation: slice,
    test: slice,
    shrinkage: float,
    horizon_hours: int,
) -> list:
    """
    Every candidate on the slate, fitted and scored.

    A candidate that fails to fit is skipped with a warning rather than
    failing the retrain — one broken family should not stop the others from
    producing a servable model.
    """

    trials = []

    for candidate in slate:
        try:
            trials.append(evaluate_candidate(
                candidate,
                frames,
                y_absolute,
                anchor,
                fit_end,
                validation,
                test,
                shrinkage,
                horizon_hours,
            ))

        except Exception as exc:
            print(f"        {candidate.name} failed to fit: {exc}")

    if not trials:
        raise RuntimeError(
            "No candidate could be fitted for this horizon."
        )

    return trials


def _relative_gain(challenger: float, incumbent: float, lower_is_better: bool) -> float:
    """How much better the challenger is, as a fraction of the incumbent."""

    if incumbent == 0:
        better = challenger < 0 if lower_is_better else challenger > 0

        return float("inf") if better else 0.0

    difference = (
        incumbent - challenger if lower_is_better else challenger - incumbent
    )

    return difference / abs(incumbent)


def _rank(trials: list, metric: str) -> list:
    lower_is_better = metric in LOWER_IS_BETTER

    return sorted(
        trials,
        key=lambda trial: trial.value(metric),
        reverse=not lower_is_better,
    )


def select(
    trials: list,
    metric: str = DEFAULT_SELECTION_METRIC,
    margin: float = DEFAULT_SELECTION_MARGIN,
    default_model: str = zoo.DEFAULT_MODEL,
    allow_reference: bool = False,
) -> tuple:
    """
    Pick the winning trial. Returns ``(trial, reason)``.

    Reference and series candidates are scored but not eligible unless
    ``allow_reference`` is set — neither offers per-feature attribution, and
    the dashboard's explanation panel is part of what ships. Use
    ``reference_advisory`` to find out whether excluding them cost anything.
    """

    eligible = [
        trial for trial in trials
        if allow_reference or trial.candidate.selectable
    ]

    note = ""

    if not eligible:
        eligible = list(trials)

        note = " (no selectable candidate was on the slate)"

    lower_is_better = metric in LOWER_IS_BETTER

    best = _rank(eligible, metric)[0]

    incumbent = next(
        (
            trial for trial in eligible
            if trial.candidate.name == default_model
        ),
        None,
    )

    if incumbent is None:
        return best, (
            f"best validation {metric} {best.value(metric):.4f}; "
            f"'{default_model}' was not on the slate{note}"
        )

    if best is incumbent:
        return best, (
            f"best validation {metric} {best.value(metric):.4f}{note}"
        )

    gain = _relative_gain(
        best.value(metric),
        incumbent.value(metric),
        lower_is_better,
    )

    if gain >= margin:
        return best, (
            f"validation {metric} {best.value(metric):.4f} beats "
            f"{incumbent.candidate.name} ({incumbent.value(metric):.4f}) "
            f"by {gain:.1%}, clearing the {margin:.1%} margin{note}"
        )

    incumbent.notes = (
        f"{best.candidate.name} led by {gain:.1%} on validation {metric}, "
        f"inside the {margin:.1%} selection margin"
    )

    return incumbent, (
        f"kept {incumbent.candidate.name}: {best.candidate.name} leads by "
        f"only {gain:.1%} on validation {metric}, inside the "
        f"{margin:.1%} margin{note}"
    )


def reference_advisory(
    trials: list,
    winner: Trial,
    metric: str = DEFAULT_SELECTION_METRIC,
) -> str:
    """
    Say so when a non-selectable candidate would have won the slot.

    Excluding the reference and series models by default is a deliberate
    trade — explanations over a small accuracy gain — but it is only an
    honest one if the run says out loud when the trade cost something.
    Returns "" when it did not.
    """

    overall = _rank(trials, metric)[0]

    if overall is winner or overall.candidate.selectable:
        return ""

    gain = _relative_gain(
        overall.value(metric),
        winner.value(metric),
        metric in LOWER_IS_BETTER,
    )

    if gain <= 0:
        return ""

    return (
        f"{overall.candidate.name} had the best validation {metric} "
        f"({overall.value(metric):.4f} vs {winner.value(metric):.4f}, "
        f"{gain:.1%} better) but is reference-only "
        f"({overall.candidate.reference_note}). "
        f"Pass --allow-reference to let it take the slot."
    )


HEADER = (
    f"        {'':2}{'CANDIDATE':<16}{'FAMILY':<12}{'FEAT':>5}{'ALPHA':>7}"
    f"{'VAL MAE':>9}{'VAL RMSE':>10}{'VAL R2':>9}"
    f"{'TEST MAE':>10}{'TEST RMSE':>10}{'TEST R2':>9}{'SKILL':>8}{'SEC':>7}"
)


def comparison_table(trials: list, metric: str = DEFAULT_SELECTION_METRIC) -> str:
    """
    The slate as a fixed-width table, ranked by the selection metric.

    Winner marked ``*``, candidates that were scored but not eligible ``~``.
    """

    lines = [HEADER]

    for trial in _rank(trials, metric):
        if trial.selected:
            mark = "*"
        elif not trial.candidate.selectable:
            mark = "~"
        else:
            mark = " "

        lines.append(
            f"        {mark:<2}"
            f"{trial.candidate.name:<16}"
            f"{trial.candidate.family:<12}"
            f"{trial.feature_count:>5}"
            f"{trial.alpha:>7.2f}"
            f"{trial.validation['mae']:>9.3f}"
            f"{trial.validation['rmse']:>10.3f}"
            f"{trial.validation['r2']:>9.4f}"
            f"{trial.test['mae']:>10.3f}"
            f"{trial.test['rmse']:>10.3f}"
            f"{trial.test['r2']:>9.4f}"
            f"{trial.test['skill_vs_persistence']:>8.3f}"
            f"{trial.fit_seconds + trial.predict_seconds:>7.1f}"
        )

    return "\n".join(lines)
