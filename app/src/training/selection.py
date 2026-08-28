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
persistence automatically rather than being punished for its variance.
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
    fit_seconds: float = 0.0
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
            "params": dict(self.candidate.params),
            "reference_only": self.candidate.is_baseline,
            "alpha": self.alpha,
            "alpha_unshrunk": self.alpha_unshrunk,
            "fit_seconds": round(self.fit_seconds, 2),
            "validation": self.validation,
            "test": self.test,
            "selected": self.selected,
        }


def evaluate_candidate(
    candidate: zoo.Candidate,
    X,
    y_absolute,
    anchor,
    fit_end: int,
    validation: slice,
    test: slice,
    shrinkage: float,
) -> Trial:
    """
    Fit one candidate, fit its alpha on validation, score both blocks.

    ``y_absolute`` is the AQI level the horizon targets; the candidate is
    fitted on the deviation of that level from ``anchor``.
    """

    y_delta = y_absolute - anchor

    model = candidate.build()

    started = time.perf_counter()

    model.fit(X.iloc[:fit_end], y_delta.iloc[:fit_end])

    fit_seconds = time.perf_counter() - started

    validation_delta = model.predict(X.iloc[validation])

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
        anchor.iloc[test].to_numpy() + alpha * model.predict(X.iloc[test]),
        anchor.iloc[test],
    )

    return Trial(
        candidate=candidate,
        model=model,
        alpha=alpha,
        alpha_unshrunk=round(unshrunk, 4),
        validation=validation_metrics,
        test=test_metrics,
        fit_seconds=fit_seconds,
    )


def evaluate_slate(
    slate,
    X,
    y_absolute,
    anchor,
    fit_end: int,
    validation: slice,
    test: slice,
    shrinkage: float,
) -> list:
    """
    Every candidate on the slate, fitted and scored.

    A candidate that fails to fit is skipped with a warning rather than
    failing the retrain — one broken family should not stop the other four
    from producing a servable model.
    """

    trials = []

    for candidate in slate:
        try:
            trials.append(evaluate_candidate(
                candidate,
                X,
                y_absolute,
                anchor,
                fit_end,
                validation,
                test,
                shrinkage,
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


def select(
    trials: list,
    metric: str = DEFAULT_SELECTION_METRIC,
    margin: float = DEFAULT_SELECTION_MARGIN,
    default_model: str = zoo.DEFAULT_MODEL,
    allow_baseline: bool = False,
) -> tuple:
    """
    Pick the winning trial. Returns ``(trial, reason)``.

    Reference candidates such as ``persistence`` are scored but not eligible
    unless ``allow_baseline`` is set — a constant has no SHAP explanation, and
    the dashboard's explanation panel is part of what ships.
    """

    eligible = [
        trial for trial in trials
        if allow_baseline or not trial.candidate.is_baseline
    ]

    note = ""

    if not eligible:
        eligible = list(trials)

        note = " (only reference candidates were on the slate)"

    lower_is_better = metric in LOWER_IS_BETTER

    ranked = sorted(
        eligible,
        key=lambda trial: trial.value(metric),
        reverse=not lower_is_better,
    )

    best = ranked[0]

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


HEADER = (
    f"        {'':2}{'CANDIDATE':<22}{'FAMILY':<10}{'ALPHA':>6}"
    f"{'VAL MAE':>10}{'VAL RMSE':>10}{'VAL R2':>9}"
    f"{'TEST MAE':>10}{'TEST RMSE':>10}{'TEST R2':>9}{'SKILL':>8}"
)


def comparison_table(trials: list, metric: str = DEFAULT_SELECTION_METRIC) -> str:
    """
    The slate as a fixed-width table, ranked by the selection metric.

    Winner marked ``*``, reference-only candidates marked ``~``.
    """

    lower_is_better = metric in LOWER_IS_BETTER

    ranked = sorted(
        trials,
        key=lambda trial: trial.value(metric),
        reverse=not lower_is_better,
    )

    lines = [HEADER]

    for trial in ranked:
        if trial.selected:
            mark = "*"
        elif trial.candidate.is_baseline:
            mark = "~"
        else:
            mark = " "

        lines.append(
            f"        {mark:<2}"
            f"{trial.candidate.name:<22}"
            f"{trial.candidate.family:<10}"
            f"{trial.alpha:>6.2f}"
            f"{trial.validation['mae']:>10.3f}"
            f"{trial.validation['rmse']:>10.3f}"
            f"{trial.validation['r2']:>9.4f}"
            f"{trial.test['mae']:>10.3f}"
            f"{trial.test['rmse']:>10.3f}"
            f"{trial.test['r2']:>9.4f}"
            f"{trial.test['skill_vs_persistence']:>8.3f}"
        )

    return "\n".join(lines)
