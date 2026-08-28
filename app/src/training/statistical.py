"""
Classical statistical forecasters for the AQI series.

The brief asks for a variety of forecasting models, "from statistical
modelling to deep learning". The candidate slate had five machine-learning
regressors, all of which learn a mapping from engineered features to a target
and know nothing about the series being a series. This module adds the other
end of that range: models that describe the AQI series itself — a seasonal
benchmark, exponential smoothing, and a seasonal autoregressive model.

**How they fit the existing pipeline.** A series model does not naturally take
a feature row, so the temptation is to bolt on a second, parallel evaluation
path — which is how the original model comparison ended up stranded in a
notebook. Instead each forecaster is an ordinary estimator with ``fit`` and
``predict``, reading its inputs from the raw history block that
``feature_engineering.create_history_features`` attaches (``aqi_hist_0`` =
now, through ``aqi_hist_167`` = a week ago). The registry already stores a
feature list *per horizon*, so a series model can be registered, promoted,
rolled back and served exactly like the trees, with no change to the serving
code — and it is measured on the same windows, with the same metrics, in the
same table, on every retrain.

**What they predict.** The same thing every other candidate predicts: the
deviation of the horizon's mean AQI from the current reading. Internally each
model forecasts a full hourly path, averages the relevant 24-hour slice
(hours 1-24, 25-48 or 49-72 ahead) and subtracts the last observation. So the
comparison against the tree models is like for like.

**Parameters are fitted once, states are rebuilt per row.** Refitting a
seasonal model for each of the ~2,600 evaluation rows, three times over,
would take hours. Instead the smoothing and autoregressive parameters are
estimated once on segments of the training block, and each prediction re-runs
the recursion over that row's own week of history to obtain the state it
forecasts from. Both training and serving therefore take the identical code
path, which is the point — deriving the state from the whole series during
evaluation and from a window at serving time would be a skew that only showed
up in production.

**Note on artifacts.** Fitted instances are pickled into the model registry,
so this module's import path is baked into those artifacts. Moving or renaming
it makes previously registered statistical versions unloadable.
"""

import numpy as np

from app.src.features.feature_engineering import HISTORY_PREFIX, is_history_column

# The daily cycle. AQI has a strong diurnal profile — the EDA notebook's
# "average AQI by hour of day" is the reason every model here is seasonal.
SEASON = 24

# Each horizon's target is the mean over a 24-hour window, so a forecast path
# is sliced into the last 24 of its steps.
WINDOW_HOURS = 24

# Roughly how many history segments to estimate parameters from. The stride is
# derived from this, so the cost of a fit does not grow with the training block.
TARGET_SEGMENTS = 32


def _finite_window(values, minimum: int):
    """
    A usable, gap-free window, or None.

    Interior missing hours are linearly interpolated and the ends are held
    flat, which is what ``np.interp`` does outside the known range. A window
    with too few real readings is rejected outright rather than invented.
    """

    values = np.asarray(values, dtype=float)

    finite = np.isfinite(values)

    if int(finite.sum()) < minimum:
        return None

    if not finite.all():
        position = np.arange(len(values))

        values = np.interp(position, position[finite], values[finite])

    return values


class WindowForecaster:
    """
    Base class: parameters fitted on training segments, state per window.

    Subclasses implement ``_estimate`` (fit parameters from a list of clean
    windows) and ``_path`` (forecast ``steps`` hours ahead from one window).
    """

    # Minimum real readings a window needs before it is forecast from.
    minimum_observations = 2 * SEASON

    def __init__(
        self,
        horizon_hours: int,
        season: int = SEASON,
        window_hours: int = WINDOW_HOURS,
    ):
        self.horizon_hours = int(horizon_hours)
        self.season = int(season)
        self.window_hours = int(window_hours)
        self.fitted_params_ = {}

    # -- input handling ----------------------------------------------------

    def _history_columns(self, X) -> list:
        """History columns ordered oldest to newest."""

        columns = [column for column in X.columns if is_history_column(column)]

        if not columns:
            raise ValueError(
                f"{type(self).__name__} needs the raw history block "
                f"('{HISTORY_PREFIX}*' columns); got {list(X.columns)[:6]}..."
            )

        return sorted(
            columns,
            key=lambda column: int(column[len(HISTORY_PREFIX):]),
            reverse=True,
        )

    def _windows(self, X) -> np.ndarray:
        return X[self._history_columns(X)].to_numpy(dtype=float)

    def _segments(self, X) -> list:
        """
        Training windows to estimate parameters from.

        The stride keeps the number of segments roughly constant, so segments
        barely overlap on a year of data and the fit stays cheap as the store
        grows.
        """

        windows = self._windows(X)

        stride = max(1, len(windows) // TARGET_SEGMENTS)

        segments = [
            cleaned
            for window in windows[::stride]
            if (cleaned := _finite_window(window, self.minimum_observations))
            is not None
        ]

        return segments

    # -- estimator interface ----------------------------------------------

    def fit(self, X, y=None):
        """
        Estimate parameters. ``y`` is accepted and ignored: the target is a
        deterministic function of the series these models already read.
        """

        segments = self._segments(X)

        if segments:
            self._estimate(segments)

        return self

    def predict(self, X):
        """The deviation of the horizon's mean AQI from the current reading."""

        windows = self._windows(X)

        start = self.horizon_hours - self.window_hours

        out = np.zeros(len(windows), dtype=float)

        for row, window in enumerate(windows):
            cleaned = _finite_window(window, self.minimum_observations)

            if cleaned is None:
                # Not enough history to say anything; zero deviation makes
                # this row a persistence forecast rather than a guess.
                continue

            path = np.asarray(
                self._path(cleaned, self.horizon_hours), dtype=float
            )

            predicted = float(path[start:self.horizon_hours].mean())

            if not np.isfinite(predicted):
                continue

            out[row] = predicted - float(cleaned[-1])

        return out

    # -- subclass hooks ----------------------------------------------------

    def _estimate(self, segments: list) -> None:
        """Fit parameters from clean training windows. Default: none to fit."""

    def _path(self, window: np.ndarray, steps: int) -> np.ndarray:
        raise NotImplementedError


class SeasonalNaiveForecaster(WindowForecaster):
    """
    The seasonal benchmark: this hour tomorrow looks like this hour today.

    No parameters. It is the classical reference every seasonal model is
    expected to beat, and on a series with a diurnal cycle it is a much
    harder target than the flat persistence baseline — which is exactly why
    it belongs in the comparison.
    """

    minimum_observations = SEASON

    def _path(self, window, steps):
        season = min(self.season, len(window))

        tail = window[-season:]

        position = (np.arange(steps) % season)

        return tail[position]


class HoltWintersForecaster(WindowForecaster):
    """
    Additive Holt-Winters with a damped trend — the ETS family.

        level_t   = a (y_t - s_{t-m}) + (1 - a)(level_{t-1} + p trend_{t-1})
        trend_t   = b (level_t - level_{t-1}) + (1 - b) p trend_{t-1}
        s_t       = g (y_t - level_{t-1} - p trend_{t-1}) + (1 - g) s_{t-m}
        forecast_k = level_n + (p + ... + p^k) trend_n + s_{n-m+((k-1) mod m)}

    The four parameters are estimated by minimising one-step-ahead squared
    error, which is the standard criterion for exponential smoothing. It is
    evaluated over a set of week-long training segments instead of one long
    run, because the series has hour-level gaps and splicing across them
    would charge the model for jumps that are not really there.

    The trend is damped (``p`` < 1) deliberately: an undamped linear trend
    extrapolated 72 hours out is what makes Holt-Winters embarrassing on
    noisy data.
    """

    minimum_observations = 3 * SEASON

    # (alpha, beta, gamma, phi) — level, trend, seasonal, damping.
    initial_guess = (0.3, 0.05, 0.10, 0.98)

    bounds = ((0.01, 0.99), (0.0001, 0.5), (0.0001, 0.99), (0.80, 0.999))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.alpha, self.beta, self.gamma, self.phi = self.initial_guess

    # -- the recursion -----------------------------------------------------

    def _recurse(self, values, alpha, beta, gamma, phi):
        """
        Run the state recursion over one window.

        Returns ``(level, trend, seasonal_tail, sse, count)`` — the state at
        the end of the window plus the one-step-ahead error it accumulated,
        so the same code both fits parameters and forecasts.
        """

        season = self.season

        n = len(values)

        first = values[:season]

        level = float(first.mean())

        if n >= 2 * season:
            trend = float(
                (values[season:2 * season].mean() - level) / season
            )
        else:
            trend = 0.0

        seasonal = np.zeros(n, dtype=float)

        seasonal[:season] = first - level

        sse = 0.0

        count = 0

        for t in range(season, n):
            damped = phi * trend

            expected = level + damped + seasonal[t - season]

            error = values[t] - expected

            sse += error * error

            count += 1

            previous_level = level

            level = alpha * (values[t] - seasonal[t - season]) + (1 - alpha) * (
                previous_level + damped
            )

            trend = beta * (level - previous_level) + (1 - beta) * damped

            seasonal[t] = gamma * (
                values[t] - previous_level - damped
            ) + (1 - gamma) * seasonal[t - season]

        return level, trend, seasonal[n - season:n], sse, count

    def _estimate(self, segments):
        from scipy.optimize import minimize

        def objective(parameters):
            alpha, beta, gamma, phi = parameters

            total = 0.0
            rows = 0

            for segment in segments:
                _, _, _, sse, count = self._recurse(
                    segment, alpha, beta, gamma, phi
                )

                total += sse
                rows += count

            if rows == 0:
                return np.inf

            return total / rows

        try:
            result = minimize(
                objective,
                x0=np.array(self.initial_guess, dtype=float),
                method="L-BFGS-B",
                bounds=self.bounds,
            )

            if result.success or np.isfinite(result.fun):
                self.alpha, self.beta, self.gamma, self.phi = (
                    float(value) for value in result.x
                )

        except Exception as exc:
            # Keep the starting values: sensible smoothing constants forecast
            # perfectly acceptably, and a failed optimiser is not a reason to
            # drop the candidate from the comparison.
            print(f"        Holt-Winters parameter fit failed ({exc}); using defaults")

        self.fitted_params_ = {
            "alpha": round(self.alpha, 5),
            "beta": round(self.beta, 5),
            "gamma": round(self.gamma, 5),
            "phi": round(self.phi, 5),
            "segments": len(segments),
        }

    def _path(self, window, steps):
        level, trend, seasonal_tail, _, _ = self._recurse(
            window, self.alpha, self.beta, self.gamma, self.phi
        )

        season = len(seasonal_tail)

        out = np.empty(steps, dtype=float)

        damping = 0.0

        for k in range(1, steps + 1):
            damping += self.phi ** k

            out[k - 1] = (
                level + damping * trend + seasonal_tail[(k - 1) % season]
            )

        return out


class SeasonalArForecaster(WindowForecaster):
    """
    Seasonal AR after a 24-hour difference — SARIMA(p,0,0)(0,1,0)[24].

    Differencing at the seasonal lag removes the daily cycle and most of the
    slow drift in pollutant levels, leaving something close to stationary for
    an autoregressive model to work on:

        z_t = y_t - y_{t-24}
        z_t = c + phi_1 z_{t-1} + ... + phi_p z_{t-p} + e_t

    Coefficients come from conditional least squares over the pooled training
    segments, and the order ``p`` is chosen by AIC from a short list rather
    than asserted — with hourly data the plausible orders span a wide range,
    and which one wins depends on how much history the store holds.

    Forecasting runs the AR recursion forward on ``z`` and then undoes the
    difference, so each hour's forecast leans on the same hour of the
    previous day.

    Order selection rejects non-stationary fits. Least squares is perfectly
    willing to return coefficients whose characteristic roots sit on or inside
    the unit circle, and running such a recursion 72 steps forward diverges
    exponentially. On the current year of Karachi data the chosen AR(24) has a
    largest root of 0.998 — stationary, but close enough to the boundary that
    leaving the check out would be relying on luck as the store grows.
    """

    minimum_observations = 3 * SEASON

    candidate_orders = (1, 2, 3, 6, 12, 24)

    # Roots must be strictly inside the unit circle, with a little headroom:
    # a root at 0.9999 is stationary on paper and a random walk in practice.
    stationarity_limit = 0.999

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.order = 2
        self.intercept = 0.0
        self.coefficients = np.zeros(self.order, dtype=float)

    def _design(self, segments, order):
        """Pooled (lags -> next value) rows from the differenced segments."""

        rows = []
        targets = []

        for segment in segments:
            if len(segment) <= self.season + order:
                continue

            z = segment[self.season:] - segment[:-self.season]

            for t in range(order, len(z)):
                rows.append(z[t - order:t][::-1])
                targets.append(z[t])

        if not rows:
            return None, None

        return np.asarray(rows, dtype=float), np.asarray(targets, dtype=float)

    def _largest_root(self, coefficients) -> float:
        """
        Modulus of the largest characteristic root of the AR polynomial.

        Below 1 the recursion is stationary and multi-step forecasts converge
        on the mean; at or above 1 they run away.
        """

        coefficients = np.asarray(coefficients, dtype=float)

        if coefficients.size == 0:
            return 0.0

        roots = np.roots(np.concatenate(([1.0], -coefficients)))

        if roots.size == 0:
            return 0.0

        return float(np.abs(roots).max())

    def _estimate(self, segments):
        best = None

        rejected = []

        for order in self.candidate_orders:
            design, targets = self._design(segments, order)

            if design is None or len(design) <= order + 2:
                continue

            # Intercept as an explicit column, so lstsq returns it with the
            # coefficients and there is no separate centring step to get wrong.
            augmented = np.column_stack(
                [np.ones(len(design)), design]
            )

            try:
                solution, *_ = np.linalg.lstsq(augmented, targets, rcond=None)

            except np.linalg.LinAlgError:
                continue

            residuals = targets - augmented @ solution

            rows = len(targets)

            sse = float(residuals @ residuals)

            if sse <= 0 or not np.isfinite(sse):
                continue

            largest_root = self._largest_root(solution[1:])

            if largest_root >= self.stationarity_limit:
                rejected.append((order, round(largest_root, 4)))

                continue

            aic = rows * np.log(sse / rows) + 2 * (order + 1)

            if best is None or aic < best[0]:
                best = (aic, order, solution, rows, largest_root)

        if rejected:
            print(
                "        Seasonal AR rejected non-stationary orders: "
                + ", ".join(f"p={order} (root {root})" for order, root in rejected)
            )

        if best is None:
            # No stationary fit. Zero coefficients leave `_path` undoing the
            # seasonal difference with no AR correction, i.e. the seasonal
            # benchmark — a defensible forecast rather than a diverging one.
            print(
                "        Seasonal AR found no stationary order; "
                "forecasting the seasonal benchmark instead"
            )

            self.coefficients = np.zeros(self.order, dtype=float)
            self.intercept = 0.0

            self.fitted_params_ = {
                "order": None,
                "stationary": False,
                "rejected_orders": rejected,
                "segments": len(segments),
            }

            return

        aic, order, solution, rows, largest_root = best

        self.order = int(order)
        self.intercept = float(solution[0])
        self.coefficients = np.asarray(solution[1:], dtype=float)

        self.fitted_params_ = {
            "order": self.order,
            "aic": round(float(aic), 2),
            "largest_root": round(float(largest_root), 4),
            "stationary": True,
            "intercept": round(self.intercept, 5),
            "coefficients": [round(float(value), 5) for value in self.coefficients],
            "training_rows": int(rows),
            "segments": len(segments),
        }

    def _path(self, window, steps):
        season = self.season

        order = self.order

        if len(window) <= season + order:
            # Fall back to the seasonal benchmark rather than nothing.
            tail = window[-min(season, len(window)):]

            return tail[np.arange(steps) % len(tail)]

        differenced = list(window[season:] - window[:-season])

        extended = list(window)

        for step in range(steps):
            lags = np.asarray(differenced[-order:][::-1], dtype=float)

            if len(lags) < order:
                lags = np.pad(lags, (0, order - len(lags)))

            differenced.append(
                self.intercept + float(self.coefficients @ lags)
            )

            extended.append(extended[-season] + differenced[-1])

        return np.asarray(extended[len(window):], dtype=float)
