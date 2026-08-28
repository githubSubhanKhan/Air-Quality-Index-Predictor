"""
Render a retrain's metadata as Markdown.

The candidate comparison is only useful if somebody sees it. The training job
prints fixed-width tables to its log, which is fine when you already know to
go looking; this renders the same run as Markdown for the GitHub Actions job
summary, where the daily retrain's choice is visible without opening the log.

    python -m app.src.training.train --city karachi --metadata-out run.json
    python -m app.src.training.report run.json >> "$GITHUB_STEP_SUMMARY"

Reads the metadata dict ``train.run`` returns, so it also works on the
``training_metadata.json`` written next to the local model copy.
"""

import argparse
import json
import sys
from pathlib import Path

# Candidate names contain underscores, which Markdown will happily read as
# emphasis; every identifier goes in backticks.
CODE = "`{}`"


def _number(value, digits=2, signed=False):
    if value is None:
        return "-"

    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def _row(cells) -> str:
    return "| " + " | ".join(cells) + " |"


def _table(headers, rows) -> list:
    return [
        _row(headers),
        _row(["---"] * len(headers)),
        *(_row(row) for row in rows),
    ]


def chosen_table(metadata: dict) -> list:
    """One row per horizon: what was picked and how it scored on test."""

    rows = []

    for horizon, chosen in metadata.get("models", {}).items():
        metrics = metadata.get("metrics", {}).get(horizon, {})

        rows.append([
            horizon,
            CODE.format(chosen.get("candidate", "?")),
            chosen.get("family", "-"),
            _number(chosen.get("alpha")),
            _number(metrics.get("mae")),
            _number(metrics.get("rmse")),
            _number(metrics.get("r2"), 4, signed=True),
            _number(metrics.get("baseline_r2"), 4, signed=True),
            _number(metrics.get("skill_vs_persistence"), 3, signed=True),
        ])

    return _table(
        [
            "Horizon", "Model", "Family", "Alpha",
            "Test MAE", "Test RMSE", "Test R²",
            "Persistence R²", "Skill",
        ],
        rows,
    )


def slate_table(candidates: list) -> list:
    """Every candidate for one horizon, in the order they were ranked."""

    rows = []

    for entry in candidates:
        if entry.get("selected"):
            mark = "**selected**"
        elif entry.get("reference_only"):
            mark = "reference"
        else:
            mark = ""

        validation = entry.get("validation", {})
        test = entry.get("test", {})

        seconds = (
            entry.get("fit_seconds", 0) or 0
        ) + (entry.get("predict_seconds", 0) or 0)

        rows.append([
            CODE.format(entry.get("candidate", "?")),
            entry.get("family", "-"),
            mark,
            str(entry.get("feature_count", "-")),
            _number(entry.get("alpha")),
            _number(validation.get("mae")),
            _number(validation.get("rmse")),
            _number(validation.get("r2"), 4, signed=True),
            _number(test.get("mae")),
            _number(test.get("rmse")),
            _number(test.get("r2"), 4, signed=True),
            _number(seconds, 1),
        ])

    return _table(
        [
            "Candidate", "Family", "", "Features", "Alpha",
            "Val MAE", "Val RMSE", "Val R²",
            "Test MAE", "Test RMSE", "Test R²", "Time (s)",
        ],
        rows,
    )


def fitted_parameters(metadata: dict) -> list:
    """
    What estimation produced for the candidates that fit parameters of their
    own — smoothing constants, AR order, stationarity.

    The ML models are omitted: their fitted state is the artifact, not a
    handful of numbers worth printing. Entries are de-duplicated because these
    parameters are estimated from the series and do not depend on the horizon,
    so all three horizons report the same fit.
    """

    seen = {}

    for entries in metadata.get("candidates", {}).values():
        for entry in entries:
            fitted = entry.get("fitted_params") or {}

            if not fitted:
                continue

            shown = ", ".join(
                f"{key}={value}"
                for key, value in fitted.items()
                # The coefficient vector can be 24 numbers long; it is on the
                # registry document for anyone who needs it.
                if key not in ("coefficients", "rejected_orders")
            )

            seen.setdefault(
                (entry.get("candidate", "?"), shown),
                f"- {CODE.format(entry.get('candidate', '?'))} — {shown}",
            )

    if not seen:
        return []

    return [
        "#### Fitted statistical parameters",
        "",
        *seen.values(),
        "",
        "> Estimated from the training series, so they are shared across the "
        "three horizons. Full AR coefficient vectors are on the registry "
        "documents.",
        "",
    ]


def render(metadata: dict) -> str:
    """The whole run as Markdown."""

    data = metadata.get("data", {})

    evaluation = metadata.get("evaluation", {})

    selection = evaluation.get("selection", {})

    registry = metadata.get("registry", {})

    lines = [
        f"### Retrain `{metadata.get('run_id', '?')}` — "
        f"{metadata.get('city', '?')}",
        "",
        f"Trained at {metadata.get('trained_at', '?')} on "
        f"{data.get('usable_rows', '?')} usable rows "
        f"({data.get('rows_in_store', '?')} in store, "
        f"{data.get('missing_hourly_rows', '?')} hourly gaps).",
        "",
        f"Split: {data.get('train_rows', '?')} train / "
        f"{data.get('validation_rows', '?')} validation / "
        f"{data.get('test_rows', '?')} test, "
        f"{evaluation.get('purge_rows', '?')} rows purged per boundary "
        f"({evaluation.get('scheme', 'chronological')}).",
        "",
        f"Selection: best validation **{selection.get('metric', 'mae')}**, "
        f"{selection.get('margin', 0):.1%} margin over "
        f"{CODE.format(selection.get('default_model', '?'))}"
        + (
            ", reference models eligible"
            if selection.get("reference_models_eligible")
            else ", reference models excluded"
        )
        + ".",
        "",
        "#### Selected per horizon",
        "",
        *chosen_table(metadata),
        "",
    ]

    for horizon, chosen in metadata.get("models", {}).items():
        lines += [f"- **{horizon}** — {chosen.get('reason', '')}"]

        advisory = chosen.get("advisory")

        if advisory:
            lines += [f"  - ⚠️ {advisory}"]

    lines += [""]

    candidates = metadata.get("candidates", {})

    if candidates:
        lines += ["#### Candidate slate", ""]

        for horizon, entries in candidates.items():
            lines += [
                "<details>",
                f"<summary>{horizon} — {len(entries)} candidates</summary>",
                "",
                *slate_table(entries),
                "",
                "</details>",
                "",
            ]

        lines += [
            "> Candidates were ranked on the validation block only. Test "
            "metrics are shown for every candidate but took no part in the "
            "choice. Rows marked *reference* were scored but were not "
            "eligible for the slot: none of them offers per-feature "
            "attribution, so promoting one would empty the dashboard's SHAP "
            "panel.",
            "",
        ]

        lines += fitted_parameters(metadata)

    published = registry.get("published", {})

    if published:
        lines += ["#### Registry", ""]

        rows = [
            [
                horizon,
                CODE.format(entry.get("name", "?")),
                f"v{entry.get('version', '?')}",
                CODE.format(entry.get("candidate", "?")),
                entry.get("stage", "-"),
                "yes" if entry.get("promoted") else "no",
                entry.get("reason", ""),
            ]
            for horizon, entry in published.items()
        ]

        lines += _table(
            ["Horizon", "Name", "Version", "Model", "Stage", "Promoted", "Reason"],
            rows,
        )

        lines += [""]

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a retrain's metadata as Markdown",
    )

    parser.add_argument(
        "metadata",
        help="Path to the JSON written by --metadata-out (or training_metadata.json)",
    )

    args = parser.parse_args()

    with open(Path(args.metadata), encoding="utf-8") as f:
        metadata = json.load(f)

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print(render(metadata))


if __name__ == "__main__":
    main()
