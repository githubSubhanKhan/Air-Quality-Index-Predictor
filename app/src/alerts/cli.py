"""
Send a 3-day AQI alert from the command line.

    # check the wording and the configuration without mailing anyone
    python -m app.src.alerts.cli --city karachi --to you@example.com --dry-run

    # print the composed message instead of sending it
    python -m app.src.alerts.cli --city karachi --to you@example.com --show

    # send it
    python -m app.src.alerts.cli --city karachi --to you@example.com

    # only mail when a day is forecast to reach "Unhealthy" or worse
    python -m app.src.alerts.cli --city karachi --to you@example.com \
        --threshold 151 --only-if-breach

``--only-if-breach`` is what a scheduled job wants: silent on clean days, mail
on bad ones. The dashboard button does the opposite by default, because a
person who just pressed "send" expects an email either way.
"""

import argparse
import sys

from app.src.alerts.mailer import (
    AlertError,
    send_alert,
    sender_hint,
)
from app.src.alerts.messages import build_alert
from app.src.features.aqi import (
    DEFAULT_ALERT_THRESHOLD,
    categorise,
    threshold_categories,
)
from app.src.prediction.forecast import forecast_for_city


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Email a city's 3-day AQI forecast as an alert",
    )

    parser.add_argument(
        "--city",
        default="karachi",
        help="City to forecast, as stored in the feature store",
    )

    parser.add_argument(
        "--to",
        required=True,
        help="Recipient email address",
    )

    parser.add_argument(
        "--threshold",
        type=int,
        default=DEFAULT_ALERT_THRESHOLD,
        help=(
            "AQI at which a day counts as an alert. Category floors: "
            + ", ".join(
                f"{category.low} ({category.label})"
                for category in threshold_categories()
            )
        ),
    )

    parser.add_argument(
        "--only-if-breach",
        action="store_true",
        help="Send nothing unless a day reaches the threshold",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compose and validate, but do not connect to the mail server",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Print the composed plain-text message (implies --dry-run)",
    )

    args = parser.parse_args()

    dry_run = args.dry_run or args.show

    try:
        current = forecast_for_city(args.city)

    except LookupError as exc:
        raise SystemExit(f"No forecast available: {exc}")

    forecast = current["forecast"]

    reading_time = current["reading_time"]

    print(
        f"City         : {current['city']}\n"
        f"Reading       : {reading_time:%Y-%m-%d %H:%M} UTC\n"
        f"Current AQI   : {forecast['current_aqi']:.0f} "
        f"({categorise(forecast['current_aqi']).label})\n"
        f"Forecast      : "
        + "  ".join(
            f"day{index} {forecast[f'day_{index}']:.0f}"
            for index in (1, 2, 3)
        )
        + f"\nThreshold     : {args.threshold}+ AQI "
        f"({categorise(args.threshold).label})"
    )

    if args.show:
        alert = build_alert(
            city=current["city"],
            forecast=forecast,
            reading_time=reading_time,
            threshold=args.threshold,
            model_details=current["models"],
        )

        print(f"\nSubject: {alert['subject']}\n")
        print(alert["text"])

    if args.only_if_breach:
        breaching = [
            index
            for index in (1, 2, 3)
            if forecast[f"day_{index}"] >= args.threshold
        ]

        if not breaching:
            print(
                f"\nNo day reaches {args.threshold} AQI; "
                f"--only-if-breach set, so nothing was sent."
            )

            return

    sender = sender_hint()

    print(f"Sender        : {sender or '(not configured)'}")

    try:
        result = send_alert(
            recipient=args.to,
            city=current["city"],
            forecast=forecast,
            reading_time=reading_time,
            threshold=args.threshold,
            model_details=current["models"],
            dry_run=dry_run,
        )

    except AlertError as exc:
        raise SystemExit(f"\n{type(exc).__name__}: {exc}")

    print(
        f"\n{'Composed (not sent)' if dry_run else 'Sent'}: "
        f"{result['subject']}\n"
        f"  to        : {result['recipient']}\n"
        f"  severity  : {result['worst']} (peak {result['worst_aqi']:.0f} AQI)\n"
        f"  breaching : {', '.join(result['breaches']) or 'none'}"
    )


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    main()
