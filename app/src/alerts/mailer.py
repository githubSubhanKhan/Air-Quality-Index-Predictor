"""
Sending the alert email.

Credentials come from the environment, never from the request:

    ALERT_SENDER_EMAIL      the account the alert is sent from
    ALERT_SENDER_PASSWORD   its password - for Gmail, an App Password
    ALERT_SMTP_HOST         default smtp.gmail.com
    ALERT_SMTP_PORT         default 587 (STARTTLS); 465 switches to SSL
    ALERT_SENDER_NAME       display name, default "AQI Predictor"

**On Gmail specifically**: the account password will not work. Google turned
off password sign-in for SMTP, so ``ALERT_SENDER_PASSWORD`` has to be a
16-character App Password, which requires 2-Step Verification on the account.
That failure returns a bare "Username and Password not accepted" from the
server, so it is translated into something actionable below rather than shown
raw.

The recipient address is the only caller-supplied value that reaches the SMTP
conversation, and it is validated and length-capped before it does. Header
injection is not possible through ``EmailMessage`` — it encodes header values
rather than concatenating them — but the address is checked anyway, because an
address that cannot be parsed is a user error worth naming rather than an
SMTP error to surface.
"""

import os
import re
import smtplib
import ssl
from email.message import EmailMessage

from dotenv import load_dotenv

from app.src.features.aqi import DEFAULT_ALERT_THRESHOLD
from app.src.alerts.messages import build_alert

load_dotenv()

DEFAULT_SMTP_HOST = "smtp.gmail.com"

DEFAULT_SMTP_PORT = 587

DEFAULT_SENDER_NAME = "AQI Predictor"

SSL_PORT = 465

TIMEOUT_SECONDS = 20

# Deliberately permissive but bounded: enough to catch a typo or an empty box,
# not an attempt to out-parse RFC 5322.
ADDRESS_PATTERN = re.compile(r"^[^@\s,;<>]+@[^@\s,;<>]+\.[A-Za-z]{2,}$")

MAX_ADDRESS_LENGTH = 254


class AlertError(RuntimeError):
    """Base class, so callers can catch everything alert-related at once."""


class AlertConfigError(AlertError):
    """The sender account is not configured."""


class AlertAddressError(AlertError):
    """The recipient address is not usable."""


class AlertSendError(AlertError):
    """The SMTP conversation failed."""


def valid_address(address: str) -> bool:
    address = (address or "").strip()

    return (
        bool(address)
        and len(address) <= MAX_ADDRESS_LENGTH
        and bool(ADDRESS_PATTERN.match(address))
    )


def smtp_settings() -> dict:
    """
    The sender configuration, or ``AlertConfigError`` naming what is missing.

    Read on every call rather than cached at import: the Streamlit app bridges
    ``st.secrets`` into the environment after import, and a cached empty value
    would make the feature look permanently unconfigured in the cloud.
    """

    sender = (os.getenv("ALERT_SENDER_EMAIL") or "").strip()

    password = os.getenv("ALERT_SENDER_PASSWORD") or ""

    missing = [
        name
        for name, value in (
            ("ALERT_SENDER_EMAIL", sender),
            ("ALERT_SENDER_PASSWORD", password),
        )
        if not value
    ]

    if missing:
        raise AlertConfigError(
            "Email alerts are not configured: "
            + " and ".join(missing)
            + " must be set in the environment (.env locally, or Streamlit "
              "secrets when deployed)."
        )

    try:
        port = int(os.getenv("ALERT_SMTP_PORT") or DEFAULT_SMTP_PORT)

    except ValueError as exc:
        raise AlertConfigError(
            f"ALERT_SMTP_PORT must be a number, got "
            f"{os.getenv('ALERT_SMTP_PORT')!r}"
        ) from exc

    return {
        "sender": sender,
        "password": password,
        "host": (os.getenv("ALERT_SMTP_HOST") or DEFAULT_SMTP_HOST).strip(),
        "port": port,
        "name": (os.getenv("ALERT_SENDER_NAME") or DEFAULT_SENDER_NAME).strip(),
    }


def is_configured() -> bool:
    """Whether a send could be attempted, for the UI to decide what to show."""

    try:
        smtp_settings()

    except AlertConfigError:
        return False

    return True


def sender_hint() -> str:
    """The configured sender, for the UI. Empty when unconfigured."""

    try:
        return smtp_settings()["sender"]

    except AlertConfigError:
        return ""


def _compose(recipient: str, alert: dict, settings: dict) -> EmailMessage:
    message = EmailMessage()

    message["Subject"] = alert["subject"]
    message["From"] = f"{settings['name']} <{settings['sender']}>"
    message["To"] = recipient

    # Plain text first, HTML as the alternative: a client that cannot render
    # the HTML still gets a readable forecast rather than markup.
    message.set_content(alert["text"])

    message.add_alternative(alert["html"], subtype="html")

    return message


def _deliver(message: EmailMessage, settings: dict) -> None:
    context = ssl.create_default_context()

    try:
        if settings["port"] == SSL_PORT:
            with smtplib.SMTP_SSL(
                settings["host"],
                settings["port"],
                timeout=TIMEOUT_SECONDS,
                context=context,
            ) as server:
                server.login(settings["sender"], settings["password"])
                server.send_message(message)

        else:
            with smtplib.SMTP(
                settings["host"],
                settings["port"],
                timeout=TIMEOUT_SECONDS,
            ) as server:
                server.starttls(context=context)
                server.login(settings["sender"], settings["password"])
                server.send_message(message)

    except smtplib.SMTPAuthenticationError as exc:
        raise AlertSendError(
            f"{settings['host']} rejected the sign-in for "
            f"{settings['sender']}. If this is a Gmail account, "
            f"ALERT_SENDER_PASSWORD must be a 16-character App Password "
            f"(Google account -> Security -> 2-Step Verification -> App "
            f"passwords), not the account password."
        ) from exc

    except smtplib.SMTPRecipientsRefused as exc:
        raise AlertAddressError(
            f"The mail server refused the recipient address: {exc.recipients}"
        ) from exc

    except (smtplib.SMTPException, OSError) as exc:
        raise AlertSendError(
            f"Could not send through {settings['host']}:{settings['port']} "
            f"- {type(exc).__name__}: {exc}"
        ) from exc


def send_alert(
    recipient: str,
    city: str,
    forecast: dict,
    reading_time=None,
    threshold: int = DEFAULT_ALERT_THRESHOLD,
    model_details: dict = None,
    dry_run: bool = False,
) -> dict:
    """
    Build and send the 3-day alert. Returns the composed alert's summary.

    ``dry_run`` composes and validates everything but does not open a
    connection, which is how the CLI lets you check the wording (and the
    configuration) without mailing anyone.
    """

    recipient = (recipient or "").strip()

    if not valid_address(recipient):
        raise AlertAddressError(
            f"'{recipient}' does not look like an email address."
        )

    settings = smtp_settings()

    alert = build_alert(
        city=city,
        forecast=forecast,
        reading_time=reading_time,
        threshold=threshold,
        model_details=model_details,
    )

    message = _compose(recipient, alert, settings)

    if not dry_run:
        _deliver(message, settings)

    return {
        "recipient": recipient,
        "subject": alert["subject"],
        "breaches": alert["breaches"],
        "worst": alert["worst"],
        "worst_aqi": alert["worst_aqi"],
        "threshold": alert["threshold"],
        "is_alert": alert["is_alert"],
        "sent": not dry_run,
        "sender": settings["sender"],
    }
