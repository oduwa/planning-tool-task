from __future__ import annotations

import base64
import getpass
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests
from cryptography.fernet import Fernet, InvalidToken
from dotenv import dotenv_values, set_key

DEFAULT_CREDENTIALS_FILE = Path("assignment-credentials.json")
DEFAULT_ENV_FILE = Path(".env")
OPENROUTER_KEY_NAME = "OPENROUTER_API_KEY"
OPENROUTER_BASE_URL_NAME = "OPENROUTER_BASE_URL"
ASSIGNMENT_DEADLINE_TIMEZONE_NAME = "ASSIGNMENT_DEADLINE_TIMEZONE"
OPENROUTER_STATUS_URL = "https://openrouter.ai/api/v1/key"


class AssignmentSetupError(RuntimeError):
    """Raised when local assignment setup cannot complete safely."""


@dataclass(frozen=True)
class AssignmentCredentials:
    """Decrypted OpenRouter configuration."""

    api_key: str
    base_url: str
    expires_at: str | None = None
    deadline_timezone: str | None = None


def _format_expiry(expires_at: str, deadline_timezone: str | None) -> str:
    """Format an expiry in the assignment pipeline's local timezone.

    Args:
        expires_at: ISO 8601 API-key expiry.
        deadline_timezone: IANA timezone used to display the deadline.

    Returns:
        A human-readable local expiry.

    Raises:
        AssignmentSetupError: If the expiry or timezone is invalid.
    """
    try:
        expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
        if expiry.utcoffset() is None:
            raise ValueError
        zone = ZoneInfo(deadline_timezone) if deadline_timezone else UTC
    except (ValueError, ZoneInfoNotFoundError) as error:
        raise AssignmentSetupError("Invalid assignment expiry configuration") from error
    local_expiry = expiry.astimezone(zone)
    return f"{local_expiry:%A}, {local_expiry.day} {local_expiry:%B %Y at %H:%M %Z}"


def decrypt_credentials(path: Path, passphrase: str) -> AssignmentCredentials:
    """Decrypt and validate a versioned assignment credential envelope.

    Args:
        path: Encrypted credential envelope path.
        passphrase: Passphrase supplied by recruitment.

    Returns:
        Validated OpenRouter credentials.

    Raises:
        AssignmentSetupError: If the file, passphrase, or payload is invalid.
    """
    if not passphrase:
        raise AssignmentSetupError("Passphrase cannot be empty")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise AssignmentSetupError(
            f"{path} is missing. Pull the latest repository changes or contact recruitment."
        ) from error
    except (OSError, json.JSONDecodeError) as error:
        raise AssignmentSetupError(f"Could not read {path} safely") from error

    if (
        not isinstance(document, dict)
        or set(document) != {"version", "token"}
        or document.get("version") != 1
        or not isinstance(document.get("token"), str)
    ):
        raise AssignmentSetupError("Unsupported assignment credentials format")
    try:
        key = hashlib.sha256(b"fernet\0" + passphrase.encode()).digest()
        plaintext = Fernet(base64.urlsafe_b64encode(key)).decrypt(
            document["token"].encode("ascii")
        )
        payload = json.loads(plaintext)
    except (
        InvalidToken,
        UnicodeEncodeError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as error:
        raise AssignmentSetupError(
            "The passphrase is incorrect or the credentials file has been changed"
        ) from error

    if not isinstance(payload, dict):
        raise AssignmentSetupError("Invalid decrypted credentials")
    api_key = payload.get("apiKey")
    base_url = payload.get("baseUrl")
    expires_at = payload.get("expiresAt")
    deadline_timezone = payload.get("deadlineTimezone")
    if not isinstance(api_key, str) or not api_key:
        raise AssignmentSetupError("Invalid decrypted credentials")
    if not isinstance(base_url, str) or not base_url.startswith("https://"):
        raise AssignmentSetupError("Invalid decrypted credentials")
    if expires_at is not None and not isinstance(expires_at, str):
        raise AssignmentSetupError("Invalid decrypted credentials")
    if deadline_timezone is not None and (
        not isinstance(deadline_timezone, str) or not deadline_timezone
    ):
        raise AssignmentSetupError("Invalid decrypted credentials")
    if expires_at:
        _format_expiry(expires_at, deadline_timezone)
    return AssignmentCredentials(api_key, base_url, expires_at, deadline_timezone)


def save_openrouter_configuration(
    path: Path, credentials: AssignmentCredentials
) -> None:
    """Save OpenRouter configuration without replacing an existing key.

    Args:
        path: Destination dotenv file.
        credentials: Validated OpenRouter credentials.

    Raises:
        AssignmentSetupError: If the file cannot be read or written safely.
    """
    try:
        current = dotenv_values(path) if path.exists() else {}
    except OSError as error:
        raise AssignmentSetupError(f"Could not read {path} safely") from error
    if current.get(OPENROUTER_KEY_NAME):
        raise AssignmentSetupError(
            f"{path} already contains {OPENROUTER_KEY_NAME}; it was not overwritten"
        )
    try:
        set_key(path, OPENROUTER_BASE_URL_NAME, credentials.base_url)
        set_key(path, OPENROUTER_KEY_NAME, credentials.api_key)
        if credentials.deadline_timezone:
            set_key(
                path,
                ASSIGNMENT_DEADLINE_TIMEZONE_NAME,
                credentials.deadline_timezone,
            )
        if os.name != "nt":
            path.chmod(0o600)
    except OSError as error:
        raise AssignmentSetupError(f"Could not save {path} safely") from error


def main() -> None:
    """Run the local, offline candidate setup flow."""
    try:
        passphrase = getpass.getpass("Passphrase from your recruitment email: ")
        credentials = decrypt_credentials(DEFAULT_CREDENTIALS_FILE, passphrase)
        save_openrouter_configuration(DEFAULT_ENV_FILE, credentials)
    except (AssignmentSetupError, OSError) as error:
        sys.stderr.write(f"Setup failed: {error}\n")
        raise SystemExit(1) from error

    sys.stdout.write(
        f"Setup complete. Your API key is stored in {DEFAULT_ENV_FILE} and was not displayed.\n"
    )
    if credentials.expires_at:
        formatted_expiry = _format_expiry(
            credentials.expires_at,
            credentials.deadline_timezone,
        )
        sys.stdout.write(f"API access expires: {formatted_expiry}\n")


def status() -> None:
    """Show spend and expiry for the API key stored in ``.env``."""
    configuration = dotenv_values(DEFAULT_ENV_FILE)
    api_key = configuration.get(OPENROUTER_KEY_NAME)
    deadline_timezone = configuration.get(ASSIGNMENT_DEADLINE_TIMEZONE_NAME)
    if not isinstance(api_key, str) or not api_key:
        sys.stderr.write("Status failed: Run assignment-setup first\n")
        raise SystemExit(1)

    try:
        response = requests.get(
            OPENROUTER_STATUS_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
            },
            allow_redirects=False,
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()["data"]
        usage = float(data["usage"])
        limit = float(data["limit"])
        remaining = float(data["limit_remaining"])
        expires_at = data.get("expires_at")
        if expires_at is not None and not isinstance(expires_at, str):
            raise TypeError
        expiry = (
            datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            if expires_at
            else None
        )
        if expiry is not None and expiry.utcoffset() is None:
            raise ValueError
        if deadline_timezone is not None and not isinstance(deadline_timezone, str):
            raise TypeError
        formatted_expiry = (
            _format_expiry(expires_at, deadline_timezone) if expires_at else "Not set"
        )
    except (
        AssignmentSetupError,
        requests.RequestException,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        unauthorized = (
            isinstance(error, requests.HTTPError)
            and error.response is not None
            and error.response.status_code == 401
        )
        message = (
            "The API key is invalid, disabled, or expired"
            if unauthorized
            else "OpenRouter could not return the key status"
        )
        sys.stderr.write(f"Status failed: {message}\n")
        raise SystemExit(1) from error

    sys.stdout.write("Assignment API key status\n")
    if expiry:
        seconds_left = (expiry - datetime.now(UTC)).total_seconds()
        if seconds_left <= 0:
            time_left = "Expired"
        else:
            total_hours = int(seconds_left // 3_600)
            days, hours = divmod(total_hours, 24)
            time_left = (
                f"{days} days, {hours} hours"
                if days
                else f"{hours} hours"
                if hours
                else "Less than 1 hour"
            )
    else:
        time_left = "Not set"
    sys.stdout.write(f"API access remaining: {time_left}\n")
    sys.stdout.write(f"Spend remaining: ${remaining:.2f} of ${limit:.2f}\n")
    sys.stdout.write(f"Spend used: ${usage:.2f}\n")
    sys.stdout.write(f"API access expires: {formatted_expiry}\n")


if __name__ == "__main__":
    main()
