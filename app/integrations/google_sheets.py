"""Google Sheets read-only access for the early-user feedback survey
responses (see app/admin/routes.py's feedback_survey_responses endpoint) --
the same service account already used for Drive access
(app/integrations/google_drive.py), just a different OAuth scope and API,
since Google Forms responses land in a linked Sheet the account has been
given Viewer access to. Same hand-rolled JWT-bearer flow as google_drive.py
(no google-auth/google-api-python-client dependency), but Sheets needs its
own scope, so this can't just call google_drive's cached token -- a
service account's OAuth token is scoped at mint time, not per-request."""

import base64
import json
import time
from typing import Any

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from app.config import Settings

SHEETS_SCOPE = "https://www.googleapis.com/auth/spreadsheets.readonly"
TOKEN_URL = "https://oauth2.googleapis.com/token"
SHEETS_API_BASE = "https://sheets.googleapis.com/v4/spreadsheets"

# Cached per-process, same reasoning as google_drive.py's _token_cache --
# separate cache key (scope isn't part of the key there since that module
# only ever requests one scope) so a Sheets-scoped token never gets handed
# out where a Drive-scoped one was expected, or vice versa.
_token_cache: dict[str, tuple[str, float]] = {}


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _sign_jwt(service_account_info: dict[str, Any]) -> str:
    now = int(time.time())
    header = {"alg": "RS256", "typ": "JWT"}
    claims = {
        "iss": service_account_info["client_email"],
        "scope": SHEETS_SCOPE,
        "aud": TOKEN_URL,
        "iat": now,
        "exp": now + 3600,
    }
    signing_input = f"{_b64url(json.dumps(header).encode())}.{_b64url(json.dumps(claims).encode())}"
    private_key = serialization.load_pem_private_key(service_account_info["private_key"].encode(), password=None)
    signature = private_key.sign(signing_input.encode(), padding.PKCS1v15(), hashes.SHA256())
    return f"{signing_input}.{_b64url(signature)}"


async def _get_access_token(settings: Settings, http: httpx.AsyncClient) -> str:
    service_account_info = json.loads(settings.google_service_account_json)
    cache_key = f"sheets:{service_account_info['client_email']}"
    cached = _token_cache.get(cache_key)
    if cached and cached[1] > time.time():
        return cached[0]

    assertion = _sign_jwt(service_account_info)
    resp = await http.post(
        TOKEN_URL,
        data={"grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer", "assertion": assertion},
    )
    resp.raise_for_status()
    body = resp.json()
    token = body["access_token"]
    expires_in = body.get("expires_in", 3600)
    _token_cache[cache_key] = (token, time.time() + expires_in - 60)
    return token


async def get_sheet_values(
    settings: Settings, http: httpx.AsyncClient, spreadsheet_id: str, sheet_range: str = "A:Z"
) -> list[list[str]]:
    """Returns raw rows (first row is Google Forms' own header row: the
    exact question text of each column) -- no assumption baked in here
    about which columns exist, since a form can gain/lose questions over
    time and the caller is what interprets column meaning."""
    token = await _get_access_token(settings, http)
    resp = await http.get(
        f"{SHEETS_API_BASE}/{spreadsheet_id}/values/{sheet_range}",
        headers={"Authorization": f"Bearer {token}"},
    )
    resp.raise_for_status()
    return resp.json().get("values", [])
