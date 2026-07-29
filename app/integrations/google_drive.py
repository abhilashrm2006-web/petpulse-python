"""Google Drive read-only access for the doctor-onboarding-drafts sync (see
app/scheduler/jobs.py sync_doctor_onboarding_drafts) -- a Google Cloud
service account (not a personal Gmail/OAuth login) with Viewer access
shared on one specific parent folder, so this app never has broader
inbox/Drive access than that one folder needs.

Implements the service-account JWT-bearer OAuth2 flow directly (RS256-sign
a short-lived assertion, exchange it for an access token) rather than
depending on the google-auth/google-api-python-client packages, matching
how every other integration in this codebase talks to its provider over
plain httpx rather than a heavier official SDK."""

import base64
import json
import time
from typing import Any

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from app.config import Settings

DRIVE_SCOPE = "https://www.googleapis.com/auth/drive.readonly"
TOKEN_URL = "https://oauth2.googleapis.com/token"
DRIVE_API_BASE = "https://www.googleapis.com/drive/v3"

# Cached per-process (each Uvicorn worker has its own -- no cross-process
# consistency needed for an OAuth access token cache). Keyed by client_email
# so distinct service accounts, if ever used, don't collide.
_token_cache: dict[str, tuple[str, float]] = {}


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _sign_jwt(service_account_info: dict[str, Any]) -> str:
    now = int(time.time())
    header = {"alg": "RS256", "typ": "JWT"}
    claims = {
        "iss": service_account_info["client_email"],
        "scope": DRIVE_SCOPE,
        "aud": TOKEN_URL,
        "iat": now,
        "exp": now + 3600,
    }
    signing_input = f"{_b64url(json.dumps(header).encode())}.{_b64url(json.dumps(claims).encode())}"
    private_key = serialization.load_pem_private_key(service_account_info["private_key"].encode(), password=None)
    signature = private_key.sign(signing_input.encode(), padding.PKCS1v15(), hashes.SHA256())
    return f"{signing_input}.{_b64url(signature)}"


async def _get_access_token(settings: Settings, http: httpx.AsyncClient) -> str:
    """Cached for its actual lifetime (~1h) minus a safety margin, so a
    sync run that makes many Drive calls doesn't re-authenticate every
    single call."""
    service_account_info = json.loads(settings.google_service_account_json)
    cache_key = service_account_info["client_email"]
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


async def list_subfolders(settings: Settings, http: httpx.AsyncClient, parent_folder_id: str) -> list[dict[str, Any]]:
    token = await _get_access_token(settings, http)
    resp = await http.get(
        f"{DRIVE_API_BASE}/files",
        headers={"Authorization": f"Bearer {token}"},
        params={
            "q": f"'{parent_folder_id}' in parents and trashed = false and mimeType = 'application/vnd.google-apps.folder'",
            "fields": "files(id,name,modifiedTime)",
            "supportsAllDrives": "true",
            "includeItemsFromAllDrives": "true",
            "pageSize": 1000,
        },
    )
    resp.raise_for_status()
    return resp.json().get("files", [])


async def list_files(settings: Settings, http: httpx.AsyncClient, folder_id: str) -> list[dict[str, Any]]:
    token = await _get_access_token(settings, http)
    resp = await http.get(
        f"{DRIVE_API_BASE}/files",
        headers={"Authorization": f"Bearer {token}"},
        params={
            "q": f"'{folder_id}' in parents and trashed = false",
            "fields": "files(id,name,mimeType,modifiedTime,size)",
            "supportsAllDrives": "true",
            "includeItemsFromAllDrives": "true",
            "pageSize": 1000,
        },
    )
    resp.raise_for_status()
    return resp.json().get("files", [])


async def download_file(settings: Settings, http: httpx.AsyncClient, file_id: str) -> bytes:
    token = await _get_access_token(settings, http)
    resp = await http.get(
        f"{DRIVE_API_BASE}/files/{file_id}",
        headers={"Authorization": f"Bearer {token}"},
        params={"alt": "media", "supportsAllDrives": "true"},
    )
    resp.raise_for_status()
    return resp.content
