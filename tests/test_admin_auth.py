"""Covers require_admin_token -- calls it directly as a plain function with
a fake Request (bypassing TestClient/lifespan, same approach as
test_main_passport.py) since it only reads request.app.state.ctx and
request.headers."""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.admin.auth import require_admin_token


def _fake_request(token_configured: str, auth_header: str | None):
    ctx = SimpleNamespace(settings=SimpleNamespace(admin_api_token=token_configured))
    headers = {"authorization": auth_header} if auth_header is not None else {}
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(ctx=ctx)), headers=headers)


def test_fails_closed_when_token_not_configured():
    request = _fake_request(token_configured="", auth_header="Bearer anything")
    with pytest.raises(HTTPException) as exc:
        require_admin_token(request)
    assert exc.value.status_code == 401


def test_rejects_missing_header():
    request = _fake_request(token_configured="secret123", auth_header=None)
    with pytest.raises(HTTPException) as exc:
        require_admin_token(request)
    assert exc.value.status_code == 401


def test_rejects_non_bearer_header():
    request = _fake_request(token_configured="secret123", auth_header="Basic secret123")
    with pytest.raises(HTTPException) as exc:
        require_admin_token(request)
    assert exc.value.status_code == 401


def test_rejects_wrong_token():
    request = _fake_request(token_configured="secret123", auth_header="Bearer wrong")
    with pytest.raises(HTTPException) as exc:
        require_admin_token(request)
    assert exc.value.status_code == 401


def test_accepts_correct_token():
    request = _fake_request(token_configured="secret123", auth_header="Bearer secret123")
    require_admin_token(request)  # must not raise
