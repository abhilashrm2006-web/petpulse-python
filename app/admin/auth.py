"""Auth gate for every /admin/* route (app/admin/routes.py) -- a single
shared bearer token, matching the "one operator for now" scope agreed with
the user. Deliberately fails closed: an unconfigured admin_api_token means
every admin request 401s, never a silent bypass."""

import hmac

from fastapi import HTTPException, Request

from app.deps import AppContext


def require_admin_token(request: Request) -> None:
    ctx: AppContext = request.app.state.ctx
    token = ctx.settings.admin_api_token
    if not token:
        raise HTTPException(status_code=401, detail="Admin API not configured")

    header = request.headers.get("authorization", "")
    if not header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")

    supplied = header.removeprefix("Bearer ")
    if not hmac.compare_digest(supplied, token):
        raise HTTPException(status_code=401, detail="Invalid token")
