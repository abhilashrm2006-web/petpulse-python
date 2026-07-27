"""Covers the public GET /passport/{token} route -- calls the handler
function directly (bypassing TestClient/lifespan, which would try to build
a real Supabase client) with a fake request carrying app.state.ctx, the
same shape FastAPI provides at runtime."""

from types import SimpleNamespace

import pytest

from app.main import public_passport
from tests.fake_supabase import FakeSupabaseClient


def _fake_request(ctx):
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(ctx=ctx)))


@pytest.mark.asyncio
async def test_valid_token_renders_passport_html():
    supabase = FakeSupabaseClient(
        initial={
            "pets": [{"id": "pet-1", "name": "Rex", "species": "Dog", "breed": "Labrador", "passport_share_token": "tok123"}],
            "vaccinations": [{"id": "v1", "pet_id": "pet-1", "vaccine_name": "Rabies", "date_administered": "2025-01-01"}],
            "medical_records": [],
        }
    )
    ctx = SimpleNamespace(supabase=supabase)

    response = await public_passport("tok123", _fake_request(ctx))

    assert response.status_code == 200
    body = response.body.decode()
    assert "Rex" in body
    assert "Rabies" in body


@pytest.mark.asyncio
async def test_unknown_token_returns_404():
    supabase = FakeSupabaseClient(initial={"pets": []})
    ctx = SimpleNamespace(supabase=supabase)

    response = await public_passport("does-not-exist", _fake_request(ctx))

    assert response.status_code == 404
