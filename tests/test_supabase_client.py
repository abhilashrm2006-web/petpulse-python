"""Covers _attach_owner_info — the piece that lets a vet's cross-household
patient list carry each pet's actual owner name/phone, so the agent (not
code) can tell two same-named pets from different owners apart."""

from app.config import Settings
from app.integrations.supabase_client import _attach_owner_info, make_supabase_client
from tests.fake_supabase import FakeSupabaseClient

FAKE_JWT_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJyb2xlIjoic2VydmljZV9yb2xlIn0.abc"


def test_make_supabase_client_retries_transient_transport_failures():
    """Reproduces a real production crash: a pooled HTTP/2 connection the
    server had already closed got reused anyway, raised
    httpcore.RemoteProtocolError("Server disconnected"), and — since
    httpx's default transport does zero retries — took down an entire
    inbound WhatsApp turn (no reply sent at all) for what a fresh
    connection would have handled fine. Both the postgrest (`.table()`)
    and storage clients must retry transient connection failures."""
    settings = Settings(supabase_url="https://x.supabase.co", supabase_service_role_key=FAKE_JWT_KEY)
    client = make_supabase_client(settings)

    assert client.postgrest.session._transport._pool._retries == 2
    assert client.storage._client._transport._pool._retries == 2


def test_attach_owner_info_fills_in_each_pets_primary_owner():
    supabase = FakeSupabaseClient(
        initial={
            "pet_members": [
                {"pet_id": "b1", "profile_id": "owner-1", "is_primary": True},
                {"pet_id": "b1", "profile_id": "vet-1", "is_primary": False},
                {"pet_id": "b2", "profile_id": "owner-2", "is_primary": True},
            ],
            "profiles": [
                {"id": "owner-1", "full_name": "Abhilash", "phone_number": "919000000001"},
                {"id": "owner-2", "full_name": "Priya", "phone_number": "919000000002"},
                {"id": "vet-1", "full_name": "Dr. Rao", "phone_number": "919111111111"},
            ],
        }
    )
    pets = [{"id": "b1", "name": "Bobby"}, {"id": "b2", "name": "Bobby"}]

    _attach_owner_info(supabase, pets)

    by_id = {p["id"]: p for p in pets}
    assert by_id["b1"]["owner_name"] == "Abhilash"
    assert by_id["b1"]["owner_phone"] == "919000000001"
    assert by_id["b2"]["owner_name"] == "Priya"
    assert by_id["b2"]["owner_phone"] == "919000000002"


def test_attach_owner_info_handles_no_pets():
    supabase = FakeSupabaseClient()
    pets = []
    _attach_owner_info(supabase, pets)  # must not raise
    assert pets == []


