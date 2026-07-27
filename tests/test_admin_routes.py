"""Covers the /admin/* route handlers directly (bypassing TestClient/lifespan,
same approach as test_main_passport.py) against FakeSupabaseClient. Auth
itself is covered separately in test_admin_auth.py."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from app.admin import routes as admin_routes
from tests.fake_supabase import FakeSupabaseClient


def _make_ctx(supabase):
    whatsapp = SimpleNamespace(send_text=AsyncMock())
    settings = SimpleNamespace(
        razorpay_key_id="x", razorpay_key_secret="y",
    )
    return SimpleNamespace(supabase=supabase, whatsapp=whatsapp, settings=settings)


def _fake_request(ctx):
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(ctx=ctx)))


@pytest.mark.asyncio
async def test_list_customers_filters_by_role_and_search():
    supabase = FakeSupabaseClient(
        initial={
            "profiles": [
                {"id": "c1", "role": "customer", "full_name": "Jane Doe", "phone_number": "919000000001", "email": "jane@x.com", "created_at": "2026-01-01"},
                {"id": "c2", "role": "customer", "full_name": "Bob Smith", "phone_number": "919000000002", "email": "bob@x.com", "created_at": "2026-01-02"},
                {"id": "v1", "role": "vet", "full_name": "Dr. Rao", "phone_number": "919000000003", "created_at": "2026-01-01"},
            ]
        }
    )
    request = _fake_request(_make_ctx(supabase))

    all_customers = await admin_routes.list_customers(request)
    assert all_customers["count"] == 2

    filtered = await admin_routes.list_customers(request, search="jane")
    assert filtered["count"] == 1
    assert filtered["customers"][0]["id"] == "c1"


@pytest.mark.asyncio
async def test_list_customers_includes_each_customers_pets():
    supabase = FakeSupabaseClient(
        initial={
            "profiles": [
                {"id": "c1", "role": "customer", "full_name": "Jane", "phone_number": "919000000001", "created_at": "2026-01-01"},
                {"id": "c2", "role": "customer", "full_name": "Bob", "phone_number": "919000000002", "created_at": "2026-01-02"},
            ],
            "pets": [
                {"id": "p1", "profile_id": "c1", "name": "Rex", "breed": "Labrador", "date_of_birth": "2022-01-01", "created_at": "2026-01-01"},
                {"id": "p2", "profile_id": "c1", "name": "Milo", "breed": "Persian", "date_of_birth": "2023-01-01", "created_at": "2026-01-02"},
            ],
        }
    )
    request = _fake_request(_make_ctx(supabase))

    result = await admin_routes.list_customers(request)

    by_id = {c["id"]: c for c in result["customers"]}
    assert len(by_id["c1"]["pets"]) == 2
    assert by_id["c1"]["pets"][0]["name"] == "Rex"
    assert by_id["c2"]["pets"] == []


@pytest.mark.asyncio
async def test_list_customers_filters_by_date_range():
    supabase = FakeSupabaseClient(
        initial={
            "profiles": [
                {"id": "c1", "role": "customer", "full_name": "Old", "phone_number": "919000000001", "created_at": "2026-01-01T00:00:00"},
                {"id": "c2", "role": "customer", "full_name": "Mid", "phone_number": "919000000002", "created_at": "2026-06-15T00:00:00"},
                {"id": "c3", "role": "customer", "full_name": "New", "phone_number": "919000000003", "created_at": "2026-12-01T00:00:00"},
            ],
        }
    )
    request = _fake_request(_make_ctx(supabase))

    result = await admin_routes.list_customers(request, date_from="2026-06-01", date_to="2026-06-30")

    assert result["count"] == 1
    assert result["customers"][0]["id"] == "c2"


@pytest.mark.asyncio
async def test_list_customers_categorizes_subscription_tier():
    supabase = FakeSupabaseClient(
        initial={
            "profiles": [
                {"id": "c1", "role": "customer", "full_name": "FreeCust", "phone_number": "919000000001", "created_at": "2026-01-01", "is_founding_member": False},
                {"id": "c2", "role": "customer", "full_name": "TrialCust", "phone_number": "919000000002", "created_at": "2026-01-02", "is_founding_member": False},
                {"id": "c3", "role": "customer", "full_name": "SubCust", "phone_number": "919000000003", "created_at": "2026-01-03", "is_founding_member": False},
                {"id": "c4", "role": "customer", "full_name": "FoundingCust", "phone_number": "919000000004", "created_at": "2026-01-04", "is_founding_member": True},
            ],
            "subscriptions": [
                {"id": "s2", "profile_id": "c2", "status": "trial", "amount": 399, "created_at": "2026-01-02"},
                {"id": "s3", "profile_id": "c3", "status": "active", "amount": 399, "created_at": "2026-01-03"},
                {"id": "s4", "profile_id": "c4", "status": "active", "amount": 99, "created_at": "2026-01-04"},
            ],
        }
    )
    request = _fake_request(_make_ctx(supabase))

    result = await admin_routes.list_customers(request)

    categories = {c["full_name"]: c["subscription_category"] for c in result["customers"]}
    assert categories == {"FreeCust": "Free", "TrialCust": "Trial", "SubCust": "Subscriber", "FoundingCust": "Founding"}


@pytest.mark.asyncio
async def test_list_customers_filters_by_tier():
    supabase = FakeSupabaseClient(
        initial={
            "profiles": [
                {"id": "c1", "role": "customer", "full_name": "FreeCust", "phone_number": "919000000001", "created_at": "2026-01-01", "is_founding_member": False},
                {"id": "c3", "role": "customer", "full_name": "SubCust", "phone_number": "919000000003", "created_at": "2026-01-03", "is_founding_member": False},
            ],
            "subscriptions": [
                {"id": "s3", "profile_id": "c3", "status": "active", "amount": 399, "created_at": "2026-01-03"},
            ],
        }
    )
    request = _fake_request(_make_ctx(supabase))

    result = await admin_routes.list_customers(request, tier="subscriber")

    assert result["count"] == 1
    assert result["customers"][0]["full_name"] == "SubCust"


@pytest.mark.asyncio
async def test_activate_customer_flips_is_active_true():
    supabase = FakeSupabaseClient(initial={"profiles": [{"id": "c1", "role": "customer", "is_active": False}]})
    request = _fake_request(_make_ctx(supabase))

    result = await admin_routes.activate_customer("c1", request)

    assert result["success"] is True
    assert supabase.rows("profiles")[0]["is_active"] is True


@pytest.mark.asyncio
async def test_activate_customer_404_when_missing():
    supabase = FakeSupabaseClient()
    request = _fake_request(_make_ctx(supabase))

    with pytest.raises(HTTPException) as exc:
        await admin_routes.activate_customer("nope", request)
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_get_customer_detail_includes_pets_subscription_and_documents():
    supabase = FakeSupabaseClient(
        initial={
            "profiles": [{"id": "c1", "role": "customer", "full_name": "Jane"}],
            "pets": [{"id": "p1", "profile_id": "c1", "name": "Rex"}],
            "subscriptions": [{"id": "s1", "profile_id": "c1", "status": "active", "amount": 399, "created_at": "2026-01-01"}],
            "doctor_sessions": [{"id": "sess1", "profile_id": "c1", "status": "accepted", "preferred_time": "2026-08-01T10:00:00+05:30", "created_at": "2026-01-01"}],
            "documents": [{"id": "d1", "pet_id": "p1"}],
        }
    )
    request = _fake_request(_make_ctx(supabase))

    result = await admin_routes.get_customer("c1", request)

    assert result["profile"]["id"] == "c1"
    assert len(result["pets"]) == 1
    assert result["subscription"]["status"] == "active"
    assert result["document_count"] == 1
    assert len(result["recent_sessions"]) == 1


@pytest.mark.asyncio
async def test_get_customer_404_when_missing():
    supabase = FakeSupabaseClient()
    request = _fake_request(_make_ctx(supabase))

    with pytest.raises(HTTPException) as exc:
        await admin_routes.get_customer("does-not-exist", request)
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_deactivate_customer_cancels_sessions_and_subscription(monkeypatch):
    supabase = FakeSupabaseClient(
        initial={
            "profiles": [{"id": "c1", "role": "customer", "full_name": "Jane", "phone_number": "919000000001"}],
            "doctor_sessions": [
                {"id": "sess1", "profile_id": "c1", "status": "pending", "doctor_phone": "919000000099"},
            ],
            "subscriptions": [{"id": "s1", "profile_id": "c1", "status": "active", "provider_subscription_id": "sub_abc", "amount": 399}],
        }
    )
    cancel_subscription = AsyncMock(return_value={})
    monkeypatch.setattr("app.admin.routes.razorpay_client.cancel_subscription", cancel_subscription)
    ctx = _make_ctx(supabase)
    request = _fake_request(ctx)

    result = await admin_routes.deactivate_customer("c1", request)

    assert result["success"] is True
    assert result["cancelled_sessions"] == 1
    assert result["subscription_cancelled"] is True
    cancel_subscription.assert_awaited_once_with(ctx.settings, "sub_abc")
    assert supabase.rows("profiles")[0]["is_active"] is False
    assert supabase.rows("doctor_sessions")[0]["status"] == "cancelled"
    assert supabase.rows("subscriptions")[0]["status"] == "cancelled"


@pytest.mark.asyncio
async def test_delete_customer_removes_the_row(monkeypatch):
    supabase = FakeSupabaseClient(initial={"profiles": [{"id": "c1", "role": "customer", "phone_number": "919000000001"}]})
    request = _fake_request(_make_ctx(supabase))

    result = await admin_routes.delete_customer("c1", request)

    assert result["success"] is True
    assert supabase.rows("profiles") == []


@pytest.mark.asyncio
async def test_delete_customer_purges_no_action_fk_references():
    """Regression test for a real live failure: deleting a customer with
    any doctor_sessions history 500'd with a Postgres FK violation, since
    doctor_sessions.profile_id (and 3 other tables) are ON DELETE NO ACTION,
    not CASCADE. All four must be cleared before the profile row itself is
    deleted, not just the ones that already have their own cancel/notify
    logic (doctor_sessions)."""
    supabase = FakeSupabaseClient(
        initial={
            "profiles": [{"id": "c1", "role": "customer", "phone_number": "919000000001"}],
            "doctor_sessions": [{"id": "sess1", "profile_id": "c1", "status": "completed", "doctor_phone": "919000000099"}],
            "new_parent_followups": [{"id": "f1", "profile_id": "c1"}],
            "new_parent_guides": [{"id": "g1", "profile_id": "c1"}],
            "pet_members": [{"pet_id": "p1", "profile_id": "other-profile", "added_by": "c1"}],
        }
    )
    request = _fake_request(_make_ctx(supabase))

    result = await admin_routes.delete_customer("c1", request)

    assert result["success"] is True
    assert supabase.rows("profiles") == []
    assert supabase.rows("doctor_sessions") == []
    assert supabase.rows("new_parent_followups") == []
    assert supabase.rows("new_parent_guides") == []
    assert supabase.rows("pet_members") == []


@pytest.mark.asyncio
async def test_onboard_doctor_creates_a_vet_profile():
    supabase = FakeSupabaseClient()
    request = _fake_request(_make_ctx(supabase))
    payload = {
        "full_name": "Dr. Rao", "phone_number": "919000000010", "qualification": "BVSc",
        "registration_number": "REG123", "specialization": "General", "experience_years": 5, "consultation_fee": 399,
    }

    result = await admin_routes.onboard_doctor(request, payload)

    assert result["success"] is True
    assert result["doctor"]["role"] == "vet"
    assert result["doctor"]["is_active"] is True
    assert supabase.rows("profiles")[0]["registration_number"] == "REG123"


@pytest.mark.asyncio
async def test_onboard_doctor_rejects_duplicate_phone_number():
    supabase = FakeSupabaseClient(initial={"profiles": [{"id": "v1", "phone_number": "919000000010", "role": "vet"}]})
    request = _fake_request(_make_ctx(supabase))

    with pytest.raises(HTTPException) as exc:
        await admin_routes.onboard_doctor(request, {"full_name": "Dr. Rao", "phone_number": "919000000010"})
    assert exc.value.status_code == 409


@pytest.mark.asyncio
async def test_onboard_doctor_requires_name_and_phone():
    supabase = FakeSupabaseClient()
    request = _fake_request(_make_ctx(supabase))

    with pytest.raises(HTTPException) as exc:
        await admin_routes.onboard_doctor(request, {"full_name": "Dr. Rao"})
    assert exc.value.status_code == 422


@pytest.mark.asyncio
async def test_deactivate_doctor_cancels_their_pending_sessions():
    supabase = FakeSupabaseClient(
        initial={
            "profiles": [{"id": "v1", "role": "vet", "phone_number": "919000000010", "full_name": "Dr. Rao"}],
            "doctor_sessions": [
                {"id": "sess1", "doctor_phone": "919000000010", "profile_id": "c1", "status": "accepted"},
                {"id": "sess2", "doctor_phone": "919000000010", "profile_id": "c1", "status": "cancelled"},
            ],
        }
    )
    request = _fake_request(_make_ctx(supabase))

    result = await admin_routes.deactivate_doctor("v1", request)

    assert result["success"] is True
    assert result["cancelled_sessions"] == 1
    assert supabase.rows("profiles")[0]["is_active"] is False
    statuses = {s["id"]: s["status"] for s in supabase.rows("doctor_sessions")}
    assert statuses["sess1"] == "cancelled"


@pytest.mark.asyncio
async def test_analytics_overview_computes_expected_counts():
    from datetime import date

    today = date.today().isoformat()
    supabase = FakeSupabaseClient(
        initial={
            "profiles": [
                {"id": "c1", "role": "customer", "created_at": today},
                {"id": "c2", "role": "customer", "created_at": "2020-01-01"},
                {"id": "v1", "role": "vet", "created_at": today, "is_founding_member": False},
            ],
            "subscriptions": [
                {"id": "s1", "status": "active", "amount": 99},
                {"id": "s2", "status": "active", "amount": 399},
                {"id": "s3", "status": "cancelled", "amount": 399},
            ],
            "doctor_sessions": [{"id": "sess1", "status": "accepted", "created_at": today}],
            "health_logs": [{"id": "h1", "created_at": today}],
            "documents": [{"id": "d1", "uploaded_at": today}],
        }
    )
    supabase._store["profiles"][0]["is_founding_member"] = True
    request = _fake_request(_make_ctx(supabase))

    result = await admin_routes.analytics_overview(request)

    assert result["total_customers"] == 2
    assert result["active_subscribers"] == 2
    assert result["founding_members"] == 1
    assert result["standard_subscribers"] == 1
    assert result["estimated_mrr"] == 498
    assert result["new_signups_this_month"] == 1
    assert result["consults_this_month"] == 1
    assert result["symptom_checks_this_month"] == 1
    assert result["documents_this_month"] == 1


@pytest.mark.asyncio
async def test_analytics_timeseries_groups_by_day():
    supabase = FakeSupabaseClient(
        initial={
            "profiles": [
                {"id": "c1", "role": "customer", "created_at": "2026-07-01T10:00:00"},
                {"id": "c2", "role": "customer", "created_at": "2026-07-01T15:00:00"},
                {"id": "c3", "role": "customer", "created_at": "2026-07-02T09:00:00"},
            ],
            "subscriptions": [
                {"id": "s1", "created_at": "2026-07-01T10:00:00", "amount": 399},
                {"id": "s2", "created_at": "2026-07-02T09:00:00", "amount": 99},
            ],
        }
    )
    request = _fake_request(_make_ctx(supabase))

    result = await admin_routes.analytics_timeseries(request, days=30)

    signups = {row["date"]: row["count"] for row in result["signups"]}
    assert signups["2026-07-01"] == 2
    assert signups["2026-07-02"] == 1
    revenue = {row["date"]: row["amount"] for row in result["revenue"]}
    assert revenue["2026-07-01"] == 399
    assert revenue["2026-07-02"] == 99
