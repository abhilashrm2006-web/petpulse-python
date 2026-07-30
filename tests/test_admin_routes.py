"""Covers the /admin/* route handlers directly (bypassing TestClient/lifespan,
same approach as test_main_passport.py) against FakeSupabaseClient. Auth
itself is covered separately in test_admin_auth.py."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from app.admin import routes as admin_routes
from tests.fake_supabase import FakeSupabaseClient


def _make_ctx(supabase):
    whatsapp = SimpleNamespace(send_text=AsyncMock())
    settings = SimpleNamespace(
        razorpay_key_id="x", razorpay_key_secret="y",
    )
    return SimpleNamespace(supabase=supabase, whatsapp=whatsapp, settings=settings, openai=SimpleNamespace())


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
async def test_list_customers_search_with_filter_syntax_characters_is_treated_as_literal_text():
    """Real bug found via audit: `search` was spliced unescaped into a raw
    PostgREST .or_() filter expression, so a value containing a comma or
    parenthesis could inject additional filter clauses instead of being
    treated as a literal search term. A search string built to look like
    injected filter syntax must match nothing (no customer's name/phone/
    email literally contains that text) rather than being parsed as extra
    clauses that could widen the result set."""
    supabase = FakeSupabaseClient(
        initial={
            "profiles": [
                {"id": "c1", "role": "customer", "full_name": "Jane Doe", "phone_number": "919000000001", "email": "jane@x.com", "created_at": "2026-01-01"},
                {"id": "c2", "role": "customer", "full_name": "Bob Smith", "phone_number": "919000000002", "email": "bob@x.com", "created_at": "2026-01-02"},
            ]
        }
    )
    request = _fake_request(_make_ctx(supabase))

    injection_attempt = 'x),phone_number.ilike.*'
    result = await admin_routes.list_customers(request, search=injection_attempt)

    assert result["count"] == 0


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_list_customers_reports_onboarding_incomplete_stage():
    supabase = FakeSupabaseClient(
        initial={
            "profiles": [
                {
                    "id": "c1", "role": "customer", "full_name": "Jane", "phone_number": "919000000001",
                    "created_at": "2026-01-01", "registration_step": "awaiting_pet_weight",
                },
            ],
        }
    )
    request = _fake_request(_make_ctx(supabase))

    result = await admin_routes.list_customers(request)

    customer = result["customers"][0]
    assert customer["stage_code"] == "onboarding"
    assert "pet's weight" in customer["stage_detail"]


@pytest.mark.asyncio
async def test_list_customers_reports_active_booking_stage():
    supabase = FakeSupabaseClient(
        initial={
            "profiles": [
                {
                    "id": "c1", "role": "customer", "full_name": "Jane", "phone_number": "919000000001",
                    "created_at": "2026-01-01", "registration_step": "completed",
                },
            ],
            "doctor_sessions": [
                {
                    "id": "s1", "profile_id": "c1", "status": "accepted", "payment_status": "awaiting",
                    "doctor_phone": "919000000099", "preferred_time": "2026-08-01T10:00:00", "created_at": "2026-07-30T00:00:00",
                },
            ],
        }
    )
    request = _fake_request(_make_ctx(supabase))

    result = await admin_routes.list_customers(request)

    customer = result["customers"][0]
    assert customer["stage_code"] == "payment"
    assert "Booked for" in customer["stage_detail"]


@pytest.mark.asyncio
async def test_list_customers_reports_inactive_stage_from_last_active_at():
    supabase = FakeSupabaseClient(
        initial={
            "profiles": [
                {
                    "id": "c1", "role": "customer", "full_name": "Jane", "phone_number": "919000000001",
                    "created_at": "2026-01-01", "registration_step": "completed",
                    "last_active_at": "2026-07-01T00:00:00+00:00",
                },
            ],
        }
    )
    request = _fake_request(_make_ctx(supabase))

    result = await admin_routes.list_customers(request)

    assert result["customers"][0]["stage_code"] == "inactive"


def test_compute_customer_stage_new_customer_with_no_activity():
    stage = admin_routes._compute_customer_stage({"registration_step": "completed"}, [])
    assert stage["code"] == "new"


@pytest.mark.asyncio
async def test_get_customer_includes_stage():
    supabase = FakeSupabaseClient(
        initial={
            "profiles": [
                {"id": "c1", "role": "customer", "full_name": "Jane", "phone_number": "919000000001", "registration_step": "awaiting_city"},
            ],
        }
    )
    request = _fake_request(_make_ctx(supabase))

    result = await admin_routes.get_customer("c1", request)

    assert result["stage_code"] == "onboarding"
    assert "city" in result["stage_detail"]


@pytest.mark.asyncio
async def test_get_customer_includes_activity_in_chronological_order():
    supabase = FakeSupabaseClient(
        initial={
            "profiles": [{"id": "c1", "role": "customer", "full_name": "Jane", "phone_number": "919000000001"}],
            "pets": [{"id": "pet1", "profile_id": "c1", "name": "Rex"}],
            "messages": [
                {"id": "m1", "profile_id": "c1", "sender_type": "user", "content": "hi", "created_at": "2026-06-01T10:00:00"},
                {"id": "m2", "profile_id": "c1", "sender_type": "assistant", "content": "hello!", "created_at": "2026-06-01T10:00:05"},
            ],
            "health_logs": [
                {"id": "h1", "profile_id": "c1", "pet_id": "pet1", "ai_risk_score": 40, "symptoms": "limping", "created_at": "2026-06-01T09:00:00"},
            ],
            "memory": [
                {"id": "mem1", "profile_id": "c1", "pet_id": "pet1", "memory_type": "Fact", "pet_name": "Rex", "species": "Dog"},
            ],
        }
    )
    request = _fake_request(_make_ctx(supabase))

    result = await admin_routes.get_customer("c1", request)

    assert [m["content"] for m in result["recent_messages"]] == ["hi", "hello!"]
    assert result["health_logs"][0]["symptoms"] == "limping"
    assert result["memory_facts"][0]["pet_name"] == "Rex"


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
async def test_list_customers_filters_by_breed():
    supabase = FakeSupabaseClient(
        initial={
            "profiles": [
                {"id": "c1", "role": "customer", "full_name": "HasLab", "phone_number": "919000000001", "created_at": "2026-01-01"},
                {"id": "c2", "role": "customer", "full_name": "HasPersian", "phone_number": "919000000002", "created_at": "2026-01-02"},
            ],
            "pets": [
                {"id": "p1", "profile_id": "c1", "name": "Rex", "breed": "Labrador", "created_at": "2026-01-01"},
                {"id": "p2", "profile_id": "c2", "name": "Milo", "breed": "Persian", "created_at": "2026-01-02"},
            ],
        }
    )
    request = _fake_request(_make_ctx(supabase))

    result = await admin_routes.list_customers(request, breed="labrador")

    assert result["count"] == 1
    assert result["customers"][0]["full_name"] == "HasLab"


@pytest.mark.asyncio
async def test_list_customers_filters_by_status():
    supabase = FakeSupabaseClient(
        initial={
            "profiles": [
                {"id": "c1", "role": "customer", "full_name": "ActiveCust", "phone_number": "919000000001", "created_at": "2026-01-01", "is_active": True},
                {"id": "c2", "role": "customer", "full_name": "InactiveCust", "phone_number": "919000000002", "created_at": "2026-01-02", "is_active": False},
            ],
        }
    )
    request = _fake_request(_make_ctx(supabase))

    active_only = await admin_routes.list_customers(request, status="active")
    assert active_only["count"] == 1
    assert active_only["customers"][0]["full_name"] == "ActiveCust"

    inactive_only = await admin_routes.list_customers(request, status="inactive")
    assert inactive_only["count"] == 1
    assert inactive_only["customers"][0]["full_name"] == "InactiveCust"


@pytest.mark.asyncio
async def test_list_customers_filters_by_stage():
    supabase = FakeSupabaseClient(
        initial={
            "profiles": [
                {"id": "c1", "role": "customer", "full_name": "Stuck", "phone_number": "919000000001", "created_at": "2026-01-01", "registration_step": "awaiting_city"},
                {"id": "c2", "role": "customer", "full_name": "Done", "phone_number": "919000000002", "created_at": "2026-01-02", "registration_step": "completed"},
            ],
        }
    )
    request = _fake_request(_make_ctx(supabase))

    result = await admin_routes.list_customers(request, stage="onboarding")

    assert result["count"] == 1
    assert result["customers"][0]["full_name"] == "Stuck"


@pytest.mark.asyncio
async def test_list_customers_filters_by_intent():
    supabase = FakeSupabaseClient(
        initial={
            "profiles": [
                {"id": "c1", "role": "customer", "full_name": "Hot", "phone_number": "919000000001", "created_at": "2026-01-01", "intent_rating": "High"},
                {"id": "c2", "role": "customer", "full_name": "Cold", "phone_number": "919000000002", "created_at": "2026-01-02", "intent_rating": "Low"},
                {"id": "c3", "role": "customer", "full_name": "Unrated", "phone_number": "919000000003", "created_at": "2026-01-03"},
            ],
        }
    )
    request = _fake_request(_make_ctx(supabase))

    high = await admin_routes.list_customers(request, intent="High")
    assert high["count"] == 1
    assert high["customers"][0]["full_name"] == "Hot"

    unrated = await admin_routes.list_customers(request, intent="unrated")
    assert unrated["count"] == 1
    assert unrated["customers"][0]["full_name"] == "Unrated"


@pytest.mark.asyncio
async def test_rate_customer_intent_endpoint_stores_result():
    supabase = FakeSupabaseClient(
        initial={
            "profiles": [{"id": "c1", "role": "customer", "full_name": "Jane", "phone_number": "919000000001"}],
            "messages": [
                {"id": "m1", "profile_id": "c1", "sender_type": "user", "content": "I want to book a vet now", "created_at": "2026-06-01T10:00:00"},
            ],
        }
    )
    request = _fake_request(_make_ctx(supabase))

    with patch.object(
        admin_routes, "rate_customer_intent",
        AsyncMock(return_value={"rating": "High", "reason": "Asked to book a vet."}),
    ):
        result = await admin_routes.rate_customer_intent_endpoint("c1", request)

    assert result["success"] is True
    assert result["profile"]["intent_rating"] == "High"
    assert result["profile"]["intent_rating_reason"] == "Asked to book a vet."
    assert result["profile"]["intent_rating_message_count"] == 1
    assert supabase.rows("profiles")[0]["intent_rating"] == "High"


@pytest.mark.asyncio
async def test_rate_customer_intent_endpoint_404s_for_unknown_customer():
    supabase = FakeSupabaseClient()
    request = _fake_request(_make_ctx(supabase))

    with pytest.raises(HTTPException) as exc:
        await admin_routes.rate_customer_intent_endpoint("nope", request)
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_list_customers_stage_filter_applies_before_pagination():
    """Regression: stage used to be computed only on the already-paginated
    page, so filtering by stage against a large customer list could miss
    matches that got cut by the page slice before stage was even known."""
    supabase = FakeSupabaseClient(
        initial={
            "profiles": [
                {"id": f"c{i}", "role": "customer", "full_name": f"Done{i}", "phone_number": f"91900000{i:04d}", "created_at": "2026-01-01", "registration_step": "completed"}
                for i in range(5)
            ]
            + [{"id": "stuck", "role": "customer", "full_name": "Stuck", "phone_number": "919999999999", "created_at": "2025-01-01", "registration_step": "awaiting_city"}],
        }
    )
    request = _fake_request(_make_ctx(supabase))

    result = await admin_routes.list_customers(request, stage="onboarding", limit=3)

    assert result["count"] == 1
    assert result["customers"][0]["full_name"] == "Stuck"


@pytest.mark.asyncio
async def test_list_customer_breeds_returns_distinct_sorted_breeds():
    supabase = FakeSupabaseClient(
        initial={
            "pets": [
                {"id": "p1", "breed": "Labrador"},
                {"id": "p2", "breed": "Persian"},
                {"id": "p3", "breed": "Labrador"},
                {"id": "p4", "breed": None},
            ],
        }
    )
    request = _fake_request(_make_ctx(supabase))

    result = await admin_routes.list_customer_breeds(request)

    assert result["breeds"] == ["Labrador", "Persian"]


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
async def test_onboard_doctor_sends_a_welcome_message():
    supabase = FakeSupabaseClient()
    ctx = _make_ctx(supabase)
    request = _fake_request(ctx)

    await admin_routes.onboard_doctor(request, {"full_name": "Dr. Rao", "phone_number": "919000000010"})

    ctx.whatsapp.send_text.assert_awaited_once()
    assert ctx.whatsapp.send_text.call_args.args[0] == "919000000010"
    assert "Dr. Rao" in ctx.whatsapp.send_text.call_args.args[1]
    assert "Veterinary Doctor" in ctx.whatsapp.send_text.call_args.args[1]


@pytest.mark.asyncio
async def test_onboard_doctor_still_succeeds_if_welcome_message_send_fails():
    supabase = FakeSupabaseClient()
    ctx = _make_ctx(supabase)
    ctx.whatsapp.send_text = AsyncMock(side_effect=RuntimeError("WhatsApp outage"))
    request = _fake_request(ctx)

    result = await admin_routes.onboard_doctor(request, {"full_name": "Dr. Rao", "phone_number": "919000000010"})

    assert result["success"] is True
    assert supabase.rows("profiles")[0]["phone_number"] == "919000000010"


# ---------------------------------------------------------------------------
# Doctor onboarding drafts
# ---------------------------------------------------------------------------


def _make_draft(**overrides):
    draft = {
        "id": "draft-1",
        "drive_folder_id": "folder-1",
        "drive_folder_name": "Dr Mounika",
        "status": "pending_review",
        "extracted_full_name": "Dr. Mounika",
        "extracted_phone_number": "919182381400",
        "extracted_qualification": "B.V.Sc & A.H.",
        "extracted_registration_number": "TSVC 02353/2024",
    }
    draft.update(overrides)
    return draft


@pytest.mark.asyncio
async def test_list_doctor_drafts_defaults_to_pending_review():
    supabase = FakeSupabaseClient(
        initial={
            "doctor_onboarding_drafts": [
                _make_draft(id="d1", status="pending_review"),
                _make_draft(id="d2", status="approved"),
            ]
        }
    )
    request = _fake_request(_make_ctx(supabase))

    result = await admin_routes.list_doctor_drafts(request)

    assert result["count"] == 1
    assert result["drafts"][0]["id"] == "d1"


@pytest.mark.asyncio
async def test_get_doctor_draft_404s_when_missing():
    supabase = FakeSupabaseClient()
    request = _fake_request(_make_ctx(supabase))

    with pytest.raises(HTTPException) as exc:
        await admin_routes.get_doctor_draft("nope", request)
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_update_doctor_draft_fills_in_missing_fields():
    supabase = FakeSupabaseClient(initial={"doctor_onboarding_drafts": [_make_draft()]})
    request = _fake_request(_make_ctx(supabase))

    result = await admin_routes.update_doctor_draft("draft-1", request, {"extracted_email": "mounika@example.com"})

    assert result["draft"]["extracted_email"] == "mounika@example.com"


@pytest.mark.asyncio
async def test_update_doctor_draft_rejects_once_reviewed():
    supabase = FakeSupabaseClient(initial={"doctor_onboarding_drafts": [_make_draft(status="approved")]})
    request = _fake_request(_make_ctx(supabase))

    with pytest.raises(HTTPException) as exc:
        await admin_routes.update_doctor_draft("draft-1", request, {"extracted_email": "x@example.com"})
    assert exc.value.status_code == 409


@pytest.mark.asyncio
async def test_approve_doctor_draft_creates_active_vet_and_sends_welcome():
    supabase = FakeSupabaseClient(initial={"doctor_onboarding_drafts": [_make_draft()]})
    ctx = _make_ctx(supabase)
    request = _fake_request(ctx)

    result = await admin_routes.approve_doctor_draft("draft-1", request)

    assert result["success"] is True
    doctor = result["doctor"]
    assert doctor["role"] == "vet"
    assert doctor["is_active"] is True
    assert doctor["phone_number"] == "919182381400"
    ctx.whatsapp.send_text.assert_awaited_once_with(
        "919182381400", admin_routes.DOCTOR_WELCOME_MESSAGE.format(name="Dr. Mounika")
    )
    draft = supabase.rows("doctor_onboarding_drafts")[0]
    assert draft["status"] == "approved"
    assert draft["created_profile_id"] == doctor["id"]


@pytest.mark.parametrize(
    "raw,expected",
    [("female", "Female"), ("Male", "Male"), ("OTHER", "Other"), ("", None), (None, None), ("nonbinary", None)],
)
def test_normalize_profile_gender(raw, expected):
    # profiles.gender has a DB check constraint accepting only the exact
    # strings "Male"/"Female"/"Other" -- confirmed live: lowercase values
    # extracted from doctor documents (e.g. "female") 400 on insert, which
    # crashed the whole approve-draft flow after the customer profile it
    # replaces had already been deleted (see approve_doctor_draft).
    assert admin_routes._normalize_profile_gender(raw) == expected


@pytest.mark.asyncio
async def test_approve_doctor_draft_normalizes_lowercase_gender():
    supabase = FakeSupabaseClient(initial={"doctor_onboarding_drafts": [_make_draft(extracted_gender="female")]})
    request = _fake_request(_make_ctx(supabase))

    result = await admin_routes.approve_doctor_draft("draft-1", request)

    assert result["doctor"]["gender"] == "Female"


@pytest.mark.asyncio
async def test_approve_doctor_draft_restores_customer_profile_if_doctor_creation_fails():
    """The replaced-customer cleanup deletes the old row before the new vet
    row can be inserted (phone_number is unique) -- if profile creation then
    fails for any reason, the deleted customer profile must come back rather
    than leaving this person with no profile at all, which happened once in
    production (a gender check-constraint violation slipped through)."""
    customer = {"id": "existing-customer", "phone_number": "919182381400", "role": "customer", "full_name": "Existing Person"}
    supabase = FakeSupabaseClient(
        initial={"doctor_onboarding_drafts": [_make_draft()], "profiles": [customer]}
    )
    request = _fake_request(_make_ctx(supabase))

    with patch.object(admin_routes, "_create_doctor_profile", AsyncMock(side_effect=RuntimeError("boom"))):
        with pytest.raises(RuntimeError):
            await admin_routes.approve_doctor_draft("draft-1", request)

    profiles = supabase.rows("profiles")
    assert any(p["id"] == "existing-customer" and p["role"] == "customer" for p in profiles)
    draft = supabase.rows("doctor_onboarding_drafts")[0]
    assert draft["status"] == "pending_review"


@pytest.mark.asyncio
async def test_approve_doctor_draft_requires_name_and_phone_filled_in():
    supabase = FakeSupabaseClient(
        initial={"doctor_onboarding_drafts": [_make_draft(extracted_phone_number=None)]}
    )
    request = _fake_request(_make_ctx(supabase))

    with pytest.raises(HTTPException) as exc:
        await admin_routes.approve_doctor_draft("draft-1", request)
    assert exc.value.status_code == 422


@pytest.mark.asyncio
async def test_approve_doctor_draft_rejects_duplicate_phone_number_for_existing_vet():
    supabase = FakeSupabaseClient(
        initial={
            "doctor_onboarding_drafts": [_make_draft()],
            "profiles": [{"id": "existing", "phone_number": "919182381400", "role": "vet", "full_name": "Existing Person"}],
        }
    )
    request = _fake_request(_make_ctx(supabase))

    with pytest.raises(HTTPException) as exc:
        await admin_routes.approve_doctor_draft("draft-1", request)
    assert exc.value.status_code == 409
    # The message must name the actual conflicting role/person so an admin
    # can tell "already a vet" apart from "already a customer" -- a bare
    # "already exists" gave no way to decide how to resolve the conflict.
    assert "vet" in exc.value.detail
    assert "Existing Person" in exc.value.detail


@pytest.mark.asyncio
async def test_approve_doctor_draft_replaces_matching_customer_profile():
    """Same phone number under a customer profile just means this person
    explored the product as a pet owner before being onboarded as a vet --
    the customer identity should be dropped in favor of the doctor one,
    not block the approval."""
    supabase = FakeSupabaseClient(
        initial={
            "doctor_onboarding_drafts": [_make_draft()],
            "profiles": [{"id": "existing-customer", "phone_number": "919182381400", "role": "customer", "full_name": "Existing Person"}],
        }
    )
    ctx = _make_ctx(supabase)
    request = _fake_request(ctx)

    result = await admin_routes.approve_doctor_draft("draft-1", request)

    assert result["success"] is True
    assert result["doctor"]["role"] == "vet"
    profiles = supabase.rows("profiles")
    assert not any(p["id"] == "existing-customer" for p in profiles)
    assert any(p["phone_number"] == "919182381400" and p["role"] == "vet" for p in profiles)


@pytest.mark.asyncio
async def test_approve_doctor_draft_rejects_already_reviewed():
    supabase = FakeSupabaseClient(initial={"doctor_onboarding_drafts": [_make_draft(status="rejected")]})
    request = _fake_request(_make_ctx(supabase))

    with pytest.raises(HTTPException) as exc:
        await admin_routes.approve_doctor_draft("draft-1", request)
    assert exc.value.status_code == 409


@pytest.mark.asyncio
async def test_reject_doctor_draft_marks_rejected_without_creating_a_profile():
    supabase = FakeSupabaseClient(initial={"doctor_onboarding_drafts": [_make_draft()]})
    ctx = _make_ctx(supabase)
    request = _fake_request(ctx)

    result = await admin_routes.reject_doctor_draft("draft-1", request)

    assert result["success"] is True
    assert result["draft"]["status"] == "rejected"
    assert supabase.rows("profiles") == []
    ctx.whatsapp.send_text.assert_not_awaited()


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
            "doctor_sessions": [
                {"id": "sess1", "status": "completed", "created_at": today},
                {"id": "sess2", "status": "accepted", "created_at": today},
            ],
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
    assert result["new_signups"] == 1
    assert result["completed_consultations"] == 1
    assert result["pending_consultations"] == 1
    assert result["symptom_checks"] == 1
    assert result["documents_uploaded"] == 1


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


@pytest.mark.asyncio
async def test_list_doctors_filters_by_area_city_hospital_treatments_status():
    supabase = FakeSupabaseClient(
        initial={
            "profiles": [
                {"id": "v1", "role": "vet", "full_name": "Dr. Match", "phone_number": "919000000001", "created_at": "2026-01-01",
                 "area": "Anna Nagar", "city": "Chennai", "hospital_name": "City Vet Hospital", "treatments": "Surgery, Dental", "is_active": True},
                {"id": "v2", "role": "vet", "full_name": "Dr. NoMatch", "phone_number": "919000000002", "created_at": "2026-01-02",
                 "area": "Velachery", "city": "Chennai", "hospital_name": "Other Clinic", "treatments": "Grooming", "is_active": False},
            ],
        }
    )
    request = _fake_request(_make_ctx(supabase))

    assert (await admin_routes.list_doctors(request, area="anna"))["count"] == 1
    assert (await admin_routes.list_doctors(request, city="chennai"))["count"] == 2
    assert (await admin_routes.list_doctors(request, hospital="city vet"))["count"] == 1
    assert (await admin_routes.list_doctors(request, treatments="dental"))["count"] == 1
    assert (await admin_routes.list_doctors(request, status="active"))["count"] == 1
    assert (await admin_routes.list_doctors(request, status="inactive"))["count"] == 1


@pytest.mark.asyncio
async def test_doctor_distinct_value_endpoints():
    supabase = FakeSupabaseClient(
        initial={
            "profiles": [
                {"id": "v1", "role": "vet", "area": "Anna Nagar", "city": "Chennai", "hospital_name": "City Vet", "treatments": "Surgery, Dental"},
                {"id": "v2", "role": "vet", "area": "Velachery", "city": "Chennai", "hospital_name": "Other Clinic", "treatments": "Dental, Grooming"},
            ],
        }
    )
    request = _fake_request(_make_ctx(supabase))

    assert (await admin_routes.list_doctor_areas(request))["areas"] == ["Anna Nagar", "Velachery"]
    assert (await admin_routes.list_doctor_cities(request))["cities"] == ["Chennai"]
    assert (await admin_routes.list_doctor_hospitals(request))["hospitals"] == ["City Vet", "Other Clinic"]
    assert (await admin_routes.list_doctor_treatments(request))["treatments"] == ["Dental", "Grooming", "Surgery"]


@pytest.mark.asyncio
async def test_onboard_doctor_saves_new_fields():
    supabase = FakeSupabaseClient()
    request = _fake_request(_make_ctx(supabase))
    payload = {
        "full_name": "Dr. Rao", "phone_number": "919000000010", "area": "Anna Nagar", "city": "Chennai",
        "hospital_name": "City Vet Hospital", "treatments": "Surgery, Dental",
        "opening_time": "09:00", "closing_time": "18:00",
    }

    result = await admin_routes.onboard_doctor(request, payload)

    doctor = result["doctor"]
    assert doctor["area"] == "Anna Nagar"
    assert doctor["hospital_name"] == "City Vet Hospital"
    assert doctor["opening_time"] == "09:00"


@pytest.mark.asyncio
async def test_update_doctor_partial_update():
    supabase = FakeSupabaseClient(initial={"profiles": [{"id": "v1", "role": "vet", "full_name": "Dr. Rao", "area": None}]})
    request = _fake_request(_make_ctx(supabase))

    result = await admin_routes.update_doctor("v1", request, {"area": "Anna Nagar", "not_a_real_field": "ignored"})

    assert result["doctor"]["area"] == "Anna Nagar"
    assert "not_a_real_field" not in result["doctor"]


@pytest.mark.asyncio
async def test_update_doctor_404_when_missing():
    supabase = FakeSupabaseClient()
    request = _fake_request(_make_ctx(supabase))

    with pytest.raises(HTTPException) as exc:
        await admin_routes.update_doctor("nope", request, {"area": "X"})
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_update_doctor_422_when_no_editable_fields():
    supabase = FakeSupabaseClient(initial={"profiles": [{"id": "v1", "role": "vet"}]})
    request = _fake_request(_make_ctx(supabase))

    with pytest.raises(HTTPException) as exc:
        await admin_routes.update_doctor("v1", request, {"not_editable": "X"})
    assert exc.value.status_code == 422


@pytest.mark.asyncio
async def test_activate_doctor_flips_is_active_true():
    supabase = FakeSupabaseClient(initial={"profiles": [{"id": "v1", "role": "vet", "is_active": False}]})
    request = _fake_request(_make_ctx(supabase))

    result = await admin_routes.activate_doctor("v1", request)

    assert result["success"] is True
    assert supabase.rows("profiles")[0]["is_active"] is True


@pytest.mark.asyncio
async def test_upload_doctor_document_stores_row_and_calls_storage(monkeypatch):
    supabase = FakeSupabaseClient(initial={"profiles": [{"id": "v1", "role": "vet"}]})
    request = _fake_request(_make_ctx(supabase))
    upload_mock = AsyncMock() if False else None  # upload_to_storage is sync
    monkeypatch.setattr("app.admin.routes.upload_to_storage", lambda *a, **k: None)

    class FakeUploadFile:
        filename = "license.pdf"
        content_type = "application/pdf"

        async def read(self):
            return b"fake-bytes"

    result = await admin_routes.upload_doctor_document("v1", request, file=FakeUploadFile())

    assert result["success"] is True
    doc = supabase.rows("doctor_documents")[0]
    assert doc["profile_id"] == "v1"
    assert doc["document_name"] == "license.pdf"


@pytest.mark.asyncio
async def test_upload_doctor_document_404_for_unknown_doctor():
    supabase = FakeSupabaseClient()
    request = _fake_request(_make_ctx(supabase))

    class FakeUploadFile:
        filename = "x.pdf"
        content_type = "application/pdf"

        async def read(self):
            return b""

    with pytest.raises(HTTPException) as exc:
        await admin_routes.upload_doctor_document("nope", request, file=FakeUploadFile())
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_list_doctor_documents_includes_signed_url(monkeypatch):
    supabase = FakeSupabaseClient(
        initial={"doctor_documents": [{"id": "d1", "profile_id": "v1", "document_name": "license.pdf", "storage_path": "v1/abc.pdf", "uploaded_at": "2026-01-01"}]}
    )
    request = _fake_request(_make_ctx(supabase))
    monkeypatch.setattr("app.admin.routes.sign_storage_url", lambda *a, **k: "https://signed.example/abc.pdf")

    result = await admin_routes.list_doctor_documents("v1", request)

    assert result["documents"][0]["url"] == "https://signed.example/abc.pdf"


@pytest.mark.asyncio
async def test_delete_doctor_document_removes_row():
    supabase = FakeSupabaseClient(
        initial={"doctor_documents": [{"id": "d1", "profile_id": "v1", "storage_path": "v1/abc.pdf"}]}
    )
    ctx = _make_ctx(supabase)
    ctx.supabase.storage = SimpleNamespace(from_=lambda bucket: SimpleNamespace(remove=lambda paths: None))
    request = _fake_request(ctx)

    result = await admin_routes.delete_doctor_document("v1", "d1", request)

    assert result["success"] is True
    assert supabase.rows("doctor_documents") == []


@pytest.mark.asyncio
async def test_delete_doctor_document_404_when_missing():
    supabase = FakeSupabaseClient()
    request = _fake_request(_make_ctx(supabase))

    with pytest.raises(HTTPException) as exc:
        await admin_routes.delete_doctor_document("v1", "nope", request)
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_analytics_overview_counts_completed_not_accepted_in_range():
    supabase = FakeSupabaseClient(
        initial={
            "profiles": [
                {"id": "c1", "role": "customer", "created_at": "2026-06-10", "is_founding_member": False},
            ],
            "doctor_sessions": [
                {"id": "s1", "status": "completed", "created_at": "2026-06-05"},
                {"id": "s2", "status": "accepted", "created_at": "2026-06-06"},
                {"id": "s3", "status": "cancelled", "created_at": "2026-06-07"},
                {"id": "s4", "status": "completed", "created_at": "2026-05-01"},
            ],
        }
    )
    request = _fake_request(_make_ctx(supabase))

    result = await admin_routes.analytics_overview(request, date_from="2026-06-01", date_to="2026-06-30")

    assert result["completed_consultations"] == 1
    assert result["pending_consultations"] == 1
    assert result["cancelled_consultations"] == 1
    assert result["new_signups"] == 1
    assert result["date_from"] == "2026-06-01"
    assert result["date_to"] == "2026-06-30"


@pytest.mark.asyncio
async def test_analytics_timeseries_includes_consultations_series():
    supabase = FakeSupabaseClient(
        initial={
            "doctor_sessions": [
                {"id": "s1", "status": "completed", "created_at": "2026-06-05T10:00:00"},
                {"id": "s2", "status": "completed", "created_at": "2026-06-05T14:00:00"},
                {"id": "s3", "status": "accepted", "created_at": "2026-06-06T10:00:00"},
            ],
        }
    )
    request = _fake_request(_make_ctx(supabase))

    result = await admin_routes.analytics_timeseries(request, date_from="2026-06-01", date_to="2026-06-30")

    assert result["consultations"] == [{"date": "2026-06-05", "count": 2}]


@pytest.mark.asyncio
async def test_analytics_reports_customer_funnel_counts_by_stage():
    supabase = FakeSupabaseClient(
        initial={
            "profiles": [
                {"id": "c1", "role": "customer", "registration_step": "awaiting_pet_name"},
                {"id": "c2", "role": "customer", "registration_step": "completed"},
                {"id": "c3", "role": "customer", "registration_step": "completed"},
            ],
            "doctor_sessions": [
                {"id": "s1", "profile_id": "c2", "status": "accepted", "payment_status": "paid", "preferred_time": "2026-08-01T10:00:00", "created_at": "2026-07-01"},
            ],
        }
    )
    request = _fake_request(_make_ctx(supabase))

    result = await admin_routes.analytics_reports(request)

    by_code = {row["code"]: row["count"] for row in result["customer_funnel"]}
    assert by_code["onboarding"] == 1
    assert by_code["upcoming"] == 1
    assert by_code["new"] == 1


@pytest.mark.asyncio
async def test_analytics_reports_booking_funnel_computes_rates():
    supabase = FakeSupabaseClient(
        initial={
            "doctor_sessions": [
                {"id": "s1", "status": "completed", "created_at": "2026-06-05"},
                {"id": "s2", "status": "completed", "created_at": "2026-06-06"},
                {"id": "s3", "status": "declined", "created_at": "2026-06-07"},
                {"id": "s4", "status": "pending", "created_at": "2026-06-08"},
            ],
        }
    )
    request = _fake_request(_make_ctx(supabase))

    result = await admin_routes.analytics_reports(request, date_from="2026-06-01", date_to="2026-06-30")

    funnel = result["booking_funnel"]
    assert funnel["total"] == 4
    assert funnel["completed_rate"] == 0.5
    assert funnel["declined_or_cancelled_rate"] == 0.25


@pytest.mark.asyncio
async def test_analytics_reports_revenue_breaks_down_by_plan_and_churn():
    supabase = FakeSupabaseClient(
        initial={
            "subscriptions": [
                {"id": "s1", "plan_name": "Premium", "amount": 399, "status": "active", "cancelled_at": None},
                {"id": "s2", "plan_name": "Premium", "amount": 399, "status": "active", "cancelled_at": None},
                {"id": "s3", "plan_name": "Basic", "amount": 99, "status": "cancelled", "cancelled_at": "2026-06-15T00:00:00"},
            ],
        }
    )
    request = _fake_request(_make_ctx(supabase))

    result = await admin_routes.analytics_reports(request, date_from="2026-06-01", date_to="2026-06-30")

    revenue = result["revenue"]
    assert revenue["by_plan"] == [{"plan_name": "Premium", "active_count": 2, "mrr": 798}]
    assert revenue["churned_in_range"] == 1


@pytest.mark.asyncio
async def test_analytics_reports_doctor_performance_excludes_broadcast_and_placeholder_phones():
    supabase = FakeSupabaseClient(
        initial={
            "profiles": [
                {"id": "v1", "role": "vet", "full_name": "Dr. Rao", "phone_number": "919000000001"},
            ],
            "doctor_sessions": [
                {"id": "s1", "doctor_phone": "919000000001", "status": "completed"},
                {"id": "s2", "doctor_phone": "919000000001", "status": "accepted"},
                {"id": "s3", "doctor_phone": "broadcast", "status": "pending"},
            ],
        }
    )
    request = _fake_request(_make_ctx(supabase))

    result = await admin_routes.analytics_reports(request)

    assert len(result["doctor_performance"]) == 1
    doc = result["doctor_performance"][0]
    assert doc["total_sessions"] == 2
    assert doc["completed_sessions"] == 1
    assert doc["completion_rate"] == 0.5
    assert doc["active_now"] == 1


def _appointment_fixture():
    return FakeSupabaseClient(
        initial={
            "doctor_sessions": [
                {
                    "id": "sess-1", "profile_id": "c1", "pet_id": "p1", "doctor_phone": "919000000001",
                    "status": "completed", "created_at": "2026-06-10T10:00:00", "completed_at": "2026-06-10T10:30:00",
                    "case_summary": "Skin rash", "pending_medications": "Cream", "awaiting_from": None,
                    "follow_up_required": True, "follow_up_date": "2026-06-24",
                },
                {
                    "id": "sess-2", "profile_id": "c2", "pet_id": "p2", "doctor_phone": "919000000002",
                    "status": "completed", "created_at": "2026-06-11T10:00:00", "completed_at": None,
                    "case_summary": "Vomiting", "pending_medications": None, "awaiting_from": "doctor_prescription",
                    "follow_up_required": False, "follow_up_date": None,
                },
            ],
            "profiles": [
                {"id": "c1", "role": "customer", "full_name": "Jane", "city": "Chennai"},
                {"id": "c2", "role": "customer", "full_name": "Bob", "city": "Bengaluru"},
                {"id": "v1", "role": "vet", "phone_number": "919000000001", "full_name": "Dr. Rao"},
                {"id": "v2", "role": "vet", "phone_number": "919000000002", "full_name": "Dr. Iyer"},
            ],
            "pets": [
                {"id": "p1", "name": "Rex", "breed": "Labrador", "age": 2},
                {"id": "p2", "name": "Milo", "breed": "Persian", "age": 8},
            ],
        }
    )


@pytest.mark.asyncio
async def test_list_appointments_returns_enriched_fields():
    supabase = _appointment_fixture()
    request = _fake_request(_make_ctx(supabase))

    result = await admin_routes.list_appointments(request)

    assert result["count"] == 2
    by_id = {a["session_id"]: a for a in result["appointments"]}
    a1 = by_id["sess-1"]
    assert a1["customer_id"] == "c1"
    assert a1["customer_name"] == "Jane"
    assert a1["pet_name"] == "Rex"
    assert a1["breed"] == "Labrador"
    assert a1["age_group"] == "Young (1-2yrs)"
    assert a1["city"] == "Chennai"
    assert a1["doctor_name"] == "Dr. Rao"
    assert a1["prescription_status"] == "Delivered"
    assert a1["follow_up_required"] is True
    assert a1["follow_up_date"] == "2026-06-24"

    a2 = by_id["sess-2"]
    assert a2["prescription_status"] == "Not filed"
    assert a2["age_group"] == "Senior (7+yrs)"


@pytest.mark.asyncio
async def test_list_appointments_filters_by_breed():
    supabase = _appointment_fixture()
    request = _fake_request(_make_ctx(supabase))

    result = await admin_routes.list_appointments(request, breed="labrador")

    assert result["count"] == 1
    assert result["appointments"][0]["session_id"] == "sess-1"


@pytest.mark.asyncio
async def test_list_appointments_filters_by_age_group():
    supabase = _appointment_fixture()
    request = _fake_request(_make_ctx(supabase))

    result = await admin_routes.list_appointments(request, age_group="Senior (7+yrs)")

    assert result["count"] == 1
    assert result["appointments"][0]["session_id"] == "sess-2"


@pytest.mark.asyncio
async def test_list_appointments_filters_by_issue():
    supabase = _appointment_fixture()
    request = _fake_request(_make_ctx(supabase))

    result = await admin_routes.list_appointments(request, issue="rash")

    assert result["count"] == 1
    assert result["appointments"][0]["session_id"] == "sess-1"


@pytest.mark.asyncio
async def test_list_appointments_filters_by_city_and_doctor():
    supabase = _appointment_fixture()
    request = _fake_request(_make_ctx(supabase))

    assert (await admin_routes.list_appointments(request, city="bengaluru"))["count"] == 1
    assert (await admin_routes.list_appointments(request, doctor="rao"))["count"] == 1


@pytest.mark.asyncio
async def test_list_appointments_filters_by_follow_up_required():
    supabase = _appointment_fixture()
    request = _fake_request(_make_ctx(supabase))

    result = await admin_routes.list_appointments(request, follow_up_required="true")

    assert result["count"] == 1
    assert result["appointments"][0]["session_id"] == "sess-1"


@pytest.mark.asyncio
async def test_list_appointment_cities():
    supabase = _appointment_fixture()
    request = _fake_request(_make_ctx(supabase))

    result = await admin_routes.list_appointment_cities(request)

    assert result["cities"] == ["Bengaluru", "Chennai"]


@pytest.mark.asyncio
async def test_prescription_status_not_completed_is_na():
    supabase = FakeSupabaseClient(
        initial={"doctor_sessions": [{"id": "s1", "profile_id": "c1", "status": "accepted", "created_at": "2026-06-01"}]}
    )
    request = _fake_request(_make_ctx(supabase))

    result = await admin_routes.list_appointments(request)

    assert result["appointments"][0]["prescription_status"] == "N/A"


@pytest.mark.asyncio
async def test_list_platform_costs_returns_sorted_rows():
    supabase = FakeSupabaseClient(
        initial={
            "platform_costs": [
                {"id": "p2", "platform_name": "Vercel", "amount_spent": 0, "credits_remaining": None, "currency": "USD"},
                {"id": "p1", "platform_name": "OpenAI", "amount_spent": 42.5, "credits_remaining": 57.5, "currency": "USD"},
            ],
        }
    )
    request = _fake_request(_make_ctx(supabase))

    result = await admin_routes.list_platform_costs(request)

    assert [p["platform_name"] for p in result["platforms"]] == ["OpenAI", "Vercel"]


@pytest.mark.asyncio
async def test_upsert_platform_cost_creates_new_platform():
    supabase = FakeSupabaseClient()
    request = _fake_request(_make_ctx(supabase))

    result = await admin_routes.upsert_platform_cost(
        "Railway", request, {"amount_spent": 12.34, "credits_remaining": 5, "currency": "USD", "notes": "Hobby plan"}
    )

    assert result["success"] is True
    assert result["platform"]["platform_name"] == "Railway"
    assert result["platform"]["amount_spent"] == 12.34
    assert result["platform"]["updated_at"] is not None


@pytest.mark.asyncio
async def test_upsert_platform_cost_updates_existing_platform_in_place():
    supabase = FakeSupabaseClient(
        initial={
            "platform_costs": [{"id": "p1", "platform_name": "Supabase", "amount_spent": 25, "credits_remaining": 0, "currency": "USD"}],
        }
    )
    request = _fake_request(_make_ctx(supabase))

    await admin_routes.upsert_platform_cost("Supabase", request, {"amount_spent": 30})

    rows = supabase.rows("platform_costs")
    assert len(rows) == 1
    assert rows[0]["amount_spent"] == 30


@pytest.mark.asyncio
async def test_delete_platform_cost():
    supabase = FakeSupabaseClient(
        initial={"platform_costs": [{"id": "p1", "platform_name": "Vercel", "amount_spent": 0, "currency": "USD"}]}
    )
    request = _fake_request(_make_ctx(supabase))

    result = await admin_routes.delete_platform_cost("p1", request)

    assert result["success"] is True
    assert supabase.rows("platform_costs") == []


@pytest.mark.asyncio
async def test_analytics_pnl_computes_revenue_and_costs():
    supabase = FakeSupabaseClient(
        initial={
            "subscriptions": [
                {"id": "s1", "amount": 399, "status": "active", "created_at": "2026-06-05"},
                {"id": "s2", "amount": 99, "status": "active", "created_at": "2026-06-10"},
                {"id": "s3", "amount": 399, "status": "cancelled", "created_at": "2020-01-01"},  # outside range, cancelled
            ],
            "platform_costs": [
                {"id": "p1", "platform_name": "OpenAI", "amount_spent": 10, "currency": "USD"},
                {"id": "p2", "platform_name": "Local Vendor", "amount_spent": 500, "currency": "INR"},
            ],
        }
    )
    request = _fake_request(_make_ctx(supabase))

    result = await admin_routes.analytics_pnl(request, date_from="2026-06-01", date_to="2026-06-30")

    assert result["revenue_in_range_inr"] == 498
    assert result["estimated_mrr_inr"] == 498
    assert {"currency": "INR", "amount": 500} in result["costs_by_currency"]
    assert {"currency": "USD", "amount": 10} in result["costs_by_currency"]
    # 10 USD * 83 approx rate + 500 INR = 1330
    assert result["total_costs_inr_approx"] == 1330.0
    assert result["net_in_range_inr_approx"] == 498 - 1330.0
    assert result["fx_rate_used"] == 83.0
