"""Covers the deterministic, safety-critical pieces of check_symptoms that
are deliberately NOT left to LLM judgement: the severity display string
(so the rating always reads the same regardless of phrasing) and the
keyword-based severity escalation (red-flag terms, and the ingestion x
harmful-substance intersection for poisoning claims)."""

from app.agent.tools.symptoms import _apply_safety_override, _severity_display


def test_severity_display_mild_is_green():
    assert _severity_display(1, "Mild") == "🟢 Mild (1/5)"


def test_severity_display_moderate_is_yellow():
    assert _severity_display(3, "Moderate") == "🟡 Moderate (3/5)"


def test_severity_display_serious_is_orange():
    assert _severity_display(4, "Serious") == "🟠 Serious (4/5)"


def test_severity_display_emergency_is_red():
    assert _severity_display(5, "Emergency") == "🔴 Emergency (5/5)"


def test_red_flag_keyword_forces_severity_5():
    verdict = {"severity": 2, "severity_label": "Mild", "requires_emergency_care": False}
    result = _apply_safety_override("he can't breathe and his gums look blue", verdict)
    assert result["severity"] == 5
    assert result["severity_label"] == "Emergency"
    assert result["requires_emergency_care"] is True


def test_poisoning_intersection_escalates():
    verdict = {"severity": 1, "severity_label": "Mild", "requires_emergency_care": False}
    result = _apply_safety_override("my dog ate a whole bar of chocolate an hour ago", verdict)
    assert result["severity"] == 5
    assert result["requires_emergency_care"] is True
    assert any("chocolate" in flag for flag in result["red_flags"])


def test_ingestion_without_harmful_substance_does_not_escalate():
    verdict = {"severity": 2, "severity_label": "Mild", "requires_emergency_care": False}
    result = _apply_safety_override("he ate his brother's kibble again", verdict)
    assert result["severity"] == 2
    assert result["requires_emergency_care"] is False


def test_no_red_flags_leaves_verdict_untouched():
    verdict = {"severity": 2, "severity_label": "Mild", "requires_emergency_care": False}
    result = _apply_safety_override("slightly less energetic than usual today", verdict)
    assert result["severity"] == 2
    assert result["requires_emergency_care"] is False


def test_poisoning_intersection_escalates_even_when_llm_sent_explicit_null_red_flags():
    """Audit bug: verdict.setdefault("red_flags", []) only fills in a
    MISSING key -- an LLM emitting a present-but-null "red_flags" (valid
    JSON, plausible when it has nothing to report on its own) left it None,
    and .append() on None crashed exactly on this poisoning/ingestion path,
    the one case this override exists to protect."""
    verdict = {"severity": 1, "severity_label": "Mild", "requires_emergency_care": False, "red_flags": None}
    result = _apply_safety_override("my dog ate a whole bar of chocolate an hour ago", verdict)
    assert result["severity"] == 5
    assert result["requires_emergency_care"] is True
    assert any("chocolate" in flag for flag in result["red_flags"])


import json as _json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.agent.tools.symptoms import FREE_SYMPTOM_QUERY_LIMIT, check_symptoms
from tests.fake_supabase import FakeSupabaseClient


def _make_ctx(supabase=None):
    return SimpleNamespace(supabase=supabase or FakeSupabaseClient(), openai=object(), settings=object())


def _make_agent_ctx(is_subscriber, pets=None):
    return SimpleNamespace(
        profile={"id": "profile-1", "phone_number": "919876543210"},
        pets=pets if pets is not None else [{"id": "pet-1", "name": "Rex", "species": "Dog"}],
        is_subscriber=is_subscriber,
    )


@pytest.mark.asyncio
async def test_free_customer_blocked_once_monthly_quota_used(monkeypatch):
    from datetime import date

    existing_logs = [
        {"id": f"log-{i}", "profile_id": "profile-1", "created_at": date.today().isoformat()}
        for i in range(FREE_SYMPTOM_QUERY_LIMIT)
    ]
    supabase = FakeSupabaseClient(initial={"health_logs": existing_logs})
    ctx = _make_ctx(supabase)
    agent_ctx = _make_agent_ctx(is_subscriber=False)

    result = await check_symptoms(ctx, agent_ctx, symptoms="vomiting")

    assert result["success"] is False
    assert result["error"] == "quota_exceeded"


@pytest.mark.asyncio
async def test_free_customer_gets_trimmed_color_and_message(monkeypatch):
    verdict = {
        "severity": 4, "severity_label": "Serious", "requires_emergency_care": False,
        "red_flags": [], "likely_categories": ["GI upset"], "recommendation": "See a vet today.",
        "reasoning": "...", "first_aid_checklist": ["Withhold food"],
    }
    monkeypatch.setattr("app.agent.tools.symptoms.json_completion", AsyncMock(return_value=_json.dumps(verdict)))
    ctx = _make_ctx()
    agent_ctx = _make_agent_ctx(is_subscriber=False)

    result = await check_symptoms(ctx, agent_ctx, symptoms="vomiting a lot today")

    assert result["success"] is True
    assert result["severity_color"] == "Red"
    assert "emergency" in result["message"].lower()
    assert "severity_display" not in result
    assert "red_flags" not in result
    assert "first_aid_checklist" not in result


@pytest.mark.asyncio
async def test_subscriber_gets_full_detail_and_checklist(monkeypatch):
    verdict = {
        "severity": 3, "severity_label": "Moderate", "requires_emergency_care": False,
        "red_flags": [], "likely_categories": ["GI upset"], "recommendation": "Monitor closely.",
        "reasoning": "mild GI signs", "first_aid_checklist": ["Withhold food for a few hours", "Offer small water sips"],
    }
    monkeypatch.setattr("app.agent.tools.symptoms.json_completion", AsyncMock(return_value=_json.dumps(verdict)))
    ctx = _make_ctx()
    agent_ctx = _make_agent_ctx(is_subscriber=True)

    result = await check_symptoms(ctx, agent_ctx, symptoms="a bit of vomiting")

    assert result["success"] is True
    assert result["severity_display"] == "🟡 Moderate (3/5)"
    assert result["first_aid_checklist"] == ["Withhold food for a few hours", "Offer small water sips"]
    assert result["severity_color"] == "Yellow"


@pytest.mark.asyncio
async def test_subscriber_never_hits_the_free_quota(monkeypatch):
    from datetime import date

    existing_logs = [
        {"id": f"log-{i}", "profile_id": "profile-1", "created_at": date.today().isoformat()}
        for i in range(FREE_SYMPTOM_QUERY_LIMIT + 5)
    ]
    supabase = FakeSupabaseClient(initial={"health_logs": existing_logs})
    verdict = {
        "severity": 1, "severity_label": "Mild", "requires_emergency_care": False,
        "red_flags": [], "likely_categories": [], "recommendation": "Keep an eye on it.",
        "reasoning": "...", "first_aid_checklist": [],
    }
    monkeypatch.setattr("app.agent.tools.symptoms.json_completion", AsyncMock(return_value=_json.dumps(verdict)))
    ctx = _make_ctx(supabase)
    agent_ctx = _make_agent_ctx(is_subscriber=True)

    result = await check_symptoms(ctx, agent_ctx, symptoms="a little sleepy")

    assert result["success"] is True
    assert result.get("error") != "quota_exceeded"
