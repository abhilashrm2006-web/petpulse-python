"""Covers is_tool_allowed_for_tier -- the Subscriber-vs-Free gating layer
that sits alongside the existing role-based gating (is_tool_allowed_for_role).
Tier gating only ever applies to role="customer"; a vet is never restricted
by a customer's subscription tier."""

from app.agent.registry import is_tool_allowed_for_tier


def test_subscriber_only_tool_blocked_for_free_customer():
    assert is_tool_allowed_for_tier("book_slot", "customer", is_subscriber=False) is False


def test_subscriber_only_tool_allowed_for_subscriber():
    assert is_tool_allowed_for_tier("book_slot", "customer", is_subscriber=True) is True


def test_non_gated_tool_allowed_for_free_customer():
    assert is_tool_allowed_for_tier("save_onboarding_field", "customer", is_subscriber=False) is True


def test_check_symptoms_is_open_to_free_and_subscriber_alike():
    """The triage assessment itself is never paywalled -- only booking an
    actual consultation is Subscriber-gated (see book_slot above)."""
    assert is_tool_allowed_for_tier("check_symptoms", "customer", is_subscriber=False) is True
    assert is_tool_allowed_for_tier("check_symptoms", "customer", is_subscriber=True) is True


def test_vet_is_never_restricted_by_tier_even_for_gated_tools():
    assert is_tool_allowed_for_tier("file_document", "vet", is_subscriber=False) is True


def test_start_subscription_is_never_tier_gated_itself():
    """start_subscription is how a Free customer becomes a Subscriber --
    it must never itself require being a Subscriber already."""
    assert is_tool_allowed_for_tier("start_subscription", "customer", is_subscriber=False) is True


def test_unknown_tool_name_defaults_to_allowed():
    assert is_tool_allowed_for_tier("not_a_real_tool", "customer", is_subscriber=False) is True
