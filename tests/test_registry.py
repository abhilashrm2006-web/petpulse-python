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
    assert is_tool_allowed_for_tier("search_documents", "vet", is_subscriber=False) is True


def test_records_vault_tools_are_open_to_free_customers():
    """send_pet_document/get_pet_passport/file_document work for Free too --
    Free is capped (5 documents, basic passport), not locked out. Only
    search/share are Subscriber-gated."""
    for name in ("send_pet_document", "get_pet_passport", "file_document"):
        assert is_tool_allowed_for_tier(name, "customer", is_subscriber=False) is True
    assert is_tool_allowed_for_tier("search_documents", "customer", is_subscriber=False) is False
    assert is_tool_allowed_for_tier("get_shareable_link", "customer", is_subscriber=False) is False


def test_nearby_vet_finder_is_open_to_free_customers():
    """Free gets the plain list; only the open_now/emergency_24h/category
    filter params are Subscriber-only (enforced inside the tool, not here)."""
    assert is_tool_allowed_for_tier("find_nearby_vets", "customer", is_subscriber=False) is True


def test_start_subscription_is_never_tier_gated_itself():
    """start_subscription is how a Free customer becomes a Subscriber --
    it must never itself require being a Subscriber already."""
    assert is_tool_allowed_for_tier("start_subscription", "customer", is_subscriber=False) is True


def test_unknown_tool_name_defaults_to_allowed():
    assert is_tool_allowed_for_tier("not_a_real_tool", "customer", is_subscriber=False) is True


def test_welcome_character_tool_is_open_to_free_customers():
    """The mascot greeting is a branding touch, not a paywalled feature --
    every customer, Free or Subscriber, should get it."""
    assert is_tool_allowed_for_tier("send_welcome_character", "customer", is_subscriber=False) is True
    assert is_tool_allowed_for_tier("send_welcome_character", "customer", is_subscriber=True) is True
