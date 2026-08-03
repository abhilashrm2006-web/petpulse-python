"""Covers build_winback_message (2026-08-04 re-engagement workstream, item
3): must pull the consultation price dynamically from settings, and must
never state or imply that consultations themselves are free."""

from types import SimpleNamespace

from scripts.paywall_winback import build_winback_message


def test_message_includes_the_live_configured_price():
    settings = SimpleNamespace(razorpay_consult_fee_inr=399)
    message = build_winback_message(settings, "Amrapali")

    assert "₹399" in message
    assert "Amrapali" in message


def test_message_reflects_a_changed_price_without_code_changes():
    """Price must come from settings, not be hardcoded, so it can't go
    stale if pricing changes later."""
    settings = SimpleNamespace(razorpay_consult_fee_inr=499)
    message = build_winback_message(settings, "Rohan")

    assert "₹499" in message
    assert "₹399" not in message


def test_message_never_claims_consultations_are_free():
    settings = SimpleNamespace(razorpay_consult_fee_inr=399)
    message = build_winback_message(settings, "Priya")

    assert "free consult" not in message.lower()
    assert "consultation is free" not in message.lower()
    assert "the only paid part" in message.lower()


def test_message_mentions_hindi_is_now_supported():
    settings = SimpleNamespace(razorpay_consult_fee_inr=399)
    message = build_winback_message(settings, "Ravi")

    assert "hindi" in message.lower()
