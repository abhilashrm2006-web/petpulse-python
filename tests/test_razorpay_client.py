import hashlib
import hmac

from app.config import Settings
from app.integrations.razorpay_client import extract_paid_session_id, extract_subscription_event, verify_webhook_signature


def _settings(secret: str = "whsec_test") -> Settings:
    return Settings(razorpay_webhook_secret=secret)


def test_verify_webhook_signature_accepts_correct_signature():
    settings = _settings()
    body = b'{"event": "payment_link.paid"}'
    signature = hmac.new(settings.razorpay_webhook_secret.encode(), body, hashlib.sha256).hexdigest()
    assert verify_webhook_signature(settings, body, signature) is True


def test_verify_webhook_signature_rejects_tampered_body():
    settings = _settings()
    body = b'{"event": "payment_link.paid"}'
    signature = hmac.new(settings.razorpay_webhook_secret.encode(), body, hashlib.sha256).hexdigest()
    assert verify_webhook_signature(settings, b'{"event": "payment_link.paid", "tampered": true}', signature) is False


def test_verify_webhook_signature_rejects_missing_header():
    settings = _settings()
    assert verify_webhook_signature(settings, b"{}", None) is False


def test_extract_paid_session_id_from_payment_link_paid_event():
    body = {
        "event": "payment_link.paid",
        "payload": {"payment_link": {"entity": {"id": "plink_123", "reference_id": "session-a"}}},
    }
    assert extract_paid_session_id(body) == "session-a"


def test_extract_paid_session_id_ignores_other_events():
    body = {"event": "payment_link.expired", "payload": {"payment_link": {"entity": {"reference_id": "session-a"}}}}
    assert extract_paid_session_id(body) is None


def test_extract_paid_session_id_handles_malformed_payload():
    assert extract_paid_session_id({"event": "payment_link.paid", "payload": {}}) is None
    assert extract_paid_session_id({}) is None


def test_extract_subscription_event_parses_activated_event():
    body = {
        "event": "subscription.activated",
        "payload": {"subscription": {"entity": {"id": "sub_123", "notes": {"reference_id": "profile-a"}}}},
    }
    assert extract_subscription_event(body) == ("subscription.activated", "sub_123", "profile-a")


def test_extract_subscription_event_ignores_non_subscription_events():
    assert extract_subscription_event({"event": "payment_link.paid", "payload": {}}) is None


def test_extract_subscription_event_handles_malformed_payload():
    assert extract_subscription_event({"event": "subscription.activated", "payload": {}}) is None
    assert extract_subscription_event({}) is None


def test_extract_subscription_event_tolerates_missing_reference_id():
    body = {"event": "subscription.charged", "payload": {"subscription": {"entity": {"id": "sub_123", "notes": {}}}}}
    assert extract_subscription_event(body) == ("subscription.charged", "sub_123", None)
