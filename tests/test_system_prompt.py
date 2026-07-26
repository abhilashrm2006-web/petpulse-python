"""Reproduces and verifies the fix for a real reported bug: a customer with
a stale open session for one pet (Thomas, awaiting a time input) said "book
a vet session for maxc" — naming a DIFFERENT pet entirely — and the agent
replied as if continuing Thomas's session, ignoring "maxc" completely. The
root cause: open_session only showed a bare pet_id (a UUID), so the agent
had to silently cross-reference it against the pets list and didn't
reliably do so. Fix: resolve the name explicitly and state in the prompt
that the open session is scoped to that pet only."""

from app.agent.system_prompt import GREETING_RULE, _pet_name_for, build_system_prompt, build_turn_context
from app.ingestion.context import AgentContext
from app.ingestion.webhook import ExtractedMessage


def test_greeting_rule_forbids_unprompted_symptom_recap():
    """Reproduces a real reported bug: a bare "Hi" got a full unsolicited
    severity re-assessment of an old on-file symptom instead of a plain
    greeting — "continue naturally" wasn't enough of a constraint on its
    own."""
    assert "never proactively" in GREETING_RULE
    assert "severity assessment" in GREETING_RULE


def test_greeting_rule_only_applies_to_customer_role():
    assert GREETING_RULE in build_system_prompt("customer")
    assert GREETING_RULE not in build_system_prompt("vet")

PETS = [
    {"id": "pet-thomas", "name": "Thomas", "species": "Dog"},
    {"id": "pet-maxc", "name": "maxc", "species": "Dog"},
]


def test_pet_name_for_resolves_known_id():
    assert _pet_name_for(PETS, "pet-thomas") == "Thomas"


def test_pet_name_for_unknown_id_is_unknown():
    assert _pet_name_for(PETS, "nonexistent-id") == "unknown"


def test_pet_name_for_none_is_unknown():
    assert _pet_name_for(PETS, None) == "unknown"


def _make_agent_ctx(active_pet=None, open_session=None):
    return AgentContext(
        profile={"id": "profile-1", "full_name": "Jane", "phone_number": "919876543210"},
        role="customer",
        is_new_profile=False,
        pets=PETS,
        active_pet=active_pet,
        active_pet_matched_from_message=bool(active_pet),
        memory_context=[],
        medical_context={},
        knowledge_base=[],
        open_session=open_session,
        pending_negotiation=None,
        onboarding={"complete": True, "missing_fields": []},
    )


def test_open_session_names_its_pet_and_warns_about_different_pet_requests():
    agent_ctx = _make_agent_ctx(
        active_pet=PETS[1],  # maxc — the pet actually named in the current message
        open_session={"id": "session-thomas", "pet_id": "pet-thomas", "status": "pending", "awaiting_from": "customer_time_input"},
    )
    extracted = ExtractedMessage(
        phone_number="919876543210", sender_name="Jane", message_id="wamid.1",
        timestamp="1700000000", message_type="text", text="i want to book a vet session for maxc",
    )

    context = build_turn_context(agent_ctx, extracted, media_context="", document_filing_status="")

    assert "Open booking session (for pet: Thomas)" in context
    assert "scoped ONLY to Thomas" in context
    assert "Active pet: maxc" in context


def test_shared_location_pin_is_surfaced_for_find_nearby_vets():
    """Bug: find_nearby_vets accepts latitude/longitude and its own tool
    description says to use a shared location pin if available, but the
    coordinates were parsed out of the webhook payload and then never
    included anywhere in the turn context the model actually sees."""
    agent_ctx = _make_agent_ctx()
    extracted = ExtractedMessage(
        phone_number="919876543210", sender_name="Jane", message_id="wamid.1",
        timestamp="1700000000", message_type="location", text="",
        latitude=13.0067, longitude=80.2206, location_text="Puzhuthivakkam, Chennai",
    )

    context = build_turn_context(agent_ctx, extracted, media_context="", document_filing_status="")

    assert "latitude=13.0067" in context
    assert "longitude=80.2206" in context
    assert "Puzhuthivakkam, Chennai" in context
    assert "find_nearby_vets" in context


def test_no_location_line_when_no_pin_shared():
    agent_ctx = _make_agent_ctx()
    extracted = ExtractedMessage(
        phone_number="919876543210", sender_name="Jane", message_id="wamid.1",
        timestamp="1700000000", message_type="text", text="find me a vet nearby",
    )

    context = build_turn_context(agent_ctx, extracted, media_context="", document_filing_status="")

    assert "Shared location pin" not in context
