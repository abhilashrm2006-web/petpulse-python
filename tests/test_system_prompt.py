"""Reproduces and verifies the fix for a real reported bug: a customer with
a stale open session for one pet (Thomas, awaiting a time input) said "book
a vet session for maxc" — naming a DIFFERENT pet entirely — and the agent
replied as if continuing Thomas's session, ignoring "maxc" completely. The
root cause: open_session only showed a bare pet_id (a UUID), so the agent
had to silently cross-reference it against the pets list and didn't
reliably do so. Fix: resolve the name explicitly and state in the prompt
that the open session is scoped to that pet only."""

from app.agent.system_prompt import (
    CASUAL_TONE_RULE,
    CUSTOMER_RULES,
    FORMATTING_RULES,
    GREETING_RULE,
    PERSONALIZATION_RULE,
    SAFETY_RULES,
    VET_RULES,
    VOICE_REPLY_LANGUAGE_RULE_TEMPLATE,
    _pet_name_for,
    build_system_prompt,
    build_turn_context,
)
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


def test_customer_gets_hindi_and_breed_aware_personalization_rule():
    prompt = build_system_prompt("customer")
    assert PERSONALIZATION_RULE in prompt


def test_vet_gets_no_personalization_rule():
    prompt = build_system_prompt("vet")
    assert PERSONALIZATION_RULE not in prompt


def test_personalization_rule_covers_major_regional_languages_not_just_hindi():
    for language in ("Hindi", "Tamil", "Telugu", "Kannada", "Malayalam", "Bengali", "Marathi", "Gujarati", "Punjabi", "Urdu"):
        assert language in PERSONALIZATION_RULE


def test_personalization_rule_covers_all_22_eighth_schedule_languages():
    """Expanded from the original 10 to all 22 -- built dynamically from
    app.agent.language.SUPPORTED_LANGUAGES so the two lists can't drift."""
    from app.agent.language import SUPPORTED_LANGUAGES

    for language in SUPPORTED_LANGUAGES.values():
        assert language in PERSONALIZATION_RULE


def test_casual_tone_rule_applies_to_customers_not_vets():
    """Vets keep a professional/brief tone (VET_RULES) -- the casual/slang
    style is a customer-facing thing only."""
    assert CASUAL_TONE_RULE in build_system_prompt("customer")
    assert CASUAL_TONE_RULE not in build_system_prompt("vet")


def test_greeting_rule_acknowledges_the_already_sent_welcome_image():
    """The mascot image is now sent deterministically in code (see
    app/agent/orchestrator.py), not via an LLM tool call -- the prompt just
    needs to know it's already been sent, not to send it itself."""
    assert "already been sent" in GREETING_RULE
    assert "Pulsy" in GREETING_RULE


def test_safety_rules_forbid_a_severity_line_without_an_actual_tool_call():
    """Found via a live model comparison during a response-quality audit: a
    plain, vague check-in ("he seems tired today") got a hallucinated
    "*Seriousness:* Moderate" line with no check_symptoms call behind it --
    a real correctness/safety risk since severity_display is supposed to be
    tool-grounded, never invented. This rule closes that loophole."""
    assert "Seriousness" in SAFETY_RULES
    assert "check_symptoms was actually called" in SAFETY_RULES


def test_formatting_rules_calibrate_length_to_the_question():
    """Found via the same audit: replies weren't calibrated to what was
    actually asked -- a simple check-in got an over-elaborate structured
    breakdown. FORMATTING_RULES now defaults explicitly to short with a
    concrete word-count anchor, instead of just "be concise", which was too
    vague to act on and still produced text-heavy replies per user feedback."""
    assert "default to SHORT" in FORMATTING_RULES
    assert "40-60 words" in FORMATTING_RULES


def test_voice_reply_language_rule_asks_for_natural_spoken_style():
    formatted = VOICE_REPLY_LANGUAGE_RULE_TEMPLATE.format(language="Tamil")
    assert "colloquialisms/slang" in formatted
    assert "real person talking" in formatted


def test_voice_reply_language_rule_only_applies_with_a_detected_language():
    prompt = build_system_prompt("customer", voice_reply_language="Tamil")
    assert VOICE_REPLY_LANGUAGE_RULE_TEMPLATE.format(language="Tamil") in prompt

    # No detected language -> no override.
    prompt_no_language = build_system_prompt("customer", voice_reply_language=None)
    assert VOICE_REPLY_LANGUAGE_RULE_TEMPLATE.format(language="Tamil") not in prompt_no_language


def test_general_pet_qa_is_explicitly_in_scope_for_customers():
    """Product requirement: the bot should act like a general conversational
    assistant for ANY pet-related question (nutrition, training, behavior,
    etc.), not just the specific workflows (onboarding/booking/documents/
    triage) — this must be stated explicitly, not left implicit, or the
    model may default to deflecting off-tool questions as out of scope."""
    assert "General pet Q&A" in CUSTOMER_RULES
    assert "nutrition" in CUSTOMER_RULES.lower()
    assert "GROUNDED FACTS ONLY still applies" in CUSTOMER_RULES
    assert "General pet Q&A" not in VET_RULES

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


def test_bare_yes_no_resolution_rule_covers_generic_offers_not_just_structured_flows():
    assert "Bare yes/no against your own last offer" in CUSTOMER_RULES
    assert "what are you saying yes to" in CUSTOMER_RULES.lower()
    assert "recording consent, prescription format" in CUSTOMER_RULES.lower() or "recording consent" in CUSTOMER_RULES


def test_pricing_sequencing_rule_forbids_pricing_in_same_turn_as_unresolved_failure():
    assert "Pricing sequencing" in CUSTOMER_RULES
    assert "unresolved tool failure" in CUSTOMER_RULES
    assert "₹399" in CUSTOMER_RULES


def test_remote_only_procedure_disclosure_rule_exists():
    assert "Remote-only expectation for procedures/surgery" in CUSTOMER_RULES
    assert "spay, neuter, surgery, operation" in CUSTOMER_RULES
    assert "remote/online-only" in CUSTOMER_RULES


def test_invalid_value_narrower_reask_rule_exists():
    assert 'error="invalid_value"' in CUSTOMER_RULES
    assert "narrower" in CUSTOMER_RULES.lower()
