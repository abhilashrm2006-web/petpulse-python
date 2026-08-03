"""Single tool registry for the one PetPulse agent, role-filtered. This
replaces n8n's `Is Doctor?` hard branch (customer tools vs. an entirely
separate vet code path) with one registry where tool *availability* is
gated by `profiles.role`, but the decision of which tool to call is always
made by the same LLM loop."""

from dataclasses import dataclass
from typing import Any, Callable

from app.agent.tools import account, booking, documents, onboarding, pet_members, pet_parent_guide, symptoms, vets


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable
    roles: set[str]


def _spec(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str],
    fn: Callable,
    roles: set[str],
) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=description,
        parameters={"type": "object", "properties": properties, "required": required},
        fn=fn,
        roles=roles,
    )


CUSTOMER = {"customer"}
VET = {"vet"}
BOTH = {"customer", "vet"}

_STR = {"type": "string"}
_NUM_OR_NULL = {"type": ["number", "null"]}

TOOL_SPECS: list[ToolSpec] = [
    _spec(
        "save_onboarding_field",
        "Save or update one piece of profile/pet information the customer just gave you (email, city, "
        "pet name, species, breed, age, dob, weight, gender). Call once per field; call multiple times in "
        "the SAME turn whenever several fields were mentioned in one message — e.g. \"my dog Max is a 2yo "
        "Labrador\" is 3 separate calls (pet_name=Max, species=Dog, breed=Labrador, age=2), all before you "
        "reply, not spread across several turns.",
        {
            "field": {"type": "string", "description": "email|city|pet_name|species|breed|age|dob|weight|gender"},
            "value": _STR,
            "pet_name": {"type": "string", "description": "Which pet this is for, if the account has more than one."},
        },
        ["field", "value"],
        onboarding.save_onboarding_field,
        CUSTOMER,
    ),
    _spec(
        "check_symptoms",
        "Run a structured triage assessment (1-5 severity rating) on a NEW symptom or health concern — "
        "whether the customer typed it, or you noticed it yourself in a photo, video, or voice note. Always "
        "call this before giving your own clinical read, for any modality. Free and unlimited for every "
        "customer — only booking an actual consultation afterward costs a fee.",
        {
            "pet_id": _STR, "pet_name": _STR, "species": _STR,
            "symptoms": {
                "type": "string",
                "description": "The symptom description — as the customer typed it, or your own summary of "
                "what you observed in an image/video/audio Media Context (e.g. 'visible limp on left hind leg' "
                "or 'labored breathing and coughing audible in voice note').",
            },
        },
        ["symptoms"],
        symptoms.check_symptoms,
        CUSTOMER,
    ),
    _spec(
        "find_nearby_vets",
        "Find nearby vet clinics. Use shared location-pin coordinates if available in context, else pass "
        "location_text from what the customer said. Pass open_now/emergency_24h/category to filter results "
        "whenever the customer asks for that.",
        {
            "location_text": _STR, "latitude": _NUM_OR_NULL, "longitude": _NUM_OR_NULL,
            "open_now": {"type": "boolean", "description": "Only show clinics open right now (24/7 clinics only, per data limits)."},
            "emergency_24h": {"type": "boolean", "description": "Only show 24/7 emergency clinics."},
            "category": {"type": "string", "description": "Filter by type: clinic|hospital|pharmacy."},
        },
        [],
        vets.find_nearby_vets,
        CUSTOMER,
    ),
    _spec(
        "send_pet_document",
        "Send previously-filed documents for a pet back to the customer over WhatsApp. If this (or any "
        "pet-lookup tool) returns error=\"ambiguous_pet\" with `candidates`, several pets share that name "
        "across DIFFERENT owners — use each candidate's owner_name/owner_phone plus conversation context to "
        "work out which one was meant, then call again with its exact pet_id. Never guess.",
        {"pet_id": _STR, "pet_name": _STR, "document_type": {"type": "string", "description": "Optional filter, e.g. 'vaccination'."}},
        [],
        documents.send_pet_document,
        CUSTOMER,
    ),
    _spec(
        "get_pet_passport",
        "Build and return the full health-passport summary for a pet (vaccinations incl. manufacturer/"
        "batch-lot number, recent records) plus certificate attachments. On error=\"ambiguous_pet\" with "
        "`candidates`, disambiguate using each candidate's owner_name/owner_phone and conversation context, "
        "then re-call with the exact pet_id.",
        {
            "pet_id": _STR, "pet_name": _STR,
            "send_certificates": {
                "type": "boolean",
                "description": "Set false only if the customer explicitly asked for just the summary, not the files.",
            },
        },
        [],
        documents.get_pet_passport,
        CUSTOMER,
    ),
    _spec(
        "file_document",
        "File the image/document from THIS turn into the pet's medical record. Call this when the system "
        "prompt's document-filing rule applies to what was just uploaded. A vet's patients can span multiple "
        "unrelated owners — if the pet name given matches more than one (error=\"ambiguous_pet\" with "
        "`candidates`), use each candidate's owner_name/owner_phone plus who the message named to pick the "
        "right one, then call again with its exact pet_id instead of pet_name. Never file to a guessed pet.",
        {
            "pet_id": _STR,
            "pet_name": _STR,
            "document_type": {"type": "string", "description": "Override the auto-detected type if the customer's own caption says otherwise."},
        },
        [],
        documents.file_document,
        BOTH,
    ),
    _spec(
        "search_documents",
        "Full-text search across a customer's filed documents (prescriptions, X-rays, lab reports) by "
        "keyword — searches document text/summary/type across all their pets unless a specific pet is given.",
        {"query": _STR, "pet_id": _STR, "pet_name": _STR},
        ["query"],
        documents.search_documents,
        CUSTOMER,
    ),
    _spec(
        "get_shareable_link",
        "Generate (or return the existing) public, no-login-required link to a pet's health summary — "
        "vaccination passport plus recent records — that a vet or boarding facility can open directly. "
        "One link per pet, stable across calls.",
        {"pet_id": _STR, "pet_name": _STR},
        [],
        documents.get_shareable_link,
        CUSTOMER,
    ),
    _spec(
        "start_new_pet_parent_guide",
        "Start the new-pet-parent onboarding guide (life-stage-specific welcome messages).",
        {"pet_id": _STR, "species": _STR, "pet_name": _STR, "age": _STR, "breed": _STR, "extra_context": _STR},
        [],
        pet_parent_guide.start_new_pet_parent_guide,
        CUSTOMER,
    ),
    _spec(
        "add_pet_member",
        "Add another person (family/caregiver/co-owner) to a pet's care team so they also get updates.",
        {
            "pet_id": _STR, "pet_name": _STR, "member_name": _STR,
            "member_phone": {"type": "string", "description": "With country code."},
            "role": {"type": "string", "description": "owner|family|caregiver"},
        },
        ["member_name", "member_phone"],
        pet_members.add_pet_member,
        CUSTOMER,
    ),
    _spec(
        "request_doctor_session",
        "Send the customer a catalogue of available vets to book a consultation. Requires pet name, "
        "species, breed, age, and email already on file. Write case_summary yourself.",
        {
            "pet_id": _STR, "pet_name": _STR,
            "case_summary": {"type": "string", "description": "2-4 sentence vet-tech-handoff summary."},
            "preferred_time": {"type": "string", "description": "ISO 8601 with +05:30 offset, if the customer stated one."},
        },
        ["case_summary"],
        booking.request_doctor_session,
        CUSTOMER,
    ),
    _spec(
        "select_doctor",
        "Customer picked a specific vet from the catalogue — compute and send that vet's open slots.",
        {"session_id": _STR, "doctor_phone": _STR},
        ["session_id", "doctor_phone"],
        booking.select_doctor,
        CUSTOMER,
    ),
    _spec(
        "book_slot",
        "Customer picked a specific time slot — finalize the booking (with a fresh double-booking check), "
        "and send a ₹399 consultation-fee payment link.",
        {"session_id": _STR, "slot_start": {"type": "string", "description": "ISO 8601 timestamp."}, "doctor_phone": _STR},
        ["session_id", "slot_start"],
        booking.book_slot,
        CUSTOMER,
    ),
    _spec(
        "propose_time",
        "Propose an alternate time for a session, to the other party.",
        {
            "session_id": _STR,
            "proposed_time": {"type": "string", "description": "ISO 8601 timestamp."},
            "proposed_by": {"type": "string", "description": "'customer' or 'vet' — whichever role you are talking to right now."},
        },
        ["session_id", "proposed_time", "proposed_by"],
        booking.propose_time,
        BOTH,
    ),
    _spec(
        "respond_to_time_proposal",
        "Respond to a time the other party proposed for a session.",
        {"session_id": _STR, "decision": {"type": "string", "description": "accept|decline|retime"}},
        ["session_id", "decision"],
        booking.respond_to_time_proposal,
        BOTH,
    ),
    _spec(
        "accept_session",
        "Vet accepts a pending session request at its currently proposed time.",
        {"session_id": _STR},
        ["session_id"],
        booking.accept_session,
        VET,
    ),
    _spec(
        "decline_session",
        "Vet declines a pending session request.",
        {"session_id": _STR},
        ["session_id"],
        booking.decline_session,
        VET,
    ),
    _spec(
        "cancel_session",
        "Cancel a booking session (either party) — also cancels the real calendar event/Meet link if one exists, "
        "not just the internal record.",
        {"session_id": _STR},
        ["session_id"],
        booking.cancel_session,
        BOTH,
    ),
    _spec(
        "reschedule_session",
        "Reschedule a session to a new time — whether it's still being negotiated or already fully confirmed "
        "with a calendar invite/Meet link. Sends the new time to the OTHER party for confirmation, same as "
        "initial booking; the calendar event only actually moves once they accept, and keeps the same Meet "
        "link rather than creating a duplicate.",
        {"session_id": _STR, "new_time": {"type": "string", "description": "ISO 8601 timestamp for the new proposed time."}},
        ["session_id", "new_time"],
        booking.reschedule_session,
        BOTH,
    ),
    _spec(
        "mark_session_done",
        "Vet marks a session as completed, moving it into the prescription-collection step.",
        {"session_id": _STR},
        ["session_id"],
        booking.mark_session_done,
        VET,
    ),
    _spec(
        "file_prescription",
        "File the vet's prescription/treatment notes for a completed session. Does NOT send anything to the "
        "customer yet — it asks them (via WhatsApp buttons) whether they want it as a text message or a PDF, "
        "and delivery happens once they answer (see deliver_prescription). Also records whether a follow-up "
        "visit is needed — if the vet's own message already says so (e.g. 'recheck in 2 weeks', 'no follow-up "
        "needed'), set follow_up_required/follow_up_date from that directly. If they haven't said either way, "
        "ask them before calling this tool rather than guessing.",
        {
            "session_id": _STR,
            "medications": _STR,
            "treatment_plan": _STR,
            "follow_up_required": {"type": "boolean", "description": "Whether the vet said a follow-up visit is needed. Defaults to false."},
            "follow_up_date": {
                "type": "string",
                "description": "ISO date (YYYY-MM-DD) for the follow-up, only if follow_up_required is true. Omit otherwise.",
            },
        },
        ["session_id", "medications"],
        booking.file_prescription,
        VET,
    ),
    _spec(
        "deliver_prescription",
        "Deliver an already-filed prescription in the format the customer just picked. Call this once they've "
        "answered the text-vs-PDF question file_prescription sent — a button tap or a plain 'text'/'pdf' reply "
        "both count. Never invents or reformats the content itself beyond the format param.",
        {"session_id": _STR, "format": {"type": "string", "description": "'text' or 'pdf', from the customer's own choice."}},
        ["session_id", "format"],
        booking.deliver_prescription,
        CUSTOMER,
    ),
    _spec(
        "list_my_appointments",
        "List the vet's upcoming accepted sessions.",
        {},
        [],
        booking.list_my_appointments,
        VET,
    ),
    _spec(
        "relay_to_customer",
        "Relay a vet's clinical note/reply to the customer on a specific session, close to verbatim. Sent "
        "with attribution to the vet's name, to EVERY member of the pet's household (owner/family/caregiver), "
        "not just whoever is chatting right now.",
        {"session_id": _STR, "message": _STR},
        ["session_id", "message"],
        booking.relay_to_customer,
        VET,
    ),
    _spec(
        "respond_to_recording_consent",
        "Record the customer's answer to the recording-consent question that's automatically sent right "
        "after payment. Call this once they've replied (a button tap or a plain yes/no) — it finalizes the "
        "booking (Calendar event + Meet link + notifications to both parties), which was held pending this "
        "answer. Declining doesn't block the booking, it just means the vet shouldn't record.",
        {"session_id": _STR, "consent": {"type": "boolean", "description": "true if they consent to recording, false if they decline."}},
        ["session_id", "consent"],
        booking.respond_to_recording_consent,
        CUSTOMER,
    ),
    _spec(
        "relay_to_doctor",
        "Relay the customer's own words to the vet on a specific session, close to verbatim, with attribution "
        "to the customer's name — for a follow-up question or detail that doesn't fit any structured action.",
        {"session_id": _STR, "message": _STR},
        ["session_id", "message"],
        booking.relay_to_doctor,
        CUSTOMER,
    ),
    _spec(
        "request_data_deletion",
        "Customer asked to delete/clear/reset their profile, chat, or data — sends a Yes/No confirmation "
        "over WhatsApp before anything happens. Never delete or clear anything yourself without this "
        "confirmation step.",
        {},
        [],
        account.request_data_deletion,
        CUSTOMER,
    ),
    _spec(
        "respond_to_deletion_confirmation",
        "Record the customer's answer to the deletion confirmation request_data_deletion sent — a button tap "
        "or a plain yes/no reply both count. 'Yes' actually clears their chat history (not their pet/account "
        "data) and confirms it; 'No' cancels and asks them why, already sent — don't call this a second time "
        "for the same request.",
        {"confirm": {"type": "boolean", "description": "true if they confirmed clearing their chat, false if they backed out."}},
        ["confirm"],
        account.respond_to_deletion_confirmation,
        CUSTOMER,
    ),
    _spec(
        "record_deletion_feedback",
        "Log the reason a customer gave for wanting to clear their chat, after they declined via "
        "respond_to_deletion_confirmation and were asked why. Call this once, using their own words, then "
        "just thank them briefly — don't restate or analyze their reason back to them.",
        {"reason": _STR},
        ["reason"],
        account.record_deletion_feedback,
        CUSTOMER,
    ),
]

_BY_NAME = {spec.name: spec for spec in TOOL_SPECS}


def get_tool_schemas(role: str) -> list[dict[str, Any]]:
    return [
        {"type": "function", "function": {"name": spec.name, "description": spec.description, "parameters": spec.parameters}}
        for spec in TOOL_SPECS
        if role in spec.roles
    ]


def get_tool_fn(name: str) -> Callable | None:
    spec = _BY_NAME.get(name)
    return spec.fn if spec else None


def is_tool_allowed_for_role(name: str, role: str) -> bool:
    spec = _BY_NAME.get(name)
    return bool(spec and role in spec.roles)
