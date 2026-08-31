"""Unifies `jCjl7XGrpW1K8kcO` (Find Available Slots / `request_doctor_session`,
spec §3.2) with the ~150-node `DocMsg -`/`CustResp -` action-switch chains
(spec §2) into explicit tools. In n8n these were reached via hardcoded
button-id classifiers that bypassed the LLM entirely; here the single agent
decides which of these to call — including for button taps, which arrive
as a plain-language system note rather than a routed action id (see
agent/orchestrator.py). Each tool still owns its own safety-critical
mechanics (slot double-booking recheck, calendar event creation) — the
agent decides *whether* to book, the tool guarantees *how*.

`doctor_sessions` state machine (unchanged from n8n):
  status: pending -> negotiating -> accepted | declined | cancelled -> completed
  awaiting_from: doctor_choice -> customer_time_input | doctor_time_input -> doctor_prescription (null once resolved)
  doctor_phone sentinel 'pending_doctor_choice' while no vet has been chosen yet.
"""

import logging
from datetime import datetime, timedelta
from typing import Any

from app.availability.slots import IST, compute_doctor_slots, is_window_free
from app.deps import AppContext
from app.ingestion.context import AgentContext
from app.integrations import google_calendar, razorpay_client
from app.integrations.openai_client import text_completion
from app.integrations.prescription_pdf import build_prescription_pdf
from app.media_pipeline.classify import DEFAULT_BUCKET, LABEL_TO_BUCKET
from app.integrations.supabase_client import get_pet_member_contacts, sign_storage_url, upload_to_storage
from app.utils.pet_resolution import AMBIGUOUS_PET, resolve_pet

logger = logging.getLogger(__name__)

MAX_DOCTORS_LISTED = 9
SESSION_DURATION_MINUTES = 30

PRESCRIPTION_FORMAT_SYSTEM_PROMPT = """You are formatting a veterinary prescription for a WhatsApp message. \
You will be given the vet's own medications and treatment-plan text plus patient details. Reorganize them \
into a clean, professional prescription layout using WhatsApp markdown (*bold* for headers, plain lines for \
content, no HTML). Do NOT invent, add, remove, or alter any drug name, dose, frequency, or instruction beyond \
what's given — only reorganize and lightly tidy the wording (e.g. expand obvious shorthand like "bid" -> \
"twice daily" ONLY if unambiguous) of the exact same clinical facts the vet gave you. Use roughly this shape, \
omitting any section with nothing to put in it:

🐾 *Prescription — <pet name>*
Vet: <doctor name>
Date: <date>
Reason for visit: <reason, if given>

*Medications*
<one line per medication>

*Treatment Plan / Advice*
<treatment plan / instructions>

Respond with ONLY the formatted message text, nothing else — no preamble, no code fences."""


async def _format_prescription_message(
    ctx: AppContext,
    pet_name: str,
    doctor_name: str,
    case_summary: str,
    medications: str,
    treatment_plan: str,
) -> str:
    """Turns the vet's own loosely-typed medications/treatment-plan text into
    a clean prescription layout instead of just echoing it back under bare
    "Medications:"/"Treatment plan:" labels. Falls back to that same plain
    layout if the formatting call fails for any reason — the customer must
    still get the actual clinical content even if the nicer formatting pass
    doesn't come through."""
    user_prompt = (
        f"Pet: {pet_name}\nVet: {doctor_name}\nDate: {datetime.now(tz=IST).strftime('%d %b %Y')}\n"
        f"Reason for visit: {case_summary or 'not stated'}\n"
        f"Medications (vet's own words): {medications}\n"
        f"Treatment plan (vet's own words): {treatment_plan or 'not stated'}"
    )
    try:
        formatted = await text_completion(ctx.openai, ctx.settings, PRESCRIPTION_FORMAT_SYSTEM_PROMPT, user_prompt, reasoning_effort="low")
        if formatted.strip():
            return formatted.strip()
    except Exception:
        logger.exception("Prescription formatting call failed for pet %s, falling back to plain layout", pet_name)

    lines = [f"*Session summary — {pet_name} with {doctor_name}*"]
    if case_summary:
        lines.append(f"Reason for visit: {case_summary}")
    lines.append(f"Medications: {medications}")
    if treatment_plan:
        lines.append(f"Treatment plan: {treatment_plan}")
    return "\n".join(lines)


def _get_session(client, session_id: str) -> dict[str, Any] | None:
    resp = client.table("doctor_sessions").select("*").eq("id", session_id).limit(1).execute()
    return resp.data[0] if resp.data else None


def _get_profile(client, profile_id: str) -> dict[str, Any] | None:
    resp = client.table("profiles").select("*").eq("id", profile_id).limit(1).execute()
    return resp.data[0] if resp.data else None


def _get_pet(client, pet_id: str) -> dict[str, Any] | None:
    if not pet_id:
        return None
    resp = client.table("pets").select("*").eq("id", pet_id).limit(1).execute()
    return resp.data[0] if resp.data else None


def _household_phones(client, session: dict[str, Any]) -> set[str]:
    """Every member of the pet's household (owner/family/caregiver), not
    just whichever one happens to be driving the current WhatsApp
    conversation — a session concerning the pet is everyone's business.
    Falls back to just the acting profile if the pet has no members on file
    (e.g. pet_id missing on an older session) so a notification never
    silently goes nowhere."""
    pet_id = session.get("pet_id")
    phones: set[str] = set()
    if pet_id:
        phones = {c["phone_number"] for c in get_pet_member_contacts(client, pet_id)}
    if not phones:
        profile = _get_profile(client, session["profile_id"])
        if profile:
            phones = {profile["phone_number"]}
    return phones


async def _notify_household(ctx: AppContext, session: dict[str, Any], message: str) -> None:
    for phone in _household_phones(ctx.supabase, session):
        try:
            await ctx.whatsapp.send_text(phone, message)
        except Exception:
            # One member's send failing (expired 24h WhatsApp window, invalid
            # number, rate limit) must never take the rest of the household —
            # or, via _finalize_booking calling this before the doctor
            # notification, the vet — down with it.
            logger.exception("Failed to notify household member %s for session %s", phone, session.get("id"))


def _pet_description(pet: dict[str, Any] | None, fallback: str = "the pet") -> str:
    if not pet:
        return fallback
    name = pet.get("name") or fallback
    extras = [v for v in (pet.get("species"), pet.get("breed")) if v]
    return f"{name} ({', '.join(extras)})" if extras else name


def _format_doctor_message(
    client, session: dict[str, Any], header: str, when_text: str | None = None, meet_link: str | None = None
) -> str:
    """Detailed, self-contained notification for the vet — enough to know
    who/what/when without scrolling back through chat history, for the
    three moments a session's state changes under them: assigned (booked),
    rescheduled, cancelled."""
    customer_profile = _get_profile(client, session["profile_id"])
    pet = _get_pet(client, session.get("pet_id"))
    customer_name = customer_profile.get("full_name", "Pet Parent") if customer_profile else "Pet Parent"
    customer_phone = customer_profile.get("phone_number") if customer_profile else None

    lines = [
        header,
        f"Patient: {_pet_description(pet)}",
        f"Owner: {customer_name}" + (f" ({customer_phone})" if customer_phone else ""),
    ]
    if when_text:
        lines.append(f"When: {when_text}")
    if session.get("case_summary"):
        lines.append(f"Reason: {session['case_summary']}")
    if session.get("recording_consent") in ("given", "declined"):
        lines.append(f"Recording consent: {'Given' if session['recording_consent'] == 'given' else 'Declined — do not record'}")
    if meet_link:
        lines.append(f"Join: {meet_link}")
    return "\n".join(lines)


def _normalize_to_date(date_str: str) -> str | None:
    """Same ambiguity problem as _normalize_to_ist, but for a bare follow-up
    date (no time component) -- accepts either a plain date or a full
    timestamp and always returns just the date portion."""
    if not date_str:
        return None
    try:
        dt = datetime.fromisoformat(date_str)
    except ValueError:
        return None
    return dt.date().isoformat()


def _normalize_to_ist(time_str: str) -> str | None:
    """Every write of a customer/vet-supplied time string must go through
    this — an LLM-emitted ISO timestamp with no offset is ambiguous, and
    Postgres/PostgREST will store it as UTC, landing real bookings 5.5
    hours off from what both parties were told over WhatsApp."""
    if not time_str:
        return None
    try:
        dt = datetime.fromisoformat(time_str)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=IST)
    return dt.isoformat()


async def _send_doctor_catalogue(ctx: AppContext, session_id: str, phone: str) -> bool:
    client = ctx.supabase
    doctors = (
        client.table("profiles").select("*").eq("role", "vet").eq("is_active", True)
        .limit(MAX_DOCTORS_LISTED).execute().data or []
    )
    if not doctors:
        return False

    rows = [
        {
            "id": f"choose_doctor|{session_id}|{doc['phone_number']}",
            "title": (doc.get("full_name") or "Vet")[:24],
            # Confirmed live bug (2026-08-27): dict.get(key, default) only
            # applies the default when the KEY is absent, not when it's
            # present with a None value -- a doctor onboarded without
            # experience_years/specialization set rendered as the literal
            # string "Noney exp • None". `or` catches both cases.
            "description": f"{doc.get('experience_years') or '?'} years exp • {doc.get('specialization') or 'General Practice'}"[:72],
        }
        for doc in doctors
    ]
    rows.append({"id": f"cancel_booking|{session_id}", "title": "Cancel", "description": "Don't book right now"})

    await ctx.whatsapp.send_interactive_list(
        phone,
        header="Choose a Vet",
        body="Here are our available vets. Pick one to see their open slots:",
        button_label="View Vets",
        sections=[{"title": "Available Vets", "rows": rows}],
    )
    return True


async def request_doctor_session(
    ctx: AppContext,
    agent_ctx: AgentContext,
    pet_id: str = "",
    pet_name: str = "",
    case_summary: str = "",
    preferred_time: str = "",
) -> dict[str, Any]:
    if pet_id == AMBIGUOUS_PET:
        names = ", ".join(p.get("name", "?") for p in agent_ctx.pets)
        return {"success": False, "error": "ambiguous_pet", "message": f"Which pet is this for? On file: {names}"}

    client = ctx.supabase
    profile_id = agent_ctx.profile["id"]
    phone = agent_ctx.profile["phone_number"]

    # Resolve the pet by id/name like every other tool does — previously this was
    # never resolved at all, so every session got created with pet_id=NULL unless
    # the caller happened to already have the exact pet_id in hand.
    resolution = resolve_pet(agent_ctx.pets, pet_id=pet_id, pet_name=pet_name, auto_resolve_single=True)
    if resolution.ambiguous:
        names = ", ".join(p.get("name", "?") for p in agent_ctx.pets)
        return {"success": False, "error": "ambiguous_pet", "message": f"Which pet is this for? On file: {names}"}
    resolved_pet_id = resolution.pet["id"] if resolution.pet else None

    # Scoped by pet, not just by account — a pending vet-choice request for one pet
    # must not block starting a fresh one for a DIFFERENT pet on the same account.
    existing_query = (
        client.table("doctor_sessions")
        .select("*")
        .eq("profile_id", profile_id)
        .eq("doctor_phone", "pending_doctor_choice")
        .eq("status", "pending")
    )
    if resolved_pet_id:
        existing_query = existing_query.eq("pet_id", resolved_pet_id)
    existing = existing_query.limit(1).execute().data

    if existing:
        session_id = existing[0]["id"]
        # Resend the actual list instead of telling the customer to go find an old
        # message — much less friction, and gives them a working "Cancel" row too.
        sent = await _send_doctor_catalogue(ctx, session_id, phone)
        if not sent:
            return {"success": False, "error": "no_doctors_available", "message": "No vets are available to book right now."}
        return {
            "success": True,
            "mode": "doctor_catalogue_sent",
            "session_id": session_id,
            "instruction_to_llm": "A vet list was already open for this pet — it has just been resent. Reply with an empty string.",
        }

    session_row = (
        client.table("doctor_sessions")
        .insert(
            {
                "profile_id": profile_id,
                "pet_id": resolved_pet_id,
                "doctor_phone": "pending_doctor_choice",
                "status": "pending",
                "awaiting_from": "doctor_choice",
                "case_summary": case_summary,
                "preferred_time": _normalize_to_ist(preferred_time),
            }
        )
        .execute()
        .data[0]
    )

    sent = await _send_doctor_catalogue(ctx, session_row["id"], phone)
    if not sent:
        return {"success": False, "error": "no_doctors_available", "message": "No vets are available to book right now."}

    return {"success": True, "mode": "doctor_catalogue_sent", "session_id": session_row["id"]}


async def select_doctor(ctx: AppContext, agent_ctx: AgentContext, session_id: str, doctor_phone: str) -> dict[str, Any]:
    client = ctx.supabase
    session = _get_session(client, session_id)
    if not session:
        return {"success": False, "error": "session_not_found"}

    client.table("doctor_sessions").update(
        {"doctor_phone": doctor_phone, "awaiting_from": "customer_time_input"}
    ).eq("id", session_id).execute()

    slots = await compute_doctor_slots(ctx.settings, client=client, doctor_phone=doctor_phone)
    if not slots:
        return {"success": True, "mode": "no_slots", "message": "That vet has no open slots in the next few days. Would you like to try another vet?"}

    by_day: dict[str, list] = {}
    for slot in slots:
        day_key = slot.start.strftime("%A, %d %b")
        by_day.setdefault(day_key, []).append(slot)

    sections = [
        {
            "title": day,
            "rows": [
                {"id": f"book_slot|{session_id}|{doctor_phone}|{s.to_iso()}", "title": s.label(), "description": ""}
                for s in day_slots
            ],
        }
        for day, day_slots in by_day.items()
    ]
    sections.append({"title": "Other", "rows": [{"id": f"cancel_booking|{session_id}", "title": "Cancel", "description": "Don't book right now"}]})

    await ctx.whatsapp.send_interactive_list(
        agent_ctx.profile["phone_number"],
        header="Choose a Time",
        body="Here are the next available slots with this vet:",
        button_label="View Slots",
        sections=sections,
    )
    return {"success": True, "mode": "slot_list_sent", "session_id": session_id}


async def book_slot(ctx: AppContext, agent_ctx: AgentContext, session_id: str, slot_start: str, doctor_phone: str = "") -> dict[str, Any]:
    client = ctx.supabase
    session = _get_session(client, session_id)
    if not session:
        return {"success": False, "error": "session_not_found"}
    doctor_phone = doctor_phone or session.get("doctor_phone")

    start = datetime.fromisoformat(slot_start)
    if start.tzinfo is None:
        start = start.replace(tzinfo=IST)
    end = start + timedelta(minutes=SESSION_DURATION_MINUTES)

    fresh_slots = await compute_doctor_slots(ctx.settings, client=client, doctor_phone=doctor_phone)
    if not any(s.start == start for s in fresh_slots):
        return await select_doctor(ctx, agent_ctx, session_id, doctor_phone)  # slot taken, resend fresh list

    return await _request_payment(ctx, session, doctor_phone, start)


async def _request_payment(ctx: AppContext, session: dict[str, Any], doctor_phone: str, start: datetime) -> dict[str, Any]:
    """Gate booking on payment: hold the slot (doctor_phone + preferred_time
    written now so handle_payment_webhook can pick up exactly where this
    left off) and send a Razorpay payment link instead of confirming
    outright. The Calendar event + both-party notifications only happen
    once Razorpay's webhook reports the link paid — never here, since
    accepting this tool call says nothing about whether the customer ever
    actually pays."""
    client = ctx.supabase
    customer_profile = _get_profile(client, session["profile_id"])
    customer_phone = customer_profile["phone_number"]
    customer_name = customer_profile.get("full_name") or "Pet Parent"

    link = await razorpay_client.create_payment_link(
        ctx.settings,
        amount_inr=ctx.settings.razorpay_consult_fee_inr,
        reference_id=session["id"],
        description="PetPulse vet consultation fee",
        customer_name=customer_name,
        customer_phone=customer_phone,
    )

    client.table("doctor_sessions").update(
        {
            "doctor_phone": doctor_phone,
            "preferred_time": start.isoformat(),
            "awaiting_from": "payment",
            "payment_status": "awaiting",
            "payment_link_id": link.get("id"),
        }
    ).eq("id", session["id"]).execute()

    await ctx.whatsapp.send_text(
        customer_phone,
        "Almost there! Please complete the consultation fee payment to confirm your slot:\n"
        f"{link['short_url']}\n\nYour session will be booked automatically as soon as payment is received.",
    )
    return {"success": True, "mode": "payment_requested", "session_id": session["id"], "payment_link": link.get("short_url")}


async def handle_payment_webhook(ctx: AppContext, event_body: dict[str, Any]) -> bool:
    """Razorpay's async confirmation that a payment link was actually paid —
    the only trustworthy signal that money changed hands (the webhook
    route's signature check happens before this is ever called). Idempotent:
    a replayed event for a session already marked 'paid' is a no-op, so
    Razorpay's at-least-once delivery can never double-book/double-notify.

    Doesn't finalize the booking directly — payment only unlocks the
    recording-consent question (see _request_recording_consent). The
    Calendar event/Meet link/notifications only actually go out once the
    customer answers that, via respond_to_recording_consent."""
    session_id = razorpay_client.extract_paid_session_id(event_body)
    if not session_id:
        logger.info("Razorpay webhook event=%s ignored: not a payment_link.paid event, or missing reference_id", event_body.get("event"))
        return False

    client = ctx.supabase
    session = _get_session(client, session_id)
    if not session:
        logger.error("Razorpay webhook: paid session_id=%s has no matching doctor_sessions row", session_id)
        return False

    # Read-then-write on payment_status was a TOCTOU race: Razorpay's at-least-once
    # webhook delivery means two deliveries for the same payment_link.paid event can
    # both read payment_status="awaiting" before either write lands, double-sending
    # the recording-consent prompt (confirmed reachable — nothing here previously
    # made the read+write atomic). Scoping the UPDATE's own WHERE clause to
    # payment_status="awaiting" makes Postgres itself the single arbiter: only the
    # delivery that finds a matching row does .execute() gets it, since Postgres
    # runs UPDATE...WHERE as one atomic statement.
    updated = (
        client.table("doctor_sessions")
        .update({"payment_status": "paid", "awaiting_from": "recording_consent", "recording_consent": "pending"})
        .eq("id", session_id)
        .eq("payment_status", "awaiting")
        .execute()
        .data
    )
    if not updated:
        logger.info("Razorpay webhook: session_id=%s already marked paid, ignoring duplicate delivery", session_id)
        return False

    await _request_recording_consent(ctx, session)
    return True


async def _request_recording_consent(ctx: AppContext, session: dict[str, Any]) -> None:
    """Ask for recording consent AFTER payment, BEFORE the Calendar event/Meet
    link is created — a deliberate ordering, not a formality tacked on
    after the fact, so the vet's notification (see _format_doctor_message)
    always carries a real answer, never a stale/blank one."""
    client = ctx.supabase
    customer_profile = _get_profile(client, session["profile_id"])
    phone = customer_profile["phone_number"]
    await ctx.whatsapp.send_interactive_buttons(
        phone,
        "Payment received! Before we confirm your vet session: this call may be recorded for quality and "
        "record-keeping purposes. Do you consent to the session being recorded?",
        [
            {"id": f"recording_consent|{session['id']}|yes", "title": "Yes, I consent"},
            {"id": f"recording_consent|{session['id']}|no", "title": "No, don't record"},
        ],
    )


async def respond_to_recording_consent(
    ctx: AppContext, agent_ctx: AgentContext, session_id: str, consent: bool
) -> dict[str, Any]:
    """The customer's answer to _request_recording_consent — this is what
    actually finalizes the booking (Calendar event + Meet link +
    notifications), which was held pending this answer. Records the
    decision either way; declining doesn't block the booking, it just
    means the vet shouldn't record."""
    client = ctx.supabase
    session = _get_session(client, session_id)
    if not session:
        return {"success": False, "error": "session_not_found"}

    recording_consent = "given" if consent else "declined"
    client.table("doctor_sessions").update({"recording_consent": recording_consent}).eq("id", session_id).execute()
    session["recording_consent"] = recording_consent

    start = datetime.fromisoformat(session["preferred_time"])
    if start.tzinfo is None:
        start = start.replace(tzinfo=IST)
    end = start + timedelta(minutes=SESSION_DURATION_MINUTES)
    return await _finalize_booking(ctx, session, session["doctor_phone"], start, end)


async def _finalize_booking(ctx: AppContext, session: dict[str, Any], doctor_phone: str, start: datetime, end: datetime) -> dict[str, Any]:
    client = ctx.supabase
    customer_profile = _get_profile(client, session["profile_id"])
    pet = _get_pet(client, session.get("pet_id"))
    doctor_profile = client.table("profiles").select("*").eq("phone_number", doctor_phone).limit(1).execute().data
    doctor_name = doctor_profile[0].get("full_name", "your vet") if doctor_profile else "your vet"
    doctor_email = doctor_profile[0].get("email") if doctor_profile else None
    customer_name = customer_profile.get("full_name", "the pet owner") if customer_profile else "the pet owner"
    pet_name = pet.get("name", "the pet") if pet else "the pet"

    household = get_pet_member_contacts(client, session.get("pet_id")) if session.get("pet_id") else []
    attendee_emails = [c["email"] for c in household if c.get("email")]
    if doctor_email:
        attendee_emails.append(doctor_email)

    # A calendar_event_id already on the session means this is a RESCHEDULE of an
    # already-confirmed booking (initial negotiation never sets it before this point)
    # — move the same event instead of creating a duplicate, so the Meet link and
    # any calendar invite the parties already have keeps working.
    existing_event_id = session.get("calendar_event_id")
    is_reschedule = bool(existing_event_id)

    if not is_reschedule:
        # The slot list this was picked from can be stale by now -- real time
        # passed waiting on payment/recording-consent/doctor-accept, with nothing
        # re-verifying the window was still free at the actual moment of booking
        # (confirmed: two different sessions picking the same open slot could both
        # reach here and double-book the same doctor+time). A reschedule doesn't
        # need this: it's moving a slot that's already this session's own event.
        if not await is_window_free(ctx.settings, start, end):
            client.table("doctor_sessions").update(
                {"status": "negotiating", "awaiting_from": "customer_time_input", "preferred_time": None}
            ).eq("id", session["id"]).execute()
            if customer_profile:
                await ctx.whatsapp.send_text(
                    customer_profile["phone_number"],
                    "Sorry, that slot was just taken by someone else. Let's find you another time.",
                )
            return {
                "success": False,
                "error": "slot_conflict",
                "session_id": session["id"],
                "message": "That slot was just taken by someone else — ask the customer to pick another time (call select_doctor for fresh availability).",
            }

    event = None
    if existing_event_id:
        try:
            event = await google_calendar.update_event_time(ctx.settings, existing_event_id, start, end, attendees=attendee_emails)
        except Exception:
            logger.exception("Failed to update existing calendar event %s, falling back to a new one", existing_event_id)

    if event is None:
        event = await google_calendar.create_event_with_meet(
            ctx.settings,
            summary=f"PetPulse consult: {customer_name} & {doctor_name} ({pet_name})",
            description=session.get("case_summary", ""),
            start=start,
            end=end,
            attendees=attendee_emails,
        )
    meet_link = google_calendar.extract_meet_link(event)

    client.table("doctor_sessions").update(
        {
            "status": "accepted",
            "awaiting_from": None,
            "doctor_phone": doctor_phone,
            "preferred_time": start.isoformat(),
            "meet_link": meet_link,
            "calendar_event_id": event.get("event_id", existing_event_id),
        }
    ).eq("id", session["id"]).execute()

    when_text = start.strftime("%a %d %b, %I:%M %p IST")
    verb = "rescheduled to" if is_reschedule else "confirmed for"
    await _notify_household(
        ctx,
        session,
        f"Your session with {doctor_name} for {pet_name} is {verb} {when_text}."
        + (f"\nJoin here: {meet_link}" if meet_link else ""),
    )
    doctor_header = "🔄 *Session rescheduled*" if is_reschedule else "📋 *New session assigned to you*"
    try:
        await ctx.whatsapp.send_text(
            doctor_phone, _format_doctor_message(client, session, doctor_header, when_text=when_text, meet_link=meet_link)
        )
    except Exception:
        # Calendar event + DB row are already committed by this point — a
        # delivery failure to the doctor is a notification problem, not a
        # reason to report the booking itself as failed.
        logger.exception("Failed to notify doctor %s for session %s", doctor_phone, session["id"])

    return {
        "success": True,
        "mode": "rescheduled" if is_reschedule else "booked",
        "session_id": session["id"],
        "when": when_text,
        "meet_link": meet_link,
    }


async def propose_time(
    ctx: AppContext, agent_ctx: AgentContext, session_id: str, proposed_time: str, proposed_by: str
) -> dict[str, Any]:
    client = ctx.supabase
    session = _get_session(client, session_id)
    if not session:
        return {"success": False, "error": "session_not_found"}

    normalized_time = _normalize_to_ist(proposed_time)
    if not normalized_time:
        return {"success": False, "error": "invalid_time", "message": "Couldn't understand that time — ask for a specific date and time."}

    awaiting_from = "doctor_time_input" if proposed_by == "customer" else "customer_time_input"
    client.table("doctor_sessions").update(
        {"status": "negotiating", "awaiting_from": awaiting_from, "preferred_time": normalized_time}
    ).eq("id", session_id).execute()

    start = datetime.fromisoformat(normalized_time)
    when_text = start.strftime("%a %d %b, %I:%M %p IST")

    customer_profile = _get_profile(client, session["profile_id"])
    recipient = session["doctor_phone"] if proposed_by == "customer" else (customer_profile["phone_number"] if customer_profile else None)
    if not recipient:
        return {"success": False, "error": "recipient_unknown"}

    await ctx.whatsapp.send_interactive_buttons(
        recipient,
        f"Proposed time: {when_text}. Does this work?",
        [
            {"id": f"accept_session:{session_id}", "title": "Accept"},
            {"id": f"decline_session:{session_id}", "title": "Decline"},
            {"id": f"retime_session:{session_id}", "title": "Suggest a time"},
        ],
    )
    return {"success": True, "mode": "proposal_sent", "session_id": session_id, "when": when_text}


async def respond_to_time_proposal(ctx: AppContext, agent_ctx: AgentContext, session_id: str, decision: str) -> dict[str, Any]:
    client = ctx.supabase
    session = _get_session(client, session_id)
    if not session:
        return {"success": False, "error": "session_not_found"}

    if decision == "accept":
        if not session.get("preferred_time"):
            return {"success": False, "error": "no_proposed_time"}
        start = datetime.fromisoformat(session["preferred_time"])
        if start.tzinfo is None:
            start = start.replace(tzinfo=IST)
        # A reschedule of an already-paid session (calendar_event_id already exists)
        # skips payment — the fee was already spent the first time this session was
        # confirmed; only a session that's never been paid for gates on payment.
        if session.get("payment_status") == "paid" or session.get("calendar_event_id"):
            end = start + timedelta(minutes=SESSION_DURATION_MINUTES)
            return await _finalize_booking(ctx, session, session["doctor_phone"], start, end)

        return await _request_payment(ctx, session, session["doctor_phone"], start)

    if decision == "decline":
        return await decline_session(ctx, agent_ctx, session_id)

    return {"success": True, "mode": "awaiting_retime", "message": "Ask them for a specific time, then call propose_time with it."}


async def accept_session(ctx: AppContext, agent_ctx: AgentContext, session_id: str) -> dict[str, Any]:
    return await respond_to_time_proposal(ctx, agent_ctx, session_id, "accept")


async def decline_session(ctx: AppContext, agent_ctx: AgentContext, session_id: str) -> dict[str, Any]:
    client = ctx.supabase
    session = _get_session(client, session_id)
    if not session:
        return {"success": False, "error": "session_not_found"}

    client.table("doctor_sessions").update({"status": "declined", "awaiting_from": None}).eq("id", session_id).execute()

    customer_profile = _get_profile(client, session["profile_id"])
    if customer_profile:
        await ctx.whatsapp.send_text(customer_profile["phone_number"], "Unfortunately that session request was declined. Would you like to pick another vet or time?")
    if session.get("doctor_phone") and session["doctor_phone"] != "pending_doctor_choice":
        await ctx.whatsapp.send_text(session["doctor_phone"], "You declined this session request.")

    return {"success": True, "mode": "declined", "session_id": session_id}


async def cancel_session(ctx: AppContext, agent_ctx: AgentContext, session_id: str) -> dict[str, Any]:
    client = ctx.supabase
    session = _get_session(client, session_id)
    if not session:
        return {"success": False, "error": "session_not_found"}

    client.table("doctor_sessions").update({"status": "cancelled", "awaiting_from": None}).eq("id", session_id).execute()

    if session.get("calendar_event_id"):
        try:
            await google_calendar.delete_event(ctx.settings, session["calendar_event_id"])
        except Exception:
            # DB status is already the source of truth for booking state — a
            # stale calendar entry left behind isn't a functional failure for
            # the customer/vet, just log it for cleanup.
            logger.exception("Failed to delete calendar event %s for cancelled session %s", session["calendar_event_id"], session_id)

    when_text = None
    if session.get("preferred_time"):
        cancelled_start = datetime.fromisoformat(session["preferred_time"])
        if cancelled_start.tzinfo is None:
            cancelled_start = cancelled_start.replace(tzinfo=IST)
        when_text = cancelled_start.strftime("%a %d %b, %I:%M %p IST")

    if agent_ctx.role == "customer":
        if session.get("doctor_phone") and session["doctor_phone"] != "pending_doctor_choice":
            message = _format_doctor_message(client, session, "❌ *Session cancelled*", when_text=when_text)
            await ctx.whatsapp.send_text(session["doctor_phone"], message)
    else:
        await _notify_household(ctx, session, "This session has been cancelled.")

    return {"success": True, "mode": "cancelled", "session_id": session_id}


async def reschedule_session(ctx: AppContext, agent_ctx: AgentContext, session_id: str, new_time: str) -> dict[str, Any]:
    """Single, clearly-named entry point for "I need to reschedule" —
    works whether the session is still being negotiated or already fully
    confirmed with a real calendar event and Meet link. Proposes the new
    time to the OTHER party for confirmation (same as initial booking
    negotiation); the calendar event only actually moves once they accept
    (see _finalize_booking, which reuses the existing event/Meet link
    instead of creating a duplicate)."""
    proposed_by = "vet" if agent_ctx.role == "vet" else "customer"
    return await propose_time(ctx, agent_ctx, session_id=session_id, proposed_time=new_time, proposed_by=proposed_by)


def _doctor_name(client, doctor_phone: str | None) -> str:
    if not doctor_phone:
        return "your vet"
    rows = client.table("profiles").select("full_name").eq("phone_number", doctor_phone).limit(1).execute().data
    return (rows[0].get("full_name") if rows else None) or "your vet"


def _get_doctor_profile(client, doctor_phone: str | None) -> dict[str, Any]:
    """Full profile row for the treating vet, used to fill in the editable
    per-doctor fields on the prescription PDF (name/qualification/contact).
    `registration_number` isn't a column on `profiles` yet -- this degrades
    to "Not on file" on the PDF until that column is added, rather than
    erroring."""
    if not doctor_phone:
        return {}
    rows = client.table("profiles").select("*").eq("phone_number", doctor_phone).limit(1).execute().data
    return rows[0] if rows else {}


async def mark_session_done(ctx: AppContext, agent_ctx: AgentContext, session_id: str) -> dict[str, Any]:
    client = ctx.supabase
    session = _get_session(client, session_id)
    if not session:
        return {"success": False, "error": "session_not_found"}
    if session.get("status") == "completed":
        # Idempotent: a re-tap, a re-sent "done"/similar message, or the vet
        # saying it twice must never re-notify the customer that the session
        # ended a second time.
        return {"success": True, "mode": "already_completed", "session_id": session_id}

    client.table("doctor_sessions").update(
        {"status": "completed", "awaiting_from": "doctor_prescription", "completed_at": datetime.now(tz=IST).isoformat()}
    ).eq("id", session_id).execute()

    pet = _get_pet(client, session.get("pet_id"))
    pet_name = (pet.get("name") if pet else None) or "your pet"
    doctor_name = _doctor_name(client, session.get("doctor_phone"))

    await _notify_household(
        ctx,
        session,
        f"Your session with {doctor_name} for {pet_name} has ended. They'll share a summary and any "
        "prescription/treatment notes shortly.",
    )

    return {"success": True, "mode": "session_completed", "instruction_to_llm": "Ask the vet for the prescription/treatment notes, then call file_prescription."}


async def file_prescription(
    ctx: AppContext,
    agent_ctx: AgentContext,
    session_id: str,
    medications: str,
    treatment_plan: str = "",
    follow_up_required: bool = False,
    follow_up_date: str = "",
) -> dict[str, Any]:
    """Files the vet's prescription/treatment notes, but does NOT deliver
    them yet — the customer picks text or PDF first (see
    deliver_prescription), so nothing is pushed in a format they didn't
    ask for. medications/treatment_plan are held on the session itself
    (pending_medications/pending_treatment_plan) rather than re-derived
    later from medical_records, so delivery is unambiguous even if the pet
    has other historical records. follow_up_required/follow_up_date come
    straight from whatever the vet said in the same message (see the tool
    description in registry.py) -- there's no separate WhatsApp question
    for it."""
    client = ctx.supabase
    session = _get_session(client, session_id)
    if not session:
        return {"success": False, "error": "session_not_found"}

    normalized_follow_up_date = _normalize_to_date(follow_up_date) if follow_up_required else None

    client.table("medical_records").insert(
        {
            "pet_id": session.get("pet_id"),
            "profile_id": session["profile_id"],
            "visit_date": datetime.now(tz=IST).date().isoformat(),
            "chief_complaint": session.get("case_summary"),
            "medications": medications,
            "treatment_plan": treatment_plan,
            "created_by": "vet",
        }
    ).execute()

    client.table("doctor_sessions").update(
        {
            "awaiting_from": "prescription_format_choice",
            "pending_medications": medications,
            "pending_treatment_plan": treatment_plan,
            "follow_up_required": follow_up_required,
            "follow_up_date": normalized_follow_up_date,
        }
    ).eq("id", session_id).execute()

    pet = _get_pet(client, session.get("pet_id"))
    pet_name = (pet.get("name") if pet else None) or "your pet"
    doctor_name = _doctor_name(client, session.get("doctor_phone"))

    for phone in _household_phones(client, session):
        try:
            await ctx.whatsapp.send_interactive_buttons(
                phone,
                f"Your prescription for {pet_name} from {doctor_name} is ready. How would you like to receive it?",
                [
                    {"id": f"prescription_format|{session_id}|text", "title": "Text message"},
                    {"id": f"prescription_format|{session_id}|pdf", "title": "PDF document"},
                ],
            )
        except Exception:
            logger.exception("Failed to ask household member %s for prescription format for session %s", phone, session_id)

    return {
        "success": True,
        "mode": "prescription_filed_awaiting_format_choice",
        "session_id": session_id,
        "instruction_to_llm": "The prescription was filed and the customer was already asked (via WhatsApp "
        "buttons) whether they want it as text or PDF — just confirm briefly to the vet that it's filed and "
        "the customer's been asked, don't restate the medications/treatment plan yourself.",
    }


async def deliver_prescription(ctx: AppContext, agent_ctx: AgentContext, session_id: str, format: str) -> dict[str, Any]:
    """The customer's answer to file_prescription's format-choice question —
    delivers the already-filed medications/treatment plan in exactly the
    format they picked, nothing more."""
    client = ctx.supabase
    session = _get_session(client, session_id)
    if not session:
        return {"success": False, "error": "session_not_found"}
    if session.get("awaiting_from") != "prescription_format_choice":
        # Idempotent: a re-tap or duplicate reply after it's already been
        # delivered must never re-send the prescription a second time.
        return {"success": True, "mode": "already_delivered", "session_id": session_id}

    fmt = (format or "").strip().lower()
    if fmt not in ("text", "pdf"):
        return {"success": False, "error": "invalid_format", "message": "format must be 'text' or 'pdf'"}

    medications = session.get("pending_medications") or ""
    treatment_plan = session.get("pending_treatment_plan") or ""
    pet = _get_pet(client, session.get("pet_id"))
    pet_name = (pet.get("name") if pet else None) or "your pet"
    doctor_name = _doctor_name(client, session.get("doctor_phone"))

    if fmt == "text":
        message = await _format_prescription_message(ctx, pet_name, doctor_name, session.get("case_summary", ""), medications, treatment_plan)
        await _notify_household(ctx, session, message)
    else:
        await _send_prescription_pdf(ctx, session, pet_name, doctor_name, medications, treatment_plan)

    client.table("doctor_sessions").update({"awaiting_from": None}).eq("id", session_id).execute()

    return {
        "success": True,
        "mode": "prescription_delivered",
        "format": fmt,
        "session_id": session_id,
        "instruction_to_llm": "The prescription was already sent to the customer in their chosen format — "
        "just confirm briefly, don't restate its contents.",
    }


async def _send_prescription_pdf(
    ctx: AppContext, session: dict[str, Any], pet_name: str, doctor_name: str, medications: str, treatment_plan: str
) -> None:
    """Best-effort, on top of the text summary that's already gone out and
    is the guaranteed record: renders a PDF copy, files it the same way any
    other document is (so it also shows up via get_pet_passport later), and
    sends it as a WhatsApp attachment. A failure anywhere here (rendering,
    storage, or the send itself) must never undo the text summary already
    sent or fail the whole tool call — log and move on."""
    client = ctx.supabase
    try:
        doctor_profile = _get_doctor_profile(client, session.get("doctor_phone"))
        pdf_bytes = build_prescription_pdf(
            pet_name=pet_name,
            doctor_name=doctor_name,
            doctor_qualification=doctor_profile.get("qualification") or "",
            doctor_registration_number=doctor_profile.get("registration_number") or "",
            date_str=datetime.now(tz=IST).strftime("%d %b %Y"),
            reason=session.get("case_summary", ""),
            medications=medications,
            treatment_plan=treatment_plan,
        )
        bucket = LABEL_TO_BUCKET.get("Prescription", DEFAULT_BUCKET)
        object_path = f"{session.get('pet_id') or 'unassigned'}/{int(datetime.now(tz=IST).timestamp())}-prescription.pdf"
        upload_to_storage(client, bucket, object_path, pdf_bytes, "application/pdf")
        signed_url = sign_storage_url(client, bucket, object_path)

        client.table("documents").insert(
            {
                "pet_id": session.get("pet_id"),
                "profile_id": session["profile_id"],
                "document_name": f"Prescription - {pet_name}",
                "document_type": "Prescription",
                "storage_path": f"{bucket}/{object_path}",
                "mime_type": "application/pdf",
                "is_verified": True,
            }
        ).execute()

        for phone in _household_phones(client, session):
            try:
                await ctx.whatsapp.send_document(
                    phone, signed_url, f"{pet_name}_prescription.pdf", f"Prescription for {pet_name} from {doctor_name}"
                )
            except Exception:
                logger.exception("Failed to send prescription PDF to %s for session %s", phone, session.get("id"))
    except Exception:
        logger.exception("Failed to generate/file prescription PDF for session %s", session.get("id"))


async def list_my_appointments(ctx: AppContext, agent_ctx: AgentContext) -> dict[str, Any]:
    client = ctx.supabase
    phone = agent_ctx.profile["phone_number"]
    now_iso = datetime.now(tz=IST).isoformat()
    rows = (
        client.table("doctor_sessions")
        .select("id, status, preferred_time, meet_link, case_summary, profile_id, pet_id")
        .eq("doctor_phone", phone)
        .eq("status", "accepted")
        .gte("preferred_time", now_iso)
        .order("preferred_time")
        .limit(20)
        .execute()
        .data
        or []
    )

    appointments = []
    for row in rows:
        customer = _get_profile(client, row["profile_id"])
        pet = _get_pet(client, row.get("pet_id"))
        appointments.append(
            {
                "session_id": row["id"],
                "when": row["preferred_time"],
                "customer_name": customer.get("full_name") if customer else "Unknown",
                "pet_name": pet.get("name") if pet else "Unknown",
                "case_summary": row.get("case_summary"),
                "meet_link": row.get("meet_link"),
            }
        )

    return {"success": True, "count": len(appointments), "appointments": appointments}


async def relay_to_customer(ctx: AppContext, agent_ctx: AgentContext, session_id: str, message: str) -> dict[str, Any]:
    client = ctx.supabase
    session = _get_session(client, session_id)
    if not session:
        return {"success": False, "error": "session_not_found"}

    pet = _get_pet(client, session.get("pet_id"))
    pet_name = (pet.get("name") if pet else None) or "your pet"
    doctor_name = _doctor_name(client, session.get("doctor_phone"))

    await _notify_household(ctx, session, f"*Message from {doctor_name}* (re: {pet_name}):\n{message}")
    return {"success": True, "mode": "relayed", "session_id": session_id}


async def relay_to_doctor(ctx: AppContext, agent_ctx: AgentContext, session_id: str, message: str) -> dict[str, Any]:
    """Symmetric to relay_to_customer — lets the customer's own words reach
    the vet verbatim (with attribution), for cases like a follow-up
    question or extra detail that doesn't fit any structured action."""
    client = ctx.supabase
    session = _get_session(client, session_id)
    if not session:
        return {"success": False, "error": "session_not_found"}
    if not session.get("doctor_phone") or session["doctor_phone"] == "pending_doctor_choice":
        return {"success": False, "error": "no_doctor_assigned"}

    customer_profile = _get_profile(client, session["profile_id"])
    customer_name = (customer_profile.get("full_name") if customer_profile else None) or "the pet owner"
    pet = _get_pet(client, session.get("pet_id"))
    pet_name = (pet.get("name") if pet else None) or "their pet"

    await ctx.whatsapp.send_text(session["doctor_phone"], f"*Message from {customer_name}* (re: {pet_name}):\n{message}")
    return {"success": True, "mode": "relayed", "session_id": session_id}
