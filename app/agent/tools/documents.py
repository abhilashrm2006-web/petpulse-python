"""Ports `HndYcT5EWackv3ul` — Pet Document Locker (`send_pet_document` /
`get_pet_passport`, spec §3.6) plus the new `file_document` tool, which
replaces n8n's always-on silent `Classify & Route Document` ->
`Should File Document?` -> insert pipeline (spec §2) with an explicit,
agent-invoked action — the one deliberate behavior shift called out in the
plan."""

import json
import secrets
from datetime import date
from typing import Any

from app.deps import AppContext
from app.ingestion.context import AgentContext
from app.integrations.openai_client import json_completion
from app.integrations.supabase_client import sign_storage_url, upload_to_storage
from app.media_pipeline.classify import LABEL_TO_KEY, VALID_DOCUMENT_TYPES
from app.utils.pet_resolution import resolve_pet

# Free tier: manual document upload works, just capped -- Subscribers get
# unlimited storage plus search and a shareable link (see search_documents /
# get_shareable_link below).
FREE_DOCUMENT_CAP = 5

VACCINATION_ROUTE_MAP = {
    "subcutaneous": "Subcutaneous", "sc": "Subcutaneous", "sq": "Subcutaneous",
    "intramuscular": "Intramuscular", "im": "Intramuscular",
    "oral": "Oral", "po": "Oral",
    "intranasal": "Intranasal", "in": "Intranasal",
}

MEDREC_EXTRACT_SYSTEM_PROMPT = """Extract structured medical data from a vet document analysis. Respond with \
strict JSON:
{"record_kind": "vaccination"|"clinical"|"none", "visit_date": "YYYY-MM-DD" or null, \
"chief_complaint": string or null, "diagnosis": string or null, "treatment_plan": string or null, \
"medications": string or null, \
"vaccinations": [{"vaccine_name": string, "manufacturer": string or null, "batch_number": string or null, \
"date_administered": "YYYY-MM-DD", "next_due_date": "YYYY-MM-DD" or null, \
"next_due_source": "document"|"inferred", "route": string or null}]}
If next_due_date isn't stated, infer it from standard intervals (puppy/kitten: 3-4 weeks; adult booster: 1 year) \
and mark next_due_source="inferred". Omit vaccination rows missing vaccine_name or date_administered."""


def _format_vaccination_line(vax: dict[str, Any], today: str) -> tuple[str, bool]:
    """Returns (display line, is_overdue). Pure/testable: includes every
    field that's actually on file (manufacturer, batch/lot number, next-due
    date) rather than just name + date, since those are on the record and
    the owner may need them (e.g. for a boarding kennel or travel)."""
    overdue = bool(vax.get("next_due_date") and vax["next_due_date"] < today)
    flag = " (OVERDUE)" if overdue else ""
    detail = f"- {vax['vaccine_name']} — {vax['date_administered']}"
    if vax.get("manufacturer"):
        detail += f" | {vax['manufacturer']}"
    if vax.get("batch_number"):
        detail += f" | Batch/Lot: {vax['batch_number']}"
    if vax.get("next_due_date"):
        detail += f" | Next due: {vax['next_due_date']}"
    return f"{detail}{flag}", overdue


def _ambiguous_pet_result(resolution, fallback_message: str) -> dict[str, Any]:
    """Never guess which same-named pet was meant — hand the candidates
    (each already carrying owner_name/owner_phone, see
    supabase_client.get_pets_for_profile) back to the LLM so IT resolves
    the ambiguity from conversation context (e.g. an owner's name someone
    mentioned) and re-calls with an exact pet_id, rather than code trying
    to fuzzy-match an owner name itself."""
    if resolution.reason == "ambiguous_pet_name" and resolution.candidates:
        return {
            "success": False,
            "error": "ambiguous_pet",
            "candidates": resolution.candidates,
            "instruction_to_llm": (
                "More than one pet has this name, belonging to DIFFERENT owners (see each candidate's "
                "owner_name/owner_phone). Do not guess — use the conversation (e.g. an owner's name someone "
                "mentioned) to work out which one is meant and call this tool again with that exact pet_id. "
                "If you genuinely can't tell, ask which owner's pet is meant before doing anything with it."
            ),
        }
    return {"success": False, "error": "ambiguous_pet", "message": fallback_message}


async def _send_document_files(ctx: AppContext, phone: str, docs: list[dict[str, Any]], pet_name: str) -> int:
    """Signs + sends each document's storage file as a WhatsApp attachment.
    Shared by send_pet_document and get_pet_passport so the passport can
    hand over the actual certificate files, not just a text summary."""
    for doc in docs:
        bucket, _, object_path = doc["storage_path"].partition("/")
        signed_url = sign_storage_url(ctx.supabase, bucket, object_path)
        caption = f"{doc['document_type']} for {pet_name}"
        if (doc.get("mime_type") or "").startswith("image/"):
            await ctx.whatsapp.send_image(phone, signed_url, caption)
        else:
            await ctx.whatsapp.send_document(phone, signed_url, doc["document_name"], caption)
    return len(docs)


async def send_pet_document(
    ctx: AppContext, agent_ctx: AgentContext, pet_id: str = "", pet_name: str = "", document_type: str = ""
) -> dict[str, Any]:
    resolution = resolve_pet(agent_ctx.pets, pet_id=pet_id, pet_name=pet_name, auto_resolve_single=True)
    if resolution.ambiguous:
        return _ambiguous_pet_result(resolution, "Which pet's documents?")
    if not resolution.pet:
        return {"success": False, "error": "no_pet_on_file"}

    query = ctx.supabase.table("documents").select("*").eq("pet_id", resolution.pet["id"])
    if document_type:
        query = query.ilike("document_type", f"%{document_type}%")
    docs = query.order("uploaded_at", desc=True).limit(3).execute().data or []

    if not docs:
        return {"success": True, "mode": "no_documents", "message": "No matching documents on file for this pet."}

    phone = agent_ctx.profile["phone_number"]
    count = await _send_document_files(ctx, phone, docs, resolution.pet["name"])

    return {
        "success": True,
        "mode": "documents_sent",
        "count": count,
        "instruction_to_llm": "Documents were already sent as WhatsApp attachments. Confirm briefly — do NOT restate or summarise their contents.",
    }


def build_full_passport_text(client, pet: dict[str, Any]) -> tuple[str, int]:
    """The full Subscriber-grade passport: vaccinations (manufacturer/batch-lot/
    next-due) plus recent medical records. Shared by get_pet_passport (WhatsApp)
    and the public GET /passport/{token} route (app/main.py) so both surfaces
    render identically from one source of truth."""
    vaccinations = (
        client.table("vaccinations").select("*").eq("pet_id", pet["id"])
        .order("date_administered", desc=True).execute().data or []
    )
    medical_records = (
        client.table("medical_records").select("*").eq("pet_id", pet["id"])
        .order("visit_date", desc=True).limit(8).execute().data or []
    )

    today = date.today().isoformat()
    lines = [f"*{pet['name']}'s Pet Passport*", f"Species/Breed: {pet.get('species', '?')} / {pet.get('breed', '?')}"]
    if pet.get("age"):
        lines.append(f"Age: {pet['age']}")
    if pet.get("weight"):
        lines.append(f"Weight: {pet['weight']} {pet.get('weight_unit', 'kg')}")
    if pet.get("date_of_birth"):
        lines.append(f"DOB: {pet['date_of_birth']}")
    if pet.get("microchip_number"):
        lines.append(f"Microchip: {pet['microchip_number']}")

    overdue_count = 0
    lines.append("\n*Vaccinations:*")
    for vax in vaccinations:
        line, overdue = _format_vaccination_line(vax, today)
        if overdue:
            overdue_count += 1
        lines.append(line)

    lines.append("\n*Recent medical records:*")
    for rec in medical_records:
        summary = (rec.get("chief_complaint") or rec.get("diagnosis") or "visit")[:140]
        lines.append(f"- {rec['visit_date']}: {summary}")
    older = len(medical_records) - 8 if len(medical_records) > 8 else 0
    if older > 0:
        lines.append(f"...and {older} older")

    return "\n".join(lines), overdue_count


def build_basic_due_date_list(client, pet: dict[str, Any]) -> str:
    """Free tier: due dates only, no manufacturer/batch-lot detail, no
    medical records, no shareable link -- just enough to know what's due."""
    vaccinations = (
        client.table("vaccinations").select("*").eq("pet_id", pet["id"])
        .order("next_due_date", desc=False).execute().data or []
    )
    today = date.today().isoformat()
    lines = [f"*{pet['name']}'s Vaccination Due Dates*"]
    if not vaccinations:
        lines.append("Nothing on file yet.")
    for vax in vaccinations:
        if not vax.get("next_due_date"):
            continue
        overdue = vax["next_due_date"] < today
        flag = " (OVERDUE)" if overdue else ""
        lines.append(f"- {vax['vaccine_name']}: due {vax['next_due_date']}{flag}")
    return "\n".join(lines)


async def get_pet_passport(
    ctx: AppContext, agent_ctx: AgentContext, pet_id: str = "", pet_name: str = "", send_certificates: bool = True
) -> dict[str, Any]:
    resolution = resolve_pet(agent_ctx.pets, pet_id=pet_id, pet_name=pet_name, auto_resolve_single=True)
    if resolution.ambiguous:
        return _ambiguous_pet_result(resolution, "Which pet's passport?")
    pet = resolution.pet
    if not pet:
        return {"success": False, "error": "no_pet_on_file"}

    if not agent_ctx.is_subscriber:
        passport_text = build_basic_due_date_list(ctx.supabase, pet)
        return {
            "success": True,
            "mode": "passport",
            "passport_text": passport_text,
            "instruction_to_llm": "Relay passport_text verbatim. This is the Free-tier due-date list — if "
            "the customer wants the full passport (manufacturer/batch details, records, a shareable link), "
            "mention that's a Subscriber feature.",
        }

    passport_text, overdue_count = build_full_passport_text(ctx.supabase, pet)

    files_sent = 0
    if send_certificates:
        cert_docs = (
            ctx.supabase.table("documents").select("*").eq("pet_id", pet["id"])
            .ilike("document_type", "%Vaccination Certificate%")
            .order("uploaded_at", desc=True).limit(5).execute().data or []
        )
        if cert_docs:
            files_sent = await _send_document_files(ctx, agent_ctx.profile["phone_number"], cert_docs, pet["name"])

    instruction = "Relay passport_text verbatim, preserving line breaks. This is ground truth — never contradict it."
    if files_sent:
        instruction += (
            f" {files_sent} vaccination certificate file(s) on file were also just sent as WhatsApp attachments — "
            "mention briefly that you've attached them, don't restate their contents."
        )

    return {
        "success": True,
        "mode": "passport",
        "passport_text": passport_text,
        "overdue_vaccinations": overdue_count,
        "certificate_files_sent": files_sent,
        "instruction_to_llm": instruction,
    }


async def file_document(
    ctx: AppContext,
    agent_ctx: AgentContext,
    pet_id: str = "",
    pet_name: str = "",
    document_type: str = "",
) -> dict[str, Any]:
    media = getattr(agent_ctx, "pending_media", None)
    if not media or not media.document_bytes:
        return {"success": False, "error": "no_pending_media", "message": "There's no document from this turn to file."}

    classification = media.document_classification
    resolved_type = document_type if document_type in VALID_DOCUMENT_TYPES else (
        classification.document_type if classification else "Other"
    )
    target_pet = None
    if pet_id or pet_name:
        resolution = resolve_pet(agent_ctx.pets, pet_id=pet_id, pet_name=pet_name, auto_resolve_single=True)
        if resolution.ambiguous:
            return _ambiguous_pet_result(resolution, "Which pet is this document for?")
        target_pet = resolution.pet
    if not target_pet and classification:
        target_pet = classification.target_pet
    if not target_pet:
        return {"success": False, "error": "ambiguous_pet", "message": "Which pet is this document for?"}

    if not agent_ctx.is_subscriber:
        pet_ids = [p["id"] for p in agent_ctx.pets if p.get("id")]
        existing_count = (
            len(ctx.supabase.table("documents").select("id").in_("pet_id", pet_ids).execute().data or [])
            if pet_ids else 0
        )
        if existing_count >= FREE_DOCUMENT_CAP:
            return {
                "success": False,
                "error": "subscriber_only_feature",
                "message": f"You've reached the {FREE_DOCUMENT_CAP}-document limit on the Free plan — "
                "subscribe for ₹399/month for unlimited storage, full-text search, and one-tap sharing with "
                "your vet, so you always have your pet's complete health record in one place.",
            }

    bucket = classification.bucket if classification else "medical-documents"
    ext = (media.document_mime_type or "").split("/")[-1] or "bin"
    timestamp = int(date.today().strftime("%Y%m%d"))
    object_path = f"{target_pet['id']}/{timestamp}-{LABEL_TO_KEY.get(resolved_type, 'other')}.{ext}"
    storage_path = f"{bucket}/{object_path}"

    upload_to_storage(ctx.supabase, bucket, object_path, media.document_bytes, media.document_mime_type or "application/octet-stream")

    is_verified = classification.is_verified if classification else False
    doc_row = (
        ctx.supabase.table("documents")
        .insert(
            {
                "pet_id": target_pet["id"],
                "profile_id": agent_ctx.profile["id"],
                "document_name": f"{resolved_type} - {target_pet['name']}",
                "document_type": resolved_type,
                "storage_path": storage_path,
                "mime_type": media.document_mime_type,
                "ocr_text": media.media_context,
                "ai_summary": media.media_context[:400],
                "is_verified": is_verified,
            }
        )
        .execute()
    )

    medrec_result = await _extract_and_store_medical_record(ctx, agent_ctx, target_pet, media.media_context, is_verified)

    return {
        "success": True,
        "document_id": doc_row.data[0]["id"],
        "document_type": resolved_type,
        "pet_name": target_pet["name"],
        "is_verified": is_verified,
        "medical_record_extraction": medrec_result,
        "instruction_to_llm": "Document has been filed. You may now confirm it was saved/noted/on file — do not restate its contents unprompted.",
    }


async def _extract_and_store_medical_record(
    ctx: AppContext, agent_ctx: AgentContext, pet: dict[str, Any], analysis_text: str, is_verified: bool
) -> dict[str, Any]:
    try:
        raw = await json_completion(
            ctx.openai, ctx.settings, MEDREC_EXTRACT_SYSTEM_PROMPT, analysis_text, reasoning_effort="low"
        )
        extracted = json.loads(raw)
    except Exception:
        return {"record_kind": "none"}

    kind = extracted.get("record_kind", "none")
    disclaimer = "" if is_verified else " (not vet-verified"

    if kind == "vaccination":
        rows = []
        for vax in extracted.get("vaccinations", []):
            if not vax.get("vaccine_name") or not vax.get("date_administered"):
                continue
            route = VACCINATION_ROUTE_MAP.get((vax.get("route") or "").lower(), "Other")
            notes_parts = []
            if not is_verified:
                notes_parts.append("not vet-verified")
            if vax.get("next_due_source") == "inferred":
                notes_parts.append("next due date inferred, not from document")
            rows.append(
                {
                    "pet_id": pet["id"],
                    "profile_id": agent_ctx.profile["id"],
                    "vaccine_name": vax["vaccine_name"],
                    "manufacturer": vax.get("manufacturer"),
                    "batch_number": vax.get("batch_number"),
                    "date_administered": vax["date_administered"],
                    "next_due_date": vax.get("next_due_date"),
                    "route": route,
                    "notes": "; ".join(notes_parts) or None,
                }
            )
        if rows:
            ctx.supabase.table("vaccinations").insert(rows).execute()
        return {"record_kind": "vaccination", "rows_inserted": len(rows)}

    if kind == "clinical":
        ctx.supabase.table("medical_records").insert(
            {
                "pet_id": pet["id"],
                "profile_id": agent_ctx.profile["id"],
                "visit_date": extracted.get("visit_date") or date.today().isoformat(),
                "chief_complaint": extracted.get("chief_complaint"),
                "diagnosis": extracted.get("diagnosis"),
                "treatment_plan": extracted.get("treatment_plan"),
                "medications": extracted.get("medications"),
                "created_by": "ai",
            }
        ).execute()
        return {"record_kind": "clinical"}

    return {"record_kind": "none"}


async def search_documents(
    ctx: AppContext, agent_ctx: AgentContext, query: str, pet_id: str = "", pet_name: str = ""
) -> dict[str, Any]:
    """Subscriber-only full-text search over filed documents -- reuses the
    ocr_text/ai_summary every document already gets populated with at
    filing time (see file_document/_extract_and_store_medical_record), no
    separate OCR/indexing pipeline needed. Scoped to a specific pet if
    given, otherwise every pet on the account."""
    if pet_id or pet_name:
        resolution = resolve_pet(agent_ctx.pets, pet_id=pet_id, pet_name=pet_name, auto_resolve_single=True)
        if resolution.ambiguous:
            return _ambiguous_pet_result(resolution, "Which pet's documents should I search?")
        pet_ids = [resolution.pet["id"]] if resolution.pet else []
    else:
        pet_ids = [p["id"] for p in agent_ctx.pets if p.get("id")]

    if not pet_ids:
        return {"success": True, "count": 0, "results": [], "message": "No pets on file to search."}

    pattern = f"%{query}%"
    matches = {}
    for column in ("ocr_text", "ai_summary", "document_name"):
        rows = (
            ctx.supabase.table("documents")
            .select("id, document_name, document_type, uploaded_at, pet_id")
            .in_("pet_id", pet_ids)
            .ilike(column, pattern)
            .execute()
            .data
            or []
        )
        for row in rows:
            matches[row["id"]] = row

    results = sorted(matches.values(), key=lambda r: r.get("uploaded_at") or "", reverse=True)[:10]
    return {
        "success": True,
        "count": len(results),
        "results": [
            {"document_id": r["id"], "document_name": r["document_name"], "document_type": r["document_type"]}
            for r in results
        ],
        "message": "Here's what I found." if results else f"No documents matching \"{query}\" on file.",
    }


async def get_shareable_link(ctx: AppContext, agent_ctx: AgentContext, pet_id: str = "", pet_name: str = "") -> dict[str, Any]:
    """Subscriber-only: a stable, public, no-login link to a pet's health
    summary (vaccination passport + recent records) -- satisfies both the
    Vault's "share with vet" and the Vaccination Tracker's "shareable
    passport" asks with one mechanism, since they'd render near-identical
    read-only content anyway. The token itself never changes once
    generated, so re-sharing later returns the same link."""
    resolution = resolve_pet(agent_ctx.pets, pet_id=pet_id, pet_name=pet_name, auto_resolve_single=True)
    if resolution.ambiguous:
        return _ambiguous_pet_result(resolution, "Which pet's link do you want to share?")
    pet = resolution.pet
    if not pet:
        return {"success": False, "error": "no_pet_on_file"}

    token = pet.get("passport_share_token")
    if not token:
        token = secrets.token_urlsafe(24)
        ctx.supabase.table("pets").update({"passport_share_token": token}).eq("id", pet["id"]).execute()
        pet["passport_share_token"] = token

    url = f"{ctx.settings.public_base_url}/passport/{token}"
    return {
        "success": True,
        "mode": "share_link",
        "url": url,
        "instruction_to_llm": f"Share this link with the customer: {url} — anyone with it (a vet, boarding "
        "facility, etc.) can view the summary without logging in. Don't restate the passport contents "
        "yourself here, the link already carries them.",
    }
