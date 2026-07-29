"""Extracts vet-onboarding fields from a doctor's Drive folder of documents
(degree certificates, veterinary-council registration certificates/ID
cards) -- one consolidated vision call across all of that folder's images
(PDFs rendered to their first page first) so the model can cross-reference
documents against each other (e.g. confirm the same name/qualification
appears on both a degree certificate and a registration certificate)
rather than extracting each file in isolation. Always best-effort: a field
that isn't visible anywhere comes back null, never guessed -- the admin
review step (see app/admin/routes.py doctor-drafts endpoints) is where
gaps get filled in and everything gets verified before a real account is
created."""

import base64
import json
import logging

from openai import AsyncOpenAI

from app.config import Settings
from app.integrations.media_processing import render_pdf_first_page
from app.integrations.openai_client import multi_image_json_completion

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM_PROMPT = """You are extracting veterinarian onboarding details from scanned documents \
(degree certificates, veterinary council registration certificates, ID cards) for an admin to review before \
creating this doctor's account. You will be shown every document filed in this doctor's folder together — \
cross-reference them (the same name/qualification should appear on more than one) rather than trusting a \
single document alone.

Respond with strict JSON, using exactly these keys — use null for any field not clearly visible in any \
document, never guess or infer a value that isn't actually shown:
{
  "full_name": string or null — the doctor's full name, prefixed with "Dr." if not already,
  "phone_number": string or null — include country code if shown; if only a 10-digit Indian number is shown "
      with no country code, prefix it with 91,
  "email": string or null,
  "qualification": string or null — e.g. "B.V.Sc & A.H.", include any postgraduate qualification too if shown,
  "registration_number": string or null — prefer a state veterinary council registration number (e.g. "
      "\"TSVC Reg. No.\") over a provisional/university one if both are present; include the issuing "
      "council's name if shown,
  "specialization": string or null — only if explicitly stated; do not infer from qualification alone,
  "gender": string or null — "male" or "female", only if explicitly stated or unambiguous from a title/salutation,
  "date_of_birth": string or null — ISO format YYYY-MM-DD,
  "city": string or null — city/town from an address shown,
  "area": string or null — more specific locality/neighborhood from an address shown, if distinguishable from city,
  "notes": string or null — one short sentence flagging anything an admin should double-check (e.g. conflicting "
      "names/numbers across documents, an expired-looking certificate, illegible handwriting) — omit if nothing "
      "stands out
}"""


async def extract_doctor_fields(
    openai_client: AsyncOpenAI, settings: Settings, files: list[tuple[bytes, str, str]]
) -> dict:
    """`files`: list of (raw_bytes, mime_type, filename). Returns the parsed
    extraction dict (see EXTRACTION_SYSTEM_PROMPT's schema) plus which
    source filenames were actually usable. Never raises -- a folder whose
    documents can't be read at all just comes back with everything null,
    left for the admin to fill in by hand rather than blocking the sync."""
    images: list[tuple[str, str]] = []
    used_filenames: list[str] = []
    for raw_bytes, mime_type, filename in files:
        try:
            if mime_type == "application/pdf":
                page_bytes = await render_pdf_first_page(raw_bytes)
                images.append((base64.b64encode(page_bytes).decode(), "image/jpeg"))
            elif mime_type.startswith("image/"):
                images.append((base64.b64encode(raw_bytes).decode(), mime_type))
            else:
                continue
            used_filenames.append(filename)
        except Exception:
            logger.exception("Failed to prepare doctor document %s for extraction", filename)

    if not images:
        return {"notes": "No readable image/PDF documents found in this folder.", "_source_filenames": []}

    try:
        raw = await multi_image_json_completion(
            openai_client, settings, EXTRACTION_SYSTEM_PROMPT,
            f"Extract onboarding details from these {len(images)} document(s).", images,
        )
        fields = json.loads(raw)
    except Exception:
        logger.exception("Doctor document extraction failed")
        fields = {"notes": "Automatic extraction failed — fill in fields manually."}

    fields["_source_filenames"] = used_filenames
    return fields
