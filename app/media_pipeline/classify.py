"""Ports `Doc - LLM Classify Type` + `Classify & Route Document` (spec §2).

Produces classification *context* for the agent — the actual filing
decision now belongs to the `file_document` tool call the agent makes
(architecture note in the plan: this is the one deliberate behavior shift
from n8n's always-on silent auto-file).
"""

import json
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI

from app.config import Settings
from app.integrations.openai_client import json_completion
from app.utils.pet_resolution import resolve_pet

# Exact CHECK-constraint labels for documents.document_type (spec §7 / memory: supabase-check-constraints)
VALID_DOCUMENT_TYPES = [
    "Prescription", "Lab Report", "X-Ray", "MRI", "CT Scan", "Ultrasound",
    "Vaccination Certificate", "Insurance", "Invoice", "Discharge Summary",
    "Medical Certificate", "Photo", "Video", "Other",
]

LABEL_TO_BUCKET = {
    "Prescription": "prescriptions",
    "X-Ray": "xrays",
    "Vaccination Certificate": "vaccination-certificates",
}
DEFAULT_BUCKET = "medical-documents"

LABEL_TO_KEY = {
    "Prescription": "prescription",
    "Lab Report": "lab_report",
    "X-Ray": "xray",
    "MRI": "mri",
    "CT Scan": "ct_scan",
    "Ultrasound": "ultrasound",
    "Vaccination Certificate": "vaccination_certificate",
    "Insurance": "insurance",
    "Invoice": "invoice",
    "Discharge Summary": "discharge_summary",
    "Medical Certificate": "medical_certificate",
    "Photo": "photo",
    "Video": "video",
    "Other": "other",
}

CLASSIFY_SYSTEM_PROMPT = f"""Classify a pet-related image/document based on the owner's caption and a \
vision-model transcription/description of it. Respond with strict JSON:
{{"document_type": one of {VALID_DOCUMENT_TYPES}, "is_medical_document": bool, "handwritten": bool, \
"pet_name": string or null, "source": "caption"|"vision"}}

Rules:
- The owner's own caption ALWAYS takes precedence over the vision reading when they conflict.
- KCI/breed-registration papers are "Other", never "Vaccination Certificate".
- Casual pet photos are "Photo" with is_medical_document=false.
- If a document contains any handwritten field, handwritten=true.
- If the prompt states a "currently active pet", return that pet's name UNLESS the caption or the media \
itself clearly names or shows a different pet on the account -- don't re-guess identity from visual \
similarity alone when a specific pet is already the conversation's context.
"""


@dataclass
class DocumentClassification:
    document_type: str
    is_medical_document: bool
    handwritten: bool
    target_pet: dict[str, Any] | None
    is_verified: bool
    bucket: str
    routing_key: str


async def classify_document(
    client: AsyncOpenAI,
    settings: Settings,
    kind: str,
    mime_type: str,
    analysis_text: str,
    caption: str,
    pets: list[dict[str, Any]],
    active_pet_name: str | None = None,
) -> DocumentClassification:
    pet_names = [p.get("name") for p in pets if p.get("name")]
    # Confirmed live bug: without this hint the pet_name guess is re-derived
    # from scratch every time, purely from the media + a flat list of every
    # pet name on the account -- ignoring whichever pet the rest of the
    # conversation is already tracking as "active" (see
    # app.ingestion.context.AgentContext.active_pet). Biasing toward it
    # (not forcing it) fixes the "misidentified him in the video" class of
    # bug without breaking the real "customer sends a photo of a DIFFERENT,
    # named pet" case.
    active_pet_hint = (
        f"\nthe conversation's currently active pet right now is {active_pet_name!r} -- prefer this pet "
        "unless the caption or vision analysis clearly names a different pet on the account"
        if active_pet_name
        else ""
    )
    user_prompt = (
        f"kind={kind}\nmime={mime_type}\ncaption={caption or '(none)'}\n"
        f"known pet names on this account: {pet_names}{active_pet_hint}\n\nvision analysis:\n{analysis_text}"
    )
    raw = await json_completion(client, settings, CLASSIFY_SYSTEM_PROMPT, user_prompt, reasoning_effort="low")
    try:
        verdict = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        verdict = {}

    document_type = verdict.get("document_type")
    if document_type not in VALID_DOCUMENT_TYPES:
        document_type = "Other"

    is_medical_document = bool(verdict.get("is_medical_document", False))
    handwritten = bool(verdict.get("handwritten", False))

    resolution = resolve_pet(pets, pet_name=verdict.get("pet_name") or active_pet_name, auto_resolve_single=True)
    target_pet = resolution.pet

    return DocumentClassification(
        document_type=document_type,
        is_medical_document=is_medical_document,
        handwritten=handwritten,
        target_pet=target_pet,
        is_verified=is_medical_document and not handwritten,
        bucket=LABEL_TO_BUCKET.get(document_type, DEFAULT_BUCKET),
        routing_key=LABEL_TO_KEY.get(document_type, "other"),
    )
