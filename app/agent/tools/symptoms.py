"""Ports `kIy0W1dzrBegxsFd` — Symptom Checker & Triage / `check_symptoms`
(spec §3.3). The hard-coded keyword safety overrides sit ON TOP of the LLM
call, deliberately not left to model judgement — this is the one tool
where "the LLM decides" is intentionally bounded by a safety net."""

import json
import re
from typing import Any

from app.deps import AppContext
from app.ingestion.context import AgentContext
from app.integrations.openai_client import json_completion
from app.utils.pet_resolution import resolve_pet

TRIAGE_SYSTEM_PROMPT = """You are a veterinary triage assistant. Be conservative — round severity up when \
uncertain. Never diagnose or prescribe. Respond with strict JSON:
{"severity": 1-5, "severity_label": string, "requires_emergency_care": bool, "red_flags": [string], \
"likely_categories": [string], "recommendation": string, "reasoning": string, \
"first_aid_checklist": [string]}
first_aid_checklist is a short ordered list of concrete first-aid/at-home steps appropriate right now \
(e.g. "Keep them calm and limit movement", "Do not give any food or water"), not a repeat of the \
recommendation sentence."""

SEVERITY_COLOR = {1: "Green", 2: "Green", 3: "Yellow", 4: "Red", 5: "Red"}

RED_FLAG_KEYWORDS = [
    r"bloat", r"distend\w* abdomen", r"can'?t breathe", r"gasping", r"blue gums", r"pale gums",
    r"white gums", r"seizure", r"collaps", r"unconscious", r"antifreeze", r"rat poison",
    r"can'?t urinate", r"hit by (a )?car", r"fell? from", r"uncontrolled bleeding", r"heatstroke",
    r"protruding bone", r"exposed bone", r"eye (popped|out of socket)", r"sudden blindness",
]

INGESTION_VERBS = [r"\bate\b", r"\bswallowed\b", r"\bingested\b", r"\blicked\b", r"\bchewed\b"]
HARMFUL_SUBSTANCES = [
    r"chocolate", r"xylitol", r"grapes?", r"raisins?", r"onion", r"garlic", r"antifreeze",
    r"rat poison", r"pesticide", r"medication", r"pills?", r"human medicine",
]

# Deterministic display string, independent of media source (text/photo/video/
# audio) — kept out of the LLM's hands so the rating always reads the same
# way instead of being reworded turn to turn.
SEVERITY_EMOJI = {1: "🟢", 2: "🟢", 3: "🟡", 4: "🟠", 5: "🔴"}


def _severity_display(severity: int, severity_label: str) -> str:
    emoji = SEVERITY_EMOJI.get(severity, "🟡")
    label = severity_label or "Assessment"
    return f"{emoji} {label} ({severity}/5)"




def _keyword_hit(patterns: list[str], text: str) -> list[str]:
    hits = []
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            hits.append(pattern)
    return hits


def _apply_safety_override(symptoms_text: str, verdict: dict[str, Any]) -> dict[str, Any]:
    text = symptoms_text.lower()
    red_flag_hits = _keyword_hit(RED_FLAG_KEYWORDS, text)
    ingestion_hit = bool(_keyword_hit(INGESTION_VERBS, text))
    substance_hit = _keyword_hit(HARMFUL_SUBSTANCES, text)
    poisoning_escalation = ingestion_hit and bool(substance_hit)

    if red_flag_hits or poisoning_escalation:
        verdict["severity"] = 5
        verdict["severity_label"] = "Emergency"
        verdict["requires_emergency_care"] = True
        # setdefault only fills in a MISSING key -- an LLM emitting a
        # present-but-null "red_flags" (valid JSON, plausible when it has
        # nothing to report) leaves it None, and .append() on None would
        # crash exactly on the poisoning/ingestion path this override exists
        # to catch.
        verdict["red_flags"] = verdict.get("red_flags") or []
        if poisoning_escalation:
            verdict["red_flags"].append(f"possible ingestion of harmful substance: {substance_hit}")
    return verdict


async def check_symptoms(
    ctx: AppContext,
    agent_ctx: AgentContext,
    pet_id: str = "",
    pet_name: str = "",
    species: str = "",
    symptoms: str = "",
) -> dict[str, Any]:
    resolution = resolve_pet(agent_ctx.pets, pet_id=pet_id, pet_name=pet_name, auto_resolve_single=True)
    if resolution.ambiguous:
        return {"success": False, "error": "ambiguous_pet", "message": "Which pet has these symptoms?"}
    pet = resolution.pet or {}
    species = species or pet.get("species", "")

    user_prompt = f"Species: {species}\nPet: {pet_name or pet.get('name', 'unknown')}\nSymptoms: {symptoms}"

    assessment_failed = False
    try:
        # xhigh, not medium -- this is the single safety-critical judgment
        # call in the whole system (severity/red-flags/emergency-care
        # decision). Confirmed live: no meaningful latency difference vs
        # medium/high for this workload (~6s either way), so there's no
        # real tradeoff against using the deepest reasoning available.
        raw = await json_completion(ctx.openai, ctx.settings, TRIAGE_SYSTEM_PROMPT, user_prompt, reasoning_effort="xhigh")
        verdict = json.loads(raw)
        if "severity" not in verdict:
            raise ValueError("missing severity")
    except Exception:
        assessment_failed = True
        verdict = {
            "severity": 3,
            "severity_label": "Caution",
            "requires_emergency_care": False,
            "red_flags": [],
            "likely_categories": [],
            "recommendation": "Unable to complete a full assessment — please monitor closely and consult a vet if symptoms persist or worsen.",
            "reasoning": "assessment_failed",
        }

    verdict = _apply_safety_override(symptoms, verdict)

    severity = verdict.get("severity", 3)
    severity_display = _severity_display(severity, verdict.get("severity_label", ""))
    if pet.get("id"):
        ctx.supabase.table("health_logs").insert(
            {
                "pet_id": pet["id"],
                "profile_id": agent_ctx.profile["id"],
                "ai_risk_score": severity * 20,
                "ai_observation": verdict.get("recommendation", ""),
                "symptoms": symptoms,
                "notes": "Auto-logged by AI Symptom Checker and Triage tool",
            }
        ).execute()

    color = SEVERITY_COLOR.get(severity, "Yellow")

    return {
        "success": True,
        "severity": severity,
        "severity_label": verdict.get("severity_label"),
        "severity_display": severity_display,
        "severity_color": color,
        "requires_emergency_care": verdict.get("requires_emergency_care", False),
        "assessment_failed": assessment_failed,
        "red_flags": verdict.get("red_flags", []),
        "likely_categories": verdict.get("likely_categories", []),
        "recommendation": verdict.get("recommendation", ""),
        "first_aid_checklist": verdict.get("first_aid_checklist", []),
        "message": verdict.get("recommendation", ""),
        "action": "emergency" if verdict.get("requires_emergency_care") else "advise",
    }
