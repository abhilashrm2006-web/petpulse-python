"""Ports the Postgres Chat Memory node (spec §4) and the long-term
`memory` upsert (spec §2 `Extract Memory Candidates` -> `Create/Update
Long-Term Memory`). Same table names as the live n8n system
(`n8n_chat_history_petpulse`, `memory`) so conversation continuity isn't
lost by switching backends."""

import json
from typing import Any

from openai import AsyncOpenAI
from supabase import Client

from app.config import Settings
from app.integrations.openai_client import json_completion

CHAT_HISTORY_TABLE = "n8n_chat_history_petpulse"
CONTEXT_WINDOW_LENGTH = 10

MEMORY_EXTRACT_SYSTEM_PROMPT = """From this WhatsApp conversation turn, extract any durable facts worth \
remembering long-term about the pet or owner. Respond with strict JSON — omit any field you don't have a \
confident value for: {"pet_name": string, "species": string, "breed": string, \
"owner_preferences": string, "communication_style": string}. If nothing durable was mentioned, \
respond with {}."""


def _message_to_row(session_id: str, role: str, content: str, pet_id: str | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "session_id": session_id,
        "message": {"type": role, "data": {"content": content, "additional_kwargs": {}, "type": role}},
    }
    if pet_id:
        row["pet_id"] = pet_id
    return row


def load_chat_history(
    client: Client, phone_number: str, pet_id: str | None = None, window: int = CONTEXT_WINDOW_LENGTH
) -> list[dict[str, str]]:
    """pet_id scopes the thread to one pet's own history instead of one
    shared account-wide stream, so each pet on a multi-pet account carries
    its own context rather than blending across pets."""
    query = client.table(CHAT_HISTORY_TABLE).select("*").eq("session_id", phone_number)
    if pet_id:
        query = query.eq("pet_id", pet_id)
    rows = query.order("id", desc=True).limit(window).execute().data or []
    rows.reverse()
    messages = []
    for row in rows:
        msg = row.get("message", {})
        role = "assistant" if msg.get("type") == "ai" else "user"
        content = msg.get("data", {}).get("content", "")
        if content:
            messages.append({"role": role, "content": content})
    return messages


def append_turn(client: Client, phone_number: str, user_text: str, assistant_text: str, pet_id: str | None = None) -> None:
    rows = []
    if user_text:
        rows.append(_message_to_row(phone_number, "human", user_text, pet_id))
    if assistant_text:
        rows.append(_message_to_row(phone_number, "ai", assistant_text, pet_id))
    if rows:
        client.table(CHAT_HISTORY_TABLE).insert(rows).execute()


async def extract_and_update_memory(
    openai_client: AsyncOpenAI,
    settings: Settings,
    client: Client,
    profile_id: str,
    user_text: str,
    assistant_text: str,
    pet_id: str | None = None,
) -> None:
    if not user_text:
        return
    try:
        raw = await json_completion(
            openai_client,
            settings,
            MEMORY_EXTRACT_SYSTEM_PROMPT,
            f"Customer: {user_text}\nAssistant: {assistant_text}",
            reasoning_effort="low",
        )
        fields = json.loads(raw)
    except Exception:
        return
    if not fields:
        return

    # Must be scoped by pet_id, not just profile_id -- a multi-pet account was
    # previously reading/writing a single profile_id-only row
    # regardless of which pet the turn was about, so the second pet discussed
    # would silently overwrite the first pet's durable facts (pet_name, species,
    # breed) with its own. app.ingestion.context._load_memory already expects
    # per-pet rows (it unions a profile_id-only query with a pet_id query), this
    # just makes the write side match that read-side contract.
    query = client.table("memory").select("*").eq("profile_id", profile_id)
    query = query.eq("pet_id", pet_id) if pet_id else query.is_("pet_id", "null")
    existing = query.limit(1).execute().data
    if existing:
        merged = {**{k: existing[0].get(k) for k in fields}, **fields}
        client.table("memory").update(merged).eq("id", existing[0]["id"]).execute()
    else:
        client.table("memory").insert(
            {
                "profile_id": profile_id,
                "pet_id": pet_id,
                "memory_type": "Fact",
                "title": "Profile fact",
                "memory_text": json.dumps(fields),
                "source": "AI",
                **fields,
            }
        ).execute()
