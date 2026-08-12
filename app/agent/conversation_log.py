"""Shared conversation/message logging, factored out of
app.agent.orchestrator.run_agent_turn so app.ingestion.registration (the
deterministic wizard, which returns before run_agent_turn ever runs — see
app/main.py) can log the SAME shape of row for a wizard-handled turn.
Root cause this fixes: 59/106 profiles stuck mid-onboarding had zero
conversations/messages rows at all, making it impossible to tell "never
replied" from "replied but nothing logged it" (2026-08 root-cause
analysis, item #2)."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def log_conversation_turn(
    client,
    *,
    profile_id: str,
    pet_id: str | None,
    sender_type: str,
    inbound_text: str,
    inbound_message_type: str,
    inbound_wamid: str | None,
    outbound_texts_with_wamid: list[tuple[str | None, str]],
) -> None:
    """Best-effort — never raises. A fresh `conversations` row per inbound
    message (matching n8n exactly, spec §2); one `messages` row per
    WhatsApp bubble (inbound + each outbound chunk), so a later tap-reply
    can resolve back to the exact bubble it quoted."""
    try:
        conversation = (
            client.table("conversations")
            .insert({"profile_id": profile_id, "pet_id": pet_id, "channel": "whatsapp", "status": "active"})
            .execute()
            .data[0]
        )

        inbound_row: dict[str, Any] = {
            "conversation_id": conversation["id"],
            "profile_id": profile_id,
            "sender_type": sender_type,
            "content": inbound_text,
            "message_type": inbound_message_type if inbound_message_type in
                ("text", "image", "audio", "video", "document", "location") else "text",
            "metadata": {"wamid": inbound_wamid} if inbound_wamid else {},
        }
        outbound_rows = [
            {
                "conversation_id": conversation["id"],
                "profile_id": profile_id,
                "sender_type": "assistant",
                "content": chunk_text,
                "message_type": "text",
                "metadata": {"wamid": wamid} if wamid else {},
            }
            for wamid, chunk_text in outbound_texts_with_wamid
        ]
        client.table("messages").insert([inbound_row, *outbound_rows]).execute()
    except Exception:
        logger.exception("Failed to log conversation/messages for profile=%s", profile_id)
