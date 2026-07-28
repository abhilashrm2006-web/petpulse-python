"""The single tool-calling loop (spec §1: maxIterations 10) that replaces
n8n's whole `Is Doctor?` / classify / switch apparatus. Every inbound
message — customer or vet — goes through build_context -> this loop ->
whatever tools the model picks. Button taps are just another line in the
turn context (see system_prompt.build_turn_context), not a routed action."""

import json
import logging
import time
from typing import Any

from app.agent import memory
from app.agent.language import SUPPORTED_LANGUAGES, detect_regional_language
from app.agent.registry import get_tool_fn, get_tool_schemas, is_tool_allowed_for_role, is_tool_allowed_for_tier
from app.agent.tools.greeting import is_bare_greeting, send_welcome_character
from app.agent.system_prompt import build_system_prompt, build_turn_context
from app.deps import AppContext
from app.ingestion.context import AgentContext, mark_onboarding_complete_if_needed
from app.ingestion.webhook import ExtractedMessage
from app.integrations.openai_client import chat_with_tools, synthesize_speech
from app.integrations.supabase_client import sign_storage_url, upload_to_storage

logger = logging.getLogger(__name__)

VOICE_REPLIES_BUCKET = "voice-replies"

# Tool result `mode` values whose side effect (an already-sent WhatsApp
# message) should suppress the agent's own reply text — ports `AI Response
# Ready`'s double-check that request_doctor_session / start_new_pet_parent_guide
# don't leave a stray duplicate message behind (spec §2). "booked"/"rescheduled"
# added after a real reported bug: _finalize_booking already messages BOTH
# parties directly (unconditionally, regardless of who initiated it), so
# without suppression the agent composed a second, redundant confirmation
# on top — the customer saw the same date/time and Meet link twice, split
# across multiple WhatsApp bubbles. "payment_requested" is the same class of
# bug waiting to happen: _request_payment already sent the payment link.
# "prescription_delivered" is that same class again: deliver_prescription
# already sends the text/PDF directly, so without suppression the customer
# gets the prescription itself plus a redundant "here it is!" bubble on top.
# "declined" found via a repeated-message audit: decline_session unconditionally
# messages BOTH the customer and the doctor directly (not just "the other
# party" — whichever of the two is the current actor gets messaged by the tool
# too), so without suppression that actor got the tool's own message plus a
# redundant agent reply on top of it.
SELF_MESSAGING_MODES = {
    "doctor_catalogue_sent", "new_parent_guide_sent", "slot_list_sent", "booked", "rescheduled", "payment_requested",
    "prescription_delivered", "subscriber_consult_confirmed", "subscription_started", "declined",
}

SUBSCRIBER_ONLY_MESSAGE = (
    "That's a Subscriber feature — subscribe for just ₹399/month to unlock it, including a free vet consult "
    "every month! Want me to send you the subscribe link?"
)


def _tool_call_to_message_dict(message: Any) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": message.content or "",
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in (message.tool_calls or [])
        ],
    }


async def run_agent_turn(
    ctx: AppContext,
    agent_ctx: AgentContext,
    extracted: ExtractedMessage,
    media_context: str = "",
    document_filing_status: str = "",
) -> str:
    client = ctx.supabase
    phone = agent_ctx.profile["phone_number"]
    role = agent_ctx.role

    # Deterministic, not an LLM tool call -- a model tool-call-based version of this
    # didn't reliably fire turn to turn (confirmed live). Runs before the model ever
    # sees the turn, so the mascot welcome sticker always goes out for a real bare
    # greeting regardless of what the model does; it still writes its own warm text
    # separately (see GREETING_RULE), so this never duplicates that content.
    if role != "vet" and extracted.message_type == "text" and is_bare_greeting(extracted.text):
        try:
            await send_welcome_character(ctx, agent_ctx)
        except Exception:
            logger.exception("Failed to send welcome character sticker for phone=%s", phone)

    # A voice-note reply is only attempted for a Subscriber's own voice note, and
    # only while the feature is enabled (settings.voice_replies_enabled, an ops-level
    # kill-switch) -- detected BEFORE the system prompt is built so the model is
    # firmly told to reply in that language this turn, guaranteeing the text and
    # the audio synthesized from it afterward are in the same language (see
    # build_system_prompt).
    voice_reply_language_code: str | None = None
    if role != "vet" and agent_ctx.is_subscriber and extracted.message_type == "audio" and ctx.settings.voice_replies_enabled:
        try:
            detected = await detect_regional_language(ctx.openai, ctx.settings, media_context)
            if detected != "en":
                voice_reply_language_code = detected
        except Exception:
            logger.exception("Voice-reply language detection failed for phone=%s", phone)

    voice_reply_language_name = SUPPORTED_LANGUAGES.get(voice_reply_language_code) if voice_reply_language_code else None
    system_prompt = build_system_prompt(
        role,
        is_subscriber=agent_ctx.is_subscriber,
        voice_reply_language=voice_reply_language_name,
    )
    turn_context = build_turn_context(agent_ctx, extracted, media_context, document_filing_status)
    # Persistent memory is a Subscriber perk for customers (vets always keep it --
    # this isn't a customer tier feature for them). A Free customer's chat starts
    # fresh every turn; a Subscriber's is scoped to the active pet so a multi-pet
    # account doesn't blend context across pets.
    memory_enabled = role == "vet" or agent_ctx.is_subscriber
    memory_pet_id = agent_ctx.active_pet["id"] if (memory_enabled and role != "vet" and agent_ctx.active_pet) else None
    history = memory.load_chat_history(client, phone, pet_id=memory_pet_id) if memory_enabled else []
    tools = get_tool_schemas(role)

    messages: list[dict[str, Any]] = (
        [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": turn_context}]
    )

    self_messaged = False
    final_text = ""

    for _ in range(ctx.settings.openai_agent_max_iterations):
        response = await chat_with_tools(ctx.openai, ctx.settings, messages, tools)
        message = response.choices[0].message

        if not message.tool_calls:
            final_text = message.content or ""
            break

        messages.append(_tool_call_to_message_dict(message))

        for tool_call in message.tool_calls:
            name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            if not is_tool_allowed_for_role(name, role):
                result: dict[str, Any] = {"success": False, "error": "tool_not_available_for_role"}
            elif not is_tool_allowed_for_tier(name, role, agent_ctx.is_subscriber):
                result = {"success": False, "error": "subscriber_only_feature", "message": SUBSCRIBER_ONLY_MESSAGE}
            else:
                fn = get_tool_fn(name)
                try:
                    result = await fn(ctx, agent_ctx, **args)
                except Exception as exc:
                    logger.exception("Tool %s failed", name)
                    result = {"success": False, "error": "tool_error", "message": str(exc)}

            if isinstance(result, dict) and result.get("mode") in SELF_MESSAGING_MODES:
                self_messaged = True

            messages.append(
                {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result, default=str)}
            )
    else:
        logger.warning("Agent hit max iterations for phone=%s", phone)

    if self_messaged:
        final_text = ""

    sent_chunks: list[tuple[str, str]] = []
    if final_text.strip():
        sent_chunks = await ctx.whatsapp.send_reply_and_chunk(phone, final_text)

    # Best-effort voice reply on top of the text reply already sent above -- never
    # lets a TTS/storage/send failure affect the turn, since the customer already
    # has their answer in text either way.
    if voice_reply_language_code and final_text.strip():
        try:
            audio_bytes = await synthesize_speech(ctx.openai, ctx.settings, final_text, language=voice_reply_language_name)
            if audio_bytes:
                object_path = f"{agent_ctx.profile['id']}/{int(time.time())}.mp3"
                upload_to_storage(client, VOICE_REPLIES_BUCKET, object_path, audio_bytes, "audio/mpeg")
                signed_url = sign_storage_url(client, VOICE_REPLIES_BUCKET, object_path)
                await ctx.whatsapp.send_audio(phone, signed_url)
        except Exception:
            logger.exception("Failed to send voice reply for phone=%s", phone)

    # From here on the customer's reply has ALREADY been sent — everything below is
    # bookkeeping (chat history, long-term memory, conversation logs, onboarding flag).
    # Each step is independently wrapped: a bug in one (e.g. the messages-insert
    # NOT-NULL violation found live in testing) must never silently prevent an
    # unrelated one (e.g. mark_onboarding_complete_if_needed) from running too.
    if memory_enabled:
        try:
            memory.append_turn(client, phone, extracted.text, final_text, pet_id=memory_pet_id)
        except Exception:
            logger.exception("Failed to append chat history for phone=%s", phone)

        try:
            await memory.extract_and_update_memory(
                ctx.openai, ctx.settings, client, agent_ctx.profile["id"], extracted.text, final_text, pet_id=memory_pet_id
            )
        except Exception:
            logger.exception("Failed to update long-term memory for profile=%s", agent_ctx.profile["id"])

    try:
        # A fresh `conversations` row per inbound message, matching n8n exactly (spec §2) —
        # continuity lives only in the chat-memory table above, not at the DB level here.
        conversation = client.table("conversations").insert(
            {
                "profile_id": agent_ctx.profile["id"],
                "pet_id": agent_ctx.active_pet["id"] if agent_ctx.active_pet else None,
                "channel": "whatsapp",
                "status": "active",
            }
        ).execute().data[0]

        # Each row's metadata.wamid is how a later "reply"/quote from the customer
        # (WhatsApp's context.id on their next message) gets resolved back to actual
        # text -- see app.ingestion.context._resolve_quoted_message. One row per
        # WhatsApp bubble, not one combined row for the whole turn, since a customer
        # can tap-reply to any single bubble of a multi-chunk answer specifically.
        inbound_row = {
            "conversation_id": conversation["id"],
            "profile_id": agent_ctx.profile["id"],
            "sender_type": "vet" if role == "vet" else "user",
            "content": extracted.text,
            "message_type": extracted.message_type if extracted.message_type in
                ("text", "image", "audio", "video", "document", "location") else "text",
            "metadata": {"wamid": extracted.message_id} if extracted.message_id else {},
        }
        outbound_rows = (
            [
                {
                    "conversation_id": conversation["id"],
                    "profile_id": agent_ctx.profile["id"],
                    "sender_type": "assistant",
                    "content": chunk_text,
                    "message_type": "text",
                    "metadata": {"wamid": wamid},
                }
                for wamid, chunk_text in sent_chunks
            ]
            if sent_chunks
            else [
                {
                    "conversation_id": conversation["id"],
                    "profile_id": agent_ctx.profile["id"],
                    "sender_type": "assistant",
                    "content": final_text,
                    "message_type": "text",
                    "metadata": {},
                }
            ]
        )
        client.table("messages").insert([inbound_row, *outbound_rows]).execute()
    except Exception:
        logger.exception("Failed to log conversation/messages for profile=%s", agent_ctx.profile["id"])

    if role != "vet":
        try:
            mark_onboarding_complete_if_needed(client, agent_ctx.profile, agent_ctx.onboarding)
        except Exception:
            logger.exception("Failed to mark onboarding complete for profile=%s", agent_ctx.profile["id"])

    return final_text
