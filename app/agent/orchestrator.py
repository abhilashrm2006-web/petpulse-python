"""The single tool-calling loop (spec §1: maxIterations 10) that replaces
n8n's whole `Is Doctor?` / classify / switch apparatus. Every inbound
message — customer or vet — goes through build_context -> this loop ->
whatever tools the model picks. Button taps are just another line in the
turn context (see system_prompt.build_turn_context), not a routed action."""

import json
import logging
from typing import Any

from app.agent import memory
from app.agent.registry import get_tool_fn, get_tool_schemas, is_tool_allowed_for_role, is_tool_allowed_for_tier
from app.agent.system_prompt import build_system_prompt, build_turn_context
from app.deps import AppContext
from app.ingestion.context import AgentContext, mark_onboarding_complete_if_needed
from app.ingestion.webhook import ExtractedMessage
from app.integrations.openai_client import chat_with_tools

logger = logging.getLogger(__name__)

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
SELF_MESSAGING_MODES = {
    "doctor_catalogue_sent", "new_parent_guide_sent", "slot_list_sent", "booked", "rescheduled", "payment_requested",
    "prescription_delivered", "subscriber_consult_confirmed", "subscription_started",
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

    system_prompt = build_system_prompt(role)
    turn_context = build_turn_context(agent_ctx, extracted, media_context, document_filing_status)
    history = memory.load_chat_history(client, phone)
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

    if final_text.strip():
        await ctx.whatsapp.send_reply_and_chunk(phone, final_text)

    # From here on the customer's reply has ALREADY been sent — everything below is
    # bookkeeping (chat history, long-term memory, conversation logs, onboarding flag).
    # Each step is independently wrapped: a bug in one (e.g. the messages-insert
    # NOT-NULL violation found live in testing) must never silently prevent an
    # unrelated one (e.g. mark_onboarding_complete_if_needed) from running too.
    try:
        memory.append_turn(client, phone, extracted.text, final_text)
    except Exception:
        logger.exception("Failed to append chat history for phone=%s", phone)

    try:
        await memory.extract_and_update_memory(ctx.openai, ctx.settings, client, agent_ctx.profile["id"], extracted.text, final_text)
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

        client.table("messages").insert(
            [
                {
                    "conversation_id": conversation["id"],
                    "profile_id": agent_ctx.profile["id"],
                    "sender_type": "vet" if role == "vet" else "user",
                    "content": extracted.text,
                    "message_type": extracted.message_type if extracted.message_type in
                        ("text", "image", "audio", "video", "document", "location") else "text",
                },
                {
                    "conversation_id": conversation["id"],
                    "profile_id": agent_ctx.profile["id"],
                    "sender_type": "assistant",
                    "content": final_text,
                    "message_type": "text",
                },
            ]
        ).execute()
    except Exception:
        logger.exception("Failed to log conversation/messages for profile=%s", agent_ctx.profile["id"])

    if role != "vet":
        try:
            mark_onboarding_complete_if_needed(client, agent_ctx.profile, agent_ctx.onboarding)
        except Exception:
            logger.exception("Failed to mark onboarding complete for profile=%s", agent_ctx.profile["id"])

    return final_text
