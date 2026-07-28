"""Customer-initiated chat-history reset. Deliberately narrow: clears the
bot's own conversational memory (the WhatsApp-message log it uses as
context, plus derived long-term memory facts) so the customer gets a fresh
start, but never touches their actual account/pet/medical records -- those
are real business data (pets, vaccinations, documents, bookings) a customer
asking to "delete my profile" in a WhatsApp chat almost never actually
means to erase permanently, and silently doing so would be irreversible.
The confirmation and completion messages say exactly this, honestly --
never claim an account/profile deletion that isn't actually happening."""

import logging
from typing import Any

from app.deps import AppContext
from app.ingestion.context import AgentContext

logger = logging.getLogger(__name__)

DELETION_CONFIRM_PROMPT = (
    "Just to confirm: this will clear your chat history with Pulsy so we start fresh — it will NOT affect "
    "your saved pet details, medical records, documents, or bookings. Are you sure?"
)

DELETION_DONE_MESSAGE = (
    "Done — your chat history has been cleared. Your pet and account info is safe and unchanged, and I'm "
    "ready to start fresh whenever you are!"
)

DELETION_DECLINED_MESSAGE = (
    "No problem, nothing's changed! Mind sharing what made you consider clearing your chat — it really "
    "helps us improve."
)


async def request_data_deletion(ctx: AppContext, agent_ctx: AgentContext) -> dict[str, Any]:
    phone = agent_ctx.profile["phone_number"]
    await ctx.whatsapp.send_interactive_buttons(
        phone,
        DELETION_CONFIRM_PROMPT,
        [
            {"id": "delete_chat|yes", "title": "Yes, clear it"},
            {"id": "delete_chat|no", "title": "No, keep it"},
        ],
    )
    return {"success": True, "mode": "deletion_confirmation_sent"}


async def respond_to_deletion_confirmation(ctx: AppContext, agent_ctx: AgentContext, confirm: bool) -> dict[str, Any]:
    client = ctx.supabase
    phone = agent_ctx.profile["phone_number"]

    if not confirm:
        await ctx.whatsapp.send_text(phone, DELETION_DECLINED_MESSAGE)
        return {
            "success": True,
            "mode": "deletion_declined",
            "instruction_to_llm": "Already asked for a reason via WhatsApp — if their NEXT message states one, "
            "call record_deletion_feedback with it, then thank them briefly. Don't ask again yourself.",
        }

    # Chat history (n8n_chat_history_petpulse) is keyed by phone_number as
    # session_id (see app/agent/memory.py load_chat_history/append_turn) --
    # this profile's own conversation only, not any household member's.
    client.table("n8n_chat_history_petpulse").delete().eq("session_id", phone).execute()
    # Long-term memory facts, scoped to this profile (may span multiple pets).
    client.table("memory").delete().eq("profile_id", agent_ctx.profile["id"]).execute()

    await ctx.whatsapp.send_text(phone, DELETION_DONE_MESSAGE)
    return {"success": True, "mode": "deletion_done"}


async def record_deletion_feedback(ctx: AppContext, agent_ctx: AgentContext, reason: str) -> dict[str, Any]:
    """Best-effort: logged as a memory row (memory_type="Feedback") rather
    than a dedicated table, so this doesn't need its own schema migration.
    Never blocks the reply if the write fails -- the customer's feedback not
    being saved shouldn't turn into a broken conversation for them."""
    if not reason.strip():
        return {"success": False, "error": "empty_reason"}
    try:
        ctx.supabase.table("memory").insert(
            {
                "profile_id": agent_ctx.profile["id"],
                "memory_type": "Feedback",
                "title": "Chat-history deletion declined — reason given",
                "memory_text": reason,
                "source": "customer",
            }
        ).execute()
    except Exception:
        logger.exception("Failed to record deletion feedback for profile=%s", agent_ctx.profile["id"])
    return {"success": True, "mode": "feedback_recorded"}
