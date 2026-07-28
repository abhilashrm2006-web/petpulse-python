"""Sends the Pulsy mascot welcome image -- called once per conversation
when the customer's message is a bare greeting (see GREETING_RULE in
system_prompt.py), so the first moment of a chat feels like meeting a
character saying hello, not a generic bot line. The model still writes its
own warm greeting text separately (this tool only sends the image, no
caption/text of its own), so no self-messaging suppression is needed --
unlike request_doctor_session and friends, this never duplicates content
the agent's own reply also says."""

from typing import Any

from app.deps import AppContext
from app.ingestion.context import AgentContext


async def send_welcome_character(ctx: AppContext, agent_ctx: AgentContext) -> dict[str, Any]:
    phone = agent_ctx.profile["phone_number"]
    await ctx.whatsapp.send_image(phone, ctx.settings.pulsy_welcome_image_url)
    return {"success": True, "mode": "welcome_character_sent"}
