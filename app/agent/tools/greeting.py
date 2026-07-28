"""Detects a bare WhatsApp greeting and sends the Pulsy mascot welcome
image for it. Deliberately NOT an LLM tool call: a first version routed
this through the tool-calling loop and the model just didn't reliably call
it turn to turn (confirmed live — some greetings got the image, most
didn't), so this runs as plain deterministic code in the orchestrator
before the model ever sees the turn, guaranteeing it fires every time
without depending on model compliance for something this simple. The
agent still writes its own warm greeting text afterward (see GREETING_RULE
in system_prompt.py) — this only sends the image half."""

import re

from app.deps import AppContext
from app.ingestion.context import AgentContext

# Mirrors GREETING_RULE's "plain greeting with no other content" definition —
# the entire message must be just a greeting word/phrase, nothing else.
_BARE_GREETING_PATTERN = re.compile(
    r"^(hi+|hello+|hey+|yo+|hola|namaste+|namaskar|sup|good\s*(morning|afternoon|evening|day))[\s!.,]*$",
    re.IGNORECASE,
)


def is_bare_greeting(text: str) -> bool:
    return bool(_BARE_GREETING_PATTERN.match((text or "").strip()))


async def send_welcome_character(ctx: AppContext, agent_ctx: AgentContext) -> None:
    phone = agent_ctx.profile["phone_number"]
    await ctx.whatsapp.send_image(phone, ctx.settings.pulsy_welcome_image_url)
