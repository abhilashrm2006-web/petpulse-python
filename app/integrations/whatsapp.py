"""WhatsApp Cloud API (Meta Graph API) client — ports spec §6.

All sends go through raw HTTP to the Graph API, matching how the live n8n
system does the vast majority of its sends (the native n8n whatsApp node was
just a thin wrapper over the same endpoint for plain text).
"""

import asyncio
import base64
import logging

import httpx

from app.config import Settings
from app.utils.formatting import split_into_chunks, to_whatsapp_markdown

logger = logging.getLogger(__name__)

CHUNK_DELAY_SECONDS = 0.6
SEND_MAX_ATTEMPTS = 3
SEND_RETRY_BACKOFF_SECONDS = 1.0


class WhatsAppClient:
    def __init__(self, settings: Settings, http: httpx.AsyncClient):
        self._settings = settings
        self._http = http
        self._base = f"https://graph.facebook.com/{settings.whatsapp_api_version}/{settings.whatsapp_phone_number_id}"
        self._headers = {"Authorization": f"Bearer {settings.whatsapp_access_token}"}

    async def _post(self, payload: dict) -> dict:
        """Confirmed live: a booking's doctor-assignment notice (case details
        + Meet link) went missing with nothing but a caught-and-logged
        exception to show for it — a transient Graph API hiccup on that one
        send, gone for good with no retry. Retries only network-level
        failures and 5xx responses, exactly like the Calendar Bridge client;
        a 4xx (invalid number, expired 24h messaging window, bad payload) is
        never transient, so those raise immediately instead of wasting
        attempts."""
        last_exc: Exception | None = None
        for attempt in range(1, SEND_MAX_ATTEMPTS + 1):
            try:
                resp = await self._http.post(f"{self._base}/messages", headers=self._headers, json=payload)
                if resp.is_error:
                    logger.error("WhatsApp Graph API %s error for %s: %s", resp.status_code, payload.get("type"), resp.text)
                resp.raise_for_status()
                return resp.json()
            except httpx.TransportError as exc:
                last_exc = exc
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code < 500:
                    raise
                last_exc = exc

            if attempt < SEND_MAX_ATTEMPTS:
                logger.warning("WhatsApp send retrying (attempt %d/%d) after: %s", attempt, SEND_MAX_ATTEMPTS, last_exc)
                await asyncio.sleep(SEND_RETRY_BACKOFF_SECONDS * attempt)
        raise last_exc

    async def send_text(self, to: str, body: str) -> dict:
        return await self._post({"messaging_product": "whatsapp", "to": to, "type": "text", "text": {"body": body}})

    async def send_template(self, to: str, template_name: str, language_code: str, body_params: list[str]) -> dict:
        """Sends a pre-approved WhatsApp message template -- the only
        reliably deliverable message type outside the 24h customer-service
        session window (a free-form send_text can fail there per WhatsApp
        policy, even though it isn't always rejected in practice). Each
        entry in body_params fills the template's {{1}}, {{2}}, ... body
        placeholders in order; the template itself is configured/approved
        in Meta Business Manager, not defined here."""
        return await self._post(
            {
                "messaging_product": "whatsapp",
                "to": to,
                "type": "template",
                "template": {
                    "name": template_name,
                    "language": {"code": language_code},
                    "components": [
                        {
                            "type": "body",
                            "parameters": [{"type": "text", "text": p} for p in body_params],
                        }
                    ],
                },
            }
        )

    async def send_reply_and_chunk(self, to: str, text: str) -> list[tuple[str, str]]:
        """Formats + chunks a long agent reply and sends each chunk with a
        throttling delay, matching `Format WhatsApp Response` -> `Split
        Response Into Chunks` -> `Chunk Send Loop` (spec §2/§6).

        Returns each chunk's (wamid, chunk_text) so the caller can persist
        the mapping -- WhatsApp's "reply"/quote feature lets a customer tap
        any one bubble and quote it later (`context.id` on their next
        message), and without this the bot has no way to look up what text
        was actually in the bubble being quoted."""
        formatted = to_whatsapp_markdown(text)
        chunks = split_into_chunks(formatted)
        sent: list[tuple[str, str]] = []
        for i, chunk in enumerate(chunks):
            result = await self.send_text(to, chunk)
            messages = result.get("messages") or []
            if messages and messages[0].get("id"):
                sent.append((messages[0]["id"], chunk))
            if i < len(chunks) - 1:
                await asyncio.sleep(CHUNK_DELAY_SECONDS)
        return sent

    async def send_interactive_list(
        self,
        to: str,
        header: str,
        body: str,
        button_label: str,
        sections: list[dict],
    ) -> dict:
        return await self._post(
            {
                "messaging_product": "whatsapp",
                "to": to,
                "type": "interactive",
                "interactive": {
                    "type": "list",
                    "header": {"type": "text", "text": header},
                    "body": {"text": body},
                    "action": {"button": button_label, "sections": sections},
                },
            }
        )

    async def send_interactive_buttons(self, to: str, body: str, buttons: list[dict]) -> dict:
        """`buttons`: list of {"id": ..., "title": ...}, max 3."""
        return await self._post(
            {
                "messaging_product": "whatsapp",
                "to": to,
                "type": "interactive",
                "interactive": {
                    "type": "button",
                    "body": {"text": body},
                    "action": {
                        "buttons": [
                            {"type": "reply", "reply": {"id": b["id"], "title": b["title"]}} for b in buttons[:3]
                        ]
                    },
                },
            }
        )

    async def send_flow(
        self,
        to: str,
        flow_id: str,
        header: str,
        body: str,
        cta: str,
        flow_token: str,
        screen: str = "WELCOME",
        footer: str = "",
    ) -> dict:
        """Opens a WhatsApp Flow (a native multi-field form) — `flow_cta` is
        required by the Graph API despite not being documented in most
        examples, or the send fails with error 131008. `flow_token` should
        be something you can trace back to this conversation/profile when
        the completed submission arrives on the webhook."""
        interactive: dict = {
            "type": "flow",
            "header": {"type": "text", "text": header},
            "body": {"text": body},
            "action": {
                "name": "flow",
                "parameters": {
                    "flow_message_version": "3",
                    "flow_id": flow_id,
                    "flow_cta": cta,
                    "flow_token": flow_token,
                    "flow_action": "navigate",
                    "flow_action_payload": {"screen": screen},
                },
            },
        }
        if footer:
            interactive["footer"] = {"text": footer}
        return await self._post({"messaging_product": "whatsapp", "to": to, "type": "interactive", "interactive": interactive})

    async def send_image(self, to: str, link: str, caption: str = "") -> dict:
        return await self._post(
            {"messaging_product": "whatsapp", "to": to, "type": "image", "image": {"link": link, "caption": caption}}
        )

    async def send_audio(self, to: str, link: str) -> dict:
        # WhatsApp's audio message object has no caption field, unlike image/document.
        return await self._post({"messaging_product": "whatsapp", "to": to, "type": "audio", "audio": {"link": link}})

    async def send_video(self, to: str, link: str, caption: str = "") -> dict:
        return await self._post(
            {"messaging_product": "whatsapp", "to": to, "type": "video", "video": {"link": link, "caption": caption}}
        )

    async def send_sticker(self, to: str, link: str) -> dict:
        # WhatsApp's sticker object has no caption field, like audio.
        return await self._post({"messaging_product": "whatsapp", "to": to, "type": "sticker", "sticker": {"link": link}})

    async def send_document(self, to: str, link: str, filename: str, caption: str = "") -> dict:
        return await self._post(
            {
                "messaging_product": "whatsapp",
                "to": to,
                "type": "document",
                "document": {"link": link, "filename": filename, "caption": caption},
            }
        )

    async def get_media_url(self, media_id: str) -> str:
        resp = await self._http.get(
            f"https://graph.facebook.com/{self._settings.whatsapp_api_version}/{media_id}",
            headers=self._headers,
        )
        resp.raise_for_status()
        return resp.json()["url"]

    async def download_media_base64(self, media_id: str) -> tuple[str, str]:
        """Returns (base64_data, mime_type)."""
        url = await self.get_media_url(media_id)
        resp = await self._http.get(url, headers=self._headers)
        resp.raise_for_status()
        mime = resp.headers.get("content-type", "application/octet-stream")
        return base64.b64encode(resp.content).decode(), mime

    async def download_media_bytes(self, media_id: str) -> tuple[bytes, str]:
        url = await self.get_media_url(media_id)
        resp = await self._http.get(url, headers=self._headers)
        resp.raise_for_status()
        mime = resp.headers.get("content-type", "application/octet-stream")
        return resp.content, mime
