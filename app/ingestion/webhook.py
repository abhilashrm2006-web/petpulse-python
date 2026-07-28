"""Ports `Extract Incoming Message` + `Validate Incoming Data` (spec §2):
pulls the inbound message out of Meta's webhook envelope."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExtractedMessage:
    phone_number: str
    sender_name: str
    message_id: str
    timestamp: str
    message_type: str
    text: str = ""
    image_media_id: str | None = None
    audio_media_id: str | None = None
    document_media_id: str | None = None
    video_media_id: str | None = None
    document_mime_type: str | None = None
    document_filename: str | None = None
    button_reply_id: str | None = None
    button_reply_title: str | None = None
    quoted_wamid: str | None = None
    interactive_type: str | None = None
    flow_response_json: dict[str, Any] | None = None
    latitude: float | None = None
    longitude: float | None = None
    location_text: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def is_valid(self) -> bool:
        return bool(self.phone_number and self.message_id and self.timestamp)


def extract_message(body: dict[str, Any]) -> ExtractedMessage | None:
    try:
        value = body["entry"][0]["changes"][0]["value"]
        message = value["messages"][0]
    except (KeyError, IndexError, TypeError):
        return None

    contacts = value.get("contacts") or [{}]
    sender_name = contacts[0].get("profile", {}).get("name", "")

    message_type = message.get("type", "text")
    text = ""
    image_id = audio_id = document_id = video_id = None
    document_mime = document_filename = None
    button_reply_id = button_reply_title = None
    quoted_wamid = None
    interactive_type = None
    flow_response_json = None
    latitude = longitude = None
    location_text = None

    if message_type == "text":
        text = message.get("text", {}).get("body", "")
    elif message_type == "image":
        image_id = message.get("image", {}).get("id")
        text = message.get("image", {}).get("caption", "")
    elif message_type == "video":
        video_id = message.get("video", {}).get("id")
        text = message.get("video", {}).get("caption", "")
    elif message_type == "audio":
        audio_id = message.get("audio", {}).get("id")
    elif message_type == "document":
        document_id = message.get("document", {}).get("id")
        document_mime = message.get("document", {}).get("mime_type")
        document_filename = message.get("document", {}).get("filename")
        text = message.get("document", {}).get("caption", "")
    elif message_type == "location":
        loc = message.get("location", {})
        latitude = loc.get("latitude")
        longitude = loc.get("longitude")
        location_text = loc.get("name") or loc.get("address")
    elif message_type == "reaction":
        # No dedicated field on ExtractedMessage for this -- folded into text so
        # the agent sees *something* instead of an entirely blank turn (previously
        # every non-text/image/video/audio/document/location/interactive type,
        # e.g. reaction/sticker/contacts/order/system, fell through with
        # text="" and confused the agent into a generic non-answer).
        emoji = message.get("reaction", {}).get("emoji", "")
        text = f"[reacted {emoji}]".strip() if emoji else "[reacted to a message]"
    elif message_type == "sticker":
        text = "[sent a sticker]"
    elif message_type == "contacts":
        names = ", ".join(c.get("name", {}).get("formatted_name", "") for c in message.get("contacts", []) if c.get("name"))
        text = f"[shared contact: {names}]" if names else "[shared a contact]"
    elif message_type == "order":
        text = "[sent an order]"
    elif message_type == "interactive":
        interactive = message.get("interactive", {})
        interactive_type = interactive.get("type")
        if interactive_type == "button_reply":
            button_reply_id = interactive.get("button_reply", {}).get("id")
            button_reply_title = interactive.get("button_reply", {}).get("title")
        elif interactive_type == "list_reply":
            button_reply_id = interactive.get("list_reply", {}).get("id")
            button_reply_title = interactive.get("list_reply", {}).get("title")
        elif interactive_type == "nfm_reply":
            raw_json = interactive.get("nfm_reply", {}).get("response_json")
            if raw_json:
                import json

                try:
                    flow_response_json = json.loads(raw_json)
                except (json.JSONDecodeError, TypeError):
                    flow_response_json = None

    if message.get("context", {}).get("id"):
        quoted_wamid = message["context"]["id"]

    return ExtractedMessage(
        phone_number=message.get("from", ""),
        sender_name=sender_name,
        message_id=message.get("id", ""),
        timestamp=message.get("timestamp", ""),
        message_type=message_type,
        text=text,
        image_media_id=image_id,
        audio_media_id=audio_id,
        document_media_id=document_id,
        video_media_id=video_id,
        document_mime_type=document_mime,
        document_filename=document_filename,
        button_reply_id=button_reply_id,
        button_reply_title=button_reply_title,
        quoted_wamid=quoted_wamid,
        interactive_type=interactive_type,
        flow_response_json=flow_response_json,
        latitude=latitude,
        longitude=longitude,
        location_text=location_text,
        raw=body,
    )


def extract_status_update(body: dict[str, Any]) -> dict[str, Any] | None:
    """Pulls Meta's async delivery-status callback (sent/delivered/read/failed)
    out of the webhook envelope — a separate event from `messages`, sent
    AFTER the fact. We don't act on these, but logging them is the only way
    to see WHY a message that our send API call already accepted (200 +
    message id) never actually reached the recipient's phone — that 200
    only means "queued", not "delivered"; real failures (wrong opt-in
    status, 24h window expired, number not on WhatsApp, etc.) show up here,
    not on the original send call."""
    try:
        return body["entry"][0]["changes"][0]["value"]["statuses"][0]
    except (KeyError, IndexError, TypeError):
        return None


def verify_webhook_challenge(mode: str | None, token: str | None, challenge: str | None, expected_token: str) -> str | None:
    if mode == "subscribe" and token == expected_token:
        return challenge
    return None
