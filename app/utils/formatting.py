"""WhatsApp reply formatting + chunking (spec §2 `Format WhatsApp Response`
and `Split Response Into Chunks`)."""

import re

CHUNK_BUDGET = 600


def to_whatsapp_markdown(text: str) -> str:
    """**bold** -> *bold*, strip markdown '#' headers."""
    text = re.sub(r"\*\*(.+?)\*\*", r"*\1*", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    return text.strip()


def strip_for_speech(text: str) -> str:
    """WhatsApp markdown (*bold*) and stray formatting reads badly aloud --
    strip it before TTS synthesis rather than speaking literal asterisks
    (see app/integrations/openai_client.py synthesize_speech)."""
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"[_~`]", "", text)
    return text.strip()


def split_into_chunks(text: str, budget: int = CHUNK_BUDGET) -> list[str]:
    """Splits on blank lines into paragraphs, then RE-ACCUMULATES consecutive
    paragraphs into as few chunks as possible up to `budget` chars each --
    only starting a new WhatsApp bubble when the next paragraph genuinely
    doesn't fit. A single paragraph over `budget` is further split on
    sentence boundaries.

    Confirmed live bug (2026-09): the previous version put every
    blank-line-separated paragraph in its own bubble regardless of size --
    a short clinic list (a one-line header + 2 short clinic blocks + a
    one-line closer) arrived as 4 separate rapid-fire messages instead of
    one clean one, reading as choppy/spammy rather than a single coherent
    reply. Merging small paragraphs back together is what "feels like one
    reply" instead of a burst of fragments."""
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""

    def _flush():
        nonlocal current
        if current:
            chunks.append(current)
            current = ""

    for para in paragraphs:
        if len(para) > budget:
            _flush()
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sentence in sentences:
                candidate = f"{current} {sentence}".strip() if current else sentence
                if len(candidate) > budget and current:
                    chunks.append(current)
                    current = sentence
                else:
                    current = candidate
            continue

        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) > budget and current:
            _flush()
            current = para
        else:
            current = candidate

    _flush()
    return chunks
