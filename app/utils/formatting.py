"""WhatsApp reply formatting + chunking (spec §2 `Format WhatsApp Response`
and `Split Response Into Chunks`)."""

import re

CHUNK_BUDGET = 300


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
    """Split on blank lines into paragraphs; any paragraph over `budget`
    chars is further split on sentence boundaries and re-accumulated up to
    the budget."""
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []

    for para in paragraphs:
        if len(para) <= budget:
            chunks.append(para)
            continue
        sentences = re.split(r"(?<=[.!?])\s+", para)
        current = ""
        for sentence in sentences:
            candidate = f"{current} {sentence}".strip() if current else sentence
            if len(candidate) > budget and current:
                chunks.append(current)
                current = sentence
            else:
                current = candidate
        if current:
            chunks.append(current)

    return chunks
