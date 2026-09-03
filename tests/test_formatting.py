from app.utils.formatting import split_into_chunks, strip_for_speech, to_whatsapp_markdown


def test_bold_conversion():
    assert to_whatsapp_markdown("This is **important**") == "This is *important*"


def test_strips_markdown_headers():
    assert to_whatsapp_markdown("# Heading\nBody text") == "Heading\nBody text"


def test_split_short_text_is_single_chunk():
    assert split_into_chunks("Hello there!") == ["Hello there!"]


def test_short_paragraphs_merge_into_one_chunk():
    """Live bug (2026-09): short paragraphs used to each become their own
    WhatsApp bubble regardless of size -- a short reply with a header, two
    brief items, and a closer arrived as 4 separate rapid-fire messages.
    Small paragraphs must merge into as few bubbles as fit the budget."""
    text = "First paragraph.\n\nSecond paragraph."
    assert split_into_chunks(text) == ["First paragraph.\n\nSecond paragraph."]


def test_a_realistic_short_clinic_list_stays_in_one_bubble():
    text = (
        "Go now — these emergency vet hospitals are open:\n\n"
        "1. Medi Paws Pet Care Hospital — 4.9 km, +91 96222 73536, 4.9★ (217 reviews)\n\n"
        "2. Pet's Care Super Specialty Hospital — 5.74 km, +91 89780 03006, 4.2★ (937 reviews)\n\n"
        "Call while travelling and mention Bobby swallowed paracetamol."
    )
    chunks = split_into_chunks(text)
    assert len(chunks) == 1


def test_paragraphs_only_split_once_the_budget_is_exceeded():
    para = "x" * 400
    text = f"{para}\n\n{para}\n\n{para}"
    chunks = split_into_chunks(text, budget=600)
    assert len(chunks) == 3  # 400+400 > 600, so each stays in its own chunk
    for chunk in chunks:
        assert len(chunk) <= 600


def test_long_paragraph_splits_on_sentence_boundaries():
    sentence = "This is a moderately long sentence that repeats. "
    long_para = sentence * 20
    chunks = split_into_chunks(long_para.strip())
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 600 or " " not in chunk  # allow a single overlong sentence through


def test_empty_text_returns_no_chunks():
    assert split_into_chunks("") == []


def test_strip_for_speech_removes_whatsapp_markdown():
    assert strip_for_speech("*Seriousness:* 🟡 Moderate (3/5)") == "Seriousness: 🟡 Moderate (3/5)"
    assert strip_for_speech("some `code` and _emphasis_ and ~strike~") == "some code and emphasis and strike"
