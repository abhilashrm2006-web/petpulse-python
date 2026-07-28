from app.utils.formatting import split_into_chunks, strip_for_speech, to_whatsapp_markdown


def test_bold_conversion():
    assert to_whatsapp_markdown("This is **important**") == "This is *important*"


def test_strips_markdown_headers():
    assert to_whatsapp_markdown("# Heading\nBody text") == "Heading\nBody text"


def test_split_short_text_is_single_chunk():
    assert split_into_chunks("Hello there!") == ["Hello there!"]


def test_split_on_blank_lines():
    text = "First paragraph.\n\nSecond paragraph."
    assert split_into_chunks(text) == ["First paragraph.", "Second paragraph."]


def test_long_paragraph_splits_on_sentence_boundaries():
    sentence = "This is a moderately long sentence that repeats. "
    long_para = sentence * 10
    chunks = split_into_chunks(long_para.strip())
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 300 or " " not in chunk  # allow a single overlong sentence through


def test_empty_text_returns_no_chunks():
    assert split_into_chunks("") == []


def test_strip_for_speech_removes_whatsapp_markdown():
    assert strip_for_speech("*Seriousness:* 🟡 Moderate (3/5)") == "Seriousness: 🟡 Moderate (3/5)"
    assert strip_for_speech("some `code` and _emphasis_ and ~strike~") == "some code and emphasis and strike"
