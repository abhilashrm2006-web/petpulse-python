"""Covers the pure vaccination-line formatting used by get_pet_passport —
must surface every field actually on file (manufacturer, batch/lot number,
next-due date), not just name + date, and flag overdue vaccinations."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.agent.tools.documents import _format_vaccination_line, file_document, send_pet_document

TODAY = "2026-07-26"


def test_full_record_includes_manufacturer_batch_and_next_due():
    vax = {
        "vaccine_name": "Rabies",
        "date_administered": "2025-07-01",
        "manufacturer": "Zoetis",
        "batch_number": "LOT-4471B",
        "next_due_date": "2026-07-01",
    }
    line, overdue = _format_vaccination_line(vax, TODAY)
    assert "Rabies" in line
    assert "2025-07-01" in line
    assert "Zoetis" in line
    assert "Batch/Lot: LOT-4471B" in line
    assert "Next due: 2026-07-01" in line
    assert overdue is True
    assert "(OVERDUE)" in line


def test_not_yet_due_is_not_flagged_overdue():
    vax = {
        "vaccine_name": "DHPP",
        "date_administered": "2026-06-01",
        "next_due_date": "2027-06-01",
    }
    line, overdue = _format_vaccination_line(vax, TODAY)
    assert overdue is False
    assert "(OVERDUE)" not in line
    assert "Next due: 2027-06-01" in line


def test_missing_optional_fields_are_omitted_not_blank():
    vax = {"vaccine_name": "Bordetella", "date_administered": "2026-01-01"}
    line, overdue = _format_vaccination_line(vax, TODAY)
    assert overdue is False
    assert "Batch/Lot" not in line
    assert "Next due" not in line
    assert line == "- Bordetella — 2026-01-01"


def _pets_with_a_name_collision_across_two_owners():
    return [
        {"id": "b1", "name": "Bobby", "owner_name": "Abhilash", "owner_phone": "919000000001"},
        {"id": "b2", "name": "Bobby", "owner_name": "Priya", "owner_phone": "919000000002"},
    ]


@pytest.mark.asyncio
async def test_file_document_surfaces_owner_disambiguation_instead_of_guessing():
    """Reproduces a real reported bug: a vet's patient list spans multiple
    unrelated households, so "pet name bobby" matched TWO different
    owners' pets, and the document silently went to the wrong one. The
    tool must now hand the LLM the candidates (with owner_name/owner_phone)
    and refuse to pick — never file to a guessed pet."""
    agent_ctx = SimpleNamespace(
        pets=_pets_with_a_name_collision_across_two_owners(),
        pending_media=SimpleNamespace(document_bytes=b"fake-bytes", document_mime_type="image/jpeg", document_classification=None, media_context=""),
    )
    ctx = SimpleNamespace(supabase=None, whatsapp=None, settings=None, openai=None)

    result = await file_document(ctx, agent_ctx, pet_name="bobby")

    assert result["success"] is False
    assert result["error"] == "ambiguous_pet"
    assert {c["owner_name"] for c in result["candidates"]} == {"Abhilash", "Priya"}
    assert "owner_name" in result["instruction_to_llm"]


@pytest.mark.asyncio
async def test_file_document_files_to_the_exact_pet_id_once_disambiguated(monkeypatch):
    from tests.fake_supabase import FakeSupabaseClient

    supabase = FakeSupabaseClient()
    agent_ctx = SimpleNamespace(
        pets=_pets_with_a_name_collision_across_two_owners(),
        pending_media=SimpleNamespace(
            document_bytes=b"fake-bytes", document_mime_type="image/jpeg", document_classification=None, media_context="vaccination card",
        ),
        profile={"id": "vet-1"},
    )
    ctx = SimpleNamespace(supabase=supabase, whatsapp=None, settings=None, openai=AsyncMock())

    monkeypatch.setattr("app.agent.tools.documents.upload_to_storage", lambda *a, **k: None)
    monkeypatch.setattr("app.agent.tools.documents.json_completion", AsyncMock(return_value='{"record_kind": "none"}'))

    result = await file_document(ctx, agent_ctx, pet_id="b1", document_type="Vaccination Certificate")

    assert result["success"] is True
    assert result["pet_name"] == "Bobby"
    doc = supabase.rows("documents")[0]
    assert doc["pet_id"] == "b1"


@pytest.mark.asyncio
async def test_two_documents_filed_same_day_get_distinct_storage_paths(monkeypatch):
    """Real bug found via audit: object_path was built from only the pet_id
    and a day-granularity timestamp, with no per-upload uniqueness. Since
    upload_to_storage uses upsert, two documents of the same detected type
    filed for the same pet on the same day (e.g. two "Lab Report" photos)
    silently overwrote each other's bytes at the same storage path, while
    each document's own `documents` row still pointed at that same path --
    so an older row would later serve the newer file's content."""
    from tests.fake_supabase import FakeSupabaseClient

    supabase = FakeSupabaseClient()
    agent_ctx = SimpleNamespace(
        pets=[{"id": "pet-a", "name": "Max"}],
        pending_media=SimpleNamespace(
            document_bytes=b"fake-bytes", document_mime_type="image/jpeg", document_classification=None, media_context="lab report",
        ),
        profile={"id": "profile-1"},
    )
    ctx = SimpleNamespace(supabase=supabase, whatsapp=None, settings=None, openai=AsyncMock())

    monkeypatch.setattr("app.agent.tools.documents.upload_to_storage", lambda *a, **k: None)
    monkeypatch.setattr("app.agent.tools.documents.json_completion", AsyncMock(return_value='{"record_kind": "none"}'))

    await file_document(ctx, agent_ctx, pet_id="pet-a", document_type="Lab Report")
    await file_document(ctx, agent_ctx, pet_id="pet-a", document_type="Lab Report")

    docs = supabase.rows("documents")
    assert len(docs) == 2
    assert docs[0]["storage_path"] != docs[1]["storage_path"]


@pytest.mark.asyncio
async def test_send_pet_document_surfaces_owner_disambiguation_instead_of_guessing():
    agent_ctx = SimpleNamespace(pets=_pets_with_a_name_collision_across_two_owners())
    ctx = SimpleNamespace(supabase=None, whatsapp=None, settings=None, openai=None)

    result = await send_pet_document(ctx, agent_ctx, pet_name="bobby")

    assert result["success"] is False
    assert result["error"] == "ambiguous_pet"
    assert {c["owner_name"] for c in result["candidates"]} == {"Abhilash", "Priya"}


@pytest.mark.asyncio
async def test_file_document_has_no_cap(monkeypatch):
    from tests.fake_supabase import FakeSupabaseClient

    existing_docs = [{"id": f"doc-{i}", "pet_id": "pet-1"} for i in range(8)]
    supabase = FakeSupabaseClient(initial={"documents": existing_docs})
    agent_ctx = SimpleNamespace(
        pets=[{"id": "pet-1", "name": "Rex"}],
        pending_media=SimpleNamespace(document_bytes=b"fake-bytes", document_mime_type="image/jpeg", document_classification=None, media_context="lab report"),
        profile={"id": "profile-1"},
    )
    ctx = SimpleNamespace(supabase=supabase, whatsapp=None, settings=None, openai=AsyncMock())
    monkeypatch.setattr("app.agent.tools.documents.upload_to_storage", lambda *a, **k: None)
    monkeypatch.setattr("app.agent.tools.documents.json_completion", AsyncMock(return_value='{"record_kind": "none"}'))

    result = await file_document(ctx, agent_ctx, pet_id="pet-1")

    assert result["success"] is True
    assert len(supabase.rows("documents")) == 9


@pytest.mark.asyncio
async def test_search_documents_finds_matches_across_ocr_text_and_summary():
    from tests.fake_supabase import FakeSupabaseClient
    from app.agent.tools.documents import search_documents

    supabase = FakeSupabaseClient(
        initial={
            "documents": [
                {"id": "doc-1", "pet_id": "pet-1", "document_name": "Lab Report", "document_type": "Lab Report", "uploaded_at": "2026-07-01", "ocr_text": "WBC count elevated", "ai_summary": ""},
                {"id": "doc-2", "pet_id": "pet-1", "document_name": "Prescription", "document_type": "Prescription", "uploaded_at": "2026-07-02", "ocr_text": "", "ai_summary": "amoxicillin twice daily"},
                {"id": "doc-3", "pet_id": "pet-1", "document_name": "Unrelated", "document_type": "Other", "uploaded_at": "2026-06-01", "ocr_text": "nothing relevant", "ai_summary": ""},
            ]
        }
    )
    agent_ctx = SimpleNamespace(pets=[{"id": "pet-1", "name": "Rex"}])
    ctx = SimpleNamespace(supabase=supabase)

    result = await search_documents(ctx, agent_ctx, query="amoxicillin")

    assert result["success"] is True
    assert result["count"] == 1
    assert result["results"][0]["document_id"] == "doc-2"


@pytest.mark.asyncio
async def test_get_shareable_link_generates_and_persists_a_token():
    from tests.fake_supabase import FakeSupabaseClient
    from app.agent.tools.documents import get_shareable_link

    supabase = FakeSupabaseClient(initial={"pets": [{"id": "pet-1", "name": "Rex", "passport_share_token": None}]})
    agent_ctx = SimpleNamespace(pets=[{"id": "pet-1", "name": "Rex", "passport_share_token": None}])
    ctx = SimpleNamespace(supabase=supabase, settings=SimpleNamespace(public_base_url="https://example.test"))

    result = await get_shareable_link(ctx, agent_ctx, pet_id="pet-1")

    assert result["success"] is True
    assert result["url"].startswith("https://example.test/passport/")
    stored_token = supabase.rows("pets")[0]["passport_share_token"]
    assert stored_token and stored_token in result["url"]


@pytest.mark.asyncio
async def test_get_shareable_link_is_stable_across_calls():
    from tests.fake_supabase import FakeSupabaseClient
    from app.agent.tools.documents import get_shareable_link

    supabase = FakeSupabaseClient(initial={"pets": [{"id": "pet-1", "name": "Rex", "passport_share_token": "already-set-token"}]})
    agent_ctx = SimpleNamespace(pets=[{"id": "pet-1", "name": "Rex", "passport_share_token": "already-set-token"}])
    ctx = SimpleNamespace(supabase=supabase, settings=SimpleNamespace(public_base_url="https://example.test"))

    result = await get_shareable_link(ctx, agent_ctx, pet_id="pet-1")

    assert result["url"] == "https://example.test/passport/already-set-token"


@pytest.mark.asyncio
async def test_get_pet_passport_includes_full_batch_details(monkeypatch):
    from tests.fake_supabase import FakeSupabaseClient
    from app.agent.tools.documents import get_pet_passport

    supabase = FakeSupabaseClient(
        initial={
            "vaccinations": [
                {"id": "v1", "pet_id": "pet-1", "vaccine_name": "Rabies", "date_administered": "2025-07-01", "next_due_date": "2026-07-01", "manufacturer": "Zoetis", "batch_number": "LOT-1"},
            ],
            "medical_records": [],
        }
    )
    agent_ctx = SimpleNamespace(pets=[{"id": "pet-1", "name": "Rex", "species": "Dog"}], profile={"phone_number": "919876543210"})
    ctx = SimpleNamespace(supabase=supabase, whatsapp=SimpleNamespace(send_document=AsyncMock(), send_image=AsyncMock()))

    result = await get_pet_passport(ctx, agent_ctx, pet_id="pet-1", send_certificates=False)

    assert result["success"] is True
    assert "Zoetis" in result["passport_text"]
    assert "Batch/Lot: LOT-1" in result["passport_text"]
    assert result["overdue_vaccinations"] == 1


@pytest.mark.asyncio
async def test_pet_passport_shows_fallback_text_for_empty_sections_not_bare_headers():
    """Live bug (2026-08-27): a pet with no vaccinations/medical records on
    file rendered just the "*Vaccinations:*"/"*Recent medical records:*"
    headers with nothing underneath -- no data and no explanation, which
    reads as broken rather than "nothing on file yet"."""
    from tests.fake_supabase import FakeSupabaseClient
    from app.agent.tools.documents import get_pet_passport

    supabase = FakeSupabaseClient(initial={"vaccinations": [], "medical_records": []})
    agent_ctx = SimpleNamespace(pets=[{"id": "pet-1", "name": "Rex", "species": "Dog"}], profile={"phone_number": "919876543210"})
    ctx = SimpleNamespace(supabase=supabase, whatsapp=SimpleNamespace(send_document=AsyncMock(), send_image=AsyncMock()))

    result = await get_pet_passport(ctx, agent_ctx, pet_id="pet-1", send_certificates=False)

    text = result["passport_text"]
    assert "*Vaccinations:*\nNo vaccination records on file yet" in text
    assert "*Recent medical records:*\nNo medical records on file yet" in text
