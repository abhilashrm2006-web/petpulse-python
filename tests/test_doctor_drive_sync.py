"""Covers sync_doctor_onboarding_drafts: turns a Google Drive folder of
per-doctor subfolders into review-able drafts, never a real account
directly. Google Drive/extraction calls are mocked (network/LLM calls
covered by the live smoke test done while building this, not re-verified
here) -- this suite covers the sync's own control flow: skip empty/final-
state folders, don't re-extract an unchanged folder, upsert by
drive_folder_id."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.scheduler.jobs import sync_doctor_onboarding_drafts
from tests.fake_supabase import FakeSupabaseClient


def _make_ctx(supabase, settings=None):
    default_settings = SimpleNamespace(google_service_account_json="{}", doctor_drive_folder_id="parent-1")
    return SimpleNamespace(supabase=supabase, settings=settings or default_settings, http=object(), openai=object())


@pytest.mark.asyncio
async def test_does_nothing_when_not_configured():
    supabase = FakeSupabaseClient()
    settings = SimpleNamespace(google_service_account_json="", doctor_drive_folder_id="")
    ctx = _make_ctx(supabase, settings)

    await sync_doctor_onboarding_drafts(ctx)  # must not raise / must not call Drive at all

    assert supabase.rows("doctor_onboarding_drafts") == []


@pytest.mark.asyncio
async def test_creates_a_pending_draft_for_a_new_folder_with_documents(monkeypatch):
    supabase = FakeSupabaseClient()
    ctx = _make_ctx(supabase)

    monkeypatch.setattr(
        "app.scheduler.jobs.google_drive.list_subfolders",
        AsyncMock(return_value=[{"id": "folder-1", "name": "Dr Mounika", "modifiedTime": "2026-01-01T00:00:00Z"}]),
    )
    monkeypatch.setattr(
        "app.scheduler.jobs.google_drive.list_files",
        AsyncMock(return_value=[{"id": "file-1", "name": "cert.pdf", "mimeType": "application/pdf", "modifiedTime": "2026-01-02T00:00:00Z"}]),
    )
    monkeypatch.setattr("app.scheduler.jobs.google_drive.download_file", AsyncMock(return_value=b"fake-pdf-bytes"))
    monkeypatch.setattr(
        "app.scheduler.jobs.extract_doctor_fields",
        AsyncMock(return_value={"full_name": "Dr. Mounika", "phone_number": "919182381400", "_source_filenames": ["cert.pdf"]}),
    )

    await sync_doctor_onboarding_drafts(ctx)

    drafts = supabase.rows("doctor_onboarding_drafts")
    assert len(drafts) == 1
    assert drafts[0]["drive_folder_id"] == "folder-1"
    assert drafts[0]["status"] == "pending_review"
    assert drafts[0]["extracted_full_name"] == "Dr. Mounika"
    assert drafts[0]["extracted_phone_number"] == "919182381400"


@pytest.mark.asyncio
async def test_skips_a_folder_with_no_files_yet(monkeypatch):
    supabase = FakeSupabaseClient()
    ctx = _make_ctx(supabase)

    monkeypatch.setattr(
        "app.scheduler.jobs.google_drive.list_subfolders",
        AsyncMock(return_value=[{"id": "folder-1", "name": "Dr Naveen", "modifiedTime": "2026-01-01T00:00:00Z"}]),
    )
    monkeypatch.setattr("app.scheduler.jobs.google_drive.list_files", AsyncMock(return_value=[]))
    extract_mock = AsyncMock()
    monkeypatch.setattr("app.scheduler.jobs.extract_doctor_fields", extract_mock)

    await sync_doctor_onboarding_drafts(ctx)

    assert supabase.rows("doctor_onboarding_drafts") == []
    extract_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_never_touches_an_approved_or_rejected_draft(monkeypatch):
    """A human already acted on this folder -- re-running the sync (even if
    the doctor adds more files to the Drive folder afterward) must never
    silently overwrite an admin's decision."""
    supabase = FakeSupabaseClient(
        initial={
            "doctor_onboarding_drafts": [
                {
                    "id": "d1", "drive_folder_id": "folder-1", "drive_folder_name": "Dr Mounika",
                    "status": "approved", "extracted_full_name": "Dr. Mounika",
                }
            ]
        }
    )
    ctx = _make_ctx(supabase)

    monkeypatch.setattr(
        "app.scheduler.jobs.google_drive.list_subfolders",
        AsyncMock(return_value=[{"id": "folder-1", "name": "Dr Mounika", "modifiedTime": "2026-02-01T00:00:00Z"}]),
    )
    list_files_mock = AsyncMock()
    monkeypatch.setattr("app.scheduler.jobs.google_drive.list_files", list_files_mock)

    await sync_doctor_onboarding_drafts(ctx)

    list_files_mock.assert_not_awaited()
    assert supabase.rows("doctor_onboarding_drafts")[0]["status"] == "approved"


@pytest.mark.asyncio
async def test_does_not_re_extract_an_unchanged_folder(monkeypatch):
    supabase = FakeSupabaseClient(
        initial={
            "doctor_onboarding_drafts": [
                {
                    "id": "d1", "drive_folder_id": "folder-1", "drive_folder_name": "Dr Mounika",
                    "status": "pending_review", "drive_folder_modified_time": "2026-01-02T00:00:00Z",
                }
            ]
        }
    )
    ctx = _make_ctx(supabase)

    monkeypatch.setattr(
        "app.scheduler.jobs.google_drive.list_subfolders",
        AsyncMock(return_value=[{"id": "folder-1", "name": "Dr Mounika", "modifiedTime": "2026-01-01T00:00:00Z"}]),
    )
    monkeypatch.setattr(
        "app.scheduler.jobs.google_drive.list_files",
        AsyncMock(return_value=[{"id": "file-1", "name": "cert.pdf", "mimeType": "application/pdf", "modifiedTime": "2026-01-02T00:00:00Z"}]),
    )
    extract_mock = AsyncMock()
    monkeypatch.setattr("app.scheduler.jobs.extract_doctor_fields", extract_mock)

    await sync_doctor_onboarding_drafts(ctx)

    extract_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_one_folders_failure_does_not_block_another(monkeypatch):
    supabase = FakeSupabaseClient()
    ctx = _make_ctx(supabase)

    monkeypatch.setattr(
        "app.scheduler.jobs.google_drive.list_subfolders",
        AsyncMock(
            return_value=[
                {"id": "folder-1", "name": "Dr Broken", "modifiedTime": "2026-01-01T00:00:00Z"},
                {"id": "folder-2", "name": "Dr Fine", "modifiedTime": "2026-01-01T00:00:00Z"},
            ]
        ),
    )

    async def fake_list_files(settings, http, folder_id):
        if folder_id == "folder-1":
            raise RuntimeError("Drive API hiccup")
        return [{"id": "file-1", "name": "cert.pdf", "mimeType": "application/pdf", "modifiedTime": "2026-01-02T00:00:00Z"}]

    monkeypatch.setattr("app.scheduler.jobs.google_drive.list_files", fake_list_files)
    monkeypatch.setattr("app.scheduler.jobs.google_drive.download_file", AsyncMock(return_value=b"bytes"))
    monkeypatch.setattr(
        "app.scheduler.jobs.extract_doctor_fields",
        AsyncMock(return_value={"full_name": "Dr. Fine", "_source_filenames": ["cert.pdf"]}),
    )

    await sync_doctor_onboarding_drafts(ctx)

    drafts = supabase.rows("doctor_onboarding_drafts")
    assert len(drafts) == 1
    assert drafts[0]["drive_folder_id"] == "folder-2"
