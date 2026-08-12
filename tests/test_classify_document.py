"""Covers classify_document's active-pet bias (2026-08 root-cause item #5):
a submitted photo/video's pet identity now defaults to whichever pet the
conversation already has as "active" when the vision/classification model
doesn't confidently name a different one, instead of re-guessing from
scratch purely off the media + a flat list of every pet name on file."""

import json
from unittest.mock import AsyncMock

import pytest

from app.media_pipeline.classify import classify_document

PETS = [
    {"id": "pet-1", "name": "Bobby", "species": "Dog"},
    {"id": "pet-2", "name": "Whiskers", "species": "Cat"},
]


@pytest.mark.asyncio
async def test_model_returns_no_pet_name_falls_back_to_active_pet(monkeypatch):
    async def fake_json_completion(client, settings, system, user_prompt, reasoning_effort=None):
        assert "currently active pet" in user_prompt
        assert "Bobby" in user_prompt
        return json.dumps({"document_type": "Photo", "is_medical_document": False, "handwritten": False, "pet_name": None})

    monkeypatch.setattr("app.media_pipeline.classify.json_completion", fake_json_completion)

    result = await classify_document(
        AsyncMock(), object(), "video", "video/mp4", "a dog running", "", PETS, active_pet_name="Bobby"
    )

    assert result.target_pet["name"] == "Bobby"


@pytest.mark.asyncio
async def test_model_confidently_names_a_different_pet_is_respected(monkeypatch):
    async def fake_json_completion(client, settings, system, user_prompt, reasoning_effort=None):
        return json.dumps({"document_type": "Photo", "is_medical_document": False, "handwritten": False, "pet_name": "Whiskers"})

    monkeypatch.setattr("app.media_pipeline.classify.json_completion", fake_json_completion)

    result = await classify_document(
        AsyncMock(), object(), "image", "image/jpeg", "a cat on a windowsill", "", PETS, active_pet_name="Bobby"
    )

    assert result.target_pet["name"] == "Whiskers"


@pytest.mark.asyncio
async def test_no_active_pet_hint_omitted_from_prompt(monkeypatch):
    async def fake_json_completion(client, settings, system, user_prompt, reasoning_effort=None):
        assert "currently active pet" not in user_prompt
        return json.dumps({"document_type": "Photo", "is_medical_document": False, "handwritten": False, "pet_name": None})

    monkeypatch.setattr("app.media_pipeline.classify.json_completion", fake_json_completion)

    result = await classify_document(
        AsyncMock(), object(), "image", "image/jpeg", "a dog", "", PETS, active_pet_name=None
    )

    assert result.target_pet is None
