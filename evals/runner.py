"""Runs evals/scenarios.py against the REAL agent loop: real OpenAI, real
Supabase (synthetic profiles/pets, cleaned up after each scenario), only
WhatsApp sends are mocked (captured instead of actually sent). This is
qualitative, not unit-test-deterministic -- a real model call can vary --
so treat an occasional single-scenario flake as worth a re-run, but a
consistently failing scenario as a real regression.

Usage (from repo root, with prod env vars injected):
    railway run python3 -m evals.runner                # all scenarios
    railway run python3 -m evals.runner emergency_no_paid_consult   # one scenario
"""

import asyncio
import sys
import time
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
from openai import AsyncOpenAI

from app.agent.orchestrator import run_agent_turn
from app.config import get_settings
from app.deps import AppContext
from app.ingestion.context import build_context
from app.ingestion.webhook import ExtractedMessage
from app.integrations.supabase_client import make_supabase_client
from evals.scenarios import SCENARIOS, Scenario


def _synthetic_phone() -> str:
    return "9199" + uuid.uuid4().hex[:8]


async def _run_scenario(ctx: AppContext, scenario: Scenario) -> tuple[bool, list[str]]:
    supabase = ctx.supabase
    phone = _synthetic_phone()
    notes: list[str] = []
    all_passed = True

    profile = supabase.table("profiles").insert(
        {"phone_number": phone, "full_name": "Eval Tester", "role": "customer", "registration_step": "completed", "city": "Chennai"}
    ).execute().data[0]

    pet_ids = []
    for i, pet_spec in enumerate(scenario.pets):
        pet = supabase.table("pets").insert({**pet_spec, "profile_id": profile["id"]}).execute().data[0]
        pet_ids.append(pet["id"])
        for log_row in scenario.health_logs.get(i, []):
            supabase.table("health_logs").insert({**log_row, "pet_id": pet["id"], "profile_id": profile["id"]}).execute()

    try:
        for turn_idx, turn in enumerate(scenario.turns):
            extracted = ExtractedMessage(
                phone_number=phone, sender_name="Eval Tester", message_id=f"wamid.eval.{uuid.uuid4().hex[:8]}",
                timestamp=str(int(time.time())), message_type="text", text=turn.text,
            )
            agent_ctx = await build_context(supabase, extracted)
            reply = await run_agent_turn(ctx, agent_ctx, extracted)

            for check in turn.checks:
                passed = check.fn(reply or "")
                if not passed:
                    all_passed = False
                notes.append(f"  turn {turn_idx} [{'PASS' if passed else 'FAIL'}] {check.label}")
            notes.append(f"  turn {turn_idx} reply: {(reply or '(empty)')[:300]!r}")
    finally:
        for pet_id in pet_ids:
            for t in ("vaccinations", "medical_records", "health_logs", "pet_members"):
                supabase.table(t).delete().eq("pet_id", pet_id).execute()
            supabase.table("pets").delete().eq("id", pet_id).execute()
        for t in ("registration_step_history", "onboarding_events", "messages", "conversations"):
            supabase.table(t).delete().eq("profile_id", profile["id"]).execute()
        supabase.table("profiles").delete().eq("id", profile["id"]).execute()

    return all_passed, notes


async def main() -> None:
    only = sys.argv[1] if len(sys.argv) > 1 else None
    scenarios = [s for s in SCENARIOS if only is None or s.id == only]
    if not scenarios:
        print(f"No scenario matching {only!r}. Known: {[s.id for s in SCENARIOS]}")
        return

    settings = get_settings()
    supabase = make_supabase_client(settings)
    http = httpx.AsyncClient(timeout=30.0)
    openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
    whatsapp = SimpleNamespace(
        send_text=AsyncMock(), send_interactive_buttons=AsyncMock(), send_interactive_list=AsyncMock(),
        send_document=AsyncMock(), send_image=AsyncMock(), send_audio=AsyncMock(), send_sticker=AsyncMock(),
        send_video=AsyncMock(), send_reply_and_chunk=AsyncMock(side_effect=lambda to, text: [("wamid.fake", text)]),
    )
    ctx = AppContext(settings=settings, http=http, whatsapp=whatsapp, supabase=supabase, openai=openai_client)

    results = []
    for scenario in scenarios:
        print(f"\n=== {scenario.id} ===")
        print(f"    {scenario.description}")
        try:
            passed, notes = await _run_scenario(ctx, scenario)
        except Exception as exc:
            passed, notes = False, [f"  ERROR: {exc!r}"]
        for line in notes:
            print(line)
        print(f"    -> {'PASS' if passed else 'FAIL'}")
        results.append((scenario.id, passed))

    await http.aclose()

    print("\n" + "=" * 50)
    print("SUMMARY")
    for sid, passed in results:
        print(f"  [{'PASS' if passed else 'FAIL'}] {sid}")
    failed = [sid for sid, passed in results if not passed]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed")
    if failed:
        print(f"Failed: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
