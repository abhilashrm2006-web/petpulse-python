from datetime import datetime, timedelta

import pytest

from app.availability.slots import IST, MAX_DAYS, MAX_PER_DAY, _parse_calendar_boundary, compute_doctor_slots, generate_slots


def _next_weekday(base: datetime, target_weekday: int) -> datetime:
    """target_weekday: Monday=0 ... Sunday=6 (datetime.weekday() convention)."""
    days_ahead = (target_weekday - base.weekday()) % 7 or 7
    return base + timedelta(days=days_ahead)


_BASE = datetime(2026, 1, 1, 9, 0, tzinfo=IST)
MONDAY_9AM = _next_weekday(_BASE, 0).replace(hour=9, minute=0)


def test_no_slots_before_10am_and_respects_buffer():
    slots = generate_slots(MONDAY_9AM, busy=[])
    assert all(s.start.hour >= 10 for s in slots)


def test_max_three_slots_per_day():
    slots = generate_slots(MONDAY_9AM, busy=[])
    by_day = {}
    for s in slots:
        by_day.setdefault(s.start.date(), []).append(s)
    for day_slots in by_day.values():
        assert len(day_slots) <= MAX_PER_DAY


def test_skips_sunday():
    sunday = MONDAY_9AM - timedelta(days=1)
    sunday = sunday.replace(hour=6, minute=0)
    slots = generate_slots(sunday, busy=[])
    assert all(s.start.weekday() != 6 for s in slots)
    # the Monday after should still be represented within the 4-day window
    assert any(s.start.date() == MONDAY_9AM.date() for s in slots)


def test_busy_event_blocks_overlapping_slot():
    busy_start = MONDAY_9AM.replace(hour=10, minute=0)
    busy_end = MONDAY_9AM.replace(hour=11, minute=0)
    slots = generate_slots(MONDAY_9AM, busy=[(busy_start, busy_end)])
    assert not any(busy_start <= s.start < busy_end for s in slots)


def test_slots_stay_within_max_days_window():
    slots = generate_slots(MONDAY_9AM, busy=[])
    latest_allowed = MONDAY_9AM + timedelta(days=MAX_DAYS)
    assert all(s.start < latest_allowed for s in slots)


def test_buffer_pushes_first_slot_later_same_day():
    late_morning = MONDAY_9AM.replace(hour=15, minute=0)  # 15:00, so buffer (17:00) skips most of today
    slots = generate_slots(late_morning, busy=[])
    today_slots = [s for s in slots if s.start.date() == late_morning.date()]
    assert all(s.start.hour >= 17 for s in today_slots)


def test_parse_calendar_boundary_anchors_all_day_events_to_ist_midnight():
    """Reproduces an audit finding: a Google Calendar all-day event reports
    its bounds as a bare "date" (e.g. "2026-07-27"), no time or offset.
    fromisoformat() on that produces a naive datetime, and calling
    .astimezone(IST) directly on a naive datetime interprets it as the
    SERVER's local system timezone first -- not literally IST midnight. On
    a UTC-default server (Docker/Railway), that used to shift a vet's
    full-day block by 5.5 hours."""
    result = _parse_calendar_boundary("2026-07-27")

    assert result == datetime(2026, 7, 27, 0, 0, tzinfo=IST)
    assert result.utcoffset() == timedelta(hours=5, minutes=30)


def test_parse_calendar_boundary_converts_timed_events_to_ist():
    result = _parse_calendar_boundary("2026-07-27T10:00:00Z")

    assert result == datetime(2026, 7, 27, 15, 30, tzinfo=IST)


@pytest.mark.asyncio
async def test_compute_doctor_slots_blocks_the_whole_day_for_an_all_day_event(monkeypatch):
    """End-to-end version of the fix above: a vet's all-day "day off" event
    must block every slot that day, regardless of the server's system
    timezone."""
    from app.availability import slots as slots_module

    tuesday_9am = MONDAY_9AM + timedelta(days=1)
    day_str = tuesday_9am.date().isoformat()
    next_day_str = (tuesday_9am.date() + timedelta(days=1)).isoformat()

    async def fake_list_busy_events(settings, time_min, time_max):
        return [{"start": {"date": day_str}, "end": {"date": next_day_str}}]

    monkeypatch.setattr(slots_module.google_calendar, "list_busy_events", fake_list_busy_events)

    result = await compute_doctor_slots(settings=None, now=tuesday_9am)

    assert not any(s.start.date() == tuesday_9am.date() for s in result)
