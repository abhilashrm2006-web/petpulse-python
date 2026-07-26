"""Ports `CustResp - Compute Doctor Slots` — the only real
availability-computation logic in the whole system (spec §2). Lived inline
in Core Engine, not in the misleadingly-named "Find Available Slots"
sub-workflow, which is purely a doctor-catalogue presenter.

Algorithm: next 4 days (skipping Sundays), 30-min slots 10:00-17:30 IST,
max 3 per day, starting no earlier than now+2h, skipping anything that
overlaps a busy Google Calendar event on the single shared "primary"
calendar.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from app.config import Settings
from app.integrations import google_calendar

IST_OFFSET = timedelta(minutes=330)
IST = timezone(IST_OFFSET)
MAX_PER_DAY = 3
MAX_DAYS = 4
DAY_START_HALF_HOUR = 20  # 10:00 IST
DAY_END_HALF_HOUR = 35  # 17:30 IST (exclusive upper bound for slot start)
BUFFER_HOURS = 2


@dataclass
class Slot:
    start: datetime
    end: datetime

    def to_iso(self) -> str:
        return self.start.isoformat()

    def label(self) -> str:
        return self.start.strftime("%a %d %b, %I:%M %p")


def _half_hour_to_time(day: datetime, half_hour: int) -> datetime:
    total_minutes = half_hour * 30
    return day.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(minutes=total_minutes)


def _overlaps(slot_start: datetime, slot_end: datetime, busy_start: datetime, busy_end: datetime) -> bool:
    return slot_start < busy_end and slot_end > busy_start


def generate_slots(now: datetime, busy: list[tuple[datetime, datetime]]) -> list[Slot]:
    """Pure slot-generation algorithm, independent of the Google Calendar
    fetch — split out so it can be unit-tested without mocking network
    calls."""
    now = now.astimezone(IST)
    buffer_start = now + timedelta(hours=BUFFER_HOURS)

    slots: list[Slot] = []
    for day_offset in range(MAX_DAYS):
        day = (now + timedelta(days=day_offset)).replace(hour=0, minute=0, second=0, microsecond=0)
        if day.weekday() == 6:  # Sunday
            continue

        day_slots: list[Slot] = []
        for half_hour in range(DAY_START_HALF_HOUR, DAY_END_HALF_HOUR):
            if len(day_slots) >= MAX_PER_DAY:
                break
            slot_start = _half_hour_to_time(day, half_hour)
            slot_end = slot_start + timedelta(minutes=30)
            if slot_start < buffer_start:
                continue
            if any(_overlaps(slot_start, slot_end, b_start, b_end) for b_start, b_end in busy):
                continue
            day_slots.append(Slot(start=slot_start, end=slot_end))

        slots.extend(day_slots)

    return slots


def _parse_calendar_boundary(raw: str) -> datetime:
    """A timed event's "dateTime" always carries a real UTC offset, so
    .astimezone(IST) converts it correctly. An all-day event's "date" (e.g.
    "2026-07-27") has no time or offset at all -- fromisoformat produces a
    naive datetime, and .astimezone() on a naive datetime assumes the
    server's LOCAL system timezone before converting, not literally IST
    midnight. On a server whose system tz isn't IST (Docker/Railway default
    to UTC), that silently shifts a vet's full-day block by hours. Attach
    IST directly instead of letting astimezone() guess."""
    parsed = datetime.fromisoformat(raw)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=IST)
    return parsed.astimezone(IST)


async def compute_doctor_slots(settings: Settings, now: datetime | None = None) -> list[Slot]:
    now = (now or datetime.now(tz=timezone.utc)).astimezone(IST)
    window_end = now + timedelta(days=MAX_DAYS)

    busy_raw = await google_calendar.list_busy_events(settings, now, window_end)
    busy: list[tuple[datetime, datetime]] = []
    for event in busy_raw:
        start_raw = event.get("start", {}).get("dateTime") or event.get("start", {}).get("date")
        end_raw = event.get("end", {}).get("dateTime") or event.get("end", {}).get("date")
        if not start_raw or not end_raw:
            continue
        busy.append((_parse_calendar_boundary(start_raw), _parse_calendar_boundary(end_raw)))

    return generate_slots(now, busy)
