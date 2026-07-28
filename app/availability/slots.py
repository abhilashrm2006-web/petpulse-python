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


def generate_slots(
    now: datetime,
    busy: list[tuple[datetime, datetime]],
    day_start_half_hour: int = DAY_START_HALF_HOUR,
    day_end_half_hour: int = DAY_END_HALF_HOUR,
) -> list[Slot]:
    """Pure slot-generation algorithm, independent of the Google Calendar
    fetch — split out so it can be unit-tested without mocking network
    calls. day_start/end_half_hour default to the global window; pass a
    doctor's own opening/closing hours (see compute_doctor_slots) to
    override it per-doctor without changing behavior for anyone who hasn't
    set hours."""
    now = now.astimezone(IST)
    buffer_start = now + timedelta(hours=BUFFER_HOURS)

    slots: list[Slot] = []
    for day_offset in range(MAX_DAYS):
        day = (now + timedelta(days=day_offset)).replace(hour=0, minute=0, second=0, microsecond=0)
        if day.weekday() == 6:  # Sunday
            continue

        day_slots: list[Slot] = []
        for half_hour in range(day_start_half_hour, day_end_half_hour):
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


def _time_to_half_hour(raw: str) -> int:
    """Postgres `time` columns come back from PostgREST as "HH:MM:SS" (or
    occasionally "HH:MM"). Converts to the same half-hour-index unit
    generate_slots already works in."""
    parts = raw.split(":")
    hour, minute = int(parts[0]), int(parts[1])
    return hour * 2 + (1 if minute >= 30 else 0)


def _doctor_hours(client, doctor_phone: str) -> tuple[int, int] | None:
    """A doctor's own opening/closing time overrides the global default
    window -- null on either field (the common case: no hours set yet)
    means "use the existing default," not "closed all day.\""""
    if client is None or not doctor_phone:
        return None
    rows = (
        client.table("profiles").select("opening_time, closing_time").eq("phone_number", doctor_phone)
        .eq("role", "vet").limit(1).execute().data
    )
    if not rows or not rows[0].get("opening_time") or not rows[0].get("closing_time"):
        return None
    return _time_to_half_hour(rows[0]["opening_time"]), _time_to_half_hour(rows[0]["closing_time"])


async def is_window_free(settings: Settings, start: datetime, end: datetime) -> bool:
    """Last-moment recheck of the shared calendar's busy status for exactly
    this [start, end) window -- used immediately before actually creating a
    booking event. The slot list a customer picked from (compute_doctor_slots)
    can be stale by the time booking actually finalizes: real time passes
    between showing that list and the moment of finalizing (waiting on a
    Razorpay payment, a recording-consent reply, a doctor's accept), and
    nothing re-verified freshness at the actual moment of creating the
    calendar event -- so two different sessions picking the same open slot
    could both finalize and double-book. This doesn't need to be doctor-scoped
    the same way compute_doctor_slots is, since every booking (any doctor)
    shares the one "primary" calendar as the source of truth for what's busy."""
    busy_raw = await google_calendar.list_busy_events(settings, start, end)
    for event in busy_raw:
        start_raw = event.get("start", {}).get("dateTime") or event.get("start", {}).get("date")
        end_raw = event.get("end", {}).get("dateTime") or event.get("end", {}).get("date")
        if not start_raw or not end_raw:
            continue
        busy_start, busy_end = _parse_calendar_boundary(start_raw), _parse_calendar_boundary(end_raw)
        if _overlaps(start, end, busy_start, busy_end):
            return False
    return True


async def compute_doctor_slots(
    settings: Settings, now: datetime | None = None, client=None, doctor_phone: str | None = None
) -> list[Slot]:
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

    hours = _doctor_hours(client, doctor_phone) if doctor_phone else None
    if hours:
        return generate_slots(now, busy, day_start_half_hour=hours[0], day_end_half_hour=hours[1])
    return generate_slots(now, busy)
